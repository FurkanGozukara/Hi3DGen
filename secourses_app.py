import gradio as gr 
import os 
os .environ ['SPCONV_ALGO']='native'
from typing import *
import traceback 
import datetime 
import shutil 
import torch 
import numpy as np 
from hi3dgen .pipelines import Hi3DGenPipeline 
import tempfile 
import hf_transfer 
import trimesh 
import open3d as o3d 
import xatlas 
import argparse 
import threading 
from logic .auto_save import auto_save_generation ,open_outputs_folder 
from logic .parameter_info import format_parameter_info_html 
from logic .processing_core import ProcessingCore ,ProcessingParameters 
from logic .batch_processing import create_batch_processor ,BatchSettings 
from logic .cancellation import (
get_cancellation_manager ,request_cancellation ,get_status_summary ,
start_single_processing ,finish_processing ,should_cancel ,
add_cleanup_callback ,remove_cleanup_callback 
)
from logic .preset_manager import (
save_preset_from_ui ,load_preset_for_ui ,delete_preset_from_ui ,
initialize_presets_for_ui ,get_preset_choices_for_ui ,
validate_preset_name ,get_preset_status_for_ui 
)
from logic .preset_file_manager import initialize_preset_file_system
from logic .preset_validation import validate_preset_for_ui ,validate_ui_parameters

try :
    from logic .system_status import get_status_summary as get_comprehensive_status_summary ,print_system_status 
    SYSTEM_STATUS_AVAILABLE =True 
except ImportError as e :
    print (f"⚠️ System status module not available: {e}")
    SYSTEM_STATUS_AVAILABLE =False 

    def get_comprehensive_status_summary (*args ,**kwargs ):
        return "🔍 System Status: Basic monitoring only (system_status module unavailable)"
    def print_system_status (*args ,**kwargs ):
        print ("⚠️ Advanced system status unavailable")

MAX_SEED =np .iinfo (np .int32 ).max 
TMP_DIR =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'tmp')
WEIGHTS_DIR =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'weights')
os .makedirs (TMP_DIR ,exist_ok =True )
os .makedirs (WEIGHTS_DIR ,exist_ok =True )

hi3dgen_pipeline =None 
normal_predictor =None 
global_cached_paths =None 

processing_core =None 
batch_processor =None 
cancellation_manager =None 

def cache_weights (weights_dir :str )->dict :
    import os 
    import sys 
    from huggingface_hub import list_repo_files ,hf_hub_download 
    from pathlib import Path 

    os .environ ["HF_HUB_ENABLE_HF_TRANSFER"]="1"

    model_ids =[
    "Stable-X/trellis-normal-v0-1","Stable-X/yoso-normal-v1-8-1","ZhengPeng7/BiRefNet"
    ]
    cached_paths ={}

    for repo_id in model_ids :
        local_repo_root =Path (weights_dir )/f"models--{repo_id.replace('/', '--')}"
        print (f"Processing {repo_id} into {local_repo_root}...")
        sys .stdout .flush ()
        local_repo_root .mkdir (parents =True ,exist_ok =True )

        try :

            files_in_repo =list_repo_files (repo_id =repo_id ,repo_type ="model")
            num_total_files =len (files_in_repo )
            print (f"  Found {num_total_files} files. Starting download/verification...")
            sys .stdout .flush ()

            for i ,filename_in_repo in enumerate (files_in_repo ):

                print (f"  [{i+1}/{num_total_files}] Processing: {filename_in_repo}")
                sys .stdout .flush ()
                try :

                    hf_hub_download (
                    repo_id =repo_id ,
                    filename =filename_in_repo ,
                    repo_type ="model",
                    local_dir =str (local_repo_root ),
                    local_dir_use_symlinks =False ,
                    force_download =False ,
                    )
                except Exception as file_e :

                    pass 

            cached_paths [repo_id ]=str (local_repo_root )
            print (f"  Finished processing {repo_id}.")
        except Exception as repo_e :
            print (f"  ERROR processing repository {repo_id}: {repo_e}")
        sys .stdout .flush ()

    return cached_paths 

def preprocess_mesh (mesh_prompt ):
    print ("Processing mesh")
    trimesh_mesh =trimesh .load_mesh (mesh_prompt )
    trimesh_mesh .export (mesh_prompt +'.glb')
    return mesh_prompt +'.glb'

def preprocess_image (image ):
    global hi3dgen_pipeline 
    if image is None :return None 

    if hi3dgen_pipeline is None :
        raise RuntimeError ("FATAL: Hi3DGenPipeline not loaded. Cannot preprocess.")

    try :
        return hi3dgen_pipeline .preprocess_image (image ,resolution =1024 )
    except Exception as e :

        if "BiRefNet"in str (e )or "401"in str (e )or "RepositoryNotFoundError"in str (e ):
            print (f"BiRefNet error detected during preprocess: {e}")
            print ("Attempting to reset BiRefNet state and retry...")
            try :
                if hasattr (hi3dgen_pipeline ,'reset_birefnet_state'):
                    hi3dgen_pipeline .reset_birefnet_state ()

                return hi3dgen_pipeline .preprocess_image (image ,resolution =1024 )
            except Exception as retry_e :
                print (f"Retry after BiRefNet reset also failed: {retry_e}")
                raise retry_e 
        else :

            raise e 

def simplify_mesh_open3d (in_mesh :trimesh .Trimesh ,
poly_count_pcnt :float =0.5 )->trimesh .Trimesh :

    if not isinstance (in_mesh ,trimesh .Trimesh ):
        print ("Simplify ERR: Invalid input type.")
        return in_mesh 

    if not (0.0 <poly_count_pcnt <1.0 ):
        print (f"Simplify skip: poly_count_pcnt {poly_count_pcnt:.2f} out of (0,1) range.")
        return in_mesh 

    current_tris =len (in_mesh .faces )
    if current_tris ==0 :return in_mesh 

    target_tris =int (current_tris *poly_count_pcnt )
    target_tris =max (1 ,target_tris )

    if target_tris >=current_tris :
        print (f"Simplify skip: Target {target_tris} >= current {current_tris}.")
        return in_mesh 

    print (f"Simplifying: {current_tris} faces -> ~{target_tris} faces ({ (1.0-poly_count_pcnt)*100:.0f}% original).")

    o3d_m =o3d .geometry .TriangleMesh ()
    o3d_m .vertices =o3d .utility .Vector3dVector (in_mesh .vertices )
    o3d_m .triangles =o3d .utility .Vector3iVector (in_mesh .faces )

    try :

        simplified_o3d_m =o3d_m .simplify_quadric_decimation (target_number_of_triangles =target_tris )

        s_verts =np .asarray (simplified_o3d_m .vertices )
        s_faces =np .asarray (simplified_o3d_m .triangles )

        if s_faces .size ==0 and current_tris >0 :
            print ("Simplify WARN: Empty mesh result. Reverting.")
            return in_mesh 

        return trimesh .Trimesh (vertices =s_verts ,faces =s_faces ,process =True )
    except Exception as e :
        print (f"Simplify ERR: Open3D failed ({e}). Reverting.")
        traceback .print_exc ()
        return in_mesh 

def unwrap_mesh_with_xatlas (input_mesh :trimesh .Trimesh ,
max_cost_param :float =8.0 ,
normal_seam_weight_param :float =1.0 ,
resolution_param :int =1024 ,
padding_param :int =2 )->trimesh .Trimesh :

    import time 
    import sys 

    print ("UV Unwrapping with xatlas: Starting process...")
    start_time =time .time ()

    input_vertices_orig =input_mesh .vertices .astype (np .float32 )
    input_faces_orig =input_mesh .faces .astype (np .uint32 )
    vertex_normals_from_trimesh =input_mesh .vertex_normals 
    input_normals_orig =np .ascontiguousarray (vertex_normals_from_trimesh ,dtype =np .float32 )

    print (f"  Input mesh: {input_vertices_orig.shape[0]} vertices, {input_faces_orig.shape[0]} faces.")

    face_count =input_faces_orig .shape [0 ]

    min_time_estimate =face_count /50000 
    max_time_estimate =face_count /15000 
    avg_time_estimate =face_count /25000 

    print (f"  Expected processing time: {min_time_estimate:.1f}-{max_time_estimate:.1f}s (avg: {avg_time_estimate:.1f}s)")
    print (f"  Complexity level: {'Low' if face_count < 100000 else 'Medium' if face_count < 500000 else 'High'} ({face_count:,} faces)")
    sys .stdout .flush ()

    print ("  Phase 1/4: Setting up xatlas Atlas...")
    phase_start =time .time ()
    atlas =xatlas .Atlas ()
    atlas .add_mesh (input_vertices_orig ,input_faces_orig ,input_normals_orig )
    print (f"  Phase 1/4: Complete ({time.time() - phase_start:.2f}s)")
    sys .stdout .flush ()

    print ("  Phase 2/4: Configuring xatlas options...")
    phase_start =time .time ()
    chart_options =xatlas .ChartOptions ()
    chart_options .max_cost =max_cost_param 
    chart_options .normal_seam_weight =normal_seam_weight_param 

    pack_options =xatlas .PackOptions ()
    pack_options .resolution =resolution_param 
    pack_options .padding =padding_param 
    print (f"  Phase 2/4: Complete ({time.time() - phase_start:.2f}s)")
    print (f"    -> ChartOptions: max_cost={chart_options.max_cost:.2f}, normal_seam_weight={chart_options.normal_seam_weight:.2f}")
    print (f"    -> PackOptions: resolution={pack_options.resolution}, padding={pack_options.padding}")
    sys .stdout .flush ()

    print ("  Phase 3/4: Running xatlas.generate() - This is the time-consuming step...")
    print ("    -> Processing mesh charts and UV packing...")
    print (f"    -> This may take some time so patiently wait for {face_count:,} faces")
    print ("    -> xatlas will now run without progress updates (this is normal)")
    print ("    -> Please wait... processing is happening in the background")
    sys .stdout .flush ()

    generation_start =time .time ()

    try :
        print ("    -> Starting xatlas.generate()...")
        sys .stdout .flush ()
        atlas .generate (chart_options =chart_options ,pack_options =pack_options )
        generation_time =time .time ()-generation_start 
        actual_performance =face_count /generation_time if generation_time >0 else 0 
        print (f"    -> ✓ xatlas.generate() completed successfully in {generation_time:.2f}s")
        print (f"    -> Actual performance: {actual_performance:.0f} faces/second")
    except Exception as e :
        generation_time =time .time ()-generation_start 
        print (f"    -> ✗ xatlas.generate() failed after {generation_time:.2f}s: {e}")
        raise 

    print (f"  Phase 3/4: Complete ({generation_time:.2f}s)")
    print (f"    -> xatlas generated atlas with dimensions: width={atlas.width}, height={atlas.height}")
    sys .stdout .flush ()

    print ("  Phase 4/4: Processing xatlas results...")
    phase_start =time .time ()

    v_out_xref_data ,f_out_indices ,uv_coords_from_xatlas =atlas .get_mesh (0 )
    print (f"    -> Retrieved mesh data: {uv_coords_from_xatlas.shape[0]} UV vertices, {f_out_indices.shape[0]} faces")

    num_new_vertices =uv_coords_from_xatlas .shape [0 ]
    if v_out_xref_data .shape ==(num_new_vertices ,):
        xref_indices =v_out_xref_data .astype (np .uint32 )
        if np .any (xref_indices >=input_vertices_orig .shape [0 ])or np .any (xref_indices <0 ):
             raise ValueError ("Invalid xref values from xatlas - out of bounds for original input vertices.")
        final_vertices_spatial =input_vertices_orig [xref_indices ]
    elif v_out_xref_data .shape ==(num_new_vertices ,3 ):
        print ("    -> Warning: xatlas.get_mesh() returned 3D vertex data directly, which is unexpected for add_mesh workflow.")
        final_vertices_spatial =v_out_xref_data .astype (np .float32 )
    else :
        raise ValueError (f"Unexpected shape for vertex/xref data from xatlas.get_mesh: {v_out_xref_data.shape}.")

    print (f"    -> Processed vertex mapping: {final_vertices_spatial.shape[0]} final vertices")

    final_uvs =uv_coords_from_xatlas .astype (np .float32 )
    if np .any (final_uvs >1.5 ):
        print ("    -> UVs appear to be in pixel coordinates. Normalizing...")
        if atlas .width >0 and atlas .height >0 :
            final_uvs /=np .array ([atlas .width ,atlas .height ],dtype =np .float32 )
        else :
            print ("    -> WARNING: Atlas width/height is 0, cannot normalize pixel UVs. Using unnormalized.")
    else :
        min_uv =final_uvs .min (axis =0 )if final_uvs .size >0 else "N/A"
        max_uv =final_uvs .max (axis =0 )if final_uvs .size >0 else "N/A"
        print (f"    -> UVs appear to be normalized. Min: {min_uv}, Max: {max_uv}")

    print (f"    -> Creating final Trimesh object...")
    output_mesh =trimesh .Trimesh (vertices =final_vertices_spatial ,faces =f_out_indices ,process =False )

    if final_uvs .shape !=(final_vertices_spatial .shape [0 ],2 ):
        raise ValueError (f"Shape mismatch for final UVs before Trimesh assignment.")

    material =trimesh .visual .material .PBRMaterial (name ='defaultXatlasMat')
    output_mesh .visual =trimesh .visual .TextureVisuals (uv =final_uvs ,material =material )

    if hasattr (output_mesh .visual ,'uv')and output_mesh .visual .uv is not None :
        print (f"    -> Trimesh object successfully created with UVs, Shape: {output_mesh.visual.uv.shape}")
    else :
        print ("    -> ERROR: Trimesh object does NOT have UVs assigned after TextureVisuals call.")
        raise RuntimeError ("Failed to assign UVs to the Trimesh object.")

    print (f"  Phase 4/4: Complete ({time.time() - phase_start:.2f}s)")

    total_time =time .time ()-start_time 
    print (f"UV Unwrapping with xatlas: Process complete! Total time: {total_time:.2f}s")
    print (f"  -> Performance: {input_faces_orig.shape[0] / total_time:.0f} faces/second")
    print (f"  -> Efficiency: {(avg_time_estimate / total_time * 100):.1f}% of estimated average time")
    sys .stdout .flush ()

    return output_mesh 

def generate_3d (image ,seed =-1 ,
ss_guidance_strength =3 ,ss_sampling_steps =50 ,
slat_guidance_strength =3 ,slat_sampling_steps =6 ,
poly_count_pcnt :float =0.5 ,
xatlas_max_cost :float =8.0 ,
xatlas_normal_seam_weight :float =1.0 ,
xatlas_resolution :int =1024 ,
xatlas_padding :int =2 ,
normal_map_resolution :int =768 ,
normal_match_input_resolution :bool =True ,
auto_save_obj :bool =True ,
auto_save_glb :bool =True ,
auto_save_ply :bool =True ,
auto_save_stl :bool =True ):

    if image is None :
        print ("Input image is None. Aborting generation.")
        return None ,None ,None 
    if seed ==-1 :seed =np .random .randint (0 ,MAX_SEED )

    if hi3dgen_pipeline is None :
        print ("FATAL: Hi3DGenPipeline not loaded. Cannot generate 3D.")
        return None ,None ,None 

    normal_image_pil =None 
    gradio_model_path =None 

    current_normal_predictor_instance =None 
    try :
        print ("Normal Prediction: Loading model from local cache...")

        yoso_model_dir_name ="models--Stable-X--yoso-normal-v1-8-1"
        yoso_local_path =os .path .join (WEIGHTS_DIR ,yoso_model_dir_name )

        if not os .path .exists (yoso_local_path ):
            print (f"ERROR: YOSO model not found at {yoso_local_path}")
            raise FileNotFoundError (f"YOSO model not found at {yoso_local_path}. Please ensure models are downloaded.")

        birefnet_local_path =os .path .join (WEIGHTS_DIR ,"models--ZhengPeng7--BiRefNet")
        if not os .path .exists (birefnet_local_path ):
            print (f"ERROR: BiRefNet model not found at {birefnet_local_path}")

            raise FileNotFoundError (f"BiRefNet model not found at {birefnet_local_path}. Please ensure models are downloaded.")

        from pathlib import Path 
        import shutil 

        hf_cache_dir =os .path .expanduser ("~/.cache/huggingface/hub")
        os .makedirs (hf_cache_dir ,exist_ok =True )

        model_mappings ={
        "models--Stable-X--yoso-normal-v1-8-1":"models--Stable-X--yoso-normal-v1-8-1",
        "models--ZhengPeng7--BiRefNet":"models--ZhengPeng7--BiRefNet"
        }

        for src_name ,dst_folder_name in model_mappings .items ():
            src_path_in_weights =os .path .join (WEIGHTS_DIR ,src_name )
            dst_path_in_hf_cache =os .path .join (hf_cache_dir ,dst_folder_name )

            if os .path .exists (src_path_in_weights )and not os .path .exists (dst_path_in_hf_cache ):
                print (f"Copying {src_name} from {WEIGHTS_DIR} to HuggingFace cache at {dst_path_in_hf_cache}...")
                shutil .copytree (src_path_in_weights ,dst_path_in_hf_cache )

        yoso_model_abs_path =os .path .abspath (yoso_local_path )
        print (f"Normal Prediction: Attempting to load YOSO model using absolute local path: {yoso_model_abs_path}")

        current_normal_predictor_instance =torch .hub .load (
        "hugoycj/StableNormal","StableNormal_turbo",trust_repo =True ,
        yoso_version =yoso_model_abs_path 
        )

        print ("Normal Prediction: Generating normal map...")
        normal_image_pil =current_normal_predictor_instance (image ,resolution =normal_map_resolution ,match_input_resolution =normal_match_input_resolution ,data_type ='object')
    except Exception as e :
        print (f"ERROR in Normal Prediction stage: {e}")
        traceback .print_exc ()
    finally :
        if current_normal_predictor_instance is not None :
            print ("Normal Prediction: Unloading model...")
            try :

                if torch .cuda .is_available ():
                    torch .cuda .synchronize ()

                if hasattr (current_normal_predictor_instance ,'model')and hasattr (current_normal_predictor_instance .model ,'cpu'):
                    current_normal_predictor_instance .model .cpu ()

                del current_normal_predictor_instance 

                if torch .cuda .is_available ():
                    torch .cuda .empty_cache ()

                print ("Normal Prediction: Model unloaded successfully")

            except RuntimeError as cuda_error :
                if "CUDA"in str (cuda_error ):
                    print (f"Normal Prediction: CUDA error during cleanup: {cuda_error}")
                    print ("Normal Prediction: Attempting force cleanup...")
                    try :

                        if current_normal_predictor_instance is not None :
                            del current_normal_predictor_instance 

                        if torch .cuda .is_available ():
                            torch .cuda .empty_cache ()
                            torch .cuda .synchronize ()

                        print ("Normal Prediction: Force cleanup completed")
                    except Exception as force_error :
                        print (f"Normal Prediction: Force cleanup also failed: {force_error}")
                else :
                    print (f"Normal Prediction: Non-CUDA error during cleanup: {cuda_error}")
            except Exception as e :
                print (f"Normal Prediction: Unexpected error during cleanup: {e}")

                if 'current_normal_predictor_instance'in locals ():
                    try :
                        del current_normal_predictor_instance 
                    except :
                        pass 

                try :
                    if torch .cuda .is_available ():
                        torch .cuda .empty_cache ()
                except :
                    pass 

    if normal_image_pil is None :
        print ("ERROR: Normal map not generated after Stage 1. Aborting 3D generation.")
        return None ,None ,None 

    pipeline_on_gpu =False 
    try :
        import time 
        import sys 
        generation_start_time =time .time ()

        if torch .cuda .is_available ():
            print ("3D Generation: Moving Hi3DGen pipeline to GPU...")
            hi3dgen_pipeline .cuda ();pipeline_on_gpu =True 

        print ("3D Generation: Running Hi3DGen pipeline...")
        pipeline_start =time .time ()
        outputs =hi3dgen_pipeline .run (
        normal_image_pil ,seed =seed ,formats =["mesh",],preprocess_image =False ,
        sparse_structure_sampler_params ={"steps":ss_sampling_steps ,"cfg_strength":ss_guidance_strength },
        slat_sampler_params ={"steps":slat_sampling_steps ,"cfg_strength":slat_guidance_strength },
        )
        pipeline_time =time .time ()-pipeline_start 
        print (f"3D Generation: Hi3DGen pipeline completed in {pipeline_time:.2f}s")
        sys .stdout .flush ()

        timestamp =datetime .datetime .now ().strftime ('%Y%m%d%H%M%S')
        output_dir =os .path .join (TMP_DIR ,timestamp )
        os .makedirs (output_dir ,exist_ok =True )

        mesh_path_glb =os .path .join (output_dir ,"mesh.glb")

        print ("Mesh Processing: Converting to Trimesh and simplifying...")
        mesh_start =time .time ()
        raw_mesh_trimesh =outputs ['mesh'][0 ].to_trimesh (transform_pose =True )
        mesh_for_uv_unwrap =simplify_mesh_open3d (raw_mesh_trimesh ,poly_count_pcnt )
        mesh_processing_time =time .time ()-mesh_start 
        print (f"Mesh Processing: Completed in {mesh_processing_time:.2f}s")
        sys .stdout .flush ()

        unwrapped_mesh_trimesh =unwrap_mesh_with_xatlas (mesh_for_uv_unwrap ,
        max_cost_param =xatlas_max_cost ,
        normal_seam_weight_param =xatlas_normal_seam_weight ,
        resolution_param =xatlas_resolution ,
        padding_param =xatlas_padding )

        print (f"File Export: Exporting GLB to {mesh_path_glb}...")
        export_start =time .time ()
        unwrapped_mesh_trimesh .export (mesh_path_glb )
        export_time =time .time ()-export_start 
        print (f"File Export: GLB exported successfully in {export_time:.2f}s")

        total_generation_time =time .time ()-generation_start_time 
        print (f"=== GENERATION COMPLETE ===")
        print (f"Total time breakdown:")
        print (f"  - Hi3DGen pipeline: {pipeline_time:.2f}s ({pipeline_time/total_generation_time*100:.1f}%)")
        print (f"  - Mesh processing: {mesh_processing_time:.2f}s ({mesh_processing_time/total_generation_time*100:.1f}%)")
        print (f"  - UV Unwrapping: (see detailed breakdown above)")
        print (f"  - File export: {export_time:.2f}s ({export_time/total_generation_time*100:.1f}%)")
        print (f"  - TOTAL: {total_generation_time:.2f}s")
        sys .stdout .flush ()

        gradio_model_path =mesh_path_glb 

    except Exception as e :
        print (f"ERROR in 3D Generation or UV Unwrapping stage: {e}")
        traceback .print_exc ()
        gradio_model_path =None 
    finally :
        if pipeline_on_gpu :
            print ("3D Generation: Moving Hi3DGen pipeline to CPU...")
            try :

                if torch .cuda .is_available ():
                    torch .cuda .synchronize ()

                hi3dgen_pipeline .cpu ()

                if torch .cuda .is_available ():
                    torch .cuda .empty_cache ()

                print ("3D Generation: Successfully moved Hi3DGen pipeline to CPU")

            except RuntimeError as cuda_error :
                if "CUDA"in str (cuda_error ):
                    print (f"3D Generation: CUDA error during pipeline cleanup: {cuda_error}")
                    print ("3D Generation: Attempting force cleanup...")
                    try :

                        if torch .cuda .is_available ():
                            torch .cuda .empty_cache ()
                            torch .cuda .synchronize ()
                        print ("3D Generation: Force cleanup completed")
                    except Exception as force_error :
                        print (f"3D Generation: Force cleanup also failed: {force_error}")
                else :
                    print (f"3D Generation: Non-CUDA error during cleanup: {cuda_error}")
            except Exception as e :
                print (f"3D Generation: Unexpected error during pipeline cleanup: {e}")

                try :
                    if torch .cuda .is_available ():
                        torch .cuda .empty_cache ()
                except :
                    pass 

    if gradio_model_path and (auto_save_obj or auto_save_glb or auto_save_ply or auto_save_stl ):
        try :
            enabled_formats ={
            "obj":auto_save_obj ,
            "glb":auto_save_glb ,
            "ply":auto_save_ply ,
            "stl":auto_save_stl 
            }

            print ("Starting auto-save process...")
            auto_save_result =auto_save_generation (
            mesh_path =gradio_model_path ,
            normal_image =normal_image_pil ,
            enabled_formats =enabled_formats 
            )

            if auto_save_result :
                print (f"✓ Auto-save successful: Saved to folder {auto_save_result['folder_number']}")
                print (f"  Saved {len(auto_save_result['saved_files'])} files")
            else :
                print ("✗ Auto-save failed")

        except Exception as e :
            print (f"Auto-save error: {e}")
            traceback .print_exc ()

    return normal_image_pil ,gradio_model_path ,gradio_model_path 

def generate_3d_with_cancellation (image ,seed =-1 ,
ss_guidance_strength =3 ,ss_sampling_steps =50 ,
slat_guidance_strength =3 ,slat_sampling_steps =6 ,
poly_count_pcnt :float =0.5 ,
xatlas_max_cost :float =8.0 ,
xatlas_normal_seam_weight :float =1.0 ,
xatlas_resolution :int =1024 ,
xatlas_padding :int =2 ,
normal_map_resolution :int =768 ,
normal_match_input_resolution :bool =True ,
auto_save_obj :bool =True ,
auto_save_glb :bool =True ,
auto_save_ply :bool =True ,
auto_save_stl :bool =True ,
progress =gr .Progress ()):

    global processing_core ,hi3dgen_pipeline 

    def reset_birefnet_state ():

        try :
            if hi3dgen_pipeline and hasattr (hi3dgen_pipeline ,'reset_birefnet_state'):
                print ("Cleanup: Resetting BiRefNet model state...")
                hi3dgen_pipeline .reset_birefnet_state ()
                print ("Cleanup: BiRefNet model state reset successfully")
        except Exception as e :
            print (f"Cleanup: Error resetting BiRefNet state: {e}")

    if not processing_core :
        print ("Error: ProcessingCore not initialized, falling back to legacy processing")
        return generate_3d (image ,seed ,ss_guidance_strength ,ss_sampling_steps ,
        slat_guidance_strength ,slat_sampling_steps ,poly_count_pcnt ,
        xatlas_max_cost ,xatlas_normal_seam_weight ,xatlas_resolution ,xatlas_padding ,
        normal_map_resolution ,normal_match_input_resolution ,
        auto_save_obj ,auto_save_glb ,auto_save_ply ,auto_save_stl )

    try :

        start_single_processing ("Starting single image processing")

        add_cleanup_callback (reset_birefnet_state )

        def single_progress_callback (stage :str ,details :str =""):

            try :

                stage_progress ={
                "Initialization":0.05 ,
                "Normal Prediction":0.25 ,
                "3D Generation":0.60 ,
                "Mesh Processing":0.75 ,
                "UV Unwrapping":0.85 ,
                "File Export":0.95 ,
                "Auto-save":0.98 ,
                "Completed":1.0 
                }

                current_progress =0.0 
                for stage_name ,prog in stage_progress .items ():
                    if stage_name .lower ()in stage .lower ():
                        current_progress =prog 
                        break 

                desc =f"{stage}: {details}"if details else stage 
                progress (current_progress ,desc =desc )

                print (f"Single Processing Progress: {desc} ({current_progress*100:.1f}%)")

            except Exception as e :
                print (f"Error in single progress callback: {e}")

        progress (0.0 ,desc ="Starting single image processing...")

        params =ProcessingParameters (
        seed =seed ,
        ss_guidance_strength =ss_guidance_strength ,
        ss_sampling_steps =ss_sampling_steps ,
        slat_guidance_strength =slat_guidance_strength ,
        slat_sampling_steps =slat_sampling_steps ,
        poly_count_pcnt =poly_count_pcnt ,
        xatlas_max_cost =xatlas_max_cost ,
        xatlas_normal_seam_weight =xatlas_normal_seam_weight ,
        xatlas_resolution =xatlas_resolution ,
        xatlas_padding =xatlas_padding ,
        normal_map_resolution =normal_map_resolution ,
        normal_match_input_resolution =normal_match_input_resolution ,
        auto_save_obj =auto_save_obj ,
        auto_save_glb =auto_save_glb ,
        auto_save_ply =auto_save_ply ,
        auto_save_stl =auto_save_stl 
        )

        result =processing_core .process_single_image (image ,params ,single_progress_callback )

        if result .success :
            progress (1.0 ,desc =f"✅ Completed successfully in {result.total_time:.1f}s")
        else :
            progress (0.0 ,desc =f"❌ Failed: {result.error_message}")

        finish_processing ("Single processing completed"if result .success else "Single processing failed")

        if result .success :
            print (f"Single processing completed successfully: {result.get_summary()}")
            return result .normal_image ,result .mesh_path ,result .mesh_path 
        else :
            print (f"Processing failed: {result.error_message}")
            return None ,None ,None 

    except Exception as e :
        progress (0.0 ,desc =f"❌ Error: {str(e)}")
        finish_processing (f"Single processing error: {str(e)}")
        print (f"Error in generate_3d_with_cancellation: {e}")
        import traceback 
        traceback .print_exc ()

        try :
            progress (0.0 ,desc ="🔄 Falling back to legacy processing...")
            result =generate_3d (image ,seed ,ss_guidance_strength ,ss_sampling_steps ,
            slat_guidance_strength ,slat_sampling_steps ,poly_count_pcnt ,
            xatlas_max_cost ,xatlas_normal_seam_weight ,xatlas_resolution ,xatlas_padding ,
            normal_map_resolution ,normal_match_input_resolution ,
            auto_save_obj ,auto_save_glb ,auto_save_ply ,auto_save_stl )
            if result [1 ]is not None :
                progress (1.0 ,desc ="✅ Legacy processing completed")
            else :
                progress (0.0 ,desc ="❌ Legacy processing failed")
            return result 
        except Exception as fallback_e :
            progress (0.0 ,desc =f"❌ All processing failed: {str(fallback_e)}")
            print (f"Fallback processing also failed: {fallback_e}")
            return None ,None ,None 
    finally :

        try :
            remove_cleanup_callback (reset_birefnet_state )
        except :
            pass 

def convert_mesh (mesh_path :str ,export_format :str )->Optional [str ]:

    if not mesh_path or not os .path .exists (mesh_path ):
        print (f"convert_mesh: Invalid input mesh_path: {mesh_path}")
        return None 

    try :

        timestamp =datetime .datetime .now ().strftime ('%Y%m%d_%H%M%S')
        filename =f"mesh_{timestamp}.{export_format.lower()}"
        temp_file_path =os .path .join (TMP_DIR ,filename )

        os .makedirs (os .path .dirname (temp_file_path ),exist_ok =True )

        if export_format .lower ()=="glb"and mesh_path .lower ().endswith (".glb"):
            print (f"convert_mesh: Copying GLB {mesh_path} to {temp_file_path}")
            shutil .copy2 (mesh_path ,temp_file_path )
            return temp_file_path 

        print (f"convert_mesh: Converting {mesh_path} to {export_format} at {temp_file_path}")
        mesh =trimesh .load_mesh (mesh_path )

        if not (hasattr (mesh .visual ,'uv')and mesh .visual .uv is not None ):
            print (f"  Warning: Loaded mesh from {mesh_path} has no UVs before export to {export_format}.")

        mesh .export (temp_file_path ,file_type =export_format .lower ())

        if not os .path .exists (temp_file_path )or os .path .getsize (temp_file_path )==0 :
            raise Exception (f"Failed to create valid {export_format} file")

        print (f"convert_mesh: Successfully created {export_format} file at {temp_file_path} ({os.path.getsize(temp_file_path)} bytes)")
        return temp_file_path 

    except Exception as e :
        print (f"convert_mesh: Error during conversion of '{mesh_path}' to '{export_format}': {e}")
        traceback .print_exc ()
        if 'temp_file_path'in locals ()and temp_file_path and os .path .exists (temp_file_path ):
            try :
                os .remove (temp_file_path )
            except Exception as rm_e :
                print (f"convert_mesh: Error removing temp file {temp_file_path}: {rm_e}")
        return None 

custom_css ="""
footer {visibility: hidden}
#parameter-guide-content {
    width: 100% !important;
    height: 80vh !important;
    max-width: none !important;
    max-height: none !important;
}
#parameter-guide-content > div {
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
}
.parameter-guide-tab {
    width: 100% !important;
    height: 100% !important;
}
/* Batch processing styles */
.batch-status {
    font-family: monospace;
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 4px;
}
.batch-progress {
    margin: 10px 0;
}
.cancel-button {
    background-color: #dc3545 !important;
    border-color: #dc3545 !important;
}
.processing-active {
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}
/* Preset Management Styles */
.preset-management {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    color: white;
}
.preset-dropdown {
    background: white !important;
    color: black !important;
}
.preset-save-btn {
    background-color: #28a745 !important;
    border-color: #28a745 !important;
}
.preset-load-btn {
    background-color: #007bff !important;
    border-color: #007bff !important;
}
.preset-delete-btn {
    background-color: #dc3545 !important;
    border-color: #dc3545 !important;
}
.preset-status-success {
    background-color: #d4edda !important;
    border-color: #c3e6cb !important;
    color: #155724 !important;
}
.preset-status-error {
    background-color: #f8d7da !important;
    border-color: #f5c6cb !important;
    color: #721c24 !important;
}
/* Enhanced progress display styles */
.progress-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.status-success {
    background-color: #d4edda !important;
    border-color: #c3e6cb !important;
    color: #155724 !important;
}
.status-error {
    background-color: #f8d7da !important;
    border-color: #f5c6cb !important;
    color: #721c24 !important;
}
.status-processing {
    background-color: #d1ecf1 !important;
    border-color: #bee5eb !important;
    color: #0c5460 !important;
}
.emoji-status {
    font-size: 1.2em;
    margin-right: 8px;
}
"""

with gr .Blocks (css =custom_css ,theme =gr .themes .Soft ())as demo :
    gr .Markdown (
    "# Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging SECourses App V12 with Auto-Save : https://www.patreon.com/posts/130766890"
    )

    with gr .Row ():
        with gr .Column (scale =1 ):
            with gr .Tabs ():

                with gr .Tab ("Single Image"):
                    with gr .Row ():
                        image_prompt =gr .Image (label ="Image Prompt",image_mode ="RGBA",type ="pil")
                        normal_output =gr .Image (label ="Normal Bridge",image_mode ="RGBA",type ="pil")
                    with gr .Row ():
                        gen_shape_btn =gr .Button ("Generate Shape",size ="lg",variant ="primary")

                with gr .Tab ("📦 Batch Processing"):
                    with gr .Row ():
                        with gr .Column (scale =1 ):
                            gr .Markdown ("### Input Configuration")
                            batch_input_folder =gr .Textbox (
                            label ="Input Folder",
                            placeholder ="Path to folder containing images",
                            info ="Folder containing images to process (jpg, png, bmp, tiff, webp)"
                            )
                            batch_output_folder =gr .Textbox (
                            label ="Output Folder",
                            placeholder ="Path to save generated 3D models",
                            info ="Folder where generated files will be saved"
                            )

                            with gr .Row ():
                                batch_skip_existing =gr .Checkbox (
                                value =True ,
                                label ="Skip Existing Files",
                                info ="Skip images that already have generated output files"
                                )
                                batch_use_current_settings =gr .Checkbox (
                                value =True ,
                                label ="Use Current Settings",
                                info ="Use the advanced settings from the current session"
                                )

                        with gr .Column (scale =1 ):
                            gr .Markdown ("### Batch Control")
                            with gr .Row ():
                                batch_start_btn =gr .Button (
                                "🚀 Start Batch Processing",
                                size ="lg",
                                variant ="primary"
                                )
                                batch_cancel_btn =gr .Button (
                                "⏹️ Cancel",
                                size ="lg",
                                variant ="stop"
                                )

                            batch_progress_bar =gr .Progress ()
                            batch_status_text =gr .Textbox (
                            label ="Status",
                            value ="Ready",
                            interactive =False ,
                            lines =3 
                            )

                            batch_results_text =gr .Textbox (
                            label ="Results Summary",
                            value ="No batch processing started yet",
                            interactive =False ,
                            lines =5 
                            )

                with gr .Tab ("📋 Parameter Guide"):
                    with gr .Row ():
                        with gr .Column (scale =1 ):
                            parameter_info_html =gr .HTML (
                            value =format_parameter_info_html (),
                            show_label =False ,
                            elem_id ="parameter-guide-content"
                            )

            with gr .Accordion ("Advanced Settings",open =True ):
                seed =gr .Slider (-1 ,MAX_SEED ,label ="Seed",value =0 ,step =1 )
                gr .Markdown ("#### Stage 1: Sparse Structure Generation")
                with gr .Row ():
                    ss_guidance_strength =gr .Slider (0.0 ,10.0 ,label ="Guidance Strength",value =3 ,step =0.1 )
                    ss_sampling_steps =gr .Slider (1 ,50 ,label ="Sampling Steps",value =50 ,step =1 )
                gr .Markdown ("#### Stage 2: Structured Latent Generation")
                with gr .Row ():
                    slat_guidance_strength =gr .Slider (0.0 ,10.0 ,label ="Guidance Strength",value =3.0 ,step =0.1 )
                    slat_sampling_steps =gr .Slider (1 ,50 ,label ="Sampling Steps",value =6 ,step =1 )

                gr .Markdown ("#### Mesh Simplification Settings")
                poly_count_slider =gr .Slider (
                minimum =0.05 ,maximum =1.0 ,value =0.5 ,step =0.05 ,
                label ="Polygon Count Percentage",
                info ="Controls polygon retention post-simplification (e.g., 0.5 = 50%). Higher values = more detail, larger files."
                )

                gr .Markdown ("#### UV Unwrapping (xatlas) Settings")
                with gr .Row ():
                    xatlas_max_cost_slider =gr .Slider (
                    minimum =1.0 ,maximum =10.0 ,value =8.0 ,step =0.1 ,
                    label ="Max Chart Cost",
                    info ="xatlas: Higher values allow more stretch/distortion within a UV chart."
                    )
                    xatlas_normal_seam_weight_slider =gr .Slider (
                    minimum =0.1 ,maximum =5.0 ,value =1.0 ,step =0.1 ,
                    label ="Normal Seam Weight",
                    info ="xatlas: Lower values reduce UV seams based on normal changes."
                    )
                with gr .Row ():
                    xatlas_resolution_slider =gr .Slider (
                    minimum =256 ,maximum =4096 ,value =1024 ,step =256 ,
                    label ="UV Atlas Resolution",
                    info ="xatlas: Resolution of the UV texture atlas (e.g., 1024 for 1K)."
                    )
                    xatlas_padding_slider =gr .Slider (
                    minimum =0 ,maximum =16 ,value =2 ,step =1 ,
                    label ="UV Chart Padding",
                    info ="xatlas: Padding in pixels between UV charts in the atlas."
                    )

                gr .Markdown ("#### Normal Map Generation Settings")
                with gr .Row ():
                    normal_map_resolution_slider =gr .Slider (
                    minimum =256 ,maximum =1024 ,value =768 ,step =128 ,
                    label ="Normal Map Resolution",
                    info ="Target resolution for the generated normal map."
                    )
                    normal_match_input_res_checkbox =gr .Checkbox (
                    value =True ,
                    label ="Match Input Resolution for Normals",
                    info ="If input image is smaller than 'Normal Map Resolution', use its resolution."
                    )

        with gr .Column (scale =1 ):

            with gr .Row ():
                universal_cancel_btn =gr .Button (
                "⏹️ Cancel Processing",
                size ="lg",
                variant ="stop",
                visible =False 
                )
                processing_status_text =gr .Textbox (
                label ="Processing Status",
                value ="Idle",
                interactive =False ,
                scale =2 
                )

            with gr .Column ():
                model_output =gr .Model3D (label ="3D Model Preview (Each model is approximately 40MB, may take around 1 minute to load)")
            with gr .Column ():
                export_format =gr .Dropdown (
                choices =["obj","glb","ply","stl"],
                value ="glb",
                label ="File Format"
                )
                download_btn =gr .DownloadButton (label ="Export Mesh",interactive =False )

                open_folder_btn =gr .Button ("📁 Open Outputs Folder",variant ="primary")

                gr .Markdown ("#### Auto-Save Settings")
                with gr .Row ():
                    auto_save_obj_cb =gr .Checkbox (value =True ,label ="Auto-save OBJ",info ="Wavefront OBJ format")
                    auto_save_glb_cb =gr .Checkbox (value =True ,label ="Auto-save GLB",info ="Binary glTF format")
                with gr .Row ():
                    auto_save_ply_cb =gr .Checkbox (value =True ,label ="Auto-save PLY",info ="Stanford Triangle format")
                    auto_save_stl_cb =gr .Checkbox (value =True ,label ="Auto-save STL",info ="Stereolithography format")
                
                gr .Markdown ("#### 📋 Preset Management")
                with gr .Row ():
                    preset_dropdown =gr .Dropdown (
                        choices =["Default"],
                        value ="Default",
                        label ="Current Preset",
                        info ="Select or load a saved preset",
                        interactive =True 
                    )
                    preset_status_text =gr .Textbox (
                        value ="📋 Presets: Ready",
                        label ="Status",
                        interactive =False ,
                        scale =2 
                    )
                
                with gr .Row ():
                    with gr .Column (scale =2 ):
                        preset_name_input =gr .Textbox (
                            label ="New Preset Name",
                            placeholder ="Enter preset name...",
                            info ="Name for saving current settings"
                        )
                    with gr .Column (scale =3 ):
                        preset_description_input =gr .Textbox (
                            label ="Description (Optional)",
                            placeholder ="Describe this preset...",
                            info ="Optional description for the preset"
                        )
                
                with gr .Row ():
                    save_preset_btn =gr .Button ("💾 Save Preset",variant ="primary",scale =1 )
                    load_preset_btn =gr .Button ("📂 Load Preset",variant ="secondary",scale =1 )
                    delete_preset_btn =gr .Button ("🗑️ Delete",variant ="stop",scale =1 )
                
                preset_message_text =gr .Textbox (
                    value ="Ready to manage presets",
                    label ="Messages",
                    interactive =False ,
                    lines =2 
                )
                
            with gr .Column ():
                examples =gr .Examples (
                examples =[
                f'assets/example_image/{image}'
                for image in os .listdir ("assets/example_image")
                ],
                inputs =image_prompt ,
                examples_per_page =96 
                )

    image_prompt .upload (
    preprocess_image ,
    inputs =[image_prompt ],
    outputs =[image_prompt ]
    )

    gen_shape_btn .click (
    generate_3d_with_cancellation ,
    inputs =[
    image_prompt ,seed ,
    ss_guidance_strength ,ss_sampling_steps ,
    slat_guidance_strength ,slat_sampling_steps ,
    poly_count_slider ,
    xatlas_max_cost_slider ,
    xatlas_normal_seam_weight_slider ,
    xatlas_resolution_slider ,
    xatlas_padding_slider ,
    normal_map_resolution_slider ,
    normal_match_input_res_checkbox ,
    auto_save_obj_cb ,
    auto_save_glb_cb ,
    auto_save_ply_cb ,
    auto_save_stl_cb 
    ],
    outputs =[normal_output ,model_output ,download_btn ],
    show_progress =True 
    ).then (
    lambda :gr .Button (interactive =True ),
    outputs =[download_btn ],
    )

    def prepare_download_file (mesh_path_from_model_output :str ,selected_format :str ):

        if not mesh_path_from_model_output :
            return None 

        converted_path =convert_mesh (mesh_path_from_model_output ,selected_format )
        if not converted_path :
            return None 

        timestamp =datetime .datetime .now ().strftime ('%Y%m%d_%H%M%S')
        final_filename =f"generated_mesh_{timestamp}.{selected_format.lower()}"
        final_path =os .path .join (TMP_DIR ,final_filename )

        try :
            shutil .copy2 (converted_path ,final_path )
            print (f"prepare_download_file: Created download file {final_path}")
            return final_path 
        except Exception as e :
            print (f"prepare_download_file: Error creating final download file: {e}")
            return converted_path 

    def update_download_button (mesh_path_from_model_output :str ,selected_format :str ):

        if not mesh_path_from_model_output :

            return gr .DownloadButton (interactive =False ,label ="Export Mesh")

        path_for_download =prepare_download_file (mesh_path_from_model_output ,selected_format )

        if path_for_download :
            print (f"update_download_button: Providing {path_for_download} for download as {selected_format}.")

            return gr .DownloadButton (value =path_for_download ,interactive =True ,label =f"Download {selected_format.upper()}")
        else :
            print (f"update_download_button: Conversion failed for {selected_format}, button inactive.")
            return gr .DownloadButton (interactive =False ,label ="Export Mesh")

    export_format .change (
    update_download_button ,
    inputs =[model_output ,export_format ],
    outputs =[download_btn ]
    )

    open_folder_btn .click (
    fn =lambda :open_outputs_folder (),
    inputs =[],
    outputs =[]
    )

    # Preset Management Functions
    def save_preset_ui_handler (preset_name ,preset_description ,
    seed ,ss_guidance_strength ,ss_sampling_steps ,
    slat_guidance_strength ,slat_sampling_steps ,poly_count_pcnt ,
    xatlas_max_cost ,xatlas_normal_seam_weight ,xatlas_resolution ,xatlas_padding ,
    normal_map_resolution ,normal_match_input_resolution ,
    auto_save_obj ,auto_save_glb ,auto_save_ply ,auto_save_stl ):
        """Handle saving a preset from the UI with validation"""
        try :
            # Validate preset name
            is_valid ,validation_message =validate_preset_name (preset_name )
            if not is_valid :
                return (
                gr .Dropdown (choices =get_preset_choices_for_ui ()),# preset_dropdown 
                f"❌ {validation_message}",# preset_message_text 
                get_preset_status_for_ui (),# preset_status_text 
                gr .Textbox (value =""),# clear preset_name_input 
                gr .Textbox (value ="")# clear preset_description_input 
                )
            
            # Validate parameters before saving
            try :
                from logic .preset_validation import validate_ui_parameters 
                
                param_validation_result =validate_ui_parameters (
                seed =seed ,ss_guidance_strength =ss_guidance_strength ,ss_sampling_steps =ss_sampling_steps ,
                slat_guidance_strength =slat_guidance_strength ,slat_sampling_steps =slat_sampling_steps ,
                poly_count_pcnt =poly_count_pcnt ,xatlas_max_cost =xatlas_max_cost ,
                xatlas_normal_seam_weight =xatlas_normal_seam_weight ,xatlas_resolution =xatlas_resolution ,
                xatlas_padding =xatlas_padding ,normal_map_resolution =normal_map_resolution ,
                normal_match_input_resolution =normal_match_input_resolution ,
                auto_save_obj =auto_save_obj ,auto_save_glb =auto_save_glb ,
                auto_save_ply =auto_save_ply ,auto_save_stl =auto_save_stl 
                )
                
                all_params_valid ,sanitized_values ,param_message =param_validation_result 
                
                if not all_params_valid :
                    print (f"⚠️ Parameter validation warnings: {param_message}")
                    # Continue with sanitized values but warn user
                    extra_message =f" (Parameter adjustments: {param_message})"
                else :
                    extra_message =""
                    
            except ImportError :
                print ("⚠️ Parameter validation not available")
                extra_message =""
            except Exception as validation_error :
                print (f"⚠️ Parameter validation error: {validation_error}")
                extra_message =f" (Validation warning: {str(validation_error)})"
            
            print (f"💾 Saving preset '{preset_name}' with current parameters...")
            print (f"   Parameters: seed={seed}, ss_guidance={ss_guidance_strength}, poly%={poly_count_pcnt:.2f}")
            
            # Save preset
            success ,message ,updated_presets ,selected_preset =save_preset_from_ui (
            preset_name ,preset_description ,
            seed ,ss_guidance_strength ,ss_sampling_steps ,
            slat_guidance_strength ,slat_sampling_steps ,poly_count_pcnt ,
            xatlas_max_cost ,xatlas_normal_seam_weight ,xatlas_resolution ,xatlas_padding ,
            normal_map_resolution ,normal_match_input_resolution ,
            auto_save_obj ,auto_save_glb ,auto_save_ply ,auto_save_stl ,
            overwrite =False # For now, don't allow overwrite without confirmation
            )
            
            if success :
                status_msg =f"✅ {message}{extra_message}"
                clear_inputs =True 
                print (f"✅ Preset '{preset_name}' saved successfully")
            else :
                status_msg =f"❌ {message}"
                clear_inputs =False 
                print (f"❌ Failed to save preset '{preset_name}': {message}")
            
            return (
            gr .Dropdown (choices =updated_presets ,value =selected_preset ),# preset_dropdown 
            status_msg ,# preset_message_text 
            get_preset_status_for_ui (),# preset_status_text 
            gr .Textbox (value =""if clear_inputs else preset_name ),# preset_name_input 
            gr .Textbox (value =""if clear_inputs else preset_description )# preset_description_input 
            )
            
        except Exception as e :
            error_msg =f"Error saving preset: {str(e)}"
            print (f"❌ {error_msg}")
            import traceback 
            traceback .print_exc ()
            
            return (
            gr .Dropdown (choices =get_preset_choices_for_ui ()),# preset_dropdown 
            f"❌ {error_msg}",# preset_message_text 
            "❌ Error in preset system",# preset_status_text 
            gr .Textbox (value =preset_name ),# preset_name_input 
            gr .Textbox (value =preset_description )# preset_description_input 
            )

    def load_preset_ui_handler (selected_preset ):
        """Handle loading a preset from the UI with enhanced feedback"""
        try :
            # Check if preset is selected
            if not selected_preset or selected_preset .strip ()=="":
                return (
                gr .Dropdown (choices =get_preset_choices_for_ui ()),
                "⚠️ No preset selected",
                get_preset_status_for_ui (),
                # Keep current parameter values (no change)
                gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),
                gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),
                gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),gr .update ()
                )
            
            print (f"🔄 Loading preset '{selected_preset}'...")
            
            success ,message ,updated_presets ,current_preset ,ui_values =load_preset_for_ui (selected_preset )
            
            if success :
                # Format success message with parameter info
                param_info =f"Parameters: seed={ui_values[0]}, quality={ui_values[1]:.1f}"
                status_msg =f"✅ {message}"
                print (f"✅ Loaded preset '{selected_preset}' successfully ({param_info})")
                
                # Log parameter changes for debugging
                print (f"   Applied parameters:")
                print (f"     Seed: {ui_values[0]}")
                print (f"     SS Guidance: {ui_values[1]}, Steps: {ui_values[2]}")
                print (f"     SLAT Guidance: {ui_values[3]}, Steps: {ui_values[4]}")
                print (f"     Polygon %: {ui_values[5]:.2f}")
                print (f"     UV Resolution: {ui_values[8]}")
                
            else :
                status_msg =f"❌ {message}"
                print (f"❌ Failed to load preset '{selected_preset}': {message}")
            
            # ui_values contains all parameter values in this order:
            # (seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps,
            #  poly_count_pcnt, xatlas_max_cost, xatlas_normal_seam_weight, xatlas_resolution, xatlas_padding,
            #  normal_map_resolution, normal_match_input_resolution, auto_save_obj, auto_save_glb, auto_save_ply, auto_save_stl)
            
            return (
            gr .Dropdown (choices =updated_presets ,value =current_preset ),# preset_dropdown 
            status_msg ,# preset_message_text 
            get_preset_status_for_ui (),# preset_status_text 
            # All parameter controls:
            ui_values [0 ],# seed 
            ui_values [1 ],# ss_guidance_strength 
            ui_values [2 ],# ss_sampling_steps 
            ui_values [3 ],# slat_guidance_strength 
            ui_values [4 ],# slat_sampling_steps 
            ui_values [5 ],# poly_count_pcnt 
            ui_values [6 ],# xatlas_max_cost 
            ui_values [7 ],# xatlas_normal_seam_weight 
            ui_values [8 ],# xatlas_resolution 
            ui_values [9 ],# xatlas_padding 
            ui_values [10 ],# normal_map_resolution 
            ui_values [11 ],# normal_match_input_resolution 
            ui_values [12 ],# auto_save_obj 
            ui_values [13 ],# auto_save_glb 
            ui_values [14 ],# auto_save_ply 
            ui_values [15 ]# auto_save_stl 
            )
            
        except Exception as e :
            error_msg =f"Error loading preset: {str(e)}"
            print (f"❌ {error_msg}")
            import traceback 
            traceback .print_exc ()
            
            # Return current state with error message
            return (
            gr .Dropdown (choices =get_preset_choices_for_ui ()),# preset_dropdown 
            f"❌ {error_msg}",# preset_message_text 
            "❌ Error in preset system",# preset_status_text 
            # Keep current parameter values (no change)
            gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),
            gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),
            gr .update (),gr .update (),gr .update (),gr .update (),gr .update (),gr .update ()
            )

    def delete_preset_ui_handler (selected_preset ):
        """Handle deleting a preset from the UI"""
        try :
            if not selected_preset or selected_preset .strip ()=="":
                return (
                gr .Dropdown (choices =get_preset_choices_for_ui ()),
                "❌ No preset selected for deletion",
                get_preset_status_for_ui ()
                )
            
            success ,message ,updated_presets ,current_preset =delete_preset_from_ui (selected_preset )
            
            if success :
                status_msg =f"✅ {message}"
                print (f"✅ Deleted preset '{selected_preset}' successfully")
            else :
                status_msg =f"❌ {message}"
                print (f"❌ Failed to delete preset '{selected_preset}': {message}")
            
            return (
            gr .Dropdown (choices =updated_presets ,value =current_preset ),# preset_dropdown 
            status_msg ,# preset_message_text 
            get_preset_status_for_ui ()# preset_status_text 
            )
            
        except Exception as e :
            error_msg =f"Error deleting preset: {str(e)}"
            print (f"❌ {error_msg}")
            import traceback 
            traceback .print_exc ()
            
            return (
            gr .Dropdown (choices =get_preset_choices_for_ui ()),
            f"❌ {error_msg}",
            "❌ Error in preset system"
            )

    def initialize_preset_ui ():
        """Initialize preset UI on app startup"""
        try :
            print ("🔧 Initializing preset UI...")
            
            # Initialize file system first
            file_success ,file_message =initialize_preset_file_system ()
            if not file_success :
                print (f"⚠️ Preset file system initialization failed: {file_message}")
            
            # Initialize preset system
            success ,message ,preset_list ,selected_preset ,ui_values =initialize_presets_for_ui ()
            
            if success :
                status_msg =f"✅ {message}"
                print (f"✅ Preset system initialized: {len(preset_list)} presets available")
            else :
                status_msg =f"❌ {message}"
                print (f"❌ Preset system initialization failed: {message}")
                # Use fallback values
                preset_list =["Default"]
                selected_preset ="Default"
                ui_values =(
                -1 ,3.0 ,50 ,3.0 ,6 ,0.5 ,8.0 ,1.0 ,1024 ,2 ,768 ,True ,True ,True ,True ,True 
                )
            
            return (
            gr .Dropdown (choices =preset_list ,value =selected_preset ),# preset_dropdown 
            f"Preset system ready",# preset_message_text 
            get_preset_status_for_ui (),# preset_status_text 
            # All parameter controls:
            ui_values [0 ],# seed 
            ui_values [1 ],# ss_guidance_strength 
            ui_values [2 ],# ss_sampling_steps 
            ui_values [3 ],# slat_guidance_strength 
            ui_values [4 ],# slat_sampling_steps 
            ui_values [5 ],# poly_count_pcnt 
            ui_values [6 ],# xatlas_max_cost 
            ui_values [7 ],# xatlas_normal_seam_weight 
            ui_values [8 ],# xatlas_resolution 
            ui_values [9 ],# xatlas_padding 
            ui_values [10 ],# normal_map_resolution 
            ui_values [11 ],# normal_match_input_resolution 
            ui_values [12 ],# auto_save_obj 
            ui_values [13 ],# auto_save_glb 
            ui_values [14 ],# auto_save_ply 
            ui_values [15 ]# auto_save_stl 
            )
            
        except Exception as e :
            error_msg =f"Critical error initializing preset UI: {str(e)}"
            print (f"❌ {error_msg}")
            import traceback 
            traceback .print_exc ()
            
            # Emergency fallback
            return (
            gr .Dropdown (choices =["Default"],value ="Default"),
            f"❌ Preset system error: {str(e)}",
            "❌ Preset system failed",
            # Default parameter values
            -1 ,3.0 ,50 ,3.0 ,6 ,0.5 ,8.0 ,1.0 ,1024 ,2 ,768 ,True ,True ,True ,True ,True 
            )

    def validate_system_integration ():

        try :
            return get_comprehensive_status_summary (
            processing_core =processing_core ,
            batch_processor =batch_processor ,
            hi3dgen_pipeline =hi3dgen_pipeline ,
            weights_dir =WEIGHTS_DIR ,
            tmp_dir =TMP_DIR 
            )
        except Exception as e :
            return f"❌ Validation error: {str(e)}"

    def start_batch_processing_ui (input_folder ,output_folder ,skip_existing ,use_current_settings ,
    seed ,ss_guidance_strength ,ss_sampling_steps ,
    slat_guidance_strength ,slat_sampling_steps ,poly_count_pcnt ,
    xatlas_max_cost ,xatlas_normal_seam_weight ,xatlas_resolution ,xatlas_padding ,
    normal_map_resolution ,normal_match_input_resolution ,
    auto_save_obj ,auto_save_glb ,auto_save_ply ,auto_save_stl ,
    progress =gr .Progress ()):

        global batch_processor 

        if not batch_processor :
            return ("Error: Batch processor not initialized","Failed","No batch processor available",
            gr .Button (visible =True ),gr .Button (visible =False ))

        if not input_folder or not output_folder :
            return ("Error: Please specify both input and output folders","Failed","Missing folder paths",
            gr .Button (visible =False ),gr .Button (visible =False ))

        if not os .path .exists (input_folder ):
            return (f"Error: Input folder does not exist: {input_folder}","Failed","Invalid input folder",
            gr .Button (visible =True ),gr .Button (visible =False ))

        try :

            def batch_progress_callback (batch_progress ):

                try :
                    if batch_progress .total_images >0 :
                        percentage =batch_progress .progress_percentage /100.0 
                        current_stage =batch_progress .current_stage 

                        progress (percentage ,desc =f"{current_stage} ({batch_progress.processed_images}/{batch_progress.total_images})")

                        if batch_progress .current_image :
                            eta_text =""
                            if batch_progress .estimated_time_remaining >0 :
                                eta_minutes =batch_progress .estimated_time_remaining /60 
                                if eta_minutes >1 :
                                    eta_text =f", ETA: {eta_minutes:.1f}min"
                                else :
                                    eta_text =f", ETA: {batch_progress.estimated_time_remaining:.0f}s"

                            print (f"Batch Progress: {current_stage} - {batch_progress.processed_images}/{batch_progress.total_images} ({batch_progress.progress_percentage:.1f}%{eta_text})")
                    else :
                        progress (0.0 ,desc =batch_progress .current_stage )

                except Exception as e :
                    print (f"Error in batch progress callback: {e}")

            batch_processor .set_progress_callback (batch_progress_callback )

            settings =BatchSettings (
            input_folder =input_folder ,
            output_folder =output_folder ,
            skip_existing =skip_existing ,
            seed =seed if use_current_settings else -1 ,
            ss_guidance_strength =ss_guidance_strength if use_current_settings else 3.0 ,
            ss_sampling_steps =ss_sampling_steps if use_current_settings else 50 ,
            slat_guidance_strength =slat_guidance_strength if use_current_settings else 3.0 ,
            slat_sampling_steps =slat_sampling_steps if use_current_settings else 6 ,
            poly_count_pcnt =poly_count_pcnt if use_current_settings else 0.5 ,
            xatlas_max_cost =xatlas_max_cost if use_current_settings else 8.0 ,
            xatlas_normal_seam_weight =xatlas_normal_seam_weight if use_current_settings else 1.0 ,
            xatlas_resolution =xatlas_resolution if use_current_settings else 1024 ,
            xatlas_padding =xatlas_padding if use_current_settings else 2 ,
            normal_map_resolution =normal_map_resolution if use_current_settings else 768 ,
            normal_match_input_resolution =normal_match_input_resolution if use_current_settings else True ,
            auto_save_obj =auto_save_obj if use_current_settings else True ,
            auto_save_glb =auto_save_glb if use_current_settings else True ,
            auto_save_ply =auto_save_ply if use_current_settings else True ,
            auto_save_stl =auto_save_stl if use_current_settings else True 
            )

            print (f"Starting batch processing:")
            print (f"  Input folder: {input_folder}")
            print (f"  Output folder: {output_folder}")
            print (f"  Skip existing: {skip_existing}")
            print (f"  Use current settings: {use_current_settings}")

            progress (0.0 ,desc ="Initializing batch processing...")

            def run_batch ():
                try :
                    print ("Starting batch processing thread...")
                    result =batch_processor .start_batch_processing (settings )

                    if result .processed_images >0 :
                        success_rate =(result .processed_images /(result .processed_images +len (result .errors )))*100 if (result .processed_images +len (result .errors ))>0 else 0 
                        progress (1.0 ,desc =f"✅ Completed: {result.processed_images} processed, {len(result.errors)} errors ({success_rate:.1f}% success)")
                    else :
                        progress (0.0 ,desc ="⚠️ No images processed")

                    print (f"Batch processing completed: {result.processed_images} processed, {len(result.errors)} errors")

                    if result .processed_images >0 :
                        print (f"✅ Successfully processed {result.processed_images} images")
                    if len (result .errors )>0 :
                        print (f"❌ {len(result.errors)} images failed processing:")
                        for error in result .errors [:5 ]:
                            print (f"  • {error}")
                        if len (result .errors )>5 :
                            print (f"  • ... and {len(result.errors) - 5} more errors")
                    if len (result .skipped )>0 :
                        print (f"⏭️ Skipped {len(result.skipped)} existing files")

                except Exception as e :
                    progress (0.0 ,desc =f"Error: {str(e)}")
                    print (f"Batch processing error: {e}")
                    import traceback 
                    traceback .print_exc ()

            batch_thread =threading .Thread (target =run_batch ,daemon =True )
            batch_thread .start ()

            return ("Batch processing started...","Running","Initializing batch processing...",
            gr .Button (visible =False ),gr .Button (visible =True ))

        except Exception as e :
            error_msg =f"Error starting batch processing: {str(e)}"
            print (error_msg )
            import traceback 
            traceback .print_exc ()
            progress (0.0 ,desc =f"Failed: {str(e)}")
            return (error_msg ,"Failed","Error occurred",
            gr .Button (visible =False ),gr .Button (visible =False ))

    def cancel_processing_ui ():

        try :
            success =request_cancellation ("User requested cancellation")
            if success :
                return ("Cancellation requested...","Cancelling","Processing will stop soon...",
                gr .Button (visible =True ),gr .Button (visible =False ))
            else :
                return ("No active processing to cancel","Idle","Ready",
                gr .Button (visible =True ),gr .Button (visible =False ))
        except Exception as e :
            error_msg =f"Error requesting cancellation: {str(e)}"
            print (error_msg )
            return (error_msg ,"Error","Cancellation failed",
            gr .Button (visible =True ),gr .Button (visible =False ))

    def update_processing_status ():

        try :
            global_status =get_status_summary ()

            if batch_processor and hasattr (batch_processor ,'get_status_summary'):
                try :
                    batch_status =batch_processor .get_status_summary ()
                    batch_progress =batch_processor .progress 

                    if hasattr (batch_progress ,'total_images')and batch_progress .total_images >0 :

                        percentage =getattr (batch_progress ,'progress_percentage',0 )
                        elapsed =getattr (batch_progress ,'elapsed_time',0 )
                        eta =getattr (batch_progress ,'estimated_time_remaining',0 )

                        results_summary =f"📊 Batch Progress Summary\n"
                        results_summary +=f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        results_summary +=f"📁 Total Images: {batch_progress.total_images}\n"
                        results_summary +=f"✅ Processed: {batch_progress.processed_images}\n"
                        results_summary +=f"⏭️ Skipped: {len(getattr(batch_progress, 'skipped', []))}\n"
                        results_summary +=f"❌ Errors: {len(getattr(batch_progress, 'errors', []))}\n"
                        results_summary +=f"📈 Progress: {percentage:.1f}%\n"
                        results_summary +=f"⏱️ Elapsed: {elapsed:.1f}s"

                        if eta >0 :
                            eta_text =f"{eta/60:.1f}min"if eta >60 else f"{eta:.0f}s"
                            results_summary +=f" | ETA: {eta_text}"

                        current_image =getattr (batch_progress ,'current_image',None )
                        if current_image :
                            results_summary +=f"\n🔄 Current: {current_image}"

                        errors =getattr (batch_progress ,'errors',[])
                        if errors :
                            results_summary +=f"\n\n❌ Recent Errors:"
                            for error in errors [-3 :]:
                                results_summary +=f"\n  • {error}"

                        skipped =getattr (batch_progress ,'skipped',[])
                        if skipped and len (skipped )<=5 :
                            results_summary +=f"\n\n⏭️ Skipped Files:"
                            for skip in skipped :
                                results_summary +=f"\n  • {skip}"
                        elif len (skipped )>5 :
                            results_summary +=f"\n\n⏭️ Skipped Files: {len(skipped)} files (check console for details)"

                    else :
                        results_summary ="🚀 Ready for batch processing\n"
                        results_summary +="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        results_summary +="Select input and output folders to begin"

                    is_running =getattr (batch_processor ,'is_running',False )
                    if is_running :
                        if hasattr (batch_progress ,'total_images')and batch_progress .total_images >0 :
                            detailed_status =f"🔄 {batch_status}\n"
                            detailed_status +=f"Processing: {batch_progress.processed_images}/{batch_progress.total_images} images"
                            current_image =getattr (batch_progress ,'current_image',None )
                            if current_image :
                                detailed_status +=f"\nCurrent: {current_image}"
                        else :
                            detailed_status =f"🔄 {batch_status}"
                    else :
                        detailed_status =f"⏸️ {batch_status}"

                    show_cancel =is_running or "processing"in global_status .lower ()

                    return (global_status ,detailed_status ,results_summary ,
                    gr .Button (visible =show_cancel ),gr .Button (visible =not show_cancel ))

                except Exception as batch_e :
                    print (f"⚠️ Batch processor status error: {batch_e}")

                    show_cancel ="processing"in global_status .lower ()
                    return (global_status ,"⚠️ Batch processor status unavailable","Batch processor initialized but status unavailable",
                    gr .Button (visible =show_cancel ),gr .Button (visible =not show_cancel ))
            else :
                show_cancel ="processing"in global_status .lower ()
                return (global_status ,"❌ Batch processor not available","⚠️ Batch processor not initialized",
                gr .Button (visible =show_cancel ),gr .Button (visible =not show_cancel ))

        except Exception as e :
            error_msg =f"❌ Error updating status: {str(e)}"
            print (error_msg )
            import traceback 
            traceback .print_exc ()
            return (error_msg ,"❌ Status Error","Status update failed - check console",
            gr .Button (visible =False ),gr .Button (visible =False ))

    batch_start_btn .click (
    fn =start_batch_processing_ui ,
    inputs =[
    batch_input_folder ,batch_output_folder ,batch_skip_existing ,batch_use_current_settings ,
    seed ,ss_guidance_strength ,ss_sampling_steps ,
    slat_guidance_strength ,slat_sampling_steps ,poly_count_slider ,
    xatlas_max_cost_slider ,xatlas_normal_seam_weight_slider ,xatlas_resolution_slider ,xatlas_padding_slider ,
    normal_map_resolution_slider ,normal_match_input_res_checkbox ,
    auto_save_obj_cb ,auto_save_glb_cb ,auto_save_ply_cb ,auto_save_stl_cb 
    ],
    outputs =[processing_status_text ,batch_status_text ,batch_results_text ,batch_start_btn ,universal_cancel_btn ],
    show_progress =True 
    )

    universal_cancel_btn .click (
    fn =cancel_processing_ui ,
    inputs =[],
    outputs =[processing_status_text ,batch_status_text ,batch_results_text ,batch_start_btn ,universal_cancel_btn ]
    )

    batch_cancel_btn .click (
    fn =cancel_processing_ui ,
    inputs =[],
    outputs =[processing_status_text ,batch_status_text ,batch_results_text ,batch_start_btn ,universal_cancel_btn ]
    )

    demo .load (
    fn =update_processing_status ,
    inputs =[],
    outputs =[processing_status_text ,batch_status_text ,batch_results_text ,universal_cancel_btn ,batch_start_btn ]
    )

    demo .load (
    fn =lambda :print (validate_system_integration ()),
    inputs =[],
    outputs =[]
    )

    # Preset Management Event Handlers
    save_preset_btn .click (
    fn =save_preset_ui_handler ,
    inputs =[
    preset_name_input ,preset_description_input ,
    seed ,ss_guidance_strength ,ss_sampling_steps ,
    slat_guidance_strength ,slat_sampling_steps ,poly_count_slider ,
    xatlas_max_cost_slider ,xatlas_normal_seam_weight_slider ,xatlas_resolution_slider ,xatlas_padding_slider ,
    normal_map_resolution_slider ,normal_match_input_res_checkbox ,
    auto_save_obj_cb ,auto_save_glb_cb ,auto_save_ply_cb ,auto_save_stl_cb 
    ],
    outputs =[
    preset_dropdown ,preset_message_text ,preset_status_text ,
    preset_name_input ,preset_description_input 
    ]
    )

    load_preset_btn .click (
    fn =load_preset_ui_handler ,
    inputs =[preset_dropdown ],
    outputs =[
    preset_dropdown ,preset_message_text ,preset_status_text ,
    # All parameter controls need to be updated when loading preset:
    seed ,ss_guidance_strength ,ss_sampling_steps ,
    slat_guidance_strength ,slat_sampling_steps ,poly_count_slider ,
    xatlas_max_cost_slider ,xatlas_normal_seam_weight_slider ,xatlas_resolution_slider ,xatlas_padding_slider ,
    normal_map_resolution_slider ,normal_match_input_res_checkbox ,
    auto_save_obj_cb ,auto_save_glb_cb ,auto_save_ply_cb ,auto_save_stl_cb 
    ]
    )

    delete_preset_btn .click (
    fn =delete_preset_ui_handler ,
    inputs =[preset_dropdown ],
    outputs =[preset_dropdown ,preset_message_text ,preset_status_text ]
    )

    # Also handle dropdown change to load preset when user selects from dropdown
    preset_dropdown .change (
    fn =load_preset_ui_handler ,
    inputs =[preset_dropdown ],
    outputs =[
    preset_dropdown ,preset_message_text ,preset_status_text ,
    # All parameter controls:
    seed ,ss_guidance_strength ,ss_sampling_steps ,
    slat_guidance_strength ,slat_sampling_steps ,poly_count_slider ,
    xatlas_max_cost_slider ,xatlas_normal_seam_weight_slider ,xatlas_resolution_slider ,xatlas_padding_slider ,
    normal_map_resolution_slider ,normal_match_input_res_checkbox ,
    auto_save_obj_cb ,auto_save_glb_cb ,auto_save_ply_cb ,auto_save_stl_cb 
    ]
    )

    # Initialize preset system on app load
    demo .load (
    fn =initialize_preset_ui ,
    inputs =[],
    outputs =[
    preset_dropdown ,preset_message_text ,preset_status_text ,
    # All parameter controls to set initial values:
    seed ,ss_guidance_strength ,ss_sampling_steps ,
    slat_guidance_strength ,slat_sampling_steps ,poly_count_slider ,
    xatlas_max_cost_slider ,xatlas_normal_seam_weight_slider ,xatlas_resolution_slider ,xatlas_padding_slider ,
    normal_map_resolution_slider ,normal_match_input_res_checkbox ,
    auto_save_obj_cb ,auto_save_glb_cb ,auto_save_ply_cb ,auto_save_stl_cb 
    ]
    )

    def cleanup_processing_system ():

        global processing_core ,batch_processor ,cancellation_manager ,hi3dgen_pipeline 

        try :
            print ("🧹 Cleaning up processing system...")

            if cancellation_manager :
                try :
                    cancellation_manager .request_cancellation ("System shutdown")
                    print ("  ✓ Cancellation requested")
                except Exception as e :
                    print (f"  ⚠️ Error during cancellation: {e}")

            if batch_processor and hasattr (batch_processor ,'cleanup'):
                try :
                    batch_processor .cleanup ()
                    print ("  ✓ Batch processor cleaned up")
                except Exception as e :
                    print (f"  ⚠️ Error cleaning batch processor: {e}")

            if processing_core and hasattr (processing_core ,'cleanup'):
                try :
                    processing_core .cleanup ()
                    print ("  ✓ Processing core cleaned up")
                except Exception as e :
                    print (f"  ⚠️ Error cleaning processing core: {e}")

            if hi3dgen_pipeline :
                try :

                    if torch .cuda .is_available ():
                        torch .cuda .synchronize ()

                    if hasattr (hi3dgen_pipeline ,'reset_birefnet_state'):
                        hi3dgen_pipeline .reset_birefnet_state ()
                        print ("  ✓ BiRefNet state reset")

                    hi3dgen_pipeline .cpu ()

                    if torch .cuda .is_available ():
                        torch .cuda .empty_cache ()

                    print ("  ✓ GPU memory cleared")

                except RuntimeError as cuda_error :
                    if "CUDA"in str (cuda_error ):
                        print (f"  ⚠️ CUDA error during cleanup: {cuda_error}")
                        print ("  ⚠️ Attempting force GPU cleanup...")
                        try :

                            if hasattr (hi3dgen_pipeline ,'reset_birefnet_state'):
                                hi3dgen_pipeline .reset_birefnet_state ()

                            if torch .cuda .is_available ():
                                torch .cuda .empty_cache ()
                                torch .cuda .synchronize ()
                            print ("  ✓ Force GPU cleanup completed")
                        except Exception as force_error :
                            print (f"  ❌ Force GPU cleanup also failed: {force_error}")
                    else :
                        print (f"  ⚠️ Non-CUDA error during GPU cleanup: {cuda_error}")
                except Exception as e :
                    print (f"  ⚠️ Unexpected error clearing GPU memory: {e}")

                    try :
                        if hasattr (hi3dgen_pipeline ,'reset_birefnet_state'):
                            hi3dgen_pipeline .reset_birefnet_state ()
                        if torch .cuda .is_available ():
                            torch .cuda .empty_cache ()
                    except :
                        pass 

            print ("🧹 Cleanup completed")

        except Exception as e :
            print (f"❌ Error during cleanup: {e}")
            import traceback 
            traceback .print_exc ()

    import atexit 
    atexit .register (cleanup_processing_system )

if __name__ =="__main__":

    print ("Caching model weights...")
    cached_paths =cache_weights (WEIGHTS_DIR )

    global_cached_paths =cached_paths 

    expected_models =[
    "models--Stable-X--trellis-normal-v0-1",
    "models--Stable-X--yoso-normal-v1-8-1",
    "models--ZhengPeng7--BiRefNet"
    ]

    print ("Verifying cached models...")
    for model_name in expected_models :
        model_path =os .path .join (WEIGHTS_DIR ,model_name )
        if os .path .exists (model_path ):
            print (f"✓ Found: {model_name}")
        else :
            print (f"✗ Missing: {model_name}")

    print ("Loading Hi3DGenPipeline to CPU...")

    trellis_local_path =os .path .join (WEIGHTS_DIR ,"models--Stable-X--trellis-normal-v0-1")
    hi3dgen_pipeline =Hi3DGenPipeline .from_pretrained (trellis_local_path )
    hi3dgen_pipeline .cpu ()
    print ("Hi3DGenPipeline loaded on CPU.")

    print ("Initializing processing system...")
    print (f"  System Info:")
    print (f"    - WEIGHTS_DIR: {WEIGHTS_DIR}")
    print (f"    - TMP_DIR: {TMP_DIR}")
    print (f"    - MAX_SEED: {MAX_SEED}")
    print (f"    - CUDA Available: {torch.cuda.is_available()}")
    if torch .cuda .is_available ():
        print (f"    - CUDA Device: {torch.cuda.get_device_name()}")
        print (f"    - CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    try :

        if hi3dgen_pipeline is None :
            raise RuntimeError ("Hi3DGenPipeline not loaded")
        if not os .path .exists (WEIGHTS_DIR ):
            raise RuntimeError (f"Weights directory not found: {WEIGHTS_DIR}")
        if not os .path .exists (TMP_DIR ):
            os .makedirs (TMP_DIR ,exist_ok =True )
            print (f"Created TMP_DIR: {TMP_DIR}")

        processing_core =ProcessingCore (
        hi3dgen_pipeline =hi3dgen_pipeline ,
        weights_dir =WEIGHTS_DIR ,
        tmp_dir =TMP_DIR ,
        max_seed =MAX_SEED 
        )
        print ("✓ ProcessingCore initialized successfully.")

        batch_processor =create_batch_processor (processing_core )
        print ("✓ BatchProcessor initialized successfully.")

        cancellation_manager =get_cancellation_manager ()
        print ("✓ Cancellation manager initialized successfully.")

        assert processing_core is not None ,"ProcessingCore initialization failed"
        assert batch_processor is not None ,"BatchProcessor initialization failed"
        assert cancellation_manager is not None ,"Cancellation manager initialization failed"

        print ("🚀 Processing system ready and verified!")

        if SYSTEM_STATUS_AVAILABLE :
            print ("\n"+"="*60 )
            print_system_status (
            processing_core =processing_core ,
            batch_processor =batch_processor ,
            hi3dgen_pipeline =hi3dgen_pipeline ,
            weights_dir =WEIGHTS_DIR ,
            tmp_dir =TMP_DIR 
            )
            print ("="*60 )
        else :
            print ("\n🚀 System ready with basic monitoring")
            print (f"✓ ProcessingCore: {'Available' if processing_core else 'Not Available'}")
            print (f"✓ BatchProcessor: {'Available' if batch_processor else 'Not Available'}")
            print (f"✓ CancellationManager: {'Available' if cancellation_manager else 'Not Available'}")
            print (f"✓ Hi3DGen Pipeline: {'Available' if hi3dgen_pipeline else 'Not Available'}")
            print (f"✓ GPU Support: {'Available' if torch.cuda.is_available() else 'Not Available'}")

    except Exception as e :
        print (f"❌ Error initializing processing system: {e}")
        import traceback 
        traceback .print_exc ()
        print ("⚠️ Some features may not work properly.")

        processing_core =None 
        batch_processor =None 
        cancellation_manager =None 

    parser =argparse .ArgumentParser (description ="Run the Hi3DGen Gradio App.")
    parser .add_argument ('--share',action ='store_true',help ='Enable Gradio sharing')
    args =parser .parse_args ()

    demo .launch (share =args .share ,inbrowser =True )

