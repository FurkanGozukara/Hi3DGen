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

MAX_SEED =np .iinfo (np .int32 ).max 
TMP_DIR =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'tmp')
WEIGHTS_DIR =os .path .join (os .path .dirname (os .path .abspath (__file__ )),'weights')
os .makedirs (TMP_DIR ,exist_ok =True )
os .makedirs (WEIGHTS_DIR ,exist_ok =True )

hi3dgen_pipeline =None 
normal_predictor =None 
global_cached_paths =None 

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

    return hi3dgen_pipeline .preprocess_image (image ,resolution =1024 )

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

    print ("UV Unwrapping with xatlas: Starting process...")

    input_vertices_orig =input_mesh .vertices .astype (np .float32 )
    input_faces_orig =input_mesh .faces .astype (np .uint32 )
    vertex_normals_from_trimesh =input_mesh .vertex_normals 
    input_normals_orig =np .ascontiguousarray (vertex_normals_from_trimesh ,dtype =np .float32 )

    print (f"  Input mesh: {input_vertices_orig.shape[0]} vertices, {input_faces_orig.shape[0]} faces.")

    atlas =xatlas .Atlas ()
    atlas .add_mesh (input_vertices_orig ,input_faces_orig ,input_normals_orig )

    chart_options =xatlas .ChartOptions ()

    chart_options .max_cost =max_cost_param 

    chart_options .normal_seam_weight =normal_seam_weight_param 

    pack_options =xatlas .PackOptions ()
    pack_options .resolution =resolution_param 
    pack_options .padding =padding_param 

    print (f"  Running xatlas.generate() with ChartOptions(max_cost={chart_options.max_cost:.2f}, normal_seam_weight={chart_options.normal_seam_weight:.2f}) and PackOptions(resolution={pack_options.resolution}, padding={pack_options.padding})...")
    atlas .generate (chart_options =chart_options ,pack_options =pack_options )
    print (f"  xatlas generated atlas with dimensions: width={atlas.width}, height={atlas.height}")

    v_out_xref_data ,f_out_indices ,uv_coords_from_xatlas =atlas .get_mesh (0 )

    num_new_vertices =uv_coords_from_xatlas .shape [0 ]
    if v_out_xref_data .shape ==(num_new_vertices ,):
        xref_indices =v_out_xref_data .astype (np .uint32 )
        if np .any (xref_indices >=input_vertices_orig .shape [0 ])or np .any (xref_indices <0 ):
             raise ValueError ("Invalid xref values from xatlas - out of bounds for original input vertices.")
        final_vertices_spatial =input_vertices_orig [xref_indices ]
    elif v_out_xref_data .shape ==(num_new_vertices ,3 ):
        print ("  Warning: xatlas.get_mesh() returned 3D vertex data directly, which is unexpected for add_mesh workflow.")
        final_vertices_spatial =v_out_xref_data .astype (np .float32 )
    else :
        raise ValueError (f"Unexpected shape for vertex/xref data from xatlas.get_mesh: {v_out_xref_data.shape}.")

    final_uvs =uv_coords_from_xatlas .astype (np .float32 )
    if np .any (final_uvs >1.5 ):
        print ("  UVs appear to be in pixel coordinates. Normalizing...")
        if atlas .width >0 and atlas .height >0 :
            final_uvs /=np .array ([atlas .width ,atlas .height ],dtype =np .float32 )
        else :
            print ("  WARNING: Atlas width/height is 0, cannot normalize pixel UVs. Using unnormalized.")
    else :
        min_uv =final_uvs .min (axis =0 )if final_uvs .size >0 else "N/A"
        max_uv =final_uvs .max (axis =0 )if final_uvs .size >0 else "N/A"
        print (f"  UVs appear to be normalized. Min: {min_uv}, Max: {max_uv}")

    output_mesh =trimesh .Trimesh (vertices =final_vertices_spatial ,faces =f_out_indices ,process =False )

    if final_uvs .shape !=(final_vertices_spatial .shape [0 ],2 ):
        raise ValueError (f"Shape mismatch for final UVs before Trimesh assignment.")

    material =trimesh .visual .material .PBRMaterial (name ='defaultXatlasMat')
    output_mesh .visual =trimesh .visual .TextureVisuals (uv =final_uvs ,material =material )

    if hasattr (output_mesh .visual ,'uv')and output_mesh .visual .uv is not None :
        print (f"  Trimesh object successfully created with UVs, Shape: {output_mesh.visual.uv.shape}")
    else :
        print ("  ERROR: Trimesh object does NOT have UVs assigned after TextureVisuals call.")
        raise RuntimeError ("Failed to assign UVs to the Trimesh object.")

    print ("UV Unwrapping with xatlas: Process complete.")
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
normal_match_input_resolution :bool =True ):

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
            if hasattr (current_normal_predictor_instance ,'model')and hasattr (current_normal_predictor_instance .model ,'cpu'):
                current_normal_predictor_instance .model .cpu ()
            del current_normal_predictor_instance 
            if torch .cuda .is_available ():torch .cuda .empty_cache ()

    if normal_image_pil is None :
        print ("ERROR: Normal map not generated after Stage 1. Aborting 3D generation.")
        return None ,None ,None 

    pipeline_on_gpu =False 
    try :
        if torch .cuda .is_available ():
            print ("3D Generation: Moving Hi3DGen pipeline to GPU...")
            hi3dgen_pipeline .cuda ();pipeline_on_gpu =True 

        print ("3D Generation: Running Hi3DGen pipeline...")
        outputs =hi3dgen_pipeline .run (
        normal_image_pil ,seed =seed ,formats =["mesh",],preprocess_image =False ,
        sparse_structure_sampler_params ={"steps":ss_sampling_steps ,"cfg_strength":ss_guidance_strength },
        slat_sampler_params ={"steps":slat_sampling_steps ,"cfg_strength":slat_guidance_strength },
        )

        timestamp =datetime .datetime .now ().strftime ('%Y%m%d%H%M%S')
        output_dir =os .path .join (TMP_DIR ,timestamp )
        os .makedirs (output_dir ,exist_ok =True )

        mesh_path_glb =os .path .join (output_dir ,"mesh.glb")

        raw_mesh_trimesh =outputs ['mesh'][0 ].to_trimesh (transform_pose =True )
        mesh_for_uv_unwrap =simplify_mesh_open3d (raw_mesh_trimesh ,poly_count_pcnt )

        unwrapped_mesh_trimesh =unwrap_mesh_with_xatlas (mesh_for_uv_unwrap ,
        max_cost_param =xatlas_max_cost ,
        normal_seam_weight_param =xatlas_normal_seam_weight ,
        resolution_param =xatlas_resolution ,
        padding_param =xatlas_padding )

        print (f"Exporting GLB to {mesh_path_glb}...")
        unwrapped_mesh_trimesh .export (mesh_path_glb )
        print (f"SUCCESS: GLB exported.")

        gradio_model_path =mesh_path_glb 

    except Exception as e :
        print (f"ERROR in 3D Generation or UV Unwrapping stage: {e}")
        traceback .print_exc ()
        gradio_model_path =None 
    finally :
        if pipeline_on_gpu :
            print ("3D Generation: Moving Hi3DGen pipeline to CPU...")
            hi3dgen_pipeline .cpu ()
            if torch .cuda .is_available ():torch .cuda .empty_cache ()

    return normal_image_pil ,gradio_model_path ,gradio_model_path 

def convert_mesh (mesh_path :str ,export_format :str )->Optional [str ]:

    if not mesh_path or not os .path .exists (mesh_path ):
        print (f"convert_mesh: Invalid input mesh_path: {mesh_path}")
        return None 

    try :
        # Create a predictable filename for better download behavior
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"mesh_{timestamp}.{export_format.lower()}"
        temp_file_path = os.path.join(TMP_DIR, filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        if export_format .lower ()=="glb"and mesh_path .lower ().endswith (".glb"):
            print (f"convert_mesh: Copying GLB {mesh_path} to {temp_file_path}")
            shutil .copy2 (mesh_path ,temp_file_path )
            return temp_file_path 

        print (f"convert_mesh: Converting {mesh_path} to {export_format} at {temp_file_path}")
        mesh =trimesh .load_mesh (mesh_path )

        if not (hasattr (mesh .visual ,'uv')and mesh .visual .uv is not None ):
            print (f"  Warning: Loaded mesh from {mesh_path} has no UVs before export to {export_format}.")

        # Export the mesh with explicit file type
        mesh .export (temp_file_path ,file_type =export_format .lower ())
        
        # Verify the file was created and has content
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
            raise Exception(f"Failed to create valid {export_format} file")

        print (f"convert_mesh: Successfully created {export_format} file at {temp_file_path} ({os.path.getsize(temp_file_path)} bytes)")
        return temp_file_path 

    except Exception as e :
        print (f"convert_mesh: Error during conversion of '{mesh_path}' to '{export_format}': {e}")
        traceback .print_exc ()
        if 'temp_file_path' in locals() and temp_file_path and os .path .exists (temp_file_path ):
            try :
                os .remove (temp_file_path )
            except Exception as rm_e :
                print (f"convert_mesh: Error removing temp file {temp_file_path}: {rm_e}")
        return None 

with gr .Blocks (css ="footer {visibility: hidden}",theme =gr .themes .Soft ())as demo :
    gr .Markdown (
    "# Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging SECourses App V1 : https://www.patreon.com/posts/123105403"
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

                with gr .Tab ("Multiple Images"):
                    gr .Markdown ("<div style='text-align: center; padding: 40px; font-size: 24px;'>Multiple Images functionality is coming soon!</div>")

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
            with gr .Column ():
                model_output =gr .Model3D (label ="3D Model Preview (Each model is approximately 40MB, may take around 1 minute to load)")
            with gr .Column ():
                export_format =gr .Dropdown (
                choices =["obj","glb","ply","stl"],
                value ="glb",
                label ="File Format"
                )
                download_btn =gr .DownloadButton (label ="Export Mesh",interactive =False )
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
    generate_3d ,
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
    normal_match_input_res_checkbox 
    ],
    outputs =[normal_output ,model_output ,download_btn ]
    ).then (
    lambda :gr .Button (interactive =True ),
    outputs =[download_btn ],
    )

    def prepare_download_file(mesh_path_from_model_output: str, selected_format: str):
        """Prepare file for download with proper naming"""
        if not mesh_path_from_model_output:
            return None
            
        converted_path = convert_mesh(mesh_path_from_model_output, selected_format)
        if not converted_path:
            return None
            
        # Create a final file with a clean name for download
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        final_filename = f"generated_mesh_{timestamp}.{selected_format.lower()}"
        final_path = os.path.join(TMP_DIR, final_filename)
        
        try:
            shutil.copy2(converted_path, final_path)
            print(f"prepare_download_file: Created download file {final_path}")
            return final_path
        except Exception as e:
            print(f"prepare_download_file: Error creating final download file: {e}")
            return converted_path  # Fallback to original converted file

    def update_download_button (mesh_path_from_model_output :str ,selected_format :str ):

        if not mesh_path_from_model_output :

            return gr .DownloadButton (interactive =False, label="Export Mesh")

        path_for_download = prepare_download_file(mesh_path_from_model_output, selected_format)

        if path_for_download :
            print (f"update_download_button: Providing {path_for_download} for download as {selected_format}.")
            # Ensure the file has the right extension for the download name
            return gr .DownloadButton (value =path_for_download ,interactive =True, label=f"Download {selected_format.upper()}")
        else :
            print (f"update_download_button: Conversion failed for {selected_format}, button inactive.")
            return gr .DownloadButton (interactive =False, label="Export Mesh")

    export_format .change (
    update_download_button ,
    inputs =[model_output ,export_format ],
    outputs =[download_btn ]
    )

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

    parser = argparse.ArgumentParser(description="Run the Hi3DGen Gradio App.")
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    args = parser.parse_args()

    demo .launch (share =args.share ,inbrowser =True )

