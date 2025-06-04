import logging
import time
import traceback
from typing import Dict, Optional, List
import asyncio
import io
import base64
import numpy as np
import os
from fastapi import APIRouter, File, Response, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from PIL import Image
import torch
import trimesh
import open3d as o3d
import xatlas

from api_spz.core.exceptions import CancelledException
from api_spz.core.files_manage import file_manager
from api_spz.core.state_manage import state
from api_spz.core.models_pydantic import (
    GenerationArgForm,
    GenerationResponse, 
    TaskStatus,
    StatusResponse
)


router = APIRouter()

logger = logging.getLogger("stable3dgen_api") 

cancel_event = asyncio.Event() # This event will be set by the endpoint /interrupt

# A single lock to ensure only one generation at a time
generation_lock = asyncio.Lock()
def is_generation_in_progress() -> bool:
    return generation_lock.locked()


# A single dictionary holding "current generation" metadata
current_generation = {
    "status": TaskStatus.FAILED, # default
    "progress": 0,
    "message": "",
    "outputs": None,       # pipeline outputs if we did partial gen.
    "model_url": None      # final model path if relevant.
}


# Helper to reset the "current_generation" dictionary
# (useful to start fresh each time we begin generating)
def reset_current_generation():
    cancel_event.clear()
    current_generation["status"] = TaskStatus.PROCESSING
    current_generation["progress"] = 0
    current_generation["message"] = ""
    current_generation["outputs"] = None
    current_generation["model_url"] = None


# Helper to update the "current_generation" dictionary
def update_current_generation(
    status: Optional[TaskStatus] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    outputs=None
):
    if status is not None:
        current_generation["status"] = status
    if progress is not None:
        current_generation["progress"] = progress
    if message is not None:
        current_generation["message"] = message
    if outputs is not None:
        current_generation["outputs"] = outputs


# Cleanup files in "current_generation" folder
async def cleanup_generation_files(keep_videos: bool = False, keep_model: bool = False):
    file_manager.cleanup_generation_files(keep_videos=keep_videos, keep_model=keep_model)


def _gen_3d_validate_params(file_or_files, b64_or_b64list, arg: GenerationArgForm):
    """Validate incoming parameters before generation."""
    if (not file_or_files or len(file_or_files) == 0) and (not b64_or_b64list or len(b64_or_b64list) == 0):
        raise HTTPException(400, "No input images provided")
    # Range checks for Hi3DGen parameters (based on app.py sliders)
    if not (0.0 <= arg.ss_guidance_strength <= 10.0):
        raise HTTPException(status_code=400, detail="Sparse Structure Guidance Strength must be between 0.0 and 10.0")
    if not (1 <= arg.ss_sampling_steps <= 50):
        raise HTTPException(status_code=400, detail="Sparse Structure Sampling Steps must be between 1 and 50")
    
    if not (0.0 <= arg.slat_guidance_strength <= 10.0):
        raise HTTPException(status_code=400, detail="Structured Latent Guidance Strength must be between 0.0 and 10.0")
    if not (1 <= arg.slat_sampling_steps <= 50):
        raise HTTPException(status_code=400, detail="Structured Latent Sampling Steps must be between 1 and 50")

    if not (0 <= arg.poly_count_pcnt <= 100): 
        raise HTTPException(status_code=400, detail="poly_count_pcnt must be between 0 and 100")
    
    if arg.apply_texture:
        logger.warning("apply_texture is True, but Hi3DGen pipeline does not have integrated texturing. This option may not have an effect unless a separate texturing module is used.")
        if not (512 <= arg.texture_size <= 4096):
             raise HTTPException(status_code=400, detail="Texture size must be between 512 and 4096")
    
    if arg.output_format not in ["glb", "obj", "ply", "stl"]: # Match app.py export options
        raise HTTPException(status_code=400, detail="Unsupported output format. Supported: glb, obj, ply, stl")



async def _gen_3d_get_image(file: Optional[UploadFile], image_base64: Optional[str]) -> Image.Image:
    if image_base64:
        try:
            # base64 branch
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            data = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(data)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")
    else:
        try:
            content = await file.read()
            pil_image = Image.open(io.BytesIO(content)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"get_image - issue opening the image file: {str(e)}")
    return pil_image



# Reads multiple images from either UploadFile objects or base64 strings,
# and returns a list of PIL.Image.
async def _load_images_into_list( files: Optional[List[UploadFile]] = None,
                                  images_base64: Optional[List[str]] = None) -> List[Image.Image]:
    all_images = []
    files = files or []
    images_base64 = images_base64 or []
    # 1) Base64-encoded:
    for b64_str in images_base64:
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,")[1]
        img = await _gen_3d_get_image(file=None, image_base64=b64_str)
        all_images.append(img)
    # 2) Files:
    for f in files:
        img = await _gen_3d_get_image(file=f, image_base64=None)
        all_images.append(img)

    if not all_images:
        raise HTTPException(status_code=400, detail="No images provided (files or base64).")

    return all_images



async def _run_pipeline_generate_3d(pil_input_image: Image.Image, arg: GenerationArgForm) -> Dict:
    """
    Generates a 3D mesh from a single PIL image using Hi3DGen.
    Manages model movement between CPU and GPU for VRAM efficiency.
    """
    def worker():
        # Get models from global state
        hi3dgen_pipeline_cpu = state.pipeline
        normal_predictor_cpu_wrapper = state.normal_predictor
        # Determine target device for active processing (e.g., "cuda" or "cpu")
        # state._device_preference should hold this (e.g., "cuda" if available, else "cpu")
        active_processing_device = state._device_preference 
        
        if hi3dgen_pipeline_cpu is None or normal_predictor_cpu_wrapper is None:
            raise RuntimeError("Models not initialized in state. Cannot generate.")

        mesh_trimesh_output = None
        
        try:
            # --- STAGE 1: Normal Prediction ---
            logger.info(f"Normal Prediction: Preparing on device '{active_processing_device}'...")
            # The normal_predictor_cpu_wrapper might have an internal 'model' (YOSONormalsPipeline)
            actual_normal_predictor_module = None
            if hasattr(normal_predictor_cpu_wrapper, 'model'): # This is typical for StableNormal_turbo
                actual_normal_predictor_module = normal_predictor_cpu_wrapper.model
            elif isinstance(normal_predictor_cpu_wrapper, torch.nn.Module): # Fallback if wrapper itself is the model
                actual_normal_predictor_module = normal_predictor_cpu_wrapper
            else:
                raise TypeError("Normal predictor in state is not a recognized PyTorch module or wrapper.")

            if actual_normal_predictor_module and hasattr(actual_normal_predictor_module, 'to'):
                actual_normal_predictor_module.to(active_processing_device)
                state._active_device = active_processing_device # Track current device
                logger.debug(f"Normal predictor module moved to {active_processing_device}")

            if torch.cuda.is_available() and active_processing_device == "cuda":
                torch.cuda.empty_cache()
            
            seed_to_use = arg.seed
            if seed_to_use == -1 or seed_to_use == 0:
                seed_to_use = np.random.randint(1, np.iinfo(np.int32).max)
            
            logger.info("Preprocessing input image for normal map generation (on CPU)...")
            # Preprocessing for normal map can often be done on CPU before moving normal predictor to GPU
            # hi3dgen_pipeline_cpu is on CPU here.
            processed_for_normal = hi3dgen_pipeline_cpu.preprocess_image(pil_input_image.convert("RGBA"), resolution=1024)

            logger.info("Generating normal map...")
            # The normal_predictor_cpu_wrapper is called; its internal model is now on active_processing_device
            normal_image_pil = normal_predictor_cpu_wrapper(processed_for_normal, resolution=768, match_input_resolution=True, data_type='object')
            logger.info("Normal map generated.")

        finally:
            # Move Normal Predictor back to CPU
            if 'actual_normal_predictor_module' in locals() and actual_normal_predictor_module is not None:
                if hasattr(actual_normal_predictor_module, 'cpu'):
                    actual_normal_predictor_module.cpu()
                    logger.debug("Normal predictor module moved back to CPU.")
            state._active_device = "cpu" # Reset active device tracker
            if torch.cuda.is_available() and active_processing_device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Normal Prediction stage finished, model returned to CPU.")

        if normal_image_pil is None: # Critical check
            raise RuntimeError("Normal map generation failed.")

        # --- STAGE 2: Hi3DGen Pipeline Execution ---
        try:
            logger.info(f"Hi3DGen Pipeline: Preparing on device '{active_processing_device}'...")
            if hasattr(hi3dgen_pipeline_cpu, 'to'):
                hi3dgen_pipeline_cpu.to(active_processing_device) # Move the main pipeline
                state._active_device = active_processing_device
                logger.debug(f"Hi3DGenPipeline moved to {active_processing_device}")
            
            if torch.cuda.is_available() and active_processing_device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Running Hi3DGen pipeline...")
            pipeline_output_internal = hi3dgen_pipeline_cpu.run( # Now hi3dgen_pipeline_cpu is on active_processing_device
                image=normal_image_pil,
                seed=seed_to_use,
                formats=["mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": arg.ss_sampling_steps,
                    "cfg_strength": arg.ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": arg.slat_sampling_steps,
                    "cfg_strength": arg.slat_guidance_strength,
                },
            )
            internal_mesh_representation = pipeline_output_internal['mesh'][0]
            if internal_mesh_representation is None:
                raise RuntimeError("Hi3DGen pipeline did not return a mesh representation.")
            
            logger.info("Converting to Trimesh object...")
            mesh_trimesh_output = internal_mesh_representation.to_trimesh(transform_pose=True)
            logger.info("Trimesh object created.")

        finally:
            # Move Hi3DGen Pipeline back to CPU
            if hasattr(hi3dgen_pipeline_cpu, 'cpu'):
                hi3dgen_pipeline_cpu.cpu()
                logger.debug("Hi3DGenPipeline moved back to CPU.")
            state._active_device = "cpu"
            if torch.cuda.is_available() and active_processing_device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Hi3DGen Pipeline stage finished, model returned to CPU.")

        if mesh_trimesh_output is None: # Critical check
            raise RuntimeError("3D mesh generation failed.")
            
        outputs = {
            "mesh": [mesh_trimesh_output],
            "original_image": pil_input_image
        }
        return outputs
            
    # Run the synchronous worker function in a separate thread
    return await asyncio.to_thread(worker)



def normalize_meshSimplify_ratio(ratio: float) -> float:
    """Normalize poly_count_pcnt to [0,1] range"""
    if ratio > 1.0:  # Detect [0,100] range
        return ratio / 100.0
    return ratio


def simplify_mesh_open3d(in_mesh:trimesh.Trimesh, 
                         poly_count_pcnt01: float=0.5) -> trimesh.Trimesh:
    """
    Simplifies trimesh using Open3D by a reduction percentage (0.0-1.0).
    E.g., reduct_pct01=0.7 means target is 30% of original faces.
    Returns original mesh on error, invalid input, or if no reduction is practical.
    """
    if not isinstance(in_mesh, trimesh.Trimesh): # Basic type check
        print("Simplify ERR: Invalid input type.")
        return in_mesh

    # reduct_pct: 0.0 = no reduction, <1.0. E.g. 0.5 means keep 30%
    if not (0.0 < poly_count_pcnt01 < 1.0):
        print(f"Simplify skip: reduct_pct {poly_count_pcnt01:.2f} out of (0,1) range.")
        return in_mesh

    current_tris = len(in_mesh.faces)
    if current_tris == 0: return in_mesh # No faces to simplify

    # Calculate target triangles: keep (1 - reduct_pct) portion
    target_tris = int(current_tris * poly_count_pcnt01)
    target_tris = max(1, target_tris) # Ensure at least 1 triangle if original had faces

    if target_tris >= current_tris: # No actual reduction
        print(f"Simplify skip: Target {target_tris} >= current {current_tris}.")
        return in_mesh

    print(f"Simplifying: {current_tris} faces -> ~{target_tris} faces ({ (1.0-poly_count_pcnt01)*100:.0f}% original).")
    
    o3d_m = o3d.geometry.TriangleMesh() # Convert to Open3D format
    o3d_m.vertices = o3d.utility.Vector3dVector(in_mesh.vertices)
    o3d_m.triangles = o3d.utility.Vector3iVector(in_mesh.faces)

    try:
        # Quadric decimation is generally a good default
        simplified_o3d_m = o3d_m.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        
        s_verts = np.asarray(simplified_o3d_m.vertices)
        s_faces = np.asarray(simplified_o3d_m.triangles)

        # Critical: check for empty mesh post-simplification
        if s_faces.size == 0 and current_tris > 0:
            print("Simplify WARN: Empty mesh result. Reverting.")
            return in_mesh
        
        # Convert back to Trimesh, process=True recomputes normals
        return trimesh.Trimesh(vertices=s_verts, faces=s_faces, process=True)
    except Exception as e:
        print(f"Simplify ERR: Open3D failed ({e}). Reverting.")
        traceback.print_exc()
        return in_mesh # Fallback to original mesh


def unwrap_mesh_with_xatlas(input_mesh: trimesh.Trimesh, atlas_resolution:int=1024) -> trimesh.Trimesh:
    """
    Unwraps a Trimesh object using xatlas and returns a new Trimesh object
    with generated UVs.
    """
    print("UV Unwrapping with xatlas: Starting process...")

    input_vertices_orig = input_mesh.vertices.astype(np.float32) 
    input_faces_orig = input_mesh.faces.astype(np.uint32)      
    vertex_normals_from_trimesh = input_mesh.vertex_normals 
    input_normals_orig = np.ascontiguousarray(vertex_normals_from_trimesh, dtype=np.float32)

    print(f"  Input mesh: {input_vertices_orig.shape[0]} vertices, {input_faces_orig.shape[0]} faces.")

    atlas = xatlas.Atlas()
    atlas.add_mesh(input_vertices_orig, input_faces_orig, input_normals_orig) 

    # Configure xatlas ChartOptions
    chart_options = xatlas.ChartOptions()
    
    # Allow more stretch/distortion within a chart. Default is 2.0.
    chart_options.max_cost = 8.0 # keep it high.
    
    # Reduce the penalty for creating seams along edges with differing normals.
    # Default is 4.0. Lower values make xatlas less likely to cut based on normal changes.
    chart_options.normal_seam_weight = 1.0 # Significantly reduced from default 4.0
    
    # chart_options.straightness_weight = 3.0 # Default 6.0; lower might allow less straight chart boundaries.
    # chart_options.roundness_weight = 0.005 # Default 0.01

    # Configure xatlas PackOptions
    pack_options = xatlas.PackOptions()
    pack_options.resolution = atlas_resolution 
    pack_options.padding = 2

    print(f"  Running xatlas.generate() with ChartOptions(max_cost={chart_options.max_cost:.2f}, normal_seam_weight={chart_options.normal_seam_weight:.2f}) and PackOptions(resolution={pack_options.resolution}, padding={pack_options.padding})...")
    atlas.generate(chart_options=chart_options, pack_options=pack_options) 
    print(f"  xatlas generated atlas with dimensions: width={atlas.width}, height={atlas.height}")
    
    # --- xatlas output processing ---
    # Important Note on xatlas-python's get_mesh() behavior for meshes added via add_mesh():
    # The first returned array (v_out_xref_data) is NOT new spatial vertex coordinates.
    # It's an array of 'cross-reference' indices (uint32) pointing to the ORIGINAL input vertices.
    # For each new vertex created by xatlas due to UV seam splitting, this xref tells us
    # which original vertex its spatial position should be copied from.
    v_out_xref_data, f_out_indices, uv_coords_from_xatlas = atlas.get_mesh(0)
    
    num_new_vertices = uv_coords_from_xatlas.shape[0]
    if v_out_xref_data.shape == (num_new_vertices,): 
        xref_indices = v_out_xref_data.astype(np.uint32) 
        if np.any(xref_indices >= input_vertices_orig.shape[0]) or np.any(xref_indices < 0):
             raise ValueError("Invalid xref values from xatlas - out of bounds for original input vertices.")
        final_vertices_spatial = input_vertices_orig[xref_indices]
    elif v_out_xref_data.shape == (num_new_vertices, 3): 
        print("  Warning: xatlas.get_mesh() returned 3D vertex data directly, which is unexpected for add_mesh workflow.")
        final_vertices_spatial = v_out_xref_data.astype(np.float32)
    else:
        raise ValueError(f"Unexpected shape for vertex/xref data from xatlas.get_mesh: {v_out_xref_data.shape}.")

    # --- UV Handling (remains the same as before) ---
    final_uvs = uv_coords_from_xatlas.astype(np.float32)
    if np.any(final_uvs > 1.5): 
        print("  UVs appear to be in pixel coordinates. Normalizing...")
        if atlas.width > 0 and atlas.height > 0: 
            final_uvs /= np.array([atlas.width, atlas.height], dtype=np.float32)
        else:
            print("  WARNING: Atlas width/height is 0, cannot normalize pixel UVs. Using unnormalized.")
    else:
        min_uv = final_uvs.min(axis=0) if final_uvs.size > 0 else "N/A"
        max_uv = final_uvs.max(axis=0) if final_uvs.size > 0 else "N/A"
        print(f"  UVs appear to be normalized. Min: {min_uv}, Max: {max_uv}")
    
    output_mesh = trimesh.Trimesh(vertices=final_vertices_spatial, faces=f_out_indices, process=False) 
    
    if final_uvs.shape != (final_vertices_spatial.shape[0], 2):
        raise ValueError(f"Shape mismatch for final UVs before Trimesh assignment.")

    material = trimesh.visual.material.PBRMaterial(name='defaultXatlasMat')
    output_mesh.visual = trimesh.visual.TextureVisuals(uv=final_uvs, material=material)
    
    if hasattr(output_mesh.visual, 'uv') and output_mesh.visual.uv is not None:
        print(f"  Trimesh object successfully created with UVs, Shape: {output_mesh.visual.uv.shape}")
    else:
        print("  ERROR: Trimesh object does NOT have UVs assigned after TextureVisuals call.")
        raise RuntimeError("Failed to assign UVs to the Trimesh object.")

    print("UV Unwrapping with xatlas: Process complete.")
    return output_mesh



async def _run_pipeline_generate_glb(
    outputs: Dict,             # Expected to contain 'mesh': [trimesh_object]
    poly_count_pcnt: float, # Received from args (0-100)
    texture_size: int,          # Used as pack_options.resolution for xatlas
    apply_texture: bool = False,# Placeholder, not used by Hi3DGen core
    output_format: str = "glb",
):
    """
    Generate final 3D model file (GLB, OBJ, etc.) from Trimesh object.
    Also unwraps UVs using xatlas.
    """

    def worker(): # This synchronous function will be run in a thread
        try:
            start_time = time.time()
            
            if torch.cuda.is_available(): # Good practice before/after GPU-intensive ops
                torch.cuda.empty_cache()
            
            raw_mesh_trimesh: trimesh.Trimesh = outputs["mesh"][0] # Start with the raw mesh

            if not isinstance(raw_mesh_trimesh, trimesh.Trimesh):
                logger.error(f"Expected a trimesh.Trimesh object, but got {type(raw_mesh_trimesh)}")
                raise ValueError("Invalid mesh object provided for final model generation.")

            logger.info(f"Received mesh for export with {len(raw_mesh_trimesh.faces)} faces and {len(raw_mesh_trimesh.vertices)} vertices.")

            # Convert poly_count_pcnt (0-100 from UI, % to reduce by) to reduct_pct (0.0-1.0 for simplify_mesh_open3d)
            reduction_percentage_float = normalize_meshSimplify_ratio(poly_count_pcnt)
            mesh_to_export = simplify_mesh_open3d(raw_mesh_trimesh, reduction_percentage_float)

            if not apply_texture or output_format.lower() == "stl": # STL doesn't support UVs
                logger.info("Skipping xatlas UV unwrapping / texturing.")
            else:
                mesh_to_export = unwrap_mesh_with_xatlas(mesh_to_export)

            # Texturing: Hi3DGen does not provide an integrated texturing pipeline.
            if apply_texture:
                logger.warning("apply_texture is True, but Hi3DGen pipeline does not have integrated texturing. This option will be ignored for model export unless a separate texturing module is implemented and called here.")
            
            # Export to the requested format
            model_filename = f"model.{output_format.lower()}"
            model_path = file_manager.get_temp_path(model_filename) # file_manager handles actual path
            
            logger.info(f"Exporting {output_format.upper()} model to {model_path}")
            # Trimesh export handles various formats and includes UVs by default if mesh.visual.uv exists.
            mesh_to_export.export(str(model_path)) 
            
            file_size_kb = model_path.stat().st_size / 1024
            logger.info(f"Exported {output_format.upper()} model to {model_path} in {time.time() - start_time:.2f}s. File size: {file_size_kb:.2f} KB")
            
            return model_path # Return path to the generated file
            
        except Exception as e:
            logger.error(f"Error generating final model file ({output_format}): {str(e)}")
            # Log the full traceback for better debugging from API logs
            logger.error(traceback.format_exc())
            raise # Re-raise the exception to be caught by the endpoint handler
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Final cleanup
    
    return await asyncio.to_thread(worker)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@router.get("/ping")
async def ping():
    """Root endpoint to check server status."""
    busy = is_generation_in_progress()
    return {
        "status": "running",
        "message": "Stable3DGen (Hi3DGen) API is operational",
        "busy": busy
    }


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get status of the single current/last generation.
    """
    return StatusResponse(
        status=current_generation["status"],
        progress=current_generation["progress"],
        message=current_generation["message"],
        busy=is_generation_in_progress(),
    )


@router.post("/generate_no_preview", response_model=GenerationResponse)
async def generate_no_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArgForm = Depends()
):
    """Generate a 3D model directly from a single input image."""
    print() 
    logger.info(f"Request: /generate_no_preview, Output format: {arg.output_format.upper()}")
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation")
    
    start_time = time.time() 
    reset_current_generation()
    try:
        _gen_3d_validate_params(file, image_base64, arg)
        single_pil_image = await _gen_3d_get_image(file, image_base64)

        update_current_generation(status=TaskStatus.PROCESSING, progress=10, message="Generating 3D structure (incl. normal map)...")
        outputs = await _run_pipeline_generate_3d(single_pil_image, arg)
        update_current_generation(progress=80, message="Trimesh object generated, preparing final model file...", outputs=outputs)

        update_current_generation(progress=95, message=f"Generating {arg.output_format.upper()} file...")
        await _run_pipeline_generate_glb(
            outputs, 
            arg.poly_count_pcnt, 
            arg.texture_size,        
            apply_texture=arg.apply_texture, 
            output_format=arg.output_format,
        )
        current_generation["model_url"] = f"/download/model?format={arg.output_format}"

        update_current_generation(status=TaskStatus.COMPLETE, progress=100, message="Generation complete")
        await cleanup_generation_files(keep_model=True)

        duration = time.time() - start_time
        logger.info(f"Generation completed in {duration:.2f}s. Format: {arg.output_format.upper()}")
        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url=current_generation["model_url"] 
        )
    except CancelledException:
        update_current_generation(status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in /generate_no_preview: {error_trace}")
        update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()




@router.post("/generate_multi_no_preview", response_model=GenerationResponse)
async def generate_multi_no_preview(
    file_list: Optional[List[UploadFile]] = File(None),
    image_list_base64: Optional[List[str]] = Form(None),
    arg: GenerationArgForm = Depends(),
):
    """
    Generate a 3D model from a list of images.
    IMPORTANT: This currently processes ONLY THE FIRST image from the list due to
    Hi3DGen's app.py focusing on single-image workflows and unclear status of its
    direct multi-image pipeline capabilities for this API's context.
    """
    print()
    logger.info(f"Request: /generate_multi_no_preview, Output format: {arg.output_format.upper()}")
    logger.warning("Current /generate_multi_no_preview processes ONLY THE FIRST image from the input list.")

    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation")
    
    start_time = time.time() 
    reset_current_generation()

    try:
        _gen_3d_validate_params(file_list, image_list_base64, arg)
        all_pil_images = await _load_images_into_list(file_list, image_list_base64)
        
        if not all_pil_images:
            raise HTTPException(status_code=400, detail="No images provided for multi-view generation.")

        first_pil_image = all_pil_images[0] 
        # To enable true multi-image: _run_pipeline_generate_3d would need to handle a list of images
        # and correctly interface with hi3dgen_pipeline.run_multi_image (if deemed ready).

        update_current_generation(status=TaskStatus.PROCESSING, progress=10, message="Generating 3D structure (using first image, incl. normal map)...")
        outputs = await _run_pipeline_generate_3d(first_pil_image, arg) # Pass only the first image
        update_current_generation(progress=50, message="Trimesh object generated, preparing final model file...", outputs=outputs)

        update_current_generation(progress=70, message=f"Generating {arg.output_format.upper()} file...")
        await _run_pipeline_generate_glb(
            outputs, 
            arg.poly_count_pcnt,
            arg.texture_size,
            apply_texture=arg.apply_texture,
            output_format=arg.output_format,
        )
        current_generation["model_url"] = f"/download/model?format={arg.output_format}"

        update_current_generation(status=TaskStatus.COMPLETE, progress=100, message="Generation complete (from first image)")
        await cleanup_generation_files(keep_model=True)

        duration = time.time() - start_time
        logger.info(f"Multi-view generation (using first image) completed in {duration:.2f}s. Format: {arg.output_format.upper()}")
        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete (processed first image from list)",
            model_url=current_generation["model_url"]
        )
    except CancelledException:
        update_current_generation(status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in /generate_multi_no_preview: {error_trace}")
        update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()



@router.post("/generate", response_model=GenerationResponse)
async def process_ui_generation_request(
    data: Dict
):
    """Process generation request from the UI panel and redirect to appropriate endpoint."""
    try:
        arg = GenerationArgForm(
            seed = int(data.get("seed", 123)),
            ss_guidance_strength = float(data.get("ss_guidance_strength", 3.0)),
            ss_sampling_steps = int(data.get("ss_sampling_steps", 50)),
            slat_guidance_strength = float(data.get("slat_guidance_strength", 3.0)),
            slat_sampling_steps = int(data.get("slat_sampling_steps", 6)),
            poly_count_pcnt = float(data.get("poly_count_pcnt", 95.0)), # UI sends 0-100
            apply_texture = bool(data.get("apply_texture", False)),
            texture_size = int(data.get("texture_size", 1024)),
            output_format = data.get("output_format", "glb") # Get output_format from UI
        )
        # Get images from input
        images_base64 = data.get("single_multi_img_input", [])
        if not images_base64:
            raise HTTPException(status_code=400, detail="No images provided")

        # For now, Hi3DGen app.py focuses on single.
        # If images_base64 always contains one image for this path:
        if len(images_base64) == 1:
            response = await generate_no_preview( # Call generate_no_preview for single image
                file=None,
                image_base64=images_base64[0],
                arg=arg
            )
        elif len(images_base64) > 1:
             # to support multi-image through this UI endpoint eventually:
            logger.info("UI request with multiple images, using generate_multi_no_preview.")
            response = await generate_multi_no_preview(
                file_list=None,
                image_list_base64=images_base64,
                arg=arg
            )
        else: # Should not happen if check above passes
            raise HTTPException(status_code=400, detail="Image list is empty after initial check.")

        return response

    except HTTPException as httpe: # Re-raise HTTP exceptions
        raise httpe
    except Exception as e:
        logger.error(f"Error processing UI generation request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in UI request processing: {str(e)}")


# for example:
#   "make_meshes_and_tex",
#   "retexture",
#   "retexture_via_masks"  etc (see api-documentation.html)
@router.get("/info/supported_operations")
async def get_supported_operation_types():
   return ["make_meshes_and_tex"]



@router.post("/interrupt")
async def interrupt_generation():
    """Interrupt the current generation if one is in progress."""
    logger.info("Client cancelled the generation.")
    if not is_generation_in_progress():
        return {"status": "no_generation_in_progress"}
    cancel_event.set()  # <-- Signal cancellation
    return {"status": "interrupt_requested"}
    


@router.get("/download/model")
async def download_model(format: str = Query("glb", enum=["glb", "obj", "ply", "stl"])):
    """Download final 3D model (GLB, OBJ, PLY, STL)."""
    logger.info(f"Client is downloading a model in {format.upper()} format.")
    model_filename = f"model.{format.lower()}"
    model_path = file_manager.get_temp_path(model_filename)

    if not model_path.exists():
        logger.error(f"Model file {model_filename} not found at {model_path}")
        # Try to find if any model exists, maybe format mismatch
        found_formats = []
        for ext in ["glb", "obj", "ply", "stl"]:
            if file_manager.get_temp_path(f"model.{ext}").exists():
                found_formats.append(ext)
        if found_formats:
            logger.error(f"Requested format {format} not found, but other formats exist: {found_formats}. Defaulting to GLB if available, or first found.")
            if "glb" in found_formats:
                model_path = file_manager.get_temp_path("model.glb")
                format = "glb"
            else:
                format = found_formats[0]
                model_path = file_manager.get_temp_path(f"model.{format}")
                model_filename = f"model.{format}"
            if not model_path.exists(): # Should not happen if found_formats is populated
                 raise HTTPException(status_code=404, detail=f"Mesh not found. No model files available.")
        else:
            raise HTTPException(status_code=404, detail=f"Mesh not found. No model file {model_filename} or any other format found.")
    media_type_map = {
        "glb": "model/gltf-binary",
        "obj": "text/plain", # Or model/obj
        "ply": "application/octet-stream", # Or model/ply
        "stl": "model/stl"
    }
    return FileResponse(
        str(model_path),
        media_type=media_type_map.get(format, "application/octet-stream"),
        filename=model_filename # e.g., model.glb, model.obj
    )

@router.get("/download/spz-ui-layout/generation-3d-panel")
async def get_generation_panel_layout():
    """Return the UI layout for the generation panel."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'layout_generation_3d_panel.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # By using Response(content=content, media_type="text/plain; charset=utf-8")
            # - Bypass the automatic JSON encoding
            # - Explicitly tell the client this is plain text (not JSON)
            # - Ensure proper UTF-8 encoding is maintained
            # This way Unity receives the layout text exactly as it appears in the file.
            # It keeps proper line breaks and formatting intact, with special characters not being escaped:
            return Response(content=content,  media_type="text/plain; charset=utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Layout file not found")
