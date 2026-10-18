# MIT License

# Copyright (c) Microsoft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) [2025] [Microsoft]
# Copyright (c) [2025] [Chongjie Ye] 
# SPDX-License-Identifier: MIT
# This file has been modified by Chongjie Ye on 2025/04/10
# Original file was released under MIT, with the full license text # available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
# This modified file is released under the same license.

import gradio as gr
import os
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import traceback
import datetime
import shutil
import torch
import numpy as np
from hi3dgen.pipelines import Hi3DGenPipeline
import tempfile
import trimesh
import open3d as o3d
import xatlas

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Initialize placeholders for models
hi3dgen_pipeline = None
normal_predictor = None


def verify_weights(weights_dir: str) -> dict:
    """
    Verify that all required models are present in the weights directory.
    Does NOT download anything - all models must be pre-downloaded.
    """
    import os
    import sys
    from pathlib import Path

    model_ids = [
        "Stable-X/trellis-normal-v0-1", "Stable-X/yoso-normal-v1-8-1", "ZhengPeng7/BiRefNet"
    ]
    cached_paths = {}

    print("Verifying locally cached models...")
    all_present = True

    for repo_id in model_ids:
        local_repo_root = Path(weights_dir) / f"models--{repo_id.replace('/', '--')}"
        
        if local_repo_root.exists():
            print(f"  ✅ Found: {repo_id} at {local_repo_root}")
            cached_paths[repo_id] = str(local_repo_root)
        else:
            print(f"  ❌ MISSING: {repo_id} (expected at {local_repo_root})")
            all_present = False
    
    # Check GitHub repos
    github_repos = [
        ("facebookresearch", "dinov2"),
    ]
    
    for github_user, repo_name in github_repos:
        local_repo_root = Path(weights_dir) / f"github--{github_user}--{repo_name}"
        if local_repo_root.exists():
            print(f"  ✅ Found: {github_user}/{repo_name} at {local_repo_root}")
        else:
            print(f"  ❌ MISSING: {github_user}/{repo_name} (expected at {local_repo_root})")
            all_present = False
    
    if not all_present:
        print("\n⚠️  Some models are missing. Please run DownloadModels.py to download them.")
        print("   The app will attempt to proceed but may fail if models are missing.")
    else:
        print("\n✅ All models verified and present locally!")
    
    sys.stdout.flush()
    return cached_paths



def preprocess_mesh(mesh_prompt):
    print("Processing mesh")
    trimesh_mesh = trimesh.load_mesh(mesh_prompt)
    trimesh_mesh.export(mesh_prompt+'.glb')
    return mesh_prompt+'.glb'


def preprocess_image(image):
    global hi3dgen_pipeline
    if image is None: return None
    # hi3dgen_pipeline is loaded on CPU at startup. Critical if None.
    if hi3dgen_pipeline is None: 
        raise RuntimeError("FATAL: Hi3DGenPipeline not loaded. Cannot preprocess.")
    # .preprocess_image is expected to work with the pipeline on CPU
    # and manage its own sub-component (e.g., BiRefNet) devices.
    return hi3dgen_pipeline.preprocess_image(image, resolution=1024)


def simplify_mesh_open3d(in_mesh:trimesh.Trimesh, 
                         poly_count_pcnt: float=0.5) -> trimesh.Trimesh:
    """
    Simplifies trimesh using Open3D to a poly_count_pcnt percentage (0.0-1.0).
    Returns original mesh on error, invalid input, or if no reduction is practical.
    """
    if not isinstance(in_mesh, trimesh.Trimesh): # Basic type check
        print("Simplify ERR: Invalid input type.")
        return in_mesh

    if not (0.0 < poly_count_pcnt < 1.0):
        print(f"Simplify skip: poly_count_pcnt {poly_count_pcnt:.2f} out of (0,1) range.")
        return in_mesh

    current_tris = len(in_mesh.faces)
    if current_tris == 0: return in_mesh # No faces to simplify

    target_tris = int(current_tris * poly_count_pcnt)
    target_tris = max(1, target_tris) # Ensure at least 1 triangle if original had faces

    if target_tris >= current_tris: # No actual changes
        print(f"Simplify skip: Target {target_tris} >= current {current_tris}.")
        return in_mesh

    print(f"Simplifying: {current_tris} faces -> ~{target_tris} faces ({ (1.0-poly_count_pcnt)*100:.0f}% original).")
    
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


def unwrap_mesh_with_xatlas(input_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
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
    pack_options.resolution = 1024 
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



def generate_3d_multi(images, seed=-1,  
                      ss_guidance_strength=3, ss_sampling_steps=50,
                      slat_guidance_strength=3, slat_sampling_steps=6,
                      poly_count_pcnt: float = 0.5,
                      mode='multidiffusion'):
    """Generate 3D model from multiple input images"""
    global hi3dgen_pipeline

    if not images or len(images) == 0: 
        print("No input images provided. Aborting generation.")
        return None, None, None
    
    # Filter out None images
    valid_images = [img for img in images if img is not None]
    if len(valid_images) == 0:
        print("No valid input images provided. Aborting generation.")
        return None, None, None
        
    print(f"Processing {len(valid_images)} images in {mode} mode...")
    
    if seed == -1: 
        seed = np.random.randint(0, MAX_SEED)
    
    if hi3dgen_pipeline is None:
        print("FATAL: Hi3DGenPipeline not loaded. Cannot generate 3D.")
        return None, None, None 

    gradio_model_path = None 
    
    # 3D Generation using multi-image pipeline
    pipeline_on_gpu = False
    try:
        if torch.cuda.is_available():
            print("Multi-Image 3D Generation: Moving Hi3DGen pipeline to GPU...")
            hi3dgen_pipeline.cuda()
            pipeline_on_gpu = True
        
        print(f"Multi-Image 3D Generation: Running Hi3DGen pipeline with {len(valid_images)} images...")
        outputs = hi3dgen_pipeline.run_multi_image(
            valid_images, 
            seed=seed, 
            formats=["mesh"], 
            preprocess_image=True,
            mode=mode,
            sparse_structure_sampler_params={"steps": ss_sampling_steps, "cfg_strength": ss_guidance_strength},
            slat_sampler_params={"steps": slat_sampling_steps, "cfg_strength": slat_guidance_strength},
        )
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        output_dir = os.path.join(TMP_DIR, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        mesh_path_glb = os.path.join(output_dir, "mesh.glb")

        raw_mesh_trimesh = outputs['mesh'][0].to_trimesh(transform_pose=True)
        mesh_for_uv_unwrap = simplify_mesh_open3d(raw_mesh_trimesh, poly_count_pcnt)
        
        # Call the new UV unwrapping function
        unwrapped_mesh_trimesh = unwrap_mesh_with_xatlas(mesh_for_uv_unwrap)
        
        # Export GLB
        print(f"Exporting GLB to {mesh_path_glb}...")
        unwrapped_mesh_trimesh.export(mesh_path_glb) 
        print(f"SUCCESS: Multi-image GLB exported.")

        gradio_model_path = mesh_path_glb 

    except Exception as e:
        print(f"ERROR in Multi-Image 3D Generation stage: {e}")
        traceback.print_exc() 
        gradio_model_path = None 
    finally:
        if pipeline_on_gpu: 
            print("Multi-Image 3D Generation: Moving Hi3DGen pipeline to CPU...")
            hi3dgen_pipeline.cpu() 
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
            
    # For multi-image, we return the first preprocessed image as the "normal" preview
    first_preprocessed = None
    if valid_images:
        try:
            first_preprocessed = hi3dgen_pipeline.preprocess_image(valid_images[0])
        except Exception as e:
            print(f"Warning: Could not preprocess first image for preview: {e}")
            
    return first_preprocessed, gradio_model_path, gradio_model_path


def generate_3d(image, seed=-1,  
                ss_guidance_strength=3, ss_sampling_steps=50,
                slat_guidance_strength=3, slat_sampling_steps=6,
                poly_count_pcnt: float = 0.5):
    # global hi3dgen_pipeline # normal_predictor is now function-local

    if image is None: 
        print("Input image is None. Aborting generation.")
        return None, None, None 
    if seed == -1: seed = np.random.randint(0, MAX_SEED)
    
    if hi3dgen_pipeline is None: # hi3dgen_pipeline is still global and loaded at startup
        print("FATAL: Hi3DGenPipeline not loaded. Cannot generate 3D.")
        return None, None, None 

    normal_image_pil = None
    gradio_model_path = None 
    
    # --- STAGE 1: Normal Prediction ---
    current_normal_predictor_instance = None 
    try:
        print("Normal Prediction: Loading model from local cache...") 
        # StableNormal ships with the app (third_party/StableNormal); the weights come from the weights directory
        stablenormal_local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "StableNormal")
        current_normal_predictor_instance = torch.hub.load(
            stablenormal_local_path, "StableNormal_turbo", source='local', trust_repo=True,
            yoso_version='yoso-normal-v1-8-1', local_cache_dir=WEIGHTS_DIR, device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        print("Normal Prediction: Generating normal map...")
        normal_image_pil = current_normal_predictor_instance(image, resolution=768, match_input_resolution=True, data_type='object')
    except Exception as e:
        print(f"ERROR in Normal Prediction stage: {e}")
        traceback.print_exc()
    finally:
        if current_normal_predictor_instance is not None: 
            print("Normal Prediction: Unloading model...")
            if hasattr(current_normal_predictor_instance, 'model') and hasattr(current_normal_predictor_instance.model, 'to'):
                current_normal_predictor_instance.model.to('cpu', silence_dtype_warnings=True)
            del current_normal_predictor_instance 
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if normal_image_pil is None: 
        print("ERROR: Normal map not generated after Stage 1. Aborting 3D generation.")
        return None, None, None

    # --- STAGE 2: 3D Generation & UV Unwrapping ---
    pipeline_on_gpu = False
    try:
        if torch.cuda.is_available():
            print("3D Generation: Moving Hi3DGen pipeline to GPU...")
            hi3dgen_pipeline.cuda(); pipeline_on_gpu = True
        
        print("3D Generation: Running Hi3DGen pipeline...")
        outputs = hi3dgen_pipeline.run(
            normal_image_pil, seed=seed, formats=["mesh",], preprocess_image=False,
            sparse_structure_sampler_params={"steps": ss_sampling_steps, "cfg_strength": ss_guidance_strength},
            slat_sampler_params={"steps": slat_sampling_steps, "cfg_strength": slat_guidance_strength},
        )
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        output_dir = os.path.join(TMP_DIR, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        mesh_path_glb = os.path.join(output_dir, "mesh.glb")

        raw_mesh_trimesh = outputs['mesh'][0].to_trimesh(transform_pose=True)
        mesh_for_uv_unwrap = simplify_mesh_open3d(raw_mesh_trimesh, poly_count_pcnt)
        
        # Call the new UV unwrapping function
        unwrapped_mesh_trimesh = unwrap_mesh_with_xatlas(mesh_for_uv_unwrap )
        
        # Export GLB
        # Trimesh is expected to include UVs in GLB by default if they are present in mesh.visual.uv.
        print(f"Exporting GLB to {mesh_path_glb}...")
        unwrapped_mesh_trimesh.export(mesh_path_glb) 
        print(f"SUCCESS: GLB exported.")

        gradio_model_path = mesh_path_glb 

    except Exception as e:
        print(f"ERROR in 3D Generation or UV Unwrapping stage: {e}")
        traceback.print_exc() 
        gradio_model_path = None 
    finally:
        if pipeline_on_gpu: 
            print("3D Generation: Moving Hi3DGen pipeline to CPU...")
            hi3dgen_pipeline.cpu() 
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
    return normal_image_pil, gradio_model_path, gradio_model_path



def process_multi_image_upload(files):
    """Process multiple uploaded image files"""
    if not files:
        return [], None
    
    images = []
    preview_images = []
    
    for file in files:
        try:
            # Load image
            img = Image.open(file.name)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            preview_images.append(img)
            print(f"Loaded image: {file.name} ({img.size})")
        except Exception as e:
            print(f"Error loading image {file.name}: {e}")
            continue
    
    print(f"Successfully loaded {len(images)} images for multi-image processing")
    return preview_images, images


def preprocess_multi_images(images_list):
    """Preprocess multiple images for generation"""
    if not images_list:
        return []
    
    processed_images = []
    for img in images_list:
        try:
            if img is not None:
                processed = preprocess_image(img)
                if processed is not None:
                    processed_images.append(processed)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
    
    return processed_images


def convert_mesh(mesh_path: str, export_format: str) -> Optional[str]:
    """
    Converts mesh at mesh_path to export_format. Returns path to new temp file.
    If export_format is GLB and input is GLB, copies input to a new temp file.
    """
    if not mesh_path or not os.path.exists(mesh_path):
        print(f"convert_mesh: Invalid input mesh_path: {mesh_path}")
        return None

    temp_file_path = None # Define outside try for cleanup
    try:
        # Use trimesh's util for a NamedTemporaryFile context manager
        with trimesh.util.NamedTemporaryFile(suffix=f".{export_format.lower()}", delete=False) as tmp_out_file_obj:
            temp_file_path = tmp_out_file_obj.name

        # If GLB to GLB, copy original to preserve UVs perfectly, avoiding re-export issues.
        if export_format.lower() == "glb" and mesh_path.lower().endswith(".glb"):
            print(f"convert_mesh: Copying GLB {mesh_path} to {temp_file_path}")
            shutil.copy2(mesh_path, temp_file_path)
            return temp_file_path

        # For other conversions, or if GLB-to-GLB copy failed (not handled here, would need more logic)
        print(f"convert_mesh: Converting {mesh_path} to {export_format} at {temp_file_path}")
        mesh = trimesh.load_mesh(mesh_path)
        
        # Brief check if loaded mesh has UVs, important for debugging conversion issues.
        if not (hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None):
            print(f"  Warning: Loaded mesh from {mesh_path} has no UVs before export to {export_format}.")
        
        mesh.export(temp_file_path, file_type=export_format.lower())
        
        # Optional: Sanity check for GLB re-export if it didn't use the copy path.
        # if export_format.lower() == "glb":
        #     reloaded_mesh = trimesh.load_mesh(temp_file_path)
        #     if not (hasattr(reloaded_mesh.visual, 'uv') and reloaded_mesh.visual.uv is not None):
        #         print(f"  CRITICAL WARNING: Re-exported GLB {temp_file_path} lost UVs.")
        return temp_file_path

    except Exception as e:
        print(f"convert_mesh: Error during conversion of '{mesh_path}' to '{export_format}': {e}")
        traceback.print_exc()
        if temp_file_path and os.path.exists(temp_file_path): # Cleanup temp file on error
            try:
                os.remove(temp_file_path)
            except Exception as rm_e:
                print(f"convert_mesh: Error removing temp file {temp_file_path}: {rm_e}")
        return None



# Create the Gradio interface with improved layout
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center;'>Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging</h1>
        <p style='text-align: center;'>
            <strong>V0.1, Introduced By 
            <a href="https://gaplab.cuhk.edu.cn/" target="_blank">GAP Lab</a> from CUHKSZ and 
            <a href="https://www.nvsgames.cn/" target="_blank">Game-AIGC Team</a> from ByteDance</strong>
        </p>
        """
    )
    
    with gr.Row():
        gr.Markdown("""
                    <p align="center">
                    <a title="Website" href="https://stable-x.github.io/Hi3DGen/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
                    </a>
                    <a title="arXiv" href="https://stable-x.github.io/Hi3DGen/hi3dgen_paper.pdf" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
                    </a>
                    <a title="Github" href="https://github.com/Stable-X/Hi3DGen" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/github/stars/Stable-X/Hi3DGen?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
                    </a>
                    <a title="Social" href="https://x.com/ychngji6" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
                    </a>
                    </p>
                    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                
                with gr.Tab("Single Image"):
                    with gr.Row():
                        image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil")
                        normal_output = gr.Image(label="Normal Bridge", image_mode="RGBA", type="pil")
                        
                with gr.Tab("Multiple Images"):
                    gr.Markdown("### Upload multiple angle views of your object (2-8 images recommended)")
                    with gr.Row():
                        with gr.Column():
                            multi_image_inputs = gr.File(
                                label="Upload Multiple Images", 
                                file_count="multiple",
                                file_types=["image"],
                                height=200
                            )
                            with gr.Row():
                                multi_mode = gr.Radio(
                                    choices=["multidiffusion", "stochastic"], 
                                    value="multidiffusion",
                                    label="Processing Mode",
                                    info="MultiDiffusion: Average all views (recommended). Stochastic: Cycle through views."
                                )
                        with gr.Column():
                            multi_preview = gr.Gallery(
                                label="Preview Images", 
                                show_label=True,
                                elem_id="multi_preview",
                                columns=3,
                                rows=2,
                                height=200
                            )
                            multi_normal_output = gr.Image(label="Preprocessed First Image", image_mode="RGBA", type="pil")
                        
            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(-1, MAX_SEED, label="Seed", value=0, step=1)
                gr.Markdown("#### Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=50, step=1)
                gr.Markdown("#### Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=6, step=1)
                    
            with gr.Group():
                with gr.Row():
                    gen_shape_btn = gr.Button("Generate from Single Image", size="lg", variant="primary")
                    gen_multi_shape_btn = gr.Button("Generate from Multiple Images", size="lg", variant="secondary")
                        
        # Right column - Output
        with gr.Column(scale=1):
            with gr.Column():
                model_output = gr.Model3D(label="3D Model Preview (Each model is approximately 40MB, may take around 1 minute to load)")
            with gr.Column():
                export_format = gr.Dropdown(
                    choices=["obj", "glb", "ply", "stl"],
                    value="glb",
                    label="File Format"
                )
                download_btn = gr.DownloadButton(label="Export Mesh", interactive=False)

    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt]
    )
    
    gen_shape_btn.click(
        generate_3d,
        inputs=[
            image_prompt, seed,  
            ss_guidance_strength, ss_sampling_steps,
            slat_guidance_strength, slat_sampling_steps
        ],
        outputs=[normal_output, model_output, download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )
    
    # Multi-image event handlers
    multi_image_inputs.upload(
        process_multi_image_upload,
        inputs=[multi_image_inputs],
        outputs=[multi_preview, gr.State()]
    )
    
    gen_multi_shape_btn.click(
        lambda files, seed_val, ss_strength, ss_steps, slat_strength, slat_steps, mode: 
            generate_3d_multi(
                [Image.open(f.name) for f in files] if files else [],
                seed_val, ss_strength, ss_steps, slat_strength, slat_steps, 0.5, mode
            ) if files else (None, None, None),
        inputs=[
            multi_image_inputs, seed,
            ss_guidance_strength, ss_sampling_steps,
            slat_guidance_strength, slat_sampling_steps, multi_mode
        ],
        outputs=[multi_normal_output, model_output, download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )
    
    
    def update_download_button(mesh_path_from_model_output: str, selected_format: str):
        """
        Callback for Gradio DownloadButton.
        Converts the primary output mesh (GLB) to the selected_format if necessary.
        """
        if not mesh_path_from_model_output:
            # Update the button to be non-interactive if there's no mesh path
            return gr.DownloadButton.update(interactive=False) 
    
        path_for_download = convert_mesh(mesh_path_from_model_output, selected_format)
    
        if path_for_download:
            print(f"update_download_button: Providing {path_for_download} for download as {selected_format}.")
            # Set the 'value' of the DownloadButton to the path of the file to be downloaded
            return gr.DownloadButton.update(value=path_for_download, interactive=True)
        else:
            print(f"update_download_button: Conversion failed for {selected_format}, button inactive.")
            return gr.DownloadButton.update(interactive=False)

    
    export_format.change(
        update_download_button,
        inputs=[model_output, export_format],
        outputs=[download_btn]
    )
    
    examples = gr.Examples(
        examples=[
            f'assets/example_image/{image}'
            for image in os.listdir("assets/example_image")
        ],
        inputs=image_prompt,
    )
    
    # Create multi-image examples by grouping the example_multi_image files
    multi_image_examples = []
    multi_image_dir = "assets/example_multi_image"
    if os.path.exists(multi_image_dir):
        # Group by object name (everything before the last underscore and number)
        grouped_images = {}
        for filename in os.listdir(multi_image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract object name (e.g., 'character' from 'character_1.png')
                base_name = '_'.join(filename.split('_')[:-1])
                if base_name not in grouped_images:
                    grouped_images[base_name] = []
                grouped_images[base_name].append(os.path.join(multi_image_dir, filename))
        
        # Create example sets (each set is a list of 3 images for one object)
        for obj_name, image_paths in grouped_images.items():
            if len(image_paths) >= 2:  # At least 2 images
                # Sort to ensure consistent order
                image_paths.sort()
                multi_image_examples.append(image_paths[:3])  # Take first 3 images
    
    if multi_image_examples:
        multi_examples = gr.Examples(
            examples=multi_image_examples,
            inputs=multi_image_inputs,
            label="Multi-view Example Sets"
        )

    gr.Markdown(
        """
        **Acknowledgments**: Hi3DGen is built on the shoulders of giants. We would like to express our gratitude to the open-source research community and the developers of these pioneering projects:
        - **3D Modeling:** Our 3D Model is finetuned from the SOTA open-source 3D foundation model [Trellis](https://github.com/microsoft/TRELLIS) and we draw inspiration from the teams behind [Rodin](https://hyperhuman.deemos.com/rodin), [Tripo](https://www.tripo3d.ai/app/home), and [Dora](https://github.com/Seed3D/Dora).
        - **Normal Estimation:** Our Normal Estimation Model builds on the leading normal estimation research such as [StableNormal](https://github.com/hugoycj/StableNormal) and [GenPercept](https://github.com/aim-uofa/GenPercept).
        
        **Your contributions and collaboration push the boundaries of 3D modeling!**
        """
    )

if __name__ == "__main__":
    # Verify that all models are present locally
    verify_weights(WEIGHTS_DIR)

    print("Loading Hi3DGenPipeline to CPU...")
    # Use the correct local path for trellis model
    trellis_model_path = os.path.join(WEIGHTS_DIR, "models--Stable-X--trellis-normal-v0-1")
    hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained(trellis_model_path)
    hi3dgen_pipeline.cpu() # Move to CPU after loading
    print("Hi3DGenPipeline loaded on CPU.")

    # normal_predictor will be loaded on demand in generate_3d function
    
    # Launch the app
    demo.launch(share=False, server_name="127.0.0.1")

