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
import hf_transfer
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


def cache_weights(weights_dir: str) -> dict:
    import os
    import sys
    from huggingface_hub import list_repo_files, hf_hub_download
    from pathlib import Path

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # Attempt to use hf-transfer

    model_ids = [ # Renamed for brevity
        "Stable-X/trellis-normal-v0-1", "Stable-X/yoso-normal-v1-8-1", "ZhengPeng7/BiRefNet"
    ]
    cached_paths = {} # Renamed for brevity

    for repo_id in model_ids: # Renamed for clarity (Hugging Face term)
        local_repo_root = Path(weights_dir) / f"models--{repo_id.replace('/', '--')}"
        print(f"Processing {repo_id} into {local_repo_root}...")
        sys.stdout.flush()
        local_repo_root.mkdir(parents=True, exist_ok=True)

        try:
            # Get all file paths within the repository
            files_in_repo = list_repo_files(repo_id=repo_id, repo_type="model")
            num_total_files = len(files_in_repo)
            print(f"  Found {num_total_files} files. Starting download/verification...")
            sys.stdout.flush()

            for i, filename_in_repo in enumerate(files_in_repo):
                # Print progress before each file operation
                print(f"  [{i+1}/{num_total_files}] Processing: {filename_in_repo}")
                sys.stdout.flush()
                try:
                    # hf_hub_download handles caching internally.
                    # It downloads to a shared HF cache then copies/symlinks to local_dir if specified.
                    # We specify local_dir to ensure files land directly in our target structure.
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename_in_repo,
                        repo_type="model",
                        local_dir=str(local_repo_root), # Ensures download into the model's specific folder
                        local_dir_use_symlinks=False, # Force copy, not symlink
                        force_download=False, # Only downloads if not present or outdated in cache/local_dir
                    )
                except Exception as file_e:
                    # Silently skip individual file errors (e.g., if it's a directory entry or unresolvable)
                    # More robust error handling would log this or retry. For compactness, we skip.
                    # print(f"    Warning: Skipping {filename_in_repo} due to: {str(file_e)[:50]}...") # Optional: short error
                    pass
            
            cached_paths[repo_id] = str(local_repo_root)
            print(f"  Finished processing {repo_id}.")
        except Exception as repo_e:
            print(f"  ERROR processing repository {repo_id}: {repo_e}")
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
        print("Normal Prediction: Loading model...") 
        current_normal_predictor_instance = torch.hub.load(
            "hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, 
            yoso_version='yoso-normal-v1-8-1', local_cache_dir=WEIGHTS_DIR
        )
        print("Normal Prediction: Generating normal map...")
        normal_image_pil = current_normal_predictor_instance(image, resolution=768, match_input_resolution=True, data_type='object')
    except Exception as e:
        print(f"ERROR in Normal Prediction stage: {e}")
        traceback.print_exc()
    finally:
        if current_normal_predictor_instance is not None: 
            print("Normal Prediction: Unloading model...")
            if hasattr(current_normal_predictor_instance, 'model') and hasattr(current_normal_predictor_instance.model, 'cpu'):
                current_normal_predictor_instance.model.cpu() 
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
                    gr.Markdown("<div style='text-align: center; padding: 40px; font-size: 24px;'>Multiple Images functionality is coming soon!</div>")
                        
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
                    gen_shape_btn = gr.Button("Generate Shape", size="lg", variant="primary")
                        
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

    gr.Markdown(
        """
        **Acknowledgments**: Hi3DGen is built on the shoulders of giants. We would like to express our gratitude to the open-source research community and the developers of these pioneering projects:
        - **3D Modeling:** Our 3D Model is finetuned from the SOTA open-source 3D foundation model [Trellis](https://github.com/microsoft/TRELLIS) and we draw inspiration from the teams behind [Rodin](https://hyperhuman.deemos.com/rodin), [Tripo](https://www.tripo3d.ai/app/home), and [Dora](https://github.com/Seed3D/Dora).
        - **Normal Estimation:** Our Normal Estimation Model builds on the leading normal estimation research such as [StableNormal](https://github.com/hugoycj/StableNormal) and [GenPercept](https://github.com/aim-uofa/GenPercept).
        
        **Your contributions and collaboration push the boundaries of 3D modeling!**
        """
    )

if __name__ == "__main__":
    # Download and cache the weights
    cache_weights(WEIGHTS_DIR)

    print("Loading Hi3DGenPipeline to CPU...")
    hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("weights/trellis-normal-v0-1")
    hi3dgen_pipeline.cpu() # Move to CPU after loading
    print("Hi3DGenPipeline loaded on CPU.")

    # normal_predictor will be loaded on demand in generate_3d function
    
    # Launch the app
    demo.launch(share=False, server_name="127.0.0.1")

