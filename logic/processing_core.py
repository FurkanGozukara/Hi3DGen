import os
import datetime
import traceback
import time
import shutil
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Import cancellation system
from .cancellation import should_cancel, update_processing_stage, add_cleanup_callback, remove_cleanup_callback

@dataclass
class ProcessingResult:
    """Result of single image processing"""
    success: bool = False
    normal_image: Optional[Image.Image] = None
    mesh_path: Optional[str] = None
    output_dir: Optional[str] = None
    auto_save_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Timing information
    total_time: float = 0.0
    normal_prediction_time: float = 0.0
    pipeline_time: float = 0.0
    mesh_processing_time: float = 0.0
    uv_unwrapping_time: float = 0.0
    export_time: float = 0.0
    auto_save_time: float = 0.0
    
    # Processing statistics
    original_face_count: int = 0
    simplified_face_count: int = 0
    final_vertex_count: int = 0
    final_face_count: int = 0
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the processing result"""
        if not self.success:
            return f"Failed: {self.error_message or 'Unknown error'}"
        
        summary = f"Success: Generated mesh in {self.total_time:.1f}s\n"
        summary += f"  - Normal prediction: {self.normal_prediction_time:.1f}s\n"
        summary += f"  - 3D generation: {self.pipeline_time:.1f}s\n"
        summary += f"  - Mesh processing: {self.mesh_processing_time:.1f}s\n"
        summary += f"  - UV unwrapping: {self.uv_unwrapping_time:.1f}s\n"
        summary += f"  - Export: {self.export_time:.1f}s"
        
        if self.auto_save_time > 0:
            summary += f"\n  - Auto-save: {self.auto_save_time:.1f}s"
        
        if self.original_face_count > 0:
            reduction = ((self.original_face_count - self.simplified_face_count) / self.original_face_count) * 100
            summary += f"\n  - Mesh: {self.original_face_count} -> {self.simplified_face_count} faces ({reduction:.1f}% reduction)"
        
        return summary

@dataclass
class ProcessingParameters:
    """Parameters for single image processing"""
    # Core parameters
    seed: int = -1
    
    # Stage 1: Sparse Structure Generation
    ss_guidance_strength: float = 3.0
    ss_sampling_steps: int = 50
    
    # Stage 2: Structured Latent Generation
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 6
    
    # Mesh Processing
    poly_count_pcnt: float = 0.5
    
    # UV Unwrapping (xatlas)
    xatlas_max_cost: float = 8.0
    xatlas_normal_seam_weight: float = 1.0
    xatlas_resolution: int = 1024
    xatlas_padding: int = 2
    
    # Normal Map Generation
    normal_map_resolution: int = 768
    normal_match_input_resolution: bool = True
    
    # Auto-save settings
    auto_save_obj: bool = True
    auto_save_glb: bool = True
    auto_save_ply: bool = True
    auto_save_stl: bool = True

class ProcessingCore:
    """Core processing functionality extracted from generate_3d"""
    
    def __init__(self, hi3dgen_pipeline, weights_dir: str, tmp_dir: str, max_seed: int):
        self.hi3dgen_pipeline = hi3dgen_pipeline
        self.weights_dir = weights_dir
        self.tmp_dir = tmp_dir
        self.max_seed = max_seed
        
    def process_single_image(self, 
                           image: Image.Image, 
                           params: ProcessingParameters,
                           progress_callback: Optional[callable] = None) -> ProcessingResult:
        """
        Process a single image through the complete 3D generation pipeline
        
        Args:
            image: Input PIL image
            params: Processing parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            ProcessingResult with all outputs and timing information
        """
        result = ProcessingResult()
        start_time = time.time()
        
        # Progress callback helper
        def update_progress(stage: str, details: str = ""):
            update_processing_stage(f"{stage}: {details}" if details else stage)
            if progress_callback:
                try:
                    progress_callback(stage, details)
                except Exception as e:
                    print(f"Error in progress callback: {e}")
        
        try:
            # Validate inputs
            if image is None:
                result.error_message = "Input image is None"
                return result
                
            if self.hi3dgen_pipeline is None:
                result.error_message = "Hi3DGenPipeline not loaded"
                return result
            
            # Set up seed
            seed = params.seed
            if seed == -1:
                seed = np.random.randint(0, self.max_seed)
            
            update_progress("Initialization", f"Using seed {seed}")
            
            # Check for cancellation before starting
            if should_cancel():
                result.error_message = "Cancelled before processing started"
                return result
            
            # Stage 1: Normal Prediction
            result.normal_image = self._generate_normal_map(
                image, params, update_progress, result
            )
            
            if result.normal_image is None or should_cancel():
                if should_cancel():
                    result.error_message = "Cancelled during normal prediction"
                return result
            
            # Stage 2: 3D Generation and Processing
            result.mesh_path, result.output_dir = self._generate_3d_mesh(
                result.normal_image, seed, params, update_progress, result
            )
            
            if result.mesh_path is None or should_cancel():
                if should_cancel():
                    result.error_message = "Cancelled during 3D generation"
                return result
            
            # Stage 3: Auto-save (if enabled)
            if any([params.auto_save_obj, params.auto_save_glb, params.auto_save_ply, params.auto_save_stl]):
                result.auto_save_result = self._perform_auto_save(
                    result.mesh_path, result.normal_image, params, update_progress, result
                )
            
            # Final success
            result.success = True
            result.total_time = time.time() - start_time
            
            update_progress("Completed", f"Total time: {result.total_time:.1f}s")
            
        except Exception as e:
            result.error_message = f"Processing failed: {str(e)}"
            result.total_time = time.time() - start_time
            print(f"Error in process_single_image: {e}")
            traceback.print_exc()
        
        return result
    
    def _generate_normal_map(self, 
                           image: Image.Image, 
                           params: ProcessingParameters,
                           update_progress: callable,
                           result: ProcessingResult) -> Optional[Image.Image]:
        """Generate normal map from input image"""
        start_time = time.time()
        update_progress("Normal Prediction", "Loading model from local cache")
        
        current_normal_predictor_instance = None
        normal_image_pil = None
        
        # Set up cleanup callback for normal predictor
        def cleanup_normal_predictor():
            nonlocal current_normal_predictor_instance
            if current_normal_predictor_instance is not None:
                print("Cleanup: Unloading normal predictor...")
                try:
                    # Synchronize CUDA operations before cleanup
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Move model to CPU if possible
                    if hasattr(current_normal_predictor_instance, 'model') and hasattr(current_normal_predictor_instance.model, 'cpu'):
                        current_normal_predictor_instance.model.cpu()
                    
                    del current_normal_predictor_instance
                    current_normal_predictor_instance = None
                    
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    print("Cleanup: Normal predictor unloaded successfully")
                    
                except RuntimeError as cuda_error:
                    if "CUDA" in str(cuda_error):
                        print(f"Cleanup: CUDA error during normal predictor cleanup: {cuda_error}")
                        print("Cleanup: Attempting force cleanup of normal predictor...")
                        try:
                            # Force cleanup
                            if current_normal_predictor_instance is not None:
                                del current_normal_predictor_instance
                                current_normal_predictor_instance = None
                            
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            
                            print("Cleanup: Force cleanup of normal predictor completed")
                        except Exception as force_error:
                            print(f"Cleanup: Force cleanup of normal predictor also failed: {force_error}")
                            # Ensure variable is reset
                            current_normal_predictor_instance = None
                    else:
                        # Re-raise non-CUDA errors
                        raise
                except Exception as e:
                    print(f"Cleanup: Unexpected error during normal predictor cleanup: {e}")
                    # Ensure variable is reset
                    current_normal_predictor_instance = None
                    # Still try to clear CUDA cache if possible
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
        
        add_cleanup_callback(cleanup_normal_predictor)
        
        try:
            # Check for model paths
            yoso_model_dir_name = "models--Stable-X--yoso-normal-v1-8-1"
            yoso_local_path = os.path.join(self.weights_dir, yoso_model_dir_name)
            
            if not os.path.exists(yoso_local_path):
                raise FileNotFoundError(f"YOSO model not found at {yoso_local_path}")
            
            birefnet_local_path = os.path.join(self.weights_dir, "models--ZhengPeng7--BiRefNet")
            if not os.path.exists(birefnet_local_path):
                raise FileNotFoundError(f"BiRefNet model not found at {birefnet_local_path}")
            
            # Check cancellation
            if should_cancel():
                return None
            
            # Set up HuggingFace cache
            update_progress("Normal Prediction", "Setting up model cache")
            self._setup_hf_cache()
            
            # Check cancellation
            if should_cancel():
                return None
            
            # Load normal predictor
            update_progress("Normal Prediction", "Loading YOSO model")
            yoso_model_abs_path = os.path.abspath(yoso_local_path)
            
            current_normal_predictor_instance = torch.hub.load(
                "hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True,
                yoso_version=yoso_model_abs_path
            )
            
            # Check cancellation
            if should_cancel():
                return None
            
            # Generate normal map
            update_progress("Normal Prediction", "Generating normal map")
            normal_image_pil = current_normal_predictor_instance(
                image,
                resolution=params.normal_map_resolution,
                match_input_resolution=params.normal_match_input_resolution,
                data_type='object'
            )
            
        except Exception as e:
            print(f"ERROR in Normal Prediction stage: {e}")
            traceback.print_exc()
            normal_image_pil = None
        finally:
            cleanup_normal_predictor()
            remove_cleanup_callback(cleanup_normal_predictor)
        
        result.normal_prediction_time = time.time() - start_time
        
        if normal_image_pil is None:
            print("ERROR: Normal map not generated")
        
        return normal_image_pil
    
    def _generate_3d_mesh(self,
                         normal_image: Image.Image,
                         seed: int,
                         params: ProcessingParameters,
                         update_progress: callable,
                         result: ProcessingResult) -> Tuple[Optional[str], Optional[str]]:
        """Generate 3D mesh from normal map"""
        generation_start_time = time.time()
        pipeline_on_gpu = False
        mesh_path_glb = None
        output_dir = None
        
        # Set up cleanup callback for pipeline
        def cleanup_pipeline():
            nonlocal pipeline_on_gpu
            if pipeline_on_gpu:
                print("Cleanup: Moving Hi3DGen pipeline to CPU...")
                try:
                    # Synchronize CUDA operations before moving to CPU
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Try to move pipeline to CPU
                    self.hi3dgen_pipeline.cpu()
                    pipeline_on_gpu = False
                    print("Cleanup: Successfully moved Hi3DGen pipeline to CPU")
                    
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("Cleanup: CUDA cache cleared")
                        
                except RuntimeError as cuda_error:
                    if "CUDA" in str(cuda_error):
                        print(f"Cleanup: CUDA error during pipeline cleanup: {cuda_error}")
                        print("Cleanup: Attempting force cleanup...")
                        try:
                            # Force clear CUDA cache and reset pipeline state
                            pipeline_on_gpu = False
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                # Try to reset CUDA context
                                torch.cuda.synchronize()
                            print("Cleanup: Force cleanup completed")
                        except Exception as force_error:
                            print(f"Cleanup: Force cleanup also failed: {force_error}")
                            # Mark as cleaned up anyway to prevent further attempts
                            pipeline_on_gpu = False
                    else:
                        # Re-raise non-CUDA errors
                        raise
                except Exception as e:
                    print(f"Cleanup: Unexpected error during pipeline cleanup: {e}")
                    # Mark as cleaned up to prevent further attempts
                    pipeline_on_gpu = False
                    # Still try to clear CUDA cache if possible
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
        
        add_cleanup_callback(cleanup_pipeline)
        
        try:
            # Move pipeline to GPU
            if torch.cuda.is_available():
                update_progress("3D Generation", "Moving Hi3DGen pipeline to GPU")
                self.hi3dgen_pipeline.cuda()
                pipeline_on_gpu = True
            
            # Check cancellation
            if should_cancel():
                return None, None
            
            # Run Hi3DGen pipeline
            update_progress("3D Generation", "Running Hi3DGen pipeline")
            pipeline_start = time.time()
            
            outputs = self.hi3dgen_pipeline.run(
                normal_image,
                seed=seed,
                formats=["mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": params.ss_sampling_steps,
                    "cfg_strength": params.ss_guidance_strength
                },
                slat_sampler_params={
                    "steps": params.slat_sampling_steps,
                    "cfg_strength": params.slat_guidance_strength
                }
            )
            
            result.pipeline_time = time.time() - pipeline_start
            print(f"3D Generation: Hi3DGen pipeline completed in {result.pipeline_time:.2f}s")
            
            # Check cancellation
            if should_cancel():
                return None, None
            
            # Prepare output directory
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            output_dir = os.path.join(self.tmp_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            mesh_path_glb = os.path.join(output_dir, "mesh.glb")
            
            # Process mesh
            update_progress("Mesh Processing", "Converting to Trimesh and simplifying")
            mesh_start = time.time()
            
            raw_mesh_trimesh = outputs['mesh'][0].to_trimesh(transform_pose=True)
            result.original_face_count = len(raw_mesh_trimesh.faces)
            
            # Import the mesh processing functions from main module
            # Note: These will be imported at runtime from the main module
            import sys
            main_module = sys.modules.get('__main__')
            if main_module:
                simplify_mesh_open3d = getattr(main_module, 'simplify_mesh_open3d')
                unwrap_mesh_with_xatlas = getattr(main_module, 'unwrap_mesh_with_xatlas')
            else:
                raise ImportError("Cannot access main module functions")
            
            mesh_for_uv_unwrap = simplify_mesh_open3d(raw_mesh_trimesh, params.poly_count_pcnt)
            result.simplified_face_count = len(mesh_for_uv_unwrap.faces)
            
            result.mesh_processing_time = time.time() - mesh_start
            print(f"Mesh Processing: Completed in {result.mesh_processing_time:.1f}s")
            
            # Check cancellation
            if should_cancel():
                return None, None
            
            # UV Unwrapping
            update_progress("UV Unwrapping", "Processing with xatlas")
            uv_start = time.time()
            
            unwrapped_mesh_trimesh = unwrap_mesh_with_xatlas(
                mesh_for_uv_unwrap,
                max_cost_param=params.xatlas_max_cost,
                normal_seam_weight_param=params.xatlas_normal_seam_weight,
                resolution_param=params.xatlas_resolution,
                padding_param=params.xatlas_padding
            )
            
            result.uv_unwrapping_time = time.time() - uv_start
            result.final_vertex_count = len(unwrapped_mesh_trimesh.vertices)
            result.final_face_count = len(unwrapped_mesh_trimesh.faces)
            
            # Check cancellation
            if should_cancel():
                return None, None
            
            # Export mesh
            update_progress("File Export", f"Exporting GLB to {mesh_path_glb}")
            export_start = time.time()
            
            unwrapped_mesh_trimesh.export(mesh_path_glb)
            
            result.export_time = time.time() - export_start
            print(f"File Export: GLB exported successfully in {result.export_time:.2f}s")
            
            # Log timing breakdown
            total_generation_time = time.time() - generation_start_time
            print(f"=== 3D GENERATION COMPLETE ===")
            print(f"Total time breakdown:")
            print(f"  - Hi3DGen pipeline: {result.pipeline_time:.2f}s ({result.pipeline_time/total_generation_time*100:.1f}%)")
            print(f"  - Mesh processing: {result.mesh_processing_time:.2f}s ({result.mesh_processing_time/total_generation_time*100:.1f}%)")
            print(f"  - UV unwrapping: {result.uv_unwrapping_time:.2f}s ({result.uv_unwrapping_time/total_generation_time*100:.1f}%)")
            print(f"  - File export: {result.export_time:.2f}s ({result.export_time/total_generation_time*100:.1f}%)")
            print(f"  - TOTAL: {total_generation_time:.2f}s")
            
        except Exception as e:
            print(f"ERROR in 3D Generation stage: {e}")
            traceback.print_exc()
            mesh_path_glb = None
            output_dir = None
        finally:
            cleanup_pipeline()
            remove_cleanup_callback(cleanup_pipeline)
        
        return mesh_path_glb, output_dir
    
    def _perform_auto_save(self,
                          mesh_path: str,
                          normal_image: Image.Image,
                          params: ProcessingParameters,
                          update_progress: callable,
                          result: ProcessingResult) -> Optional[Dict[str, Any]]:
        """Perform auto-save if enabled"""
        start_time = time.time()
        
        try:
            enabled_formats = {
                "obj": params.auto_save_obj,
                "glb": params.auto_save_glb,
                "ply": params.auto_save_ply,
                "stl": params.auto_save_stl
            }
            
            update_progress("Auto-save", "Saving generated files")
            
            # Import auto_save function from logic module
            from .auto_save import auto_save_generation
            
            auto_save_result = auto_save_generation(
                mesh_path=mesh_path,
                normal_image=normal_image,
                enabled_formats=enabled_formats
            )
            
            if auto_save_result:
                print(f"✓ Auto-save successful: Saved to folder {auto_save_result['folder_number']}")
                print(f"  Saved {len(auto_save_result['saved_files'])} files")
            else:
                print("✗ Auto-save failed")
            
            result.auto_save_time = time.time() - start_time
            return auto_save_result
            
        except Exception as e:
            print(f"Auto-save error: {e}")
            traceback.print_exc()
            result.auto_save_time = time.time() - start_time
            return None
    
    def _setup_hf_cache(self):
        """Set up HuggingFace cache by copying models"""
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        os.makedirs(hf_cache_dir, exist_ok=True)
        
        model_mappings = {
            "models--Stable-X--yoso-normal-v1-8-1": "models--Stable-X--yoso-normal-v1-8-1",
            "models--ZhengPeng7--BiRefNet": "models--ZhengPeng7--BiRefNet"
        }
        
        for src_name, dst_folder_name in model_mappings.items():
            src_path_in_weights = os.path.join(self.weights_dir, src_name)
            dst_path_in_hf_cache = os.path.join(hf_cache_dir, dst_folder_name)
            
            if os.path.exists(src_path_in_weights) and not os.path.exists(dst_path_in_hf_cache):
                print(f"Copying {src_name} from {self.weights_dir} to HuggingFace cache at {dst_path_in_hf_cache}...")
                shutil.copytree(src_path_in_weights, dst_path_in_hf_cache)

# Convenience function for backward compatibility
def process_single_image_legacy(image, seed=-1, ss_guidance_strength=3, ss_sampling_steps=50,
                               slat_guidance_strength=3, slat_sampling_steps=6, poly_count_pcnt=0.5,
                               xatlas_max_cost=8.0, xatlas_normal_seam_weight=1.0, xatlas_resolution=1024,
                               xatlas_padding=2, normal_map_resolution=768, normal_match_input_resolution=True,
                               auto_save_obj=True, auto_save_glb=True, auto_save_ply=True, auto_save_stl=True,
                               processing_core: ProcessingCore = None) -> Tuple[Optional[Image.Image], Optional[str], Optional[str]]:
    """
    Legacy function signature compatibility for existing generate_3d calls
    
    Returns:
        Tuple of (normal_image, mesh_path, mesh_path) for Gradio compatibility
    """
    if processing_core is None:
        raise ValueError("processing_core must be provided")
    
    # Convert parameters to ProcessingParameters
    params = ProcessingParameters(
        seed=seed,
        ss_guidance_strength=ss_guidance_strength,
        ss_sampling_steps=ss_sampling_steps,
        slat_guidance_strength=slat_guidance_strength,
        slat_sampling_steps=slat_sampling_steps,
        poly_count_pcnt=poly_count_pcnt,
        xatlas_max_cost=xatlas_max_cost,
        xatlas_normal_seam_weight=xatlas_normal_seam_weight,
        xatlas_resolution=xatlas_resolution,
        xatlas_padding=xatlas_padding,
        normal_map_resolution=normal_map_resolution,
        normal_match_input_resolution=normal_match_input_resolution,
        auto_save_obj=auto_save_obj,
        auto_save_glb=auto_save_glb,
        auto_save_ply=auto_save_ply,
        auto_save_stl=auto_save_stl
    )
    
    # Process the image
    result = processing_core.process_single_image(image, params)
    
    # Return in legacy format
    if result.success:
        return result.normal_image, result.mesh_path, result.mesh_path
    else:
        print(f"Processing failed: {result.error_message}")
        return None, None, None 