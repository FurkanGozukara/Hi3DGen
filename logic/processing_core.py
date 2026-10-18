import os
import datetime
import traceback
import time
import uuid
from typing import Callable, List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image

from hi3dgen.pipelines.samplers import SamplingCancelled, step_callback

# Import cancellation system
from .cancellation import should_cancel, update_processing_stage
from .mesh_jobs import MeshJobCancelled, cleanup_stale_jobs, run_mesh_job
from .model_manager import ModelMemory

@dataclass
class ProcessingResult:
    """Result of single image processing"""
    success: bool = False
    normal_image: Optional[Image.Image] = None
    mesh_path: Optional[str] = None
    output_dir: Optional[str] = None
    auto_save_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    seed: int = -1

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

@dataclass
class PreparedMesh:
    """A raw mesh from the GPU stages, waiting for the CPU stages (decimation, UV unwrapping, export, auto-save)."""
    result: ProcessingResult
    params: ProcessingParameters
    started: float = 0.0
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    to_trimesh_time: float = 0.0
    custom_output_folder: Optional[str] = None
    custom_filename_base: Optional[str] = None

    @property
    def ready(self) -> bool:
        """False when the GPU stages failed or were cancelled (`result` then holds the reason)."""
        return self.vertices is not None

class ProcessingCore:
    """Core processing functionality extracted from generate_3d

    The work of one image has two parts that can be run apart: `prepare_image` (GPU: normal map and Hi3DGen) and
    `finish_mesh` (CPU: decimation, UV unwrapping, export, auto-save). `process_single_image` runs both one after
    the other; the batch processor overlaps the CPU part of one image with the GPU part of the next.
    """

    def __init__(self, hi3dgen_pipeline, weights_dir: str, tmp_dir: str, max_seed: int,
                 memory: Optional[ModelMemory] = None):
        self.hi3dgen_pipeline = hi3dgen_pipeline
        self.weights_dir = weights_dir
        self.tmp_dir = tmp_dir
        self.max_seed = max_seed
        # GPU memory policy: the StableNormal predictor is created once and models stay on the GPU when there is room
        self.memory = memory or ModelMemory(hi3dgen_pipeline, weights_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        cleanup_stale_jobs(self.tmp_dir)

    @staticmethod
    def _progress_updater(progress_callback: Optional[Callable]) -> Callable:
        def update_progress(stage: str, details: str = ""):
            update_processing_stage(f"{stage}: {details}" if details else stage)
            if progress_callback:
                try:
                    progress_callback(stage, details)
                except Exception as e:
                    print(f"Error in progress callback: {e}")
        return update_progress

    @staticmethod
    def _sampling_step_reporter(update_progress: Callable) -> Callable:
        """Called by the samplers before every step: live progress and cooperative cancellation"""
        def on_sampling_step(step: int, total: int, stage: str):
            if should_cancel():
                raise SamplingCancelled()
            if step < total:
                update_progress("3D Generation", f"{stage}, step {step + 1}/{total}")
        return on_sampling_step

    def _new_seed(self, params: ProcessingParameters) -> int:
        return params.seed if params.seed != -1 else int(np.random.randint(0, self.max_seed))

    # ------------------------------------------------------------------------------------------
    # Whole job
    # ------------------------------------------------------------------------------------------
    def process_single_image(self,
                           image: Image.Image,
                           params: ProcessingParameters,
                           progress_callback: Optional[callable] = None,
                           custom_output_folder: Optional[str] = None,
                           custom_filename_base: Optional[str] = None,
                           preview_callback: Optional[callable] = None,
                           mesh_preview_callback: Optional[callable] = None) -> ProcessingResult:
        """
        Process a single image to generate 3D model

        Args:
            image: Input image
            params: Processing parameters
            progress_callback: Optional progress callback function
            custom_output_folder: Optional custom output folder (for batch processing)
            custom_filename_base: Optional custom filename base (for batch processing)
            preview_callback: Optional function that receives the normal map as soon as it exists
            mesh_preview_callback: Optional function that receives the path of a GLB of the decimated mesh as soon as
                it exists, a minute before the UV unwrapped final mesh

        Returns:
            ProcessingResult with success status and generated files
        """
        prepared = self.prepare_image(image, params, progress_callback, custom_output_folder,
                                      custom_filename_base, preview_callback)
        return self.finish_mesh(prepared, progress_callback, mesh_preview_callback)

    def process_multi_image(self,
                            images: List[Image.Image],
                            mode: str,
                            params: ProcessingParameters,
                            progress_callback: Optional[callable] = None,
                            custom_output_folder: Optional[str] = None,
                            custom_filename_base: Optional[str] = None,
                            preview_callback: Optional[callable] = None,
                            mesh_preview_callback: Optional[callable] = None) -> ProcessingResult:
        """Several views of one object -> one 3D model (`mode` is "multidiffusion" or "stochastic")."""
        prepared = self.prepare_multi_image(images, mode, params, progress_callback, custom_output_folder,
                                            custom_filename_base, preview_callback)
        return self.finish_mesh(prepared, progress_callback, mesh_preview_callback)

    # ------------------------------------------------------------------------------------------
    # GPU part
    # ------------------------------------------------------------------------------------------
    def prepare_image(self,
                      image: Image.Image,
                      params: ProcessingParameters,
                      progress_callback: Optional[callable] = None,
                      custom_output_folder: Optional[str] = None,
                      custom_filename_base: Optional[str] = None,
                      preview_callback: Optional[callable] = None) -> PreparedMesh:
        """Normal map and Hi3DGen pipeline for one image."""
        result = ProcessingResult()
        prepared = PreparedMesh(result=result, params=params, started=time.time(),
                                custom_output_folder=custom_output_folder, custom_filename_base=custom_filename_base)
        update_progress = self._progress_updater(progress_callback)

        try:
            # Validate inputs
            if image is None:
                result.error_message = "Input image is None"
                return prepared

            if self.hi3dgen_pipeline is None:
                result.error_message = "Hi3DGenPipeline not loaded"
                return prepared

            # Set up seed
            seed = self._new_seed(params)
            result.seed = seed

            update_progress("Initialization", f"Using seed {seed}")

            # Check for cancellation before starting
            if should_cancel():
                result.error_message = "Cancelled before processing started"
                return prepared

            # Stage 1: Normal Prediction
            result.normal_image = self._generate_normal_map(
                image, params, update_progress, result
            )

            if result.normal_image is None or should_cancel():
                if should_cancel():
                    result.error_message = "Cancelled during normal prediction"
                elif not result.error_message:
                    result.error_message = "The normal map could not be generated (see the console for details)"
                return prepared

            if preview_callback is not None:
                try:
                    preview_callback(result.normal_image)
                except Exception as e:
                    print(f"Error in preview callback: {e}")

            # Stage 2: 3D Generation (GPU)
            with step_callback(self._sampling_step_reporter(update_progress)):
                self._generate_3d_mesh(result.normal_image, seed, params, update_progress, result, prepared)

            if not prepared.ready:
                if should_cancel():
                    result.error_message = "Cancelled during 3D generation"
                elif not result.error_message:
                    result.error_message = "3D generation failed (see the console for details)"

        except Exception as e:
            result.error_message = f"Processing failed: {str(e)}"
            result.total_time = time.time() - prepared.started
            print(f"Error in prepare_image: {e}")
            traceback.print_exc()
            prepared.vertices = prepared.faces = None

        return prepared

    def prepare_multi_image(self,
                            images: List[Image.Image],
                            mode: str,
                            params: ProcessingParameters,
                            progress_callback: Optional[callable] = None,
                            custom_output_folder: Optional[str] = None,
                            custom_filename_base: Optional[str] = None,
                            preview_callback: Optional[callable] = None) -> PreparedMesh:
        """Hi3DGen pipeline with several views of one object as the condition."""
        result = ProcessingResult()
        prepared = PreparedMesh(result=result, params=params, started=time.time(),
                                custom_output_folder=custom_output_folder, custom_filename_base=custom_filename_base)
        update_progress = self._progress_updater(progress_callback)
        outputs = None

        try:
            if not images:
                result.error_message = "No input images"
                return prepared
            if self.hi3dgen_pipeline is None:
                result.error_message = "Hi3DGenPipeline not loaded"
                return prepared

            seed = self._new_seed(params)
            result.seed = seed
            update_progress("Initialization", f"Using seed {seed}")
            if should_cancel():
                result.error_message = "Cancelled before processing started"
                return prepared

            # Remove the backgrounds first, so the GPU holds one model at a time on small cards
            started = time.time()
            preprocessed = []
            for index, image in enumerate(images):
                update_progress("Preprocessing", f"Removing the background of view {index + 1}/{len(images)}")
                preprocessed.append(self.hi3dgen_pipeline.preprocess_image(image))
                if should_cancel():
                    result.error_message = "Cancelled during preprocessing"
                    return prepared
            result.normal_image = preprocessed[0]
            result.normal_prediction_time = time.time() - started
            if preview_callback is not None:
                try:
                    preview_callback(result.normal_image)
                except Exception as e:
                    print(f"Error in preview callback: {e}")

            with step_callback(self._sampling_step_reporter(update_progress)):
                update_progress("3D Generation", "Preparing the Hi3DGen pipeline")
                with self.memory.use_pipeline() as pipeline:
                    update_progress("3D Generation", f"Running Hi3DGen pipeline ({len(preprocessed)} views, {mode})")
                    pipeline_start = time.time()
                    outputs = pipeline.run_multi_image(
                        preprocessed,
                        seed=seed,
                        formats=["mesh"],
                        preprocess_image=False,
                        mode=mode,
                        sparse_structure_sampler_params={"steps": params.ss_sampling_steps,
                                                         "cfg_strength": params.ss_guidance_strength},
                        slat_sampler_params={"steps": params.slat_sampling_steps,
                                             "cfg_strength": params.slat_guidance_strength},
                    )
                    result.pipeline_time = time.time() - pipeline_start
                    print(f"3D Generation: Hi3DGen pipeline completed in {result.pipeline_time:.2f}s")
                self._extract_raw_mesh(outputs, prepared)
                outputs = None

            if not prepared.ready and should_cancel():
                result.error_message = "Cancelled during 3D generation"

        except SamplingCancelled:
            print("3D Generation: cancelled by user")
            result.error_message = "Cancelled during 3D generation"
            prepared.vertices = prepared.faces = None
        except Exception as e:
            result.error_message = f"Processing failed: {str(e)}"
            print(f"Error in prepare_multi_image: {e}")
            traceback.print_exc()
            self._report_out_of_memory(e)
            prepared.vertices = prepared.faces = None
        finally:
            outputs = None
            self.memory.release_cached_memory()

        return prepared

    def _generate_normal_map(self,
                           image: Image.Image,
                           params: ProcessingParameters,
                           update_progress: callable,
                           result: ProcessingResult) -> Optional[Image.Image]:
        """Generate normal map from input image"""
        start_time = time.time()
        update_progress("Normal Prediction", "Preparing the normal map model")

        normal_image_pil = None

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

            # The predictor is created once (vendored StableNormal, local weights) and moved to the GPU for the call
            with self.memory.use_normal_predictor() as predictor:
                # Check cancellation
                if should_cancel():
                    return None

                # Generate normal map
                update_progress("Normal Prediction", "Generating normal map")
                normal_image_pil = predictor(
                    image,
                    resolution=params.normal_map_resolution,
                    match_input_resolution=params.normal_match_input_resolution,
                    data_type='object'
                )

        except Exception as e:
            print(f"ERROR in Normal Prediction stage: {e}")
            traceback.print_exc()
            result.error_message = f"Normal prediction failed: {e}"
            self._report_out_of_memory(e)
            normal_image_pil = None

        result.normal_prediction_time = time.time() - start_time

        if normal_image_pil is None:
            print("ERROR: Normal map not generated")

        return normal_image_pil

    def _generate_3d_mesh(self,
                         normal_image: Image.Image,
                         seed: int,
                         params: ProcessingParameters,
                         update_progress: callable,
                         result: ProcessingResult,
                         prepared: PreparedMesh) -> None:
        """Run the Hi3DGen pipeline on the normal map and keep the raw mesh in `prepared`"""
        outputs = None

        try:
            update_progress("3D Generation", "Preparing the Hi3DGen pipeline")

            # Move the pipeline to the GPU (unless it is resident there) and run it
            with self.memory.use_pipeline() as pipeline:
                # Check cancellation
                if should_cancel():
                    return

                # Run Hi3DGen pipeline
                update_progress("3D Generation", "Running Hi3DGen pipeline")
                pipeline_start = time.time()

                outputs = pipeline.run(
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
                return

            self._extract_raw_mesh(outputs, prepared)

        except SamplingCancelled:
            print("3D Generation: cancelled by user")
            prepared.vertices = prepared.faces = None
        except Exception as e:
            print(f"ERROR in 3D Generation stage: {e}")
            traceback.print_exc()
            result.error_message = f"3D generation failed: {e}"
            self._report_out_of_memory(e)
            prepared.vertices = prepared.faces = None
        finally:
            outputs = None
            self.memory.release_cached_memory()

    def _extract_raw_mesh(self, outputs, prepared: PreparedMesh) -> None:
        """Copy the mesh of the pipeline output to the CPU; the GPU tensors are released by the caller."""
        start = time.time()
        raw_mesh_trimesh = outputs['mesh'][0].to_trimesh(transform_pose=True)
        prepared.vertices = np.asarray(raw_mesh_trimesh.vertices)
        prepared.faces = np.asarray(raw_mesh_trimesh.faces)
        prepared.result.original_face_count = len(prepared.faces)
        prepared.to_trimesh_time = time.time() - start

    def _report_out_of_memory(self, error: Exception):
        """Explain GPU out-of-memory errors and give the memory back."""
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
            print("The GPU ran out of memory. Set the model memory mode to 'Offload' (Performance settings) "
                  "or close other programs that use the GPU, then try again.")
            self.memory.note_out_of_memory()

    # ------------------------------------------------------------------------------------------
    # CPU part
    # ------------------------------------------------------------------------------------------
    def finish_mesh(self, prepared: PreparedMesh, progress_callback: Optional[callable] = None,
                    mesh_preview_callback: Optional[callable] = None) -> ProcessingResult:
        """Decimation, UV unwrapping, GLB export and auto-save of a prepared mesh.

        The heavy steps run in a child process (see logic/mesh_jobs.py), so the app stays responsive meanwhile and a
        cancel stops them at once. `mesh_preview_callback(path)` gets a GLB of the decimated mesh as soon as it exists.
        """
        result, params = prepared.result, prepared.params
        if not prepared.ready:
            result.total_time = time.time() - prepared.started if prepared.started else 0.0
            return result

        update_progress = self._progress_updater(progress_callback)
        try:
            # Prepare output directory (unique, also for results that finish within the same second)
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            output_dir = os.path.join(self.tmp_dir, f"{timestamp}_{uuid.uuid4().hex[:6]}")
            os.makedirs(output_dir, exist_ok=True)
            mesh_path_glb = os.path.join(output_dir, "mesh.glb")

            def on_stage(name: str):
                if name == "simplify":
                    update_progress("Mesh Processing", "Simplifying the mesh")
                elif name == "unwrap":
                    update_progress("UV Unwrapping", "Processing with xatlas")

            update_progress("Mesh Processing", "Simplifying the mesh")
            job = run_mesh_job(
                prepared.vertices, prepared.faces, params.poly_count_pcnt,
                xatlas_max_cost=params.xatlas_max_cost,
                xatlas_normal_seam_weight=params.xatlas_normal_seam_weight,
                xatlas_resolution=params.xatlas_resolution,
                xatlas_padding=params.xatlas_padding,
                cancel_check=should_cancel, on_stage=on_stage, tmp_root=self.tmp_dir,
                on_preview=mesh_preview_callback,
                preview_path=os.path.join(output_dir, "preview.glb") if mesh_preview_callback else None,
            )
            prepared.vertices = prepared.faces = None  # the raw mesh is no longer needed

            unwrapped_mesh_trimesh = job.mesh
            result.original_face_count = job.original_faces
            result.simplified_face_count = job.simplified_faces
            result.mesh_processing_time = prepared.to_trimesh_time + job.simplify_s
            print(f"Mesh Processing: Completed in {result.mesh_processing_time:.1f}s")
            result.uv_unwrapping_time = job.unwrap_s
            result.final_vertex_count = len(unwrapped_mesh_trimesh.vertices)
            result.final_face_count = len(unwrapped_mesh_trimesh.faces)

            # Check cancellation
            if should_cancel():
                result.error_message = "Cancelled during mesh processing"
                return self._close(result, prepared)

            # Export mesh
            update_progress("File Export", f"Exporting GLB to {mesh_path_glb}")
            export_start = time.time()

            unwrapped_mesh_trimesh.export(mesh_path_glb)

            result.export_time = time.time() - export_start
            print(f"File Export: GLB exported successfully in {result.export_time:.2f}s")
            result.mesh_path, result.output_dir = mesh_path_glb, output_dir

            # Log timing breakdown
            total_generation_time = max(time.time() - prepared.started, 1e-6)
            print(f"=== 3D GENERATION COMPLETE ===")
            print(f"Total time breakdown:")
            print(f"  - Normal prediction: {result.normal_prediction_time:.2f}s")
            print(f"  - Hi3DGen pipeline: {result.pipeline_time:.2f}s")
            print(f"  - Mesh processing: {result.mesh_processing_time:.2f}s")
            print(f"  - UV unwrapping: {result.uv_unwrapping_time:.2f}s")
            print(f"  - File export: {result.export_time:.2f}s")
            print(f"  - TOTAL: {total_generation_time:.2f}s")

            # Stage 3: Auto-save (if enabled)
            if any([params.auto_save_obj, params.auto_save_glb, params.auto_save_ply, params.auto_save_stl]):
                result.auto_save_result = self._perform_auto_save(
                    result.mesh_path, result.normal_image, params, update_progress, result,
                    prepared.custom_output_folder, prepared.custom_filename_base
                )

            # Final success
            result.success = True
            result.total_time = time.time() - prepared.started

            update_progress("Completed", f"Total time: {result.total_time:.1f}s")

        except MeshJobCancelled:
            print("Mesh Processing: cancelled by user")
            result.error_message = "Cancelled during mesh processing"
            result.mesh_path = result.output_dir = None
        except Exception as e:
            result.error_message = f"Processing failed: {str(e)}"
            print(f"Error in finish_mesh: {e}")
            traceback.print_exc()
            result.mesh_path = result.output_dir = None
        finally:
            prepared.vertices = prepared.faces = None
            if not result.success:
                result.total_time = time.time() - prepared.started

        return result

    @staticmethod
    def _close(result: ProcessingResult, prepared: PreparedMesh) -> ProcessingResult:
        result.mesh_path = result.output_dir = None
        result.total_time = time.time() - prepared.started
        return result

    def _perform_auto_save(self,
                          mesh_path: str,
                          normal_image: Image.Image,
                          params: ProcessingParameters,
                          update_progress: callable,
                          result: ProcessingResult,
                          custom_output_folder: Optional[str] = None,
                          custom_filename_base: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Perform auto-save if enabled

        Args:
            mesh_path: Path to the generated mesh
            normal_image: Generated normal image
            params: Processing parameters
            update_progress: Progress callback
            result: Processing result object
            custom_output_folder: Optional custom output folder (for batch processing)
            custom_filename_base: Optional custom filename base (for batch processing)
        """
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

            # Call auto_save with batch parameters if provided
            auto_save_result = auto_save_generation(
                mesh_path=mesh_path,
                normal_image=normal_image,
                enabled_formats=enabled_formats,
                custom_output_folder=custom_output_folder,
                custom_filename_base=custom_filename_base
            )

            if auto_save_result:
                if custom_output_folder:
                    print(f"✓ Batch auto-save successful: Saved files to {custom_output_folder}")
                else:
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

# Convenience function for backward compatibility
def process_single_image_legacy(image, seed=-1, ss_guidance_strength=3, ss_sampling_steps=50,
                               slat_guidance_strength=3, slat_sampling_steps=6, poly_count_pcnt=0.5,
                               xatlas_max_cost=8.0, xatlas_normal_seam_weight=1.0, xatlas_resolution=1024,
                               xatlas_padding=2, normal_map_resolution=768, normal_match_input_resolution=True,
                               auto_save_obj=True, auto_save_glb=True, auto_save_ply=True, auto_save_stl=True,
                               processing_core: ProcessingCore = None):
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
