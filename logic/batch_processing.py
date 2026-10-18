import os
import glob
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import List, Dict, Callable, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import time
from PIL import Image
import threading
import traceback

# Import our logic modules
from .cancellation import (
    batch_processing_context, should_cancel, update_processing_stage,
    add_cleanup_callback, remove_cleanup_callback, get_processing_status
)
from .processing_core import ProcessingCore, ProcessingParameters, ProcessingResult, PreparedMesh


def default_mesh_jobs() -> int:
    """How many results may be decimated and UV-unwrapped at the same time while the GPU works on the next image.

    Each of them runs in its own process (about 3 cores and 1 GB of memory), so the default follows the CPU and the
    free memory. Environment variable HI3DGEN_MESH_JOBS overrides it.
    """
    value = os.environ.get("HI3DGEN_MESH_JOBS", "").strip()
    if value.isdigit() and int(value) > 0:
        return min(8, int(value))
    cpus = os.cpu_count() or 4
    jobs = 1 if cpus < 8 else 2 if cpus < 16 else 3
    try:
        import psutil
        jobs = min(jobs, int(psutil.virtual_memory().available / 1024 ** 3 // 3))
    except Exception:
        pass
    return max(1, jobs)


@dataclass
class BatchProgress:
    """Tracks progress of batch processing"""
    total_images: int = 0
    processed_images: int = 0
    current_image: str = ""
    current_stage: str = ""
    start_time: float = 0.0
    errors: List[str] = None
    skipped: List[str] = None
    # Stage text of every image that is being worked on right now (the GPU part and the CPU parts overlap)
    active: Dict[str, str] = None
    cancelled_images: int = 0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.skipped is None:
            self.skipped = []
        if self.active is None:
            self.active = {}

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_images == 0:
            return 0.0
        return (self.processed_images / self.total_images) * 100

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def estimated_time_remaining(self) -> float:
        """Estimate remaining time in seconds"""
        if self.processed_images == 0 or self.elapsed_time == 0:
            return 0.0

        avg_time_per_image = self.elapsed_time / self.processed_images
        remaining_images = self.total_images - self.processed_images - len(self.errors)
        return avg_time_per_image * max(remaining_images, 0)

@dataclass
class BatchSettings:
    """Configuration for batch processing"""
    input_folder: str
    output_folder: str
    skip_existing: bool = True
    auto_save_obj: bool = True
    auto_save_glb: bool = True
    auto_save_ply: bool = True
    auto_save_stl: bool = True
    # Processing parameters (will be passed through to single processing)
    seed: int = -1
    ss_guidance_strength: float = 3.0
    ss_sampling_steps: int = 50
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 6
    poly_count_pcnt: float = 0.5
    xatlas_max_cost: float = 8.0
    xatlas_normal_seam_weight: float = 1.0
    xatlas_resolution: int = 1024
    xatlas_padding: int = 2
    normal_map_resolution: int = 768
    normal_match_input_resolution: bool = True

class BatchProcessor:
    """Handles batch processing of images for 3D generation

    The GPU part of an image (normal map, Hi3DGen) runs here while the CPU part of the previous images (decimation,
    UV unwrapping, export, auto-save) runs in worker processes, so a batch takes about as long as its GPU work
    instead of the sum of both. `mesh_jobs` is the number of images whose CPU part may run at the same time.
    """

    # Supported image extensions
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(self, processing_core: ProcessingCore, mesh_jobs: Optional[int] = None):
        self.processing_core = processing_core
        self.progress = BatchProgress()
        self.is_running = False
        self.progress_callback: Optional[Callable[[BatchProgress], None]] = None
        self.processing_lock = threading.Lock()
        self._progress_lock = threading.RLock()
        self.mesh_jobs = max(1, min(8, int(mesh_jobs))) if mesh_jobs else default_mesh_jobs()

        # Set up cleanup callback for batch processing
        def batch_cleanup():
            print("Batch Processor: Cleanup called")
            with self.processing_lock:
                self.is_running = False

        self.batch_cleanup_callback = batch_cleanup

    def set_progress_callback(self, callback: Callable[[BatchProgress], None]):
        """Set callback function for progress updates"""
        self.progress_callback = callback

    def set_mesh_jobs(self, count: int) -> int:
        """Number of CPU jobs that may overlap (applies to the next batch)"""
        self.mesh_jobs = max(1, min(8, int(count)))
        return self.mesh_jobs

    def _update_progress(self):
        """Internal method to trigger progress callback"""
        if self.progress_callback:
            try:
                self.progress_callback(self.progress)
            except Exception as e:
                print(f"Error in progress callback: {e}")

    def _set_stage(self, image_name: str, stage: str, details: str = ""):
        """Record what is happening to one image and refresh the combined status text"""
        text = f"{stage}: {details}" if details else stage
        with self._progress_lock:
            self.progress.active[image_name] = text
            self.progress.current_stage = " | ".join(f"{name} - {state}" for name, state in self.progress.active.items())
            update_processing_stage(self.progress.current_stage)
        self._update_progress()

    def _clear_stage(self, image_name: str):
        with self._progress_lock:
            self.progress.active.pop(image_name, None)

    def discover_images(self, input_folder: str) -> List[str]:
        """
        Discover all supported image files in the input folder

        Args:
            input_folder: Path to folder containing images

        Returns:
            List of image file paths
        """
        if not os.path.exists(input_folder):
            raise ValueError(f"Input folder does not exist: {input_folder}")

        image_files_set = set()  # Use set to avoid duplicates
        input_path = Path(input_folder)

        # Search for all supported image extensions
        for ext in self.SUPPORTED_EXTENSIONS:
            # Case insensitive search
            pattern = f"*{ext}"
            files = list(input_path.glob(pattern))
            # Also search uppercase
            files.extend(list(input_path.glob(pattern.upper())))
            # Add to set to automatically deduplicate
            for file in files:
                image_files_set.add(str(file))

        # Convert to list and sort
        image_files = list(image_files_set)
        image_files.sort()

        print(f"Discovered {len(image_files)} image files in {input_folder}")
        return image_files

    def should_skip_image(self, image_path: str, output_folder: str,
                         enabled_formats: Dict[str, bool]) -> Tuple[bool, List[str]]:
        """
        Check if an image should be skipped based on existing output files

        Args:
            image_path: Path to input image
            output_folder: Path to output folder
            enabled_formats: Dict of format -> enabled status

        Returns:
            Tuple of (should_skip, existing_files)
        """
        image_name = Path(image_path).stem
        existing_files = []

        # With the new subfolder structure, files are saved in {output_folder}/{image_name}/{image_name}.{format}
        image_subfolder = os.path.join(output_folder, image_name)

        # Check for each enabled format
        for format_name, is_enabled in enabled_formats.items():
            if is_enabled:
                output_file = os.path.join(image_subfolder, f"{image_name}.{format_name}")
                if os.path.exists(output_file):
                    existing_files.append(output_file)

        # Skip if any enabled format already exists
        should_skip = len(existing_files) > 0
        return should_skip, existing_files

    def prepare_output_folder(self, output_folder: str):
        """Ensure output folder exists"""
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder prepared: {output_folder}")

    def filter_images_for_processing(self, image_files: List[str], settings: BatchSettings) -> List[str]:
        """
        Filter images based on skip_existing setting

        Args:
            image_files: List of discovered image files
            settings: Batch processing settings

        Returns:
            List of images to process
        """
        if not settings.skip_existing:
            return image_files

        enabled_formats = {
            "obj": settings.auto_save_obj,
            "glb": settings.auto_save_glb,
            "ply": settings.auto_save_ply,
            "stl": settings.auto_save_stl
        }

        images_to_process = []
        for image_path in image_files:
            should_skip, existing_files = self.should_skip_image(
                image_path, settings.output_folder, enabled_formats
            )

            if should_skip:
                self.progress.skipped.append(f"{Path(image_path).name} (existing: {len(existing_files)} files)")
                print(f"Skipping {Path(image_path).name} - {len(existing_files)} output files already exist")
            else:
                images_to_process.append(image_path)

        return images_to_process

    def get_output_filename_base(self, input_image_path: str) -> str:
        """
        Generate output filename base from input image path

        Args:
            input_image_path: Path to input image

        Returns:
            Base filename (without extension) for outputs
        """
        return Path(input_image_path).stem

    def cancel_processing(self):
        """Cancel the current batch processing"""
        print("Batch processing cancellation requested...")
        from .cancellation import request_cancellation
        request_cancellation("Batch processing cancelled by user")

    def is_cancelled(self) -> bool:
        """Check if processing should be cancelled"""
        return should_cancel()

    def reset_cancellation(self):
        """Reset cancellation flag"""
        # Cancellation reset is handled by the global cancellation manager
        pass

    def start_batch_processing(self, settings: BatchSettings) -> BatchProgress:
        """
        Start batch processing of images

        Args:
            settings: Batch processing configuration

        Returns:
            Final batch progress state
        """
        with self.processing_lock:
            if self.is_running:
                raise RuntimeError("Batch processing is already running")

            self.is_running = True

        # Use the cancellation system context manager
        try:
            with batch_processing_context("Starting batch processing"):
                # Register cleanup callback
                add_cleanup_callback(self.batch_cleanup_callback)

                return self._execute_batch_processing(settings)
        finally:
            # Remove cleanup callback
            remove_cleanup_callback(self.batch_cleanup_callback)
            with self.processing_lock:
                self.is_running = False

    def _execute_batch_processing(self, settings: BatchSettings) -> BatchProgress:
        """
        Internal method to execute batch processing

        Args:
            settings: Batch processing configuration

        Returns:
            Final batch progress state
        """
        # Initialize progress
        self.progress = BatchProgress()
        self.progress.start_time = time.time()
        self.progress.current_stage = "Discovering images..."
        update_processing_stage("Discovering images...")
        self._update_progress()

        try:
            # Discover images
            all_images = self.discover_images(settings.input_folder)

            if not all_images:
                self.progress.current_stage = "No images found"
                update_processing_stage("No images found")
                self._update_progress()
                return self.progress

            # Check for cancellation
            if should_cancel():
                self.progress.current_stage = "Cancelled during discovery"
                return self.progress

            # Prepare output folder
            self.prepare_output_folder(settings.output_folder)

            # Filter images based on skip_existing setting
            self.progress.current_stage = "Filtering images..."
            update_processing_stage("Filtering images...")
            self._update_progress()

            images_to_process = self.filter_images_for_processing(all_images, settings)

            self.progress.total_images = len(images_to_process)
            self.progress.current_stage = f"Processing {len(images_to_process)} images..."

            print(f"Total images to process: {len(images_to_process)}")
            print(f"Skipped images: {len(self.progress.skipped)}")
            print(f"CPU jobs that may overlap with the GPU work: {self.mesh_jobs}")

            self._update_progress()

            # GPU part of each image here, CPU part in the pool (a child process does the heavy work)
            in_flight = {}  # future -> image name
            executor = ThreadPoolExecutor(max_workers=self.mesh_jobs, thread_name_prefix="hi3dgen-mesh")
            try:
                for i, image_path in enumerate(images_to_process):
                    if should_cancel():
                        self.progress.current_stage = "Cancelled by user"
                        print("Batch processing cancelled by user")
                        break

                    # The GPU may work one image ahead of the CPU workers (its raw mesh waits for a free worker), not more
                    while len(in_flight) > self.mesh_jobs:
                        self._collect(in_flight, block=True)
                    if should_cancel():
                        self.progress.current_stage = "Cancelled by user"
                        print("Batch processing cancelled by user")
                        break

                    image_name = Path(image_path).name
                    self.progress.current_image = image_name
                    self._set_stage(image_name, "Starting", f"{i + 1}/{len(images_to_process)}")

                    prepared = self._prepare_image_in_batch(image_path, settings)
                    if not prepared.ready:
                        self._clear_stage(image_name)
                        self._record(image_name, False, prepared.result)
                    else:
                        future = executor.submit(self._finish_image_in_batch, prepared, image_path, settings)
                        in_flight[future] = image_name

                    self._collect(in_flight, block=False)

                # The last images are still being unwrapped
                while in_flight:
                    self._collect(in_flight, block=True)
            finally:
                executor.shutdown(wait=True)

            # Final status
            if should_cancel():
                self.progress.current_stage = "Cancelled"
                update_processing_stage("Batch processing cancelled")
            else:
                self.progress.current_stage = "Completed"
                update_processing_stage("Batch processing completed")
                print(f"\n=== BATCH PROCESSING COMPLETE ===")
                print(f"Total processed: {self.progress.processed_images}/{self.progress.total_images}")
                print(f"Skipped: {len(self.progress.skipped)}")
                print(f"Errors: {len(self.progress.errors)}")
                print(f"Total time: {self.progress.elapsed_time:.1f}s")

            self._update_progress()

        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            self.progress.errors.append(error_msg)
            self.progress.current_stage = "Failed"
            print(f"✗ {error_msg}")
            traceback.print_exc()
            self._update_progress()

        return self.progress

    def _collect(self, in_flight: Dict[Any, str], block: bool):
        """Take the finished CPU parts out of `in_flight` and book their results (waits for one if `block`)."""
        if not in_flight:
            return
        done, _ = wait(list(in_flight), timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for future in done:
            image_name = in_flight.pop(future)
            success, result = future.result()
            self._record(image_name, success, result)

    def _record(self, image_name: str, success: bool, result: Optional[ProcessingResult]):
        """Book the outcome of one image"""
        with self._progress_lock:
            if success:
                self.progress.processed_images += 1
                print(f"✓ Successfully processed {image_name} "
                      f"({self.progress.processed_images}/{self.progress.total_images})")
                if result:
                    print(f"  {result.get_summary()}")
            elif should_cancel() and result and (result.error_message or "").startswith("Cancelled"):
                self.progress.cancelled_images += 1
                print(f"⏹ Cancelled {image_name}")
            else:
                error_msg = f"Failed to process {image_name}"
                if result and result.error_message:
                    error_msg += f": {result.error_message}"
                self.progress.errors.append(error_msg)
                print(f"✗ {error_msg}")
        self._update_progress()

    def _batch_parameters(self, settings: BatchSettings) -> ProcessingParameters:
        """Convert BatchSettings to ProcessingParameters"""
        return ProcessingParameters(
            seed=settings.seed,
            ss_guidance_strength=settings.ss_guidance_strength,
            ss_sampling_steps=settings.ss_sampling_steps,
            slat_guidance_strength=settings.slat_guidance_strength,
            slat_sampling_steps=settings.slat_sampling_steps,
            poly_count_pcnt=settings.poly_count_pcnt,
            xatlas_max_cost=settings.xatlas_max_cost,
            xatlas_normal_seam_weight=settings.xatlas_normal_seam_weight,
            xatlas_resolution=settings.xatlas_resolution,
            xatlas_padding=settings.xatlas_padding,
            normal_map_resolution=settings.normal_map_resolution,
            normal_match_input_resolution=settings.normal_match_input_resolution,
            auto_save_obj=settings.auto_save_obj,
            auto_save_glb=settings.auto_save_glb,
            auto_save_ply=settings.auto_save_ply,
            auto_save_stl=settings.auto_save_stl
        )

    def _prepare_image_in_batch(self, image_path: str, settings: BatchSettings) -> PreparedMesh:
        """GPU part of one batch image (normal map and Hi3DGen)"""
        image_name = Path(image_path).name
        try:
            # Load image
            image = Image.open(image_path).convert('RGBA')

            # Get the base name for output files (filename without extension)
            input_name = Path(image_path).stem

            def batch_progress_callback(stage: str, details: str = ""):
                self._set_stage(image_name, stage, details)

            return self.processing_core.prepare_image(
                image=image,
                params=self._batch_parameters(settings),
                progress_callback=batch_progress_callback,
                custom_output_folder=settings.output_folder,
                custom_filename_base=input_name
            )
        except Exception as e:
            print(f"Error in single image processing for {image_path}: {e}")
            traceback.print_exc()
            error_result = ProcessingResult()
            error_result.error_message = str(e)
            return PreparedMesh(result=error_result, params=self._batch_parameters(settings))

    def _finish_image_in_batch(self, prepared: PreparedMesh, image_path: str,
                               settings: BatchSettings) -> Tuple[bool, Optional[ProcessingResult]]:
        """CPU part of one batch image (decimation, UV unwrapping, export, auto-save); runs in the pool"""
        image_name = Path(image_path).name
        try:
            def batch_progress_callback(stage: str, details: str = ""):
                self._set_stage(image_name, stage, details)

            result = self.processing_core.finish_mesh(prepared, batch_progress_callback)

            # Handle batch-specific output file management
            if result.success and result.mesh_path:
                self._handle_batch_output(image_path, result, settings)

            return result.success, result
        except Exception as e:
            print(f"Error finishing {image_path}: {e}")
            traceback.print_exc()
            error_result = ProcessingResult()
            error_result.error_message = str(e)
            return False, error_result
        finally:
            self._clear_stage(image_name)

    def _process_single_image_in_batch(self, image_path: str, settings: BatchSettings) -> Tuple[bool, Optional[ProcessingResult]]:
        """
        Process a single image within batch processing context (both parts, one after the other)

        Args:
            image_path: Path to image to process
            settings: Batch processing settings

        Returns:
            Tuple of (success, ProcessingResult)
        """
        prepared = self._prepare_image_in_batch(image_path, settings)
        if not prepared.ready:
            return False, prepared.result
        return self._finish_image_in_batch(prepared, image_path, settings)

    def _handle_batch_output(self, input_image_path: str, result: ProcessingResult, settings: BatchSettings):
        """
        Handle batch-specific output file management

        Args:
            input_image_path: Path to the input image
            result: Processing result
            settings: Batch settings
        """
        try:
            # Get the base name for output files
            input_name = Path(input_image_path).stem

            # Log the output location with subfolder information
            if result.auto_save_result:
                folder_path = result.auto_save_result.get('folder_path', 'unknown')
                print(f"  Batch output for {input_name}: Subfolder '{input_name}' in {settings.output_folder}")
                print(f"    Full path: {folder_path}")
                saved_files = result.auto_save_result.get('saved_files', [])
                for format_name, file_path in saved_files.items():
                    if format_name != 'info':  # Don't log the info file
                        print(f"    - {format_name.upper()}: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Warning: Error handling batch output for {input_image_path}: {e}")

    def get_status_summary(self) -> str:
        """Get a human-readable status summary"""
        if not self.is_running:
            # Check global cancellation status
            cancellation_status = get_processing_status()
            if cancellation_status.processing_state.value != "idle":
                return f"Global: {cancellation_status.current_stage}"
            return "Idle"

        if self.progress.total_images == 0:
            return self.progress.current_stage

        percentage = self.progress.progress_percentage
        eta = self.progress.estimated_time_remaining

        if eta > 0:
            eta_str = f", ETA: {eta/60:.1f}min" if eta > 60 else f", ETA: {eta:.0f}s"
        else:
            eta_str = ""

        return (f"{self.progress.current_stage} - "
                f"{self.progress.processed_images}/{self.progress.total_images} "
                f"({percentage:.1f}%{eta_str})")

# Factory function for creating BatchProcessor
def create_batch_processor(processing_core: ProcessingCore, mesh_jobs: Optional[int] = None) -> BatchProcessor:
    """
    Create a new BatchProcessor instance

    Args:
        processing_core: ProcessingCore instance for image processing
        mesh_jobs: Number of CPU jobs that may overlap with the GPU work (default: follows the CPU and free memory)

    Returns:
        Configured BatchProcessor instance
    """
    return BatchProcessor(processing_core, mesh_jobs)
