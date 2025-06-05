import os
import glob
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
from .processing_core import ProcessingCore, ProcessingParameters, ProcessingResult

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
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.skipped is None:
            self.skipped = []
    
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
        remaining_images = self.total_images - self.processed_images
        return avg_time_per_image * remaining_images

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
    """Handles batch processing of images for 3D generation"""
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self, processing_core: ProcessingCore):
        self.processing_core = processing_core
        self.progress = BatchProgress()
        self.is_running = False
        self.progress_callback: Optional[Callable[[BatchProgress], None]] = None
        self.processing_lock = threading.Lock()
        
        # Set up cleanup callback for batch processing
        def batch_cleanup():
            print("Batch Processor: Cleanup called")
            with self.processing_lock:
                self.is_running = False
        
        self.batch_cleanup_callback = batch_cleanup
        
    def set_progress_callback(self, callback: Callable[[BatchProgress], None]):
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    def _update_progress(self):
        """Internal method to trigger progress callback"""
        if self.progress_callback:
            try:
                self.progress_callback(self.progress)
            except Exception as e:
                print(f"Error in progress callback: {e}")
    
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
        
        # Check for each enabled format
        for format_name, is_enabled in enabled_formats.items():
            if is_enabled:
                output_file = os.path.join(output_folder, f"{image_name}.{format_name}")
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
            
            self._update_progress()
            
            # Process each image
            for i, image_path in enumerate(images_to_process):
                if should_cancel():
                    self.progress.current_stage = "Cancelled by user"
                    print("Batch processing cancelled by user")
                    break
                
                self.progress.current_image = Path(image_path).name
                self.progress.current_stage = f"Processing {self.progress.current_image}..."
                update_processing_stage(f"Processing {self.progress.current_image} ({i+1}/{len(images_to_process)})")
                self._update_progress()
                
                try:
                    success, result = self._process_single_image_in_batch(
                        image_path, settings
                    )
                    
                    if success:
                        self.progress.processed_images += 1
                        print(f"✓ Successfully processed {self.progress.current_image} "
                              f"({self.progress.processed_images}/{self.progress.total_images})")
                        if result:
                            print(f"  {result.get_summary()}")
                    else:
                        error_msg = f"Failed to process {self.progress.current_image}"
                        if result and result.error_message:
                            error_msg += f": {result.error_message}"
                        self.progress.errors.append(error_msg)
                        print(f"✗ {error_msg}")
                        
                except Exception as e:
                    error_msg = f"Error processing {self.progress.current_image}: {str(e)}"
                    self.progress.errors.append(error_msg)
                    print(f"✗ {error_msg}")
                    traceback.print_exc()
                
                self._update_progress()
            
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
    
    def _process_single_image_in_batch(self, image_path: str, settings: BatchSettings) -> Tuple[bool, Optional[ProcessingResult]]:
        """
        Process a single image within batch processing context
        
        Args:
            image_path: Path to image to process
            settings: Batch processing settings
            
        Returns:
            Tuple of (success, ProcessingResult)
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGBA')
            
            # Get the base name for output files (filename without extension)
            input_name = Path(image_path).stem
            
            # Convert BatchSettings to ProcessingParameters
            params = ProcessingParameters(
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
            
            # Create a progress callback for batch processing
            def batch_progress_callback(stage: str, details: str = ""):
                current_file = Path(image_path).name
                full_stage = f"{current_file}: {stage}"
                if details:
                    full_stage += f" - {details}"
                update_processing_stage(full_stage)
            
            # Use ProcessingCore to process the image with batch output settings
            result = self.processing_core.process_single_image(
                image=image,
                params=params,
                progress_callback=batch_progress_callback,
                custom_output_folder=settings.output_folder,
                custom_filename_base=input_name
            )
            
            # Handle batch-specific output file management
            if result.success and result.mesh_path:
                self._handle_batch_output(image_path, result, settings)
            
            return result.success, result
                
        except Exception as e:
            print(f"Error in single image processing for {image_path}: {e}")
            traceback.print_exc()
            
            # Create error result
            error_result = ProcessingResult()
            error_result.error_message = str(e)
            return False, error_result
    
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
            
            # For batch processing, the auto-save already handles file naming correctly
            # based on the logic in auto_save.py, but we could add additional 
            # batch-specific naming logic here if needed
            
            # Log the output location
            if result.auto_save_result:
                print(f"  Batch output for {input_name}: Folder {result.auto_save_result.get('folder_number', 'unknown')}")
                saved_files = result.auto_save_result.get('saved_files', [])
                for saved_file in saved_files:
                    print(f"    - {saved_file}")
            
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
def create_batch_processor(processing_core: ProcessingCore) -> BatchProcessor:
    """
    Create a new BatchProcessor instance
    
    Args:
        processing_core: ProcessingCore instance for image processing
        
    Returns:
        Configured BatchProcessor instance
    """
    return BatchProcessor(processing_core) 