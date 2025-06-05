import threading
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import gc
import torch

class ProcessingState(Enum):
    """Enum for different processing states"""
    IDLE = "idle"
    SINGLE_PROCESSING = "single_processing"
    BATCH_PROCESSING = "batch_processing"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

@dataclass
class CancellationStatus:
    """Status information about cancellation"""
    is_cancelled: bool = False
    cancellation_requested_at: float = 0.0
    current_stage: str = ""
    processing_state: ProcessingState = ProcessingState.IDLE
    cleanup_completed: bool = False
    
    @property
    def cancellation_elapsed_time(self) -> float:
        """Get time elapsed since cancellation was requested"""
        if self.cancellation_requested_at == 0.0:
            return 0.0
        return time.time() - self.cancellation_requested_at

class GlobalCancellationManager:
    """
    Global cancellation manager for coordinating cancellation across 
    single processing and batch processing operations
    """
    
    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._status = CancellationStatus()
        self._cleanup_callbacks = []
        self._progress_callback: Optional[Callable[[CancellationStatus], None]] = None
        
        # Processing tracking
        self._current_operation = None
        self._operation_start_time = 0.0
        
    def set_progress_callback(self, callback: Callable[[CancellationStatus], None]):
        """Set callback for cancellation status updates"""
        with self._lock:
            self._progress_callback = callback
    
    def _notify_progress(self):
        """Internal method to notify progress callback"""
        if self._progress_callback:
            try:
                self._progress_callback(self._status)
            except Exception as e:
                print(f"Error in cancellation progress callback: {e}")
    
    def start_operation(self, operation_type: ProcessingState, stage: str = ""):
        """
        Start a new processing operation
        
        Args:
            operation_type: Type of operation starting
            stage: Current stage description
        """
        with self._lock:
            if self._status.processing_state != ProcessingState.IDLE:
                raise RuntimeError(f"Cannot start {operation_type.value} - already in {self._status.processing_state.value} state")
            
            self._status = CancellationStatus(
                processing_state=operation_type,
                current_stage=stage
            )
            self._operation_start_time = time.time()
            
            print(f"Cancellation Manager: Started {operation_type.value} operation - {stage}")
            self._notify_progress()
    
    def update_stage(self, stage: str):
        """
        Update the current processing stage
        
        Args:
            stage: Description of current stage
        """
        with self._lock:
            if self._status.processing_state != ProcessingState.IDLE:
                self._status.current_stage = stage
                self._notify_progress()
    
    def request_cancellation(self, stage: str = "Cancellation requested"):
        """
        Request cancellation of current operation
        
        Args:
            stage: Description of cancellation reason
        """
        with self._lock:
            if self._status.processing_state == ProcessingState.IDLE:
                print("Cancellation Manager: No active operation to cancel")
                return False
            
            if self._status.is_cancelled:
                print("Cancellation Manager: Cancellation already requested")
                return True
            
            self._status.is_cancelled = True
            self._status.cancellation_requested_at = time.time()
            self._status.current_stage = stage
            self._status.processing_state = ProcessingState.CANCELLING
            
            print(f"Cancellation Manager: Cancellation requested - {stage}")
            self._notify_progress()
            
            # Execute cleanup callbacks
            self._execute_cleanup_callbacks()
            
            return True
    
    def is_cancellation_requested(self) -> bool:
        """
        Check if cancellation has been requested
        
        Returns:
            True if cancellation requested, False otherwise
        """
        with self._lock:
            return self._status.is_cancelled
    
    def should_cancel_now(self) -> bool:
        """
        Check if processing should be cancelled immediately
        This is the main method that processing loops should call
        
        Returns:
            True if processing should stop immediately
        """
        return self.is_cancellation_requested()
    
    def finish_operation(self, final_stage: str = "Completed"):
        """
        Mark the current operation as finished
        
        Args:
            final_stage: Final stage description
        """
        with self._lock:
            if self._status.processing_state == ProcessingState.IDLE:
                return
            
            operation_time = time.time() - self._operation_start_time
            
            if self._status.is_cancelled:
                self._status.processing_state = ProcessingState.CANCELLED
                self._status.current_stage = "Cancelled"
                print(f"Cancellation Manager: Operation cancelled after {operation_time:.1f}s")
            else:
                self._status.processing_state = ProcessingState.IDLE
                self._status.current_stage = final_stage
                print(f"Cancellation Manager: Operation completed in {operation_time:.1f}s")
            
            self._notify_progress()
            
            # Reset for next operation after a brief delay
            self._schedule_reset()
    
    def _schedule_reset(self):
        """Schedule reset of cancellation state after brief delay"""
        def reset_after_delay():
            time.sleep(1.0)  # Brief delay to allow UI updates
            with self._lock:
                if self._status.processing_state in [ProcessingState.CANCELLED, ProcessingState.IDLE]:
                    self._status = CancellationStatus()
                    self._cleanup_callbacks.clear()
                    print("Cancellation Manager: State reset")
                    self._notify_progress()
        
        reset_thread = threading.Thread(target=reset_after_delay, daemon=True)
        reset_thread.start()
    
    def add_cleanup_callback(self, callback: Callable[[], None]):
        """
        Add a cleanup callback to be executed on cancellation
        
        Args:
            callback: Function to call during cleanup
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def remove_cleanup_callback(self, callback: Callable[[], None]):
        """
        Remove a cleanup callback
        
        Args:
            callback: Function to remove from cleanup callbacks
        """
        with self._lock:
            if callback in self._cleanup_callbacks:
                self._cleanup_callbacks.remove(callback)
    
    def _execute_cleanup_callbacks(self):
        """Execute all registered cleanup callbacks"""
        with self._lock:
            if self._status.cleanup_completed:
                return
            
            print("Cancellation Manager: Executing cleanup callbacks...")
            
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in cleanup callback: {e}")
            
            # Always perform standard cleanup
            self._perform_standard_cleanup()
            
            self._status.cleanup_completed = True
            print("Cancellation Manager: Cleanup completed")
    
    def _perform_standard_cleanup(self):
        """Perform standard cleanup operations"""
        try:
            # Clear GPU memory if available
            if torch.cuda.is_available():
                print("Cancellation Manager: Clearing CUDA cache...")
                torch.cuda.empty_cache()
            
            # Force garbage collection
            print("Cancellation Manager: Running garbage collection...")
            gc.collect()
            
        except Exception as e:
            print(f"Error during standard cleanup: {e}")
    
    def get_status(self) -> CancellationStatus:
        """
        Get current cancellation status
        
        Returns:
            Copy of current cancellation status
        """
        with self._lock:
            # Return a copy to prevent external modification
            return CancellationStatus(
                is_cancelled=self._status.is_cancelled,
                cancellation_requested_at=self._status.cancellation_requested_at,
                current_stage=self._status.current_stage,
                processing_state=self._status.processing_state,
                cleanup_completed=self._status.cleanup_completed
            )
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary"""
        with self._lock:
            state = self._status.processing_state.value.replace('_', ' ').title()
            
            if self._status.current_stage:
                return f"{state}: {self._status.current_stage}"
            else:
                return state
    
    def force_reset(self):
        """
        Force reset of the cancellation manager (emergency use only)
        """
        with self._lock:
            print("Cancellation Manager: Force reset requested")
            self._execute_cleanup_callbacks()
            self._status = CancellationStatus()
            self._cleanup_callbacks.clear()
            print("Cancellation Manager: Force reset completed")
            self._notify_progress()

# Global instance - singleton pattern
_global_cancellation_manager = None
_manager_lock = threading.Lock()

def get_cancellation_manager() -> GlobalCancellationManager:
    """
    Get the global cancellation manager instance (singleton)
    
    Returns:
        Global cancellation manager instance
    """
    global _global_cancellation_manager
    
    with _manager_lock:
        if _global_cancellation_manager is None:
            _global_cancellation_manager = GlobalCancellationManager()
            print("Cancellation Manager: Global instance created")
        
        return _global_cancellation_manager

# Convenience functions for common operations
def start_single_processing(stage: str = "Starting single image processing"):
    """Start single processing operation"""
    get_cancellation_manager().start_operation(ProcessingState.SINGLE_PROCESSING, stage)

def start_batch_processing(stage: str = "Starting batch processing"):
    """Start batch processing operation"""
    get_cancellation_manager().start_operation(ProcessingState.BATCH_PROCESSING, stage)

def update_processing_stage(stage: str):
    """Update current processing stage"""
    get_cancellation_manager().update_stage(stage)

def request_cancellation(reason: str = "User requested cancellation"):
    """Request cancellation of current operation"""
    return get_cancellation_manager().request_cancellation(reason)

def should_cancel() -> bool:
    """Check if processing should be cancelled (main method for processing loops)"""
    return get_cancellation_manager().should_cancel_now()

def finish_processing(final_stage: str = "Completed"):
    """Mark current processing as finished"""
    get_cancellation_manager().finish_operation(final_stage)

def add_cleanup_callback(callback: Callable[[], None]):
    """Add cleanup callback for cancellation"""
    get_cancellation_manager().add_cleanup_callback(callback)

def remove_cleanup_callback(callback: Callable[[], None]):
    """Remove cleanup callback"""
    get_cancellation_manager().remove_cleanup_callback(callback)

def get_processing_status() -> CancellationStatus:
    """Get current processing status"""
    return get_cancellation_manager().get_status()

def get_status_summary() -> str:
    """Get human-readable status summary"""
    return get_cancellation_manager().get_status_summary()

def force_reset_cancellation():
    """Force reset cancellation state (emergency use)"""
    get_cancellation_manager().force_reset()

# Context manager for automatic operation lifecycle management
class ProcessingContext:
    """Context manager for automatic processing lifecycle management"""
    
    def __init__(self, operation_type: ProcessingState, initial_stage: str = ""):
        self.operation_type = operation_type
        self.initial_stage = initial_stage
        self.manager = get_cancellation_manager()
    
    def __enter__(self):
        self.manager.start_operation(self.operation_type, self.initial_stage)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Exception occurred
            self.manager.finish_operation("Failed due to exception")
        elif self.manager.is_cancellation_requested():
            # Was cancelled
            self.manager.finish_operation("Cancelled")
        else:
            # Completed successfully
            self.manager.finish_operation("Completed")
    
    def update_stage(self, stage: str):
        """Update processing stage within context"""
        self.manager.update_stage(stage)
    
    def should_cancel(self) -> bool:
        """Check if should cancel within context"""
        return self.manager.should_cancel_now()

# Convenience context managers
def single_processing_context(initial_stage: str = "Starting single processing"):
    """Context manager for single processing"""
    return ProcessingContext(ProcessingState.SINGLE_PROCESSING, initial_stage)

def batch_processing_context(initial_stage: str = "Starting batch processing"):
    """Context manager for batch processing"""
    return ProcessingContext(ProcessingState.BATCH_PROCESSING, initial_stage) 