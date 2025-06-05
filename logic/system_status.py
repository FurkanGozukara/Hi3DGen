"""
System Status Module for Hi3DGen
Provides comprehensive status checking and debugging information
"""

import os
import torch
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Optional import for psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - system performance monitoring will be limited")

from .cancellation import get_cancellation_manager
from .processing_core import ProcessingCore
from .batch_processing import BatchProcessor

@dataclass
class SystemStatus:
    """System status information"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    overall_health: str = "Unknown"
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    directories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

class SystemStatusChecker:
    """Comprehensive system status checker"""
    
    def __init__(self):
        self.last_check = None
        self.check_history = []
    
    def get_comprehensive_status(self, 
                               processing_core: Optional[ProcessingCore] = None,
                               batch_processor: Optional[BatchProcessor] = None,
                               hi3dgen_pipeline = None,
                               weights_dir: str = None,
                               tmp_dir: str = None) -> SystemStatus:
        """Get comprehensive system status"""
        
        status = SystemStatus()
        
        try:
            # Check system components
            self._check_core_components(status, processing_core, batch_processor, hi3dgen_pipeline)
            
            # Check directories
            self._check_directories(status, weights_dir, tmp_dir)
            
            # Check performance
            self._check_performance(status)
            
            # Check GPU status
            self._check_gpu_status(status)
            
            # Check cancellation system
            self._check_cancellation_system(status)
            
            # Determine overall health
            self._determine_overall_health(status)
            
            # Update check history
            self.last_check = datetime.now()
            self.check_history.append({
                'timestamp': status.timestamp,
                'health': status.overall_health,
                'component_count': len(status.components),
                'warnings': len(status.warnings),
                'errors': len(status.errors)
            })
            
            # Keep only last 10 checks
            if len(self.check_history) > 10:
                self.check_history = self.check_history[-10:]
                
        except Exception as e:
            status.errors.append(f"Status check failed: {str(e)}")
            status.overall_health = "Critical"
        
        return status
    
    def _check_core_components(self, status: SystemStatus, 
                             processing_core: Optional[ProcessingCore],
                             batch_processor: Optional[BatchProcessor],
                             hi3dgen_pipeline) -> None:
        """Check core system components"""
        
        # ProcessingCore
        if processing_core is not None:
            status.components["ProcessingCore"] = {
                "status": "Available",
                "details": "ProcessingCore initialized and ready"
            }
        else:
            status.components["ProcessingCore"] = {
                "status": "Missing",
                "details": "ProcessingCore not initialized"
            }
            status.errors.append("ProcessingCore not available")
        
        # BatchProcessor
        if batch_processor is not None:
            batch_status = "Running" if hasattr(batch_processor, 'is_running') and batch_processor.is_running else "Ready"
            status.components["BatchProcessor"] = {
                "status": batch_status,
                "details": f"BatchProcessor initialized, status: {batch_status}"
            }
            
            # Add batch-specific info
            if hasattr(batch_processor, 'progress'):
                progress = batch_processor.progress
                status.components["BatchProcessor"]["progress_info"] = {
                    "total_images": progress.total_images,
                    "processed_images": progress.processed_images,
                    "errors": len(progress.errors),
                    "skipped": len(progress.skipped)
                }
        else:
            status.components["BatchProcessor"] = {
                "status": "Missing",
                "details": "BatchProcessor not initialized"
            }
            status.errors.append("BatchProcessor not available")
        
        # Hi3DGen Pipeline
        if hi3dgen_pipeline is not None:
            try:
                # Try multiple methods to determine device location for Hi3DGenPipeline
                device = "Unknown"
                
                # Method 1: Check if pipeline has a device attribute
                if hasattr(hi3dgen_pipeline, 'device'):
                    device = str(hi3dgen_pipeline.device)
                # Method 2: Check if pipeline has individual model components
                elif hasattr(hi3dgen_pipeline, 'model') and hasattr(hi3dgen_pipeline.model, 'device'):
                    device = str(hi3dgen_pipeline.model.device)
                # Method 3: Try to check if pipeline has parameters (for PyTorch models)
                elif hasattr(hi3dgen_pipeline, 'parameters'):
                    try:
                        device = "GPU" if next(hi3dgen_pipeline.parameters()).is_cuda else "CPU"
                    except (StopIteration, AttributeError):
                        pass
                # Method 4: Check for common pipeline components
                elif hasattr(hi3dgen_pipeline, 'sparse_structure_sampler'):
                    if hasattr(hi3dgen_pipeline.sparse_structure_sampler, 'device'):
                        device = str(hi3dgen_pipeline.sparse_structure_sampler.device)
                # Method 5: Fallback - assume CPU unless proven otherwise
                else:
                    device = "CPU (assumed)"
                
                status.components["Hi3DGenPipeline"] = {
                    "status": "Available",
                    "device": device,
                    "details": f"Pipeline loaded on {device}"
                }
            except Exception as e:
                status.components["Hi3DGenPipeline"] = {
                    "status": "Available",
                    "device": "Unknown",
                    "details": f"Pipeline available, device unknown: {str(e)}"
                }
                # Don't add this as a warning since it's not critical
        else:
            status.components["Hi3DGenPipeline"] = {
                "status": "Missing", 
                "details": "Hi3DGen pipeline not loaded"
            }
            status.errors.append("Hi3DGen pipeline not available")
    
    def _check_directories(self, status: SystemStatus, weights_dir: str, tmp_dir: str) -> None:
        """Check directory status"""
        
        directories_to_check = {}
        if weights_dir:
            directories_to_check["weights"] = weights_dir
        if tmp_dir:
            directories_to_check["tmp"] = tmp_dir
            
        for dir_type, dir_path in directories_to_check.items():
            if not dir_path:
                continue
                
            path_obj = Path(dir_path)
            
            if path_obj.exists():
                # Get directory info
                size_bytes = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
                file_count = len(list(path_obj.rglob('*')))
                
                status.directories[dir_type] = {
                    "path": str(path_obj),
                    "exists": True,
                    "readable": os.access(dir_path, os.R_OK),
                    "writable": os.access(dir_path, os.W_OK),
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "file_count": file_count
                }
                
                # Check permissions
                if not os.access(dir_path, os.R_OK):
                    status.warnings.append(f"{dir_type} directory not readable: {dir_path}")
                if not os.access(dir_path, os.W_OK) and dir_type == "tmp":
                    status.warnings.append(f"{dir_type} directory not writable: {dir_path}")
                    
            else:
                status.directories[dir_type] = {
                    "path": str(path_obj),
                    "exists": False
                }
                status.errors.append(f"{dir_type} directory missing: {dir_path}")
    
    def _check_performance(self, status: SystemStatus) -> None:
        """Check system performance metrics"""
        
        if not PSUTIL_AVAILABLE:
            status.performance = {
                "note": "Performance monitoring unavailable (psutil not installed)"
            }
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage (current directory)
            disk = psutil.disk_usage('.')
            
            status.performance = {
                "cpu_percent": cpu_percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_percent": memory.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 1)
            }
            
            # Performance warnings
            if cpu_percent > 80:
                status.warnings.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 85:
                status.warnings.append(f"High memory usage: {memory.percent}%")
            if (disk.used / disk.total) > 0.9:
                status.warnings.append(f"Low disk space: {100 - (disk.used / disk.total) * 100:.1f}% free")
                
        except Exception as e:
            status.warnings.append(f"Performance check failed: {str(e)}")
    
    def _check_gpu_status(self, status: SystemStatus) -> None:
        """Check GPU status"""
        
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                
                gpu_info = {
                    "available": True,
                    "device_count": gpu_count,
                    "current_device": current_device,
                    "device_name": torch.cuda.get_device_name(current_device),
                    "memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                    "memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                    "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                }
                
                # Calculate memory usage percentage
                if gpu_info["memory_total_gb"] > 0:
                    memory_usage_percent = (gpu_info["memory_allocated_gb"] / gpu_info["memory_total_gb"]) * 100
                    gpu_info["memory_usage_percent"] = round(memory_usage_percent, 1)
                    
                    if memory_usage_percent > 85:
                        status.warnings.append(f"High GPU memory usage: {memory_usage_percent:.1f}%")
                
                # Add status key to gpu_info
                gpu_info["status"] = "Available"
                status.components["GPU"] = gpu_info
                
            else:
                status.components["GPU"] = {
                    "status": "Missing",
                    "available": False,
                    "details": "CUDA not available"
                }
                status.warnings.append("GPU not available - processing will be slower")
                
        except Exception as e:
            status.warnings.append(f"GPU check failed: {str(e)}")
    
    def _check_cancellation_system(self, status: SystemStatus) -> None:
        """Check cancellation system"""
        
        try:
            manager = get_cancellation_manager()
            if manager:
                cancel_status = manager.get_status()
                status.components["CancellationManager"] = {
                    "status": "Available",
                    "current_state": cancel_status.processing_state.name,
                    "is_cancelled": cancel_status.is_cancelled,
                    "details": f"State: {cancel_status.processing_state.name}"
                }
            else:
                status.components["CancellationManager"] = {
                    "status": "Missing",
                    "details": "Cancellation manager not available"
                }
                status.errors.append("Cancellation manager not available")
                
        except Exception as e:
            status.warnings.append(f"Cancellation system check failed: {str(e)}")
    
    def _determine_overall_health(self, status: SystemStatus) -> None:
        """Determine overall system health"""
        
        if status.errors:
            status.overall_health = "Critical"
        elif len(status.warnings) > 3:
            status.overall_health = "Poor"
        elif status.warnings:
            status.overall_health = "Fair"
        else:
            # Check if core components are available
            core_components = ["ProcessingCore", "BatchProcessor", "Hi3DGenPipeline"]
            available_components = sum(1 for comp in core_components 
                                     if comp in status.components and 
                                     status.components[comp].get("status", "Unknown") not in ["Missing", "Error", "Unknown"])
            
            if available_components == len(core_components):
                status.overall_health = "Excellent"
            elif available_components >= len(core_components) - 1:
                status.overall_health = "Good"
            else:
                status.overall_health = "Fair"
    
    def get_status_summary(self, status: SystemStatus) -> str:
        """Get formatted status summary"""
        
        summary = f"🔍 System Status Report - {status.overall_health}\n"
        summary += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"📅 Generated: {status.timestamp}\n"
        summary += f"🎯 Overall Health: {status.overall_health}\n\n"
        
        # Components
        summary += "🔧 Components:\n"
        for comp_name, comp_info in status.components.items():
            # Get status safely with fallback
            comp_status = comp_info.get("status", "Unknown")
            status_icon = "✅" if comp_status not in ["Missing", "Error", "Unknown"] else "❌"
            summary += f"  {status_icon} {comp_name}: {comp_status}\n"
            if "details" in comp_info:
                summary += f"     {comp_info['details']}\n"
        
        # Performance
        if status.performance:
            perf = status.performance
            summary += "\n📊 Performance:\n"
            summary += f"  🖥️  CPU: {perf.get('cpu_percent', 'N/A')}%\n"
            summary += f"  🧠 Memory: {perf.get('memory_used_gb', 'N/A')}/{perf.get('memory_total_gb', 'N/A')} GB ({perf.get('memory_percent', 'N/A')}%)\n"
            summary += f"  💾 Disk: {perf.get('disk_used_gb', 'N/A')}/{perf.get('disk_total_gb', 'N/A')} GB ({perf.get('disk_percent', 'N/A')}%)\n"
        
        # GPU
        if "GPU" in status.components:
            gpu = status.components["GPU"]
            summary += f"  🎮 GPU: {'Available' if gpu.get('available') else 'Not Available'}\n"
            if gpu.get('available'):
                summary += f"     {gpu.get('device_name', 'Unknown')} - {gpu.get('memory_allocated_gb', 0)}/{gpu.get('memory_total_gb', 0)} GB\n"
        
        # Directories
        if status.directories:
            summary += "\n📁 Directories:\n"
            for dir_type, dir_info in status.directories.items():
                status_icon = "✅" if dir_info.get("exists") else "❌"
                summary += f"  {status_icon} {dir_type}: {dir_info['path']}\n"
                if dir_info.get("exists"):
                    summary += f"     {dir_info.get('file_count', 0)} files, {dir_info.get('size_mb', 0)} MB\n"
        
        # Warnings and errors
        if status.warnings:
            summary += f"\n⚠️ Warnings ({len(status.warnings)}):\n"
            for warning in status.warnings:
                summary += f"  • {warning}\n"
        
        if status.errors:
            summary += f"\n❌ Errors ({len(status.errors)}):\n"
            for error in status.errors:
                summary += f"  • {error}\n"
        
        return summary

# Global status checker instance
_status_checker = SystemStatusChecker()

def get_system_status(processing_core=None, batch_processor=None, hi3dgen_pipeline=None, 
                     weights_dir=None, tmp_dir=None) -> SystemStatus:
    """Get current system status"""
    return _status_checker.get_comprehensive_status(
        processing_core=processing_core,
        batch_processor=batch_processor,
        hi3dgen_pipeline=hi3dgen_pipeline,
        weights_dir=weights_dir,
        tmp_dir=tmp_dir
    )

def get_status_summary(processing_core=None, batch_processor=None, hi3dgen_pipeline=None,
                      weights_dir=None, tmp_dir=None) -> str:
    """Get formatted status summary"""
    status = get_system_status(processing_core, batch_processor, hi3dgen_pipeline, weights_dir, tmp_dir)
    return _status_checker.get_status_summary(status)

def print_system_status(processing_core=None, batch_processor=None, hi3dgen_pipeline=None,
                       weights_dir=None, tmp_dir=None) -> None:
    """Print system status to console"""
    summary = get_status_summary(processing_core, batch_processor, hi3dgen_pipeline, weights_dir, tmp_dir)
    print(summary) 