import torch
import psutil
import logging
from typing import Dict, Optional

class DeviceManager:
    """Manages compute device selection and memory optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = self._detect_best_device()
        self.memory_info = self._get_memory_info()
        
    def _detect_best_device(self) -> str:
        """Detect the best available compute device."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"CUDA available: {gpu_count} GPUs, {gpu_memory:.1f}GB VRAM")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            self.logger.info("Apple MPS available")
        else:
            device = "cpu"
            self.logger.info("Using CPU device")
            
        return device
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        
        info = {
            "total_ram_gb": memory.total / 1e9,
            "available_ram_gb": memory.available / 1e9,
            "ram_usage_percent": memory.percent
        }
        
        if self.device == "cuda":
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
            
        return info
    
    def optimize_for_model(self, model_size: str) -> Dict[str, int]:
        """Optimize batch size and workers based on available resources."""
        available_memory = self.memory_info["available_ram_gb"]
        
        # Adjust based on model size and available memory
        if model_size in ["large", "large-v2", "large-v3"]:
            if available_memory > 16:
                batch_size, num_workers = 4, 4
            elif available_memory > 8:
                batch_size, num_workers = 2, 2
            else:
                batch_size, num_workers = 1, 1
        else:  # smaller models
            if available_memory > 8:
                batch_size, num_workers = 8, 4
            elif available_memory > 4:
                batch_size, num_workers = 4, 2
            else:
                batch_size, num_workers = 2, 1
                
        # Further adjust for GPU
        if self.device == "cuda" and "gpu_memory_gb" in self.memory_info:
            gpu_memory = self.memory_info["gpu_memory_gb"]
            if gpu_memory < 6:  # Low VRAM
                batch_size = max(1, batch_size // 2)
                
        return {"batch_size": batch_size, "num_workers": num_workers}
    
    def clear_memory(self):
        """Clear GPU memory cache if using CUDA."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def get_device_info(self) -> Dict[str, any]:
        """Get comprehensive device information."""
        info = {
            "device": self.device,
            "memory_info": self.memory_info
        }
        
        if self.device == "cuda":
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            
        return info