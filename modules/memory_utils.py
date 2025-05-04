# modules/memory_utils.py
"""
Memory monitoring and optimization utilities for the Tourism RAG system.
"""
import psutil
import torch
import gc
import time
import streamlit as st
from typing import Dict, Any, Optional
from modules.utils import log_error

class MemoryMonitor:
    """System-wide memory monitoring service."""
    
    def __init__(self, warning_threshold=0.8, critical_threshold=0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.last_check = 0
        self.check_interval = 10  # seconds
        
    def check(self, force=False):
        """Check memory status and take action if needed."""
        current_time = time.time()
        if not force and current_time - self.last_check < self.check_interval:
            return
            
        self.last_check = current_time
        
        # Check RAM usage
        ram_usage = psutil.virtual_memory().percent / 100.0
        
        # Check GPU usage if available
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.mem_get_info()
            gpu_usage = 1.0 - (gpu_memory[0] / gpu_memory[1])
        
        # Take action based on thresholds
        if ram_usage > self.critical_threshold or (gpu_usage and gpu_usage > self.critical_threshold):
            self._critical_action()
        elif ram_usage > self.warning_threshold or (gpu_usage and gpu_usage > self.warning_threshold):
            self._warning_action()
            
    def _warning_action(self):
        """Actions to take on warning level."""
        log_error("Memory warning threshold reached. Running garbage collection.")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _critical_action(self):
        """Actions to take on critical level."""
        log_error("Memory critical threshold reached. Aggressive cleanup.")
        # Clear all caches
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    return psutil.virtual_memory().available / (1024 * 1024)

def get_available_gpu_memory_mb() -> float:
    """Get available GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0
    
    gpu_memory = torch.cuda.mem_get_info()
    return gpu_memory[0] / (1024 * 1024)

def optimize_tensor_precision(model, precision="mixed"):
    """Optimize model precision based on available resources."""
    if precision == "mixed" and torch.cuda.is_available():
        try:
            import torch.cuda.amp as amp
            return amp.autocast(), model
        except ImportError:
            return None, model
    elif precision == "fp16" and torch.cuda.is_available():
        try:
            model = model.half()
            return None, model
        except Exception as e:
            log_error(f"Failed to convert model to fp16: {str(e)}")
            return None, model
    return None, model

def process_with_progress(items, processor_func, batch_size=10, progress_bar=None, memory_monitor=None):
    """Process items in batches with progress tracking and memory optimization."""
    results = []
    total = len(items)
    
    for i in range(0, total, batch_size):
        batch = items[i:min(i+batch_size, total)]
        batch_results = processor_func(batch)
        results.extend(batch_results)
        
        # Update progress
        if progress_bar:
            progress_bar.progress(min(1.0, (i + len(batch)) / total))
            
        # Check memory and cleanup if needed
        if memory_monitor:
            memory_monitor.check()
            
    return results

def get_gpu_memory_info() -> Dict[str, float]:
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        return {"available": False, "total_mb": 0, "free_mb": 0, "used_mb": 0}
    
    try:
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory / (1024 * 1024)  # Convert to MB
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 * 1024)
        free_memory = total_memory - reserved_memory
        
        return {
            "available": True,
            "device_name": gpu_properties.name,
            "total_mb": total_memory,
            "free_mb": free_memory,
            "used_mb": allocated_memory,
            "reserved_mb": reserved_memory
        }
    except Exception as e:
        log_error(f"Error getting GPU memory info: {str(e)}")
        return {"available": False, "total_mb": 0, "free_mb": 0, "used_mb": 0}

# Global memory monitor instance
memory_monitor = MemoryMonitor()