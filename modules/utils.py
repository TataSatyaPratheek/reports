"""
Utilities Module - Helper functions and classes.
"""
import time
import streamlit as st
import psutil
from typing import Dict

def log_error(error_msg: str):
    """
    Add error message to the session state error log.
    """
    # Initialize error log if not exists
    if "error_log" not in st.session_state:
        st.session_state.error_log = []
        
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.error_log.append(f"[{timestamp}] {error_msg}")
    
    # Keep log size reasonable
    if len(st.session_state.error_log) > 100:
        st.session_state.error_log = st.session_state.error_log[-100:]

class PerformanceMonitor:
    """Monitor and optimize application performance."""
    
    @staticmethod
    def get_system_resources() -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory_gb = memory.available / (1024 ** 3)
            disk = psutil.disk_usage('.')
            disk_percent = disk.percent
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "available_memory_gb": available_memory_gb,
                "disk_percent": disk_percent
            }
        except Exception as e:
            log_error(f"Error getting system resources: {str(e)}")
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "available_memory_gb": 4,
                "disk_percent": 0
            }
    
    @staticmethod
    def recommended_batch_size(available_memory_gb: float) -> int:
        """Recommend a batch size based on available memory."""
        if available_memory_gb > 8:
            return 32
        elif available_memory_gb > 4:
            return 16
        elif available_memory_gb > 2:
            return 8
        else:
            return 4
    
    @staticmethod
    def optimize_cpu_usage(cpu_percent: float) -> int:
        """Return optimal number of worker threads based on CPU usage."""
        available_cpu = max(1, psutil.cpu_count(logical=False))
        if cpu_percent < 50:
            return max(1, available_cpu - 1)
        else:
            return max(1, available_cpu // 2)

def create_directory_if_not_exists(directory_path: str) -> bool:
    """
    Create a directory if it doesn't exist.
    Returns True if directory exists or was created successfully.
    """
    import os
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return True
    except Exception as e:
        log_error(f"Failed to create directory {directory_path}: {str(e)}")
        return False