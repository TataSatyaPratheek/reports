"""
Enhanced Utilities Module - Helper functions and classes for tourism RAG chatbot.
"""
import time
import streamlit as st
import psutil
import os
import re
import json
import datetime
import asyncio
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable

class PerformanceMonitor:
    """
    Enhanced monitor and optimizer for application performance.
    Optimized for tourism document processing workloads.
    """
    
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
            
            # Get GPU info if available
            gpu_percent = 0.0
            gpu_memory_percent = 0.0
            
            try:
                # Try to get GPU info from nvidia-smi
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2
                )
                if result.stdout.strip():
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 2:
                        gpu_percent = float(parts[0].strip())
                        gpu_memory_percent = float(parts[1].strip())
            except (ImportError, FileNotFoundError, subprocess.SubprocessError):
                # GPU monitoring not available
                pass
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "available_memory_gb": available_memory_gb,
                "disk_percent": disk_percent,
                "gpu_percent": gpu_percent,
                "gpu_memory_percent": gpu_memory_percent,
                "timestamp": time.time()
            }
        except Exception as e:
            log_error(f"Error getting system resources: {str(e)}")
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "available_memory_gb": 4,
                "disk_percent": 0,
                "gpu_percent": 0,
                "gpu_memory_percent": 0,
                "timestamp": time.time()
            }
    
    @staticmethod
    def recommended_batch_size(available_memory_gb: float, document_type: str = "tourism") -> int:
        """
        Recommend a batch size based on available memory and document type.
        
        Args:
            available_memory_gb: Available system memory in gigabytes
            document_type: Type of documents being processed
            
        Returns:
            Recommended batch size for document processing
        """
        # Tourism documents often have more images and rich content, adjust accordingly
        if document_type.lower() == "tourism":
            if available_memory_gb > 12:
                return 32
            elif available_memory_gb > 8:
                return 24
            elif available_memory_gb > 4:
                return 16
            elif available_memory_gb > 2:
                return 8
            else:
                return 4
        else:
            # Default batch sizes for other document types
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
        """
        Return optimal number of worker threads based on CPU usage.
        
        Args:
            cpu_percent: Current CPU usage percentage
            
        Returns:
            Recommended number of worker threads
        """
        available_cpu = max(1, psutil.cpu_count(logical=False))
        
        if cpu_percent < 30:
            return available_cpu
        elif cpu_percent < 50:
            return max(1, available_cpu - 1)
        elif cpu_percent < 70:
            return max(1, available_cpu // 2)
        else:
            return max(1, available_cpu // 4)
    
    @staticmethod
    def should_use_gpu(available_memory_gb: float, gpu_percent: float) -> bool:
        """
        Determine if GPU should be used for embeddings based on system state.
        
        Args:
            available_memory_gb: Available system memory in gigabytes
            gpu_percent: Current GPU usage percentage
            
        Returns:
            Boolean indicating whether to use GPU
        """
        # Skip if GPU usage is already high
        if gpu_percent > 70:
            return False
            
        # Only use GPU if we have enough system memory
        return available_memory_gb > 4
    
    @staticmethod
    async def adaptive_delay(cpu_percent: float) -> None:
        """
        Introduce adaptive delay based on system load to prevent overloading.
        
        Args:
            cpu_percent: Current CPU usage percentage
        """
        if cpu_percent > 90:
            await asyncio.sleep(0.5)
        elif cpu_percent > 80:
            await asyncio.sleep(0.2)
        elif cpu_percent > 70:
            await asyncio.sleep(0.1)
        elif cpu_percent > 60:
            await asyncio.sleep(0.05)
    
    @staticmethod
    def get_tourism_processing_capacity(available_memory_gb: float) -> Dict[str, int]:
        """
        Estimate how many tourism documents can be processed given available memory.
        
        Args:
            available_memory_gb: Available system memory in gigabytes
            
        Returns:
            Dictionary with estimates for different document types
        """
        return {
            "brochures": int(available_memory_gb * 3),  # Typically image-heavy
            "reports": int(available_memory_gb * 5),    # Text-heavy
            "guides": int(available_memory_gb * 4),     # Mix of text and images
            "reviews": int(available_memory_gb * 8)     # Short text documents
        }

def log_error(error_msg: str):
    """
    Add error message to the session state error log with timestamp.
    
    Args:
        error_msg: Error message to log
    """
    # Initialize error log if not exists
    if "error_log" not in st.session_state:
        st.session_state.error_log = []
        
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.error_log.append(f"[{timestamp}] {error_msg}")
    
    # Keep log size reasonable
    if len(st.session_state.error_log) > 100:
        st.session_state.error_log = st.session_state.error_log[-100:]

def create_directory_if_not_exists(directory_path: str) -> bool:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return True
    except Exception as e:
        log_error(f"Failed to create directory {directory_path}: {str(e)}")
        return False

def extract_date_from_text(text: str) -> Optional[datetime.datetime]:
    """
    Extract date references from text using pattern matching.
    Useful for tourism content with seasonal information.
    
    Args:
        text: Text to extract date from
        
    Returns:
        Extracted date as datetime object or None if not found
    """
    # Common date patterns in tourism documents
    date_patterns = [
        # ISO format: YYYY-MM-DD
        r'(\d{4}-\d{2}-\d{2})',
        # US format: MM/DD/YYYY
        r'(\d{1,2}/\d{1,2}/\d{4})',
        # European format: DD/MM/YYYY
        r'(\d{1,2}\.\d{1,2}\.\d{4})',
        # Written format: Month DD, YYYY
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        # Season references: Summer/Winter YYYY
        r'(Summer|Winter|Spring|Fall|Autumn)\s+\d{4}',
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            date_text = matches[0]
            try:
                # Handle different formats
                if re.match(r'\d{4}-\d{2}-\d{2}', date_text):
                    return datetime.datetime.strptime(date_text, '%Y-%m-%d')
                elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_text):
                    return datetime.datetime.strptime(date_text, '%m/%d/%Y')
                elif re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', date_text):
                    return datetime.datetime.strptime(date_text, '%d.%m.%Y')
                elif re.match(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', date_text, re.IGNORECASE):
                    return datetime.datetime.strptime(date_text, '%B %d, %Y')
                elif re.match(r'(Summer|Winter|Spring|Fall|Autumn)\s+\d{4}', date_text, re.IGNORECASE):
                    year = int(re.search(r'\d{4}', date_text).group())
                    season = re.search(r'(Summer|Winter|Spring|Fall|Autumn)', date_text, re.IGNORECASE).group().lower()
                    
                    # Map seasons to months
                    if season == 'winter':
                        return datetime.datetime(year, 1, 15)  # Mid-winter
                    elif season == 'spring':
                        return datetime.datetime(year, 4, 15)  # Mid-spring
                    elif season == 'summer':
                        return datetime.datetime(year, 7, 15)  # Mid-summer
                    elif season in ('fall', 'autumn'):
                        return datetime.datetime(year, 10, 15)  # Mid-fall
            except ValueError:
                continue
    
    return None

def calculate_token_count(text: str) -> int:
    """
    Estimate token count for a text string (for LLM context management).
    
    Args:
        text: Text to estimate token count for
        
    Returns:
        Estimated token count
    """
    # Simple estimation based on word count
    # Note: This is an approximation, actual tokenization depends on the specific model
    words = text.split()
    # Add 30% overhead for tokenization differences
    return int(len(words) * 1.3)

def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amounts for tourism content display.
    
    Args:
        amount: Amount to format
        currency: Currency code (e.g., USD, EUR, GBP)
        
    Returns:
        Formatted currency string
    """
    currency_symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$",
        "CNY": "¥",
        "INR": "₹",
        "THB": "฿",
        "MXN": "Mex$"
    }
    
    symbol = currency_symbols.get(currency, currency)
    
    if currency in ["JPY", "KRW"]:
        # No decimal places for yen and won
        return f"{symbol}{int(amount):,}"
    else:
        # Two decimal places for most currencies
        return f"{symbol}{amount:,.2f}"

def parse_tourism_ratings(text: str) -> Dict[str, float]:
    """
    Extract tourism ratings from text (e.g., hotel stars, attraction ratings).
    
    Args:
        text: Text to extract ratings from
        
    Returns:
        Dictionary of rating categories and values
    """
    ratings = {}
    
    # Common rating patterns in tourism documents
    patterns = [
        (r'(\d+)[-\s]star', "stars"),
        (r'rated\s+(\d+\.?\d*)\s+out of\s+(\d+)', "rating"),
        (r'(\d+\.?\d*)/(\d+)', "score"),
        (r'(\d+\.?\d*)%\s+satisfaction', "satisfaction"),
    ]
    
    for pattern, category in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                if category == "stars":
                    ratings["stars"] = int(matches[0])
                elif category == "rating":
                    value = float(matches[0][0])
                    scale = float(matches[0][1])
                    ratings["rating"] = (value, scale)
                elif category == "score":
                    value = float(matches[0][0])
                    scale = float(matches[0][1])
                    ratings["score"] = (value, scale)
                elif category == "satisfaction":
                    ratings["satisfaction"] = float(matches[0]) / 100.0
            except (ValueError, IndexError):
                continue
    
    return ratings

def extract_tourism_metrics_from_text(text: str) -> Dict[str, Any]:
    """
    Extract tourism-specific metrics from text.
    
    Args:
        text: Text to extract metrics from
        
    Returns:
        Dictionary of extracted metrics
    """
    metrics = {}
    
    # Extract visitor numbers
    visitor_pattern = r'(\d+[,.]?\d*)\s*(million|thousand|hundred|billion)?\s*(visitors|tourists|travelers|guests)'
    visitor_matches = re.findall(visitor_pattern, text, re.IGNORECASE)
    
    if visitor_matches:
        try:
            value = float(visitor_matches[0][0].replace(',', ''))
            multiplier = visitor_matches[0][1].lower()
            
            if multiplier == 'billion':
                value *= 1_000_000_000
            elif multiplier == 'million':
                value *= 1_000_000
            elif multiplier == 'thousand':
                value *= 1_000
            elif multiplier == 'hundred':
                value *= 100
                
            metrics['visitors'] = int(value)
        except (ValueError, IndexError):
            pass
    
    # Extract revenue figures
    revenue_pattern = r'(\$|€|£|¥)(\d+[,.]?\d*)\s*(million|thousand|hundred|billion)?\s*(revenue|income|earnings|spend)'
    revenue_matches = re.findall(revenue_pattern, text, re.IGNORECASE)
    
    if revenue_matches:
        try:
            currency = revenue_matches[0][0]
            value = float(revenue_matches[0][1].replace(',', ''))
            multiplier = revenue_matches[0][2].lower()
            
            if multiplier == 'billion':
                value *= 1_000_000_000
            elif multiplier == 'million':
                value *= 1_000_000
            elif multiplier == 'thousand':
                value *= 1_000
            elif multiplier == 'hundred':
                value *= 100
                
            metrics['revenue'] = {
                'value': value,
                'currency': currency
            }
        except (ValueError, IndexError):
            pass
    
    # Extract growth percentages
    growth_pattern = r'(\d+\.?\d*)%\s*(growth|increase|decrease|decline)'
    growth_matches = re.findall(growth_pattern, text, re.IGNORECASE)
    
    if growth_matches:
        try:
            value = float(growth_matches[0][0])
            direction = growth_matches[0][1].lower()
            
            if 'decrease' in direction or 'decline' in direction:
                value = -value
                
            metrics['growth_percent'] = value
        except (ValueError, IndexError):
            pass
    
    # Extract average stay duration
    stay_pattern = r'average\s+stay\s+of\s+(\d+\.?\d*)\s*(days|nights|weeks)'
    stay_matches = re.findall(stay_pattern, text, re.IGNORECASE)
    
    if stay_matches:
        try:
            value = float(stay_matches[0][0])
            unit = stay_matches[0][1].lower()
            
            if unit == 'weeks':
                value *= 7
                
            metrics['average_stay_days'] = value
        except (ValueError, IndexError):
            pass
    
    return metrics

class TourismLogger:
    """Extended logging capability specifically for tourism application."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize tourism logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        self.create_log_dir()
        
    def create_log_dir(self):
        """Create log directory if it doesn't exist."""
        create_directory_if_not_exists(self.log_dir)
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message to the appropriate log file.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # Also send to session state error log for UI display
        if level in ["WARNING", "ERROR"]:
            log_error(message)
        
        # Determine log file based on date
        today = time.strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"tourism_{today}.log")
        
        try:
            with open(log_file, "a") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            # Fallback to error log if file writing fails
            log_error(f"Failed to write to log file: {str(e)}")
    
    def info(self, message: str):
        """Log an info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Log a warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log an error message."""
        self.log(message, "ERROR")
    
    def debug(self, message: str):
        """Log a debug message."""
        self.log(message, "DEBUG")

# Initialize global tourism logger
tourism_logger = TourismLogger()