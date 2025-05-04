"""
Optimized Utilities Module - Helper functions for tourism RAG chatbot.
"""
import time
import streamlit as st
import os
import re
from typing import Dict, Any, Optional

def log_error(error_msg: str):
    """Log error message to session state."""
    if "error_log" not in st.session_state:
        st.session_state.error_log = []
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.error_log.append(f"[{timestamp}] {error_msg}")
    
    # Keep log size reasonable
    if len(st.session_state.error_log) > 100:
        st.session_state.error_log = st.session_state.error_log[-100:]

def create_directory_if_not_exists(directory_path: str) -> bool:
    """Create a directory if it doesn't exist."""
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return True
    except Exception as e:
        log_error(f"Failed to create directory {directory_path}: {str(e)}")
        return False

def extract_tourism_metrics_from_text(text: str) -> Dict[str, Any]:
    """Extract tourism-specific metrics from text."""
    metrics = {}
    
    # Extract visitor numbers
    visitor_pattern = r'(\d+[,.]?\d*)\s*(million|thousand|hundred|billion)?\s*(visitors|tourists|travelers|guests)'
    visitor_matches = re.findall(visitor_pattern, text, re.IGNORECASE)
    
    if visitor_matches:
        try:
            value = float(visitor_matches[0][0].replace(',', ''))
            multiplier = visitor_matches[0][1].lower() if visitor_matches[0][1] else ""
            
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
            multiplier = revenue_matches[0][2].lower() if revenue_matches[0][2] else ""
            
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
    
    return metrics

class TourismLogger:
    """Extended logging capability specifically for tourism application."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize tourism logger."""
        self.log_dir = log_dir
        create_directory_if_not_exists(log_dir)
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message to the appropriate log file."""
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