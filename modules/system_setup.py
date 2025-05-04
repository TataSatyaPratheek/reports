"""
Optimized System setup module - Handles dependency checking and initialization.
"""
import subprocess
import sys
import streamlit as st
import pkg_resources
import os
from typing import Dict, List, Tuple, Any, Optional

from modules.utils import log_error

# Tourism-optimized LLM models
TOURISM_RECOMMENDED_MODELS = {
    "llama3.2:latest": {
        "description": "General purpose model with good tourism knowledge",
        "strengths": ["General knowledge", "Good context handling", "Multi-language support"],
        "recommend_for": ["General tourism queries", "Multi-language travel content"]
    },
    "mistral:latest": {
        "description": "Strong reasoning for complex travel planning",
        "strengths": ["Strong reasoning", "Detail orientation"],
        "recommend_for": ["Complex itinerary analysis", "Detailed travel recommendations"]
    },
    "solar:latest": {
        "description": "Well-balanced model with good price/performance",
        "strengths": ["Balance of capabilities", "Moderate resource requirements"],
        "recommend_for": ["Balanced general-purpose travel assistant"]
    }
}

# Default model name
DEFAULT_MODEL_NAME = "llama3.2:latest"

def ensure_dependencies() -> list:
    """Check if required Python packages are installed."""
    required_packages = {
        "streamlit": "1.33.0",
        "chromadb": "0.4.24",
        "ollama": "0.1.14",
        "PyMuPDF": "1.24.0",
        "sentence-transformers": "2.7.0",
        "psutil": "5.9.8",
        "openparse": "0.5.6",
        "rank-bm25": "0.2.2",
        "FlagEmbedding": "1.2.10",
        "pandas": "2.1.0",
        "altair": "5.1.0",
        "matplotlib": "3.7.2"  # Added for pandas styling
     }
    
    mismatched_packages = []
    
    for package, required_version in required_packages.items():
        try:
            pkg_resources.get_distribution(package)
        except pkg_resources.DistributionNotFound:
            mismatched_packages.append((package, required_version, "Missing"))
    
    return mismatched_packages

def install_package(package_spec: str) -> bool:
    """Install a Python package using pip."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install {package_spec}: {e.stderr}")
        return False

def setup_ollama(install: bool = False) -> bool:
    """Check if Ollama is installed and install if requested."""
    try:
        # Check if Ollama is installed
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except FileNotFoundError:
        if not install:
            return False
        
        # Try to install Ollama
        try:
            if sys.platform == 'darwin' or sys.platform.startswith('linux'):
                install_command = "curl -fsSL https://ollama.com/install.sh | sh"
                subprocess.run(install_command, shell=True, check=True)
                return True
            else:
                log_error(f"Automated Ollama installation not supported for {sys.platform}")
                return False
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to install Ollama: {str(e)}")
            return False

def refresh_available_models() -> List[str]:
    """Check available Ollama models."""
    models = []
    
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().splitlines()
        if len(lines) > 1:
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
    except:
        pass
    
    # Ensure recommended models are included
    for model in TOURISM_RECOMMENDED_MODELS.keys():
        if model not in models:
            models.append(model)
    
    return models

def download_model(model_name: str) -> tuple:
    """Download an Ollama model."""
    try:
        subprocess.run(
            ['ollama', 'pull', model_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=600
        )
        return True, f"Model '{model_name}' downloaded successfully"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to download model: {e.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Model download timed out"