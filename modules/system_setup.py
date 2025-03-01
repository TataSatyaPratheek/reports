"""
System setup module - Handles dependency checking and system initialization.
"""
import subprocess
import sys
import streamlit as st

def ensure_dependencies() -> list:
    """
    Check if all required Python packages are installed.
    Returns a list of missing packages.
    """
    required_packages = [
        "streamlit", "nltk", "spacy", "openparse", "sentence_transformers",
        "numpy", "chromadb", "ollama", "psutil"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    # Check for spaCy model
    if "spacy" not in missing_packages:
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except:
            missing_packages.append("en_core_web_sm")
    
    return missing_packages

def install_package(package_name: str) -> bool:
    """Install a Python package using pip."""
    try:
        if package_name == "en_core_web_sm":
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                        check=True, capture_output=True)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                        check=True, capture_output=True)
        return True
    except Exception as e:
        st.error(f"Failed to install {package_name}: {str(e)}")
        return False

def setup_ollama(install: bool = False) -> bool:
    """
    Check if Ollama is installed and install it if requested.
    Returns: True if Ollama is available, False otherwise
    """
    try:
        # Check if Ollama is already installed
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except:
        pass
    
    # Ollama not installed, attempt to install if allowed
    if not install:
        return False
    
    try:
        if sys.platform == 'win32':  # Windows
            # Download Windows installer and run it
            st.info("Downloading Ollama for Windows...")
            result = subprocess.run(
                ['powershell', '-Command', 
                 "Invoke-WebRequest -Uri 'https://ollama.com/download/ollama-windows-amd64.msi' -OutFile 'ollama-installer.msi'"], 
                capture_output=True, text=True, check=True
            )
            st.info("Installing Ollama...")
            result = subprocess.run(['msiexec', '/i', 'ollama-installer.msi', '/qn'], 
                                  capture_output=True, text=True, check=True)
            
            # Clean up installer
            import os
            os.remove('ollama-installer.msi')
            
        elif sys.platform == 'darwin':  # MacOS
            st.info("Installing Ollama for macOS...")
            result = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", 
                                  shell=True, capture_output=True, text=True, check=True)
        else:  # Linux
            st.info("Installing Ollama for Linux...")
            result = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", 
                                  shell=True, capture_output=True, text=True, check=True)
        
        # Verify installation
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
            
    except Exception as e:
        st.error(f"Failed to install Ollama: {str(e)}")
        return False

def refresh_available_models() -> list:
    """
    Check available Ollama models and return them.
    """
    default_model = "llama3.2:latest"
    models = [default_model]  # Always include default model
    
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse output to get model names
            lines = result.stdout.splitlines()
            for line in lines[1:]:  # Skip header line
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        if model_name not in models:
                            models.append(model_name)
    except Exception as e:
        st.warning(f"Error retrieving Ollama models: {str(e)}")
    
    return models

def download_model(model_name: str) -> tuple:
    """
    Download an Ollama model if not already available.
    Returns: (success, message)
    """
    try:
        # Check if model exists first
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            if model_name in result.stdout:
                return True, f"Model {model_name} is already available"
        
        # Download the model
        with st.spinner(f"Downloading model {model_name}. This may take several minutes..."):
            result = subprocess.run(['ollama', 'pull', model_name], 
                                 capture_output=True, text=True, check=True)
        return True, f"Model {model_name} downloaded successfully"
    except Exception as e:
        return False, f"Failed to download model {model_name}: {str(e)}"