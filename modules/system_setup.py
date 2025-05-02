"""
Enhanced System setup module - Handles dependency checking and system initialization.
Optimized for tourism RAG chatbot application.
"""
import subprocess
import sys
import streamlit as st
import pkg_resources
import os
import platform
import json
import asyncio
from typing import Dict, List, Tuple, Any, Optional

from modules.utils import log_error, create_directory_if_not_exists

# Define the path to the pyproject.toml file for dependency checking
PYPROJECT_FILE = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')

# Tourism-optimized LLM models recommended for the application
TOURISM_RECOMMENDED_MODELS = {
    "llama3.2:latest": {
        "description": "General purpose model with good knowledge of tourism concepts",
        "strengths": ["General knowledge", "Good context handling", "Multi-language support"],
        "recommend_for": ["General tourism queries", "Multi-language travel content"]
    },
    "llama3.1-8b:latest": {
        "description": "Smaller model for faster responses with decent travel knowledge",
        "strengths": ["Speed", "Lower resource usage"],
        "recommend_for": ["Simple queries", "Resource-constrained systems"]
    },
    "mistral:latest": {
        "description": "Strong reasoning capabilities for complex travel planning",
        "strengths": ["Strong reasoning", "Detail orientation"],
        "recommend_for": ["Complex itinerary analysis", "Detailed travel recommendations"]
    },
    "mixtralja:latest": {
        "description": "Excellent for Japanese tourism content and language queries",
        "strengths": ["Japanese language expertise", "Cultural context"],
        "recommend_for": ["Japanese tourism", "East Asian travel content"]
    },
    "solar:latest": {
        "description": "Well-balanced model with good price/performance for tourism applications",
        "strengths": ["Balance of capabilities", "Moderate resource requirements"],
        "recommend_for": ["Balanced general-purpose travel assistant"]
    }
}

# Default model name
DEFAULT_MODEL_NAME = "llama3.2:latest"

def parse_pyproject_dependencies(filepath: str) -> dict:
    """
    Parse dependencies from a pyproject.toml file.
    
    Args:
        filepath: Path to pyproject.toml file
        
    Returns:
        Dictionary of package names and version constraints
    """
    requirements = {}
    try:
        # Simple TOML parsing for dependencies section
        in_dependencies = False
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check for dependencies section
                if line == "dependencies = [":
                    in_dependencies = True
                    continue
                elif in_dependencies and line == "]":
                    in_dependencies = False
                    continue
                
                # Parse dependency lines
                if in_dependencies and line and line.startswith('"') and "=" in line:
                    # Clean up the line: remove quotes, commas, etc.
                    line = line.strip('",')
                    parts = line.split('=')
                    if len(parts) >= 2:
                        package = parts[0].strip().strip('"')
                        version = parts[1].strip().strip('"')
                        # Extract version constraint
                        if ">=" in version:
                            version = version.split(">=")[1]
                        requirements[package] = version
    except FileNotFoundError:
        err_msg = f"Error: pyproject.toml not found at {filepath}"
        st.error(err_msg)
        log_error(err_msg)
    except Exception as e:
        err_msg = f"Error parsing pyproject.toml: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
    
    return requirements

def check_pdm_installed() -> bool:
    """Check if PDM is installed on the system."""
    try:
        subprocess.run(["pdm", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def ensure_dependencies() -> list:
    """
    Check if required Python packages match versions in pyproject.toml.
    Returns a list of tuples: (package, required_version, installed_version or 'Missing').
    """
    required_packages = parse_pyproject_dependencies(PYPROJECT_FILE)
    if not required_packages:
        # Try to use PDM to check dependencies if installed
        if check_pdm_installed():
            try:
                result = subprocess.run(["pdm", "list", "--json"], capture_output=True, text=True, check=True)
                packages_data = json.loads(result.stdout)
                # PDM is installed, but we'll still return empty to let normal install flow happen
                return []
            except Exception as pdm_err:
                log_error(f"PDM check error: {str(pdm_err)}")
                return []
        return []

    mismatched_packages = []
    for package, required_version in required_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            # Use parse_version for robust comparison
            if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(required_version):
                mismatched_packages.append((package, required_version, installed_version))
        except pkg_resources.DistributionNotFound:
            mismatched_packages.append((package, required_version, "Missing"))
        except Exception as e:
            warn_msg = f"Could not verify version for {package}: {str(e)}"
            st.warning(warn_msg)
            log_error(warn_msg)
            mismatched_packages.append((package, required_version, "Unknown Error"))

    # Check for spaCy model separately
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except ImportError:
         # Spacy itself might be missing if check failed above
         mismatched_packages.append(("spacy", "Required", "Missing"))
         mismatched_packages.append(("en_core_web_sm", "Latest", "Missing (SpaCy missing)"))
    except IOError:
        mismatched_packages.append(("en_core_web_sm", "Latest", "Missing"))
    except Exception as e:
         warn_msg = f"Error checking spaCy model 'en_core_web_sm': {str(e)}"
         st.warning(warn_msg)
         log_error(warn_msg)
         mismatched_packages.append(("en_core_web_sm", "Latest", "Check Error"))

    return mismatched_packages

def install_package(package_spec: str) -> bool:
    """
    Install a Python package using pip or PDM.
    
    Args:
        package_spec: Package specification (e.g., 'package==version')
        
    Returns:
        True if installation succeeded, False otherwise
    """
    package_name = package_spec.split('==')[0]
    st.info(f"Attempting to install/download {package_name}...")
    
    # Try PDM first if installed
    if check_pdm_installed() and package_name != "en_core_web_sm":
        try:
            st.info(f"Installing {package_name} using PDM...")
            command = ["pdm", "add", package_spec]
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
            st.success(f"Successfully installed {package_name} with PDM.")
            return True
        except Exception as pdm_err:
            log_error(f"PDM install failed for {package_name}: {str(pdm_err)}")
            # Fall back to pip
    
    # Use pip as fallback or for spaCy model
    try:
        if package_name == "en_core_web_sm":
            command = [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        else:
            command = [sys.executable, "-m", "pip", "install", package_spec]

        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
        st.success(f"Successfully installed/downloaded {package_name}.")
        return True

    except subprocess.CalledProcessError as e:
        err_msg = f"Failed to install {package_name}."
        st.error(err_msg)
        st.error(f"Command: {' '.join(e.cmd)}")
        st.error(f"Return Code: {e.returncode}")
        error_output = f"Error Output:\n{e.stderr}\n{e.stdout}".strip()
        st.error(error_output)
        log_error(f"{err_msg} Command: {' '.join(e.cmd)}. Output: {e.stderr}")
        return False
    except subprocess.TimeoutExpired as e:
        err_msg = f"Timeout occurred while trying to install/download {package_name} after {e.timeout} seconds."
        st.error(err_msg)
        log_error(err_msg)
        return False
    except Exception as e:
        err_msg = f"An unexpected error occurred during installation of {package_name}: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        return False

def install_tourism_data_dependencies() -> bool:
    """
    Install recommended data packages for tourism analysis.
    
    Returns:
        True if installation succeeded, False otherwise
    """
    # List of recommended packages for tourism data processing
    tourism_packages = [
        "pycountry",          # Country data and codes
        "forex-python",       # Currency exchange rates
        "geopy",              # Geocoding and location data
        "python-dateutil",    # Enhanced date handling for seasonal analysis
        "nltk",               # For additional corpora
    ]
    
    success = True
    for package in tourism_packages:
        try:
            if not install_package(package):
                success = False
                st.warning(f"Optional tourism package {package} couldn't be installed. Some features may be limited.")
        except Exception as e:
            log_error(f"Error installing tourism package {package}: {str(e)}")
            success = False
    
    # Download NLTK corpora useful for tourism text
    try:
        import nltk
        for corpus in ["words", "stopwords", "wordnet", "names"]:
            try:
                nltk.download(corpus, quiet=True)
            except Exception as nltk_err:
                log_error(f"Error downloading NLTK corpus {corpus}: {str(nltk_err)}")
    except Exception as e:
        log_error(f"Error downloading NLTK corpora: {str(e)}")
        success = False
    
    return success

def setup_ollama(install: bool = False) -> bool:
    """
    Check if Ollama is installed and install it if requested.
    
    Args:
        install: Whether to install Ollama if not found
        
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        # Check if Ollama is already installed and runnable
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, check=True, timeout=10)
        st.info(f"Ollama found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
         status_msg = "Ollama command not found."
         if not install: st.warning(status_msg)
    except subprocess.CalledProcessError as e:
         status_msg = f"Ollama command failed: {e.stderr}"
         if not install: st.warning(status_msg)
    except subprocess.TimeoutExpired:
         status_msg = "Timeout checking for Ollama version."
         if not install: st.warning(status_msg)
    except Exception as e:
         status_msg = f"Error checking for Ollama: {str(e)}"
         if not install: st.warning(status_msg)
         log_error(status_msg)

    # If we reached here, Ollama is not ready. Proceed with install logic if allowed.
    if not install:
        return False

    st.info(f"{status_msg} Attempting installation (requires permissions)...")
    try:
        install_command_str = ""
        if sys.platform == 'win32':
            st.warning("Automated Ollama installation on Windows is experimental. Manual installation recommended: https://ollama.com/download")
            
            # Try to use winget for Windows installation
            try:
                # Check if winget is available
                winget_check = subprocess.run(["winget", "--version"], capture_output=True, text=True, check=True, timeout=10)
                
                # If winget is available, try to install Ollama
                st.info("Attempting to install Ollama using winget...")
                install_result = subprocess.run(["winget", "install", "Ollama.Ollama"], capture_output=True, text=True, check=True, timeout=300)
                
                if "successfully installed" in install_result.stdout.lower():
                    st.success("Ollama installed successfully via winget.")
                    return True
                else:
                    st.warning("Winget installation attempt completed, but success could not be verified.")
                    st.error("Please install Ollama manually: https://ollama.com/download")
                    return False
                    
            except (FileNotFoundError, subprocess.CalledProcessError):
                st.error("Winget is not available or installation failed. Please install Ollama manually: https://ollama.com/download")
                return False
                
        elif sys.platform == 'darwin' or sys.platform.startswith('linux'):
            st.info(f"Running Ollama installation script for {sys.platform}...")
            install_command_str = "curl -fsSL https://ollama.com/install.sh | sh"
            # Increased timeout for install script
            result = subprocess.run(install_command_str, shell=True, capture_output=True, text=True, check=True, timeout=600)
            st.info("Ollama installation script executed.")
            if result.stdout: st.text(f"Script output:\n{result.stdout}")
            if result.stderr: st.warning(f"Script stderr:\n{result.stderr}")
        else:
            st.error(f"Unsupported operating system for automated Ollama installation: {sys.platform}")
            return False

        # Verify installation after attempt
        try:
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, check=True, timeout=10)
            st.success(f"Ollama installed successfully: {result.stdout.strip()}")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as post_install_err:
            err_msg = f"Ollama installation script ran, but verification failed: {post_install_err}"
            st.error(err_msg)
            log_error(err_msg)
            return False

    except subprocess.CalledProcessError as e:
        err_msg = f"Failed to run Ollama install command: {install_command_str}. Error: {e.stderr}"
        st.error(err_msg)
        log_error(err_msg)
        return False
    except subprocess.TimeoutExpired as e:
        err_msg = f"Timeout running Ollama install command: {install_command_str} after {e.timeout} seconds."
        st.error(err_msg)
        log_error(err_msg)
        return False
    except Exception as e:
        err_msg = f"An unexpected error occurred during Ollama installation: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        return False

def refresh_available_models() -> list:
    """
    Check available Ollama models and return them with tourism model recommendations.
    
    Returns:
        List of available model names
    """
    models = []
    try:
        # Short timeout for listing models
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True, timeout=30)
        lines = result.stdout.strip().splitlines()
        if len(lines) > 1:
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
    except FileNotFoundError:
        st.warning("Ollama command not found. Cannot list models.")
        return [DEFAULT_MODEL_NAME]
    except subprocess.CalledProcessError as e:
        warn_msg = f"Error running 'ollama list': {e.stderr}"
        st.warning(warn_msg)
        log_error(warn_msg)
        return [DEFAULT_MODEL_NAME]
    except subprocess.TimeoutExpired:
        warn_msg = "Timeout occurred while listing Ollama models."
        st.warning(warn_msg)
        log_error(warn_msg)
        return [DEFAULT_MODEL_NAME]
    except Exception as e:
        warn_msg = f"Unexpected error retrieving Ollama models: {str(e)}"
        st.warning(warn_msg)
        log_error(warn_msg)
        return [DEFAULT_MODEL_NAME]

    # Add recommended tourism models to beginning
    for model in list(TOURISM_RECOMMENDED_MODELS.keys()):
        if model not in models:
            # Add to beginning with note that it needs downloading
            models.insert(0, model)

    return models

async def async_refresh_available_models() -> list:
    """Async version of refresh_available_models."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, refresh_available_models)

def download_model(model_name: str) -> tuple:
    """
    Download an Ollama model if not already available.
    
    Args:
        model_name: Name of model to download
        
    Returns:
        (success: bool, message: str)
    """
    available_models = refresh_available_models()
    is_available = model_name in available_models and (
        model_name != DEFAULT_MODEL_NAME or 
        len(available_models) > 1 or 
        "command not found" not in st.session_state.get("last_ollama_list_warning", "")
    )

    if is_available:
         return True, f"Model '{model_name}' is already available or listed."

    st.info(f"Attempting to download model '{model_name}'...")
    
    # Display model recommendation info if available
    if model_name in TOURISM_RECOMMENDED_MODELS:
        info = TOURISM_RECOMMENDED_MODELS[model_name]
        st.info(f"📝 {info['description']}")
        st.info(f"✅ Recommended for: {', '.join(info['recommend_for'])}")
    
    try:
        with st.spinner(f"Downloading model '{model_name}'. This can take several minutes..."):
            # Increased timeout for model download
            result = subprocess.run(['ollama', 'pull', model_name],
                                 capture_output=True, text=True, check=True, timeout=1800)

        st.success(f"Model '{model_name}' download command completed.")
        if result.stderr:
            st.warning(f"Ollama pull stderr for {model_name}:\n{result.stderr}")

        # Verify by refreshing the list again
        final_models = refresh_available_models()
        if model_name in final_models:
            return True, f"Model '{model_name}' downloaded and verified successfully."
        else:
             warn_msg = f"Model '{model_name}' download command finished, but couldn't verify in list immediately."
             st.warning(warn_msg)
             log_error(warn_msg)
             return True, warn_msg

    except FileNotFoundError:
         err_msg = "Ollama command not found. Cannot download model."
         st.error(err_msg)
         log_error(err_msg)
         return False, err_msg
    except subprocess.TimeoutExpired as e:
        err_msg = f"Failed to download model '{model_name}': Timeout occurred after {e.timeout} seconds."
        st.error(err_msg)
        log_error(err_msg)
        return False, err_msg
    except subprocess.CalledProcessError as e:
        # Provide more specific feedback if possible
        stderr_lower = e.stderr.lower()
        if "manifest for" in stderr_lower and "not found" in stderr_lower:
            error_message = f"Failed to download model '{model_name}': Model not found in Ollama registry."
        elif "connection refused" in stderr_lower:
             error_message = f"Failed to download model '{model_name}': Could not connect to Ollama service. Is it running?"
        elif "pull access denied" in stderr_lower:
             error_message = f"Failed to download model '{model_name}': Access denied. Check permissions or model name."
        else:
             error_message = f"Failed to download model '{model_name}'. Error: {e.stderr}"

        st.error(error_message)
        log_error(error_message)
        return False, error_message
    except Exception as e:
        err_msg = f"An unexpected error occurred while downloading model '{model_name}': {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        return False, err_msg

def get_system_info() -> Dict[str, Any]:
    """
    Get detailed system information for diagnostics.
    
    Returns:
        Dictionary of system information
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "ram_total_gb": 0,
        "disk_total_gb": 0,
        "ollama_version": "Not installed",
        "recommended_models": []
    }
    
    # Get RAM info
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["ram_total_gb"] = round(memory.total / (1024**3), 2)
        info["disk_total_gb"] = round(psutil.disk_usage('/').total / (1024**3), 2)
    except ImportError:
        pass
    
    # Get Ollama version
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, check=True, timeout=5)
        info["ollama_version"] = result.stdout.strip()
    except:
        pass
    
    # Add model recommendations based on system specs
    if info["ram_total_gb"] < 8:
        info["recommended_models"] = ["llama3.1-8b:latest"]
    elif info["ram_total_gb"] < 16:
        info["recommended_models"] = ["llama3.1-8b:latest", "solar:latest"]
    else:
        info["recommended_models"] = ["llama3.2:latest", "mistral:latest"]
    
    return info