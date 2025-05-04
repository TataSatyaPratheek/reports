"""
Enhanced System setup module - Handles dependency checking and system initialization.
Optimized for tourism RAG chatbot application.
"""
import subprocess
import os
NLTK_DATA_PATH = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
import nltk
nltk.data.path = [NLTK_DATA_PATH]  # Override all other paths

import sys
import streamlit as st
import pkg_resources
import os
import platform
import json
import asyncio
import re  # Import the regex module
from typing import Dict, List, Tuple, Any, Optional
from modules.nlp_models import load_nltk_resources, load_spacy_model, load_embedding_model
from modules.vector_store import initialize_vector_db

# Use the logger from utils
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
        # Use regex for more robust parsing of package names from pyproject.toml
        package_pattern = re.compile(r'^\s*"([^"]+)"')  # Pattern to find package name in quotes at line start
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
                elif in_dependencies and line:
                    match = package_pattern.match(line)
                    if match:
                        package_name = match.group(1)
                        # We primarily need the name for checking existence/version.
                        # Storing a placeholder version or '*' is sufficient for this check function's purpose.
                        requirements[package_name] = "*" # Store package name, version detail less critical here
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
            # Note: Simple comparison might fail if required_version is '*' or complex specifier.
            # For this check, we mainly care if it's installed. Version mismatch warning is secondary.
            # Let's simplify: just report if missing or error, skip complex version compare for now.
            if required_version != "*" and pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(required_version):
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
    
    Verifies client and server versions match.
    
    Args:
        install: Whether to install Ollama if not found
        
    Returns:
        True if Ollama is available, False otherwise
    """
    func_name = "setup_ollama" # Added for logging consistency
    try:
        # Check if Ollama is already installed and runnable
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, check=True, timeout=10)
        version_info = result.stdout.strip()
        st.info(f"Ollama found: {version_info}")
        
        # Check for version mismatch warning
        if "Warning: client version" in version_info:
            warning_msg = "Ollama client/server version mismatch detected. This may cause issues."
            st.warning(warning_msg)
            log_error(f"{func_name}: {warning_msg}")
            
            # Extract versions for more specific guidance
            server_match = re.search(r'ollama version is (\d+\.\d+\.\d+)', version_info)
            client_match = re.search(r'client version is (\d+\.\d+\.\d+)', version_info)
            
            if server_match and client_match:
                server_version = server_match.group(1)
                client_version = client_match.group(1)
                guidance = f"To fix this issue, consider upgrading Ollama (server is {server_version}, client expects {client_version}) or adjusting the 'ollama' Python package version (`pip install ollama=={server_version}`)."
                st.info(guidance)
            
            # Return true but with warning - system can still function
            return True
        
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
         log_error(f"{func_name}: {status_msg}") # Added func_name

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
            log_error(f"{func_name}: {err_msg}") # Added func_name
            return False

    except subprocess.CalledProcessError as e:
        err_msg = f"Failed to run Ollama install command: {install_command_str}. Error: {e.stderr}"
        st.error(err_msg)
        log_error(f"{func_name}: {err_msg}") # Added func_name
        return False
    except subprocess.TimeoutExpired as e:
        err_msg = f"Timeout running Ollama install command: {install_command_str} after {e.timeout} seconds."
        st.error(err_msg)
        log_error(f"{func_name}: {err_msg}") # Added func_name
        return False
    except Exception as e:
        err_msg = f"An unexpected error occurred during Ollama installation: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        return False

def refresh_available_models() -> List[str]:
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

    # Ensure recommended models are present, add if missing
    final_models = list(models) # Create a mutable copy
    for model in reversed(list(TOURISM_RECOMMENDED_MODELS.keys())): # Add recommended to the front if missing
        if model not in models:
            final_models.insert(0, model)

    return final_models

async def async_refresh_available_models() -> List[str]:
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
    func_name = "download_model" # For logging
    available_models = refresh_available_models()
    is_available = model_name in available_models and (
        model_name != DEFAULT_MODEL_NAME or
        len(available_models) > 1 or
        "command not found" not in st.session_state.get("last_ollama_list_warning", "")
    )

    if is_available and "needs downloading" not in model_name: # Check if it's just a placeholder
         return True, f"Model '{model_name}' is already available or listed."

    st.info(f"Attempting to download model '{model_name}'...")
    
    # Display model recommendation info if available
    if model_name in TOURISM_RECOMMENDED_MODELS:
        info = TOURISM_RECOMMENDED_MODELS[model_name]
        st.info(f"📝 {info['description']}")
        st.info(f"✅ Recommended for: {', '.join(info.get('recommend_for', ['N/A']))}")
    
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
             warn_msg = f"{func_name}: Model '{model_name}' download command finished, but couldn't verify in list immediately."
             st.warning(warn_msg)
             log_error(warn_msg)
             return True, warn_msg # Still return True, as download likely succeeded but list refresh might lag

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

# Use the global logger from utils
from modules.utils import tourism_logger as logger

def initialize_system():
    """Initialize system components for tourism analysis with better error handling."""
    func_name = "initialize_system"
    logger.info(f"Starting tourism system initialization...")
    st.session_state.system_initialized = False
    st.session_state.initialization_status = "In Progress..."
    overall_success = True
    error_messages = []
    initialization_results = {
        "dependencies": False,
        "ollama": False,
        "nltk": False,
        "nlp_model": False,
        "embedding_model": False,
        "vector_db": False
    }

    # Ensure permissions state exists
    if "permissions" not in st.session_state:
        st.session_state.permissions = {
            "allow_package_install": False,
            "allow_ollama_install": False,
        }

    # --- Dependency Check ---
    logger.info("Checking tourism analysis dependencies...")
    mismatched = ensure_dependencies()
    dependencies_ok = True # Assume OK initially for this step
    if mismatched:
        allow_install = st.session_state.permissions.get("allow_package_install", False)
        for pkg, req_v, installed_v in mismatched:
            # Determine package spec for installation command
            if pkg == "en_core_web_sm":
                pkg_spec = "en_core_web_sm" # Special command handled by install_package
            elif req_v != "*" and installed_v != "Missing":
                 # If a specific version is required and we know the installed one (or it's missing)
                 pkg_spec = f"{pkg}=={req_v}"
            else:
                 # If version is '*' or installed version is unknown/missing, just install the package name
                 pkg_spec = pkg

            if allow_install:
                logger.info(f"Attempting install/download for missing/mismatched: {pkg_spec}")
                success = install_package(pkg_spec)
                if not success:
                    dependencies_ok = False
                    error_messages.append(f"Failed to install/download {pkg}.")
            else:
                dependencies_ok = False
                error_messages.append(f"Missing/mismatched dependency: {pkg} (Required: {req_v}, Found: {installed_v}). Installation permission denied.")

    initialization_results["dependencies"] = dependencies_ok
    if not dependencies_ok:
        overall_success = False
        logger.error(f"Tourism dependency check failed. Errors: {' | '.join(error_messages)}")
        # Continue to check other components if possible
    else:
        logger.info("Tourism dependencies OK.")

    # --- Ollama Check ---
    logger.info("Checking Ollama for tourism AI models...")
    allow_ollama_install = st.session_state.permissions.get("allow_ollama_install", False)
    ollama_ready = setup_ollama(install=allow_ollama_install)
    initialization_results["ollama"] = ollama_ready
    if not ollama_ready:
        overall_success = False
        error_messages.append("Tourism AI engine (Ollama) setup failed or not found. Check installation or grant permission.")
        logger.error("Ollama setup failed.")
        # Continue to check other components
    else:
        logger.info("Tourism AI engine OK.")

    # --- Load NLP Resources ---
    logger.info("Loading tourism analysis models...")
    # NLTK (Treat as non-critical for overall success, but log failure)
    nltk_success = load_nltk_resources()
    initialization_results["nltk"] = nltk_success
    if not nltk_success:
        msg = "NLTK resources could not be loaded. Some text processing features might be limited."
        error_messages.append(msg)
        logger.warning(msg)
        # Do not set overall_success to False for NLTK failure

    # SpaCy (Critical)
    nlp_model = load_spacy_model()
    initialization_results["nlp_model"] = bool(nlp_model)
    if not nlp_model:
        overall_success = False
        error_messages.append("Critical: Tourism NLP (SpaCy) model initialization failed.")
        logger.error("Critical: SpaCy model failed to load.")

    # Embedding Model (Critical)
    embedding_model = load_embedding_model()
    initialization_results["embedding_model"] = bool(embedding_model)
    if not embedding_model:
        overall_success = False
        error_messages.append("Critical: Embedding model initialization failed.")
        logger.error("Critical: Embedding model failed to load.")

    if nlp_model and embedding_model:
         logger.info("Core NLP/Embedding models OK.")

    # --- Vector DB Initialization (Critical) ---
    logger.info("Initializing tourism knowledge base...")
    db_success = initialize_vector_db()
    initialization_results["vector_db"] = db_success
    if not db_success:
        overall_success = False
        error_messages.append("Critical: Tourism knowledge base (Vector DB) initialization failed.")
        logger.error("Critical: Tourism knowledge base initialization failed.")
    else:
        logger.info("Tourism knowledge base OK.")

    # --- Refresh Models (Run even if other steps failed, if Ollama is OK) ---
    if ollama_ready:
        logger.info("Checking available tourism AI models...")
        st.session_state.available_models = refresh_available_models()
        logger.info(f"Found {len(st.session_state.get('available_models', []))} tourism AI models.")

    # --- Final Status ---
    st.session_state.initialization_results = initialization_results # Store results regardless of success
    if overall_success:
        st.session_state.initialization_complete = True
        st.session_state.system_initialized = True
        st.session_state.initialization_status = "Completed Successfully"
        logger.info("Tourism Explorer initialized successfully!")
        return True, "Tourism Explorer initialized successfully!"
    else:
        st.session_state.initialization_complete = False # Explicitly set to false on failure
        st.session_state.system_initialized = False
        st.session_state.initialization_status = "Failed"
        final_error_message = "Tourism Explorer initialization failed. Issues found:\n- " + "\n- ".join(error_messages)
        logger.error(final_error_message)
        return False, final_error_message