"""
System setup module - Handles dependency checking and system initialization.
"""
import subprocess
import sys
import streamlit as st
import pkg_resources  # Used for version checking
import os
from modules.utils import log_error # Ensure log_error is imported

# Define the path to the requirements file relative to this script's location
REQUIREMENTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')

def parse_requirements(filepath: str) -> dict:
    """Parses a requirements.txt file."""
    requirements = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('==')
                    if len(parts) == 2:
                        requirements[parts[0]] = parts[1]
    except FileNotFoundError:
        err_msg = f"Error: requirements.txt not found at {filepath}"
        st.error(err_msg)
        log_error(err_msg) # Log the error
    except IOError as e: # More specific for file reading issues
        err_msg = f"Error reading requirements file {filepath}: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
    except Exception as e: # Catch other unexpected errors during parsing
        err_msg = f"Unexpected error parsing requirements file: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
    return requirements

def ensure_dependencies() -> list:
    """
    Check if required Python packages match versions in requirements.txt.
    Returns a list of tuples: (package, required_version, installed_version or 'Missing').
    """
    required_packages = parse_requirements(REQUIREMENTS_FILE)
    if not required_packages: # If parsing failed, return empty
        return []

    mismatched_packages = []
    for package, required_version in required_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            # Use parse_version for robust comparison
            if pkg_resources.parse_version(installed_version) != pkg_resources.parse_version(required_version):
                mismatched_packages.append((package, required_version, installed_version))
        except pkg_resources.DistributionNotFound:
            mismatched_packages.append((package, required_version, "Missing"))
        except Exception as e: # Catch other potential errors during version check
            warn_msg = f"Could not verify version for {package}: {str(e)}"
            st.warning(warn_msg)
            log_error(warn_msg) # Log the warning as an error for tracking
            mismatched_packages.append((package, required_version, "Unknown Error"))

    # Check for spaCy model separately
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except ImportError:
         # Spacy itself might be missing if check failed above
         mismatched_packages.append(("spacy", "Required", "Missing")) # Add spacy if import fails
         mismatched_packages.append(("en_core_web_sm", "Latest", "Missing (SpaCy missing)"))
    except IOError: # Specific error if model data not found
        mismatched_packages.append(("en_core_web_sm", "Latest", "Missing"))
    except Exception as e: # Other spacy errors
         warn_msg = f"Error checking spaCy model 'en_core_web_sm': {str(e)}"
         st.warning(warn_msg)
         log_error(warn_msg)
         mismatched_packages.append(("en_core_web_sm", "Latest", "Check Error"))

    return mismatched_packages

def install_package(package_spec: str) -> bool:
    """
    Install a Python package using pip, specifying the version.
    package_spec should be like 'package==version' or just 'package' for spaCy model.
    """
    package_name = package_spec.split('==')[0]
    st.info(f"Attempting to install/download {package_name}...")
    try:
        if package_name == "en_core_web_sm":
            command = [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        else:
            command = [sys.executable, "-m", "pip", "install", package_spec]

        # Use timeout for pip/spacy download
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600) # 10 min timeout

        st.success(f"Successfully installed/downloaded {package_name}.")
        # Optional: Log success output if needed for debugging
        # print(f"Install output for {package_name}:\n{result.stdout}")
        # if result.stderr: print(f"Install stderr for {package_name}:\n{result.stderr}")
        return True

    except subprocess.CalledProcessError as e:
        err_msg = f"Failed to install {package_name}."
        st.error(err_msg)
        st.error(f"Command: {' '.join(e.cmd)}")
        st.error(f"Return Code: {e.returncode}")
        # Display stderr first as it usually contains the core error message
        error_output = f"Error Output:\n{e.stderr}\n{e.stdout}".strip()
        st.error(error_output)
        log_error(f"{err_msg} Command: {' '.join(e.cmd)}. Output: {e.stderr}") # Log concise error
        return False
    except subprocess.TimeoutExpired as e:
        err_msg = f"Timeout occurred while trying to install/download {package_name} after {e.timeout} seconds."
        st.error(err_msg)
        log_error(err_msg)
        return False
    except Exception as e: # Catch-all for other unexpected errors
        err_msg = f"An unexpected error occurred during installation of {package_name}: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        return False


def setup_ollama(install: bool = False) -> bool:
    """
    Check if Ollama is installed and install it if requested.
    Returns: True if Ollama is available, False otherwise
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
    except Exception as e: # Catch other errors during check
         status_msg = f"Error checking for Ollama: {str(e)}"
         if not install: st.warning(status_msg)
         log_error(status_msg) # Log unexpected check errors

    # If we reached here, Ollama is not ready. Proceed with install logic if allowed.
    if not install:
        return False # Not ready and install not allowed

    st.info(f"{status_msg} Attempting installation (requires permissions)...")
    try:
        install_command_str = "" # For logging
        if sys.platform == 'win32':
            st.warning("Automated Ollama installation on Windows is experimental. Manual installation recommended: https://ollama.com/download")
            st.error("Automated Windows installation not implemented. Please install Ollama manually.")
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

DEFAULT_MODEL_NAME = "llama3.2:latest"

def refresh_available_models() -> list:
    """
    Check available Ollama models and return them. Ensures default is present.
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
        return [DEFAULT_MODEL_NAME] # Return default only
    except subprocess.CalledProcessError as e:
        warn_msg = f"Error running 'ollama list': {e.stderr}"
        st.warning(warn_msg)
        log_error(warn_msg) # Log the error
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

    # Ensure the default model is always in the list for selection
    if DEFAULT_MODEL_NAME not in models:
        models.insert(0, DEFAULT_MODEL_NAME)

    return models


def download_model(model_name: str) -> tuple:
    """
    Download an Ollama model if not already available.
    Returns: (success: bool, message: str)
    """
    available_models = refresh_available_models()
    # Check if model is genuinely available (covers cases where refresh failed but returned default)
    is_available = model_name in available_models and (model_name != DEFAULT_MODEL_NAME or len(available_models) > 1 or "command not found" not in st.session_state.get("last_ollama_list_warning", ""))

    if is_available:
         return True, f"Model '{model_name}' is already available or listed."

    st.info(f"Attempting to download model '{model_name}'...")
    try:
        with st.spinner(f"Downloading model '{model_name}'. This can take several minutes..."):
            # Increased timeout for model download
            result = subprocess.run(['ollama', 'pull', model_name],
                                 capture_output=True, text=True, check=True, timeout=1800) # 30 min timeout

        st.success(f"Model '{model_name}' download command completed.")
        if result.stderr:
            st.warning(f"Ollama pull stderr for {model_name}:\n{result.stderr}") # Show stderr as warning

        # Verify by refreshing the list again
        final_models = refresh_available_models()
        if model_name in final_models:
            return True, f"Model '{model_name}' downloaded and verified successfully."
        else:
             # This case might happen if pull succeeds but list fails right after
             warn_msg = f"Model '{model_name}' download command finished, but couldn't verify in list immediately."
             st.warning(warn_msg)
             log_error(warn_msg)
             return True, warn_msg # Return True as download likely succeeded

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
