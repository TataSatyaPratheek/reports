"""
NLP Models Module - Handles loading and managing NLP models.
"""
import streamlit as st
import nltk
import spacy
from sentence_transformers import SentenceTransformer, util # util might be needed for future ops
import subprocess
import sys
import os # For checking path existence
from modules.utils import log_error # Import log_error

# Define model names as constants
SPACY_MODEL_NAME = "en_core_web_sm"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource(show_spinner="Loading NLTK resources...") # Added spinner text
def load_nltk_resources():
    """Load required NLTK resources (punkt)."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception as e: # Catch potential download errors (network, permissions)
            warn_msg = f"Failed to download NLTK 'punkt' resource: {str(e)}"
            st.warning(warn_msg)
            log_error(warn_msg)

@st.cache_resource(show_spinner=f"Loading SpaCy model ({SPACY_MODEL_NAME})...") # Added spinner text
def load_spacy_model():
    """Load spaCy model with error handling and download attempt."""
    try:
        # Check if already loaded (though cache_resource handles this)
        # if SPACY_MODEL_NAME in spacy.util.get_installed_models():
        return spacy.load(SPACY_MODEL_NAME)
    except OSError: # Model not found locally
        st.warning(f"SpaCy model '{SPACY_MODEL_NAME}' not found locally.")
        st.info(f"Attempting to download '{SPACY_MODEL_NAME}'...")
        try:
            # Use install_package from system_setup for consistency? Or keep direct call?
            # Keeping direct call for now, but ensure permissions are handled in main init flow
            result = subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL_NAME],
                                 check=True, capture_output=True, text=True, timeout=300) # 5 min timeout
            st.success(f"Successfully downloaded '{SPACY_MODEL_NAME}'.")
            # Attempt to load again after download
            return spacy.load(SPACY_MODEL_NAME)
        except subprocess.CalledProcessError as e:
            err_msg = f"Failed to download SpaCy model '{SPACY_MODEL_NAME}'. Error: {e.stderr}"
            st.error(err_msg)
            log_error(err_msg)
            return None
        except subprocess.TimeoutExpired:
            err_msg = f"Timeout downloading SpaCy model '{SPACY_MODEL_NAME}'."
            st.error(err_msg)
            log_error(err_msg)
            return None
        except Exception as e_inner: # Catch other download/load errors
            err_msg = f"Failed during SpaCy model download/load: {str(e_inner)}"
            st.error(err_msg)
            log_error(err_msg)
            return None
    except ImportError:
        # This happens if spacy itself is not installed
        err_msg = "SpaCy library not installed. Cannot load model."
        st.error(err_msg)
        log_error(err_msg)
        return None
    except Exception as e_outer: # Catch other unexpected spacy.load errors
        err_msg = f"Unexpected error loading SpaCy model '{SPACY_MODEL_NAME}': {str(e_outer)}"
        st.error(err_msg)
        log_error(err_msg)
        return None

@st.cache_resource(show_spinner=f"Loading embedding model ({EMBEDDING_MODEL_NAME})...") # Added spinner text
def load_embedding_model():
    """Load sentence transformer model with error handling."""
    try:
        # Check if model files exist locally (optional, SentenceTransformer handles download)
        # model_path = util.snapshot_download(EMBEDDING_MODEL_NAME) # This downloads if needed
        # if not os.path.exists(model_path):
        #    st.info(f"Downloading embedding model '{EMBEDDING_MODEL_NAME}'...")

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        st.success(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.") # Add success message
        return model
    except ImportError:
        err_msg = "Sentence Transformers library not installed. Cannot load embedding model."
        st.error(err_msg)
        log_error(err_msg)
        return None
    except OSError as e: # Often related to network or cache dir issues
         err_msg = f"OS error loading embedding model '{EMBEDDING_MODEL_NAME}': {str(e)}. Check network connection and cache permissions."
         st.error(err_msg)
         log_error(err_msg)
         return None
    except Exception as e: # Catch other potential errors (e.g., corrupted download)
        err_msg = f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        return None

