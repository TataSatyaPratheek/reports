# Optimized embedding service module

import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import gc
from typing import Optional, List
from modules.utils import log_error

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def clear_memory():
    """Helper function to clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load embedding model with minimal configuration."""
    try:
        clear_memory()
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        model = SentenceTransformer(model_name, device=device)
        
        log_error(f"Successfully loaded {model_name} on {device}")
        
        # Clear memory after loading
        clear_memory()
        
        return model
    
    except Exception as e:
        log_error(f"Failed to load embedding model: {str(e)}")
        return None

def get_embedding_dimensions(embedding_model) -> int:
    """Get the dimensions of the embedding model."""
    try:
        # Get dimensions by testing with a sample text
        test_text = "test"
        test_embedding = embedding_model.encode([test_text])
        
        if hasattr(test_embedding, 'shape'):
            return test_embedding.shape[1] if len(test_embedding.shape) > 1 else len(test_embedding[0])
        else:
            return len(test_embedding[0])
    
    except Exception as e:
        log_error(f"Error getting embedding dimensions: {str(e)}")
        return 384  # Default dimension