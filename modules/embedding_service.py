# Optimized embedding service module with proper tokenizer usage

import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import gc
from typing import Optional, List, Dict, Any
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
def load_embedding_model(model_name: str = None):
    """Load embedding model with minimal configuration."""
    try:
        clear_memory()
        
        # Use default if not specified
        if model_name is None:
            model_name = DEFAULT_EMBEDDING_MODEL
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model with optimized settings
        model = SentenceTransformer(model_name, device=device)
        
        # Configure the model for efficient operation
        if hasattr(model, 'tokenizer'):
            # Set tokenizer to use fast tokenization
            if hasattr(model.tokenizer, 'use_fast'):
                model.tokenizer.use_fast = True
        
        # Adjust batch processing settings if available
        if hasattr(model, 'max_seq_length'):
            model.max_seq_length = min(model.max_seq_length, 256)  # Limit sequence length for efficiency
        
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
        
        # Use the model's encode method with optimal settings
        if hasattr(embedding_model, 'encode'):
            # Encode with batch_size=1 and show_progress_bar=False for faster processing
            test_embedding = embedding_model.encode(
                [test_text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        else:
            # Fallback to basic encoding
            test_embedding = embedding_model([test_text])
        
        if hasattr(test_embedding, 'shape'):
            return test_embedding.shape[1] if len(test_embedding.shape) > 1 else len(test_embedding[0])
        else:
            return len(test_embedding[0])
    
    except Exception as e:
        log_error(f"Error getting embedding dimensions: {str(e)}")
        return 384  # Default dimension

def encode_texts(embedding_model, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
    """
    Encode texts using the embedding model with proper batching and tokenization.
    
    Args:
        embedding_model: The SentenceTransformer model
        texts: List of texts to encode
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar
    
    Returns:
        List of embeddings
    """
    try:
        # Use the encode method with proper parameters
        embeddings = embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings.tolist()
    
    except Exception as e:
        log_error(f"Error encoding texts: {str(e)}")
        # Fallback to one-by-one encoding
        embeddings = []
        for text in texts:
            try:
                embedding = embedding_model.encode(
                    [text],
                    batch_size=1,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.append(embedding[0].tolist())
            except Exception as e2:
                log_error(f"Error encoding single text: {str(e2)}")
                embeddings.append([0.0] * 384)  # Default zero embedding
        
        return embeddings