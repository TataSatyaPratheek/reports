# modules/embedding_service.py
"""
Optimized embedding service module with adaptive memory management.
"""
import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import gc
from typing import Optional, List, Dict, Any
from modules.utils import log_error
from modules.memory_utils import (
    get_available_memory_mb, 
    get_available_gpu_memory_mb, 
    memory_monitor,
    get_gpu_memory_info
)

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
    """Load embedding model with memory monitoring."""
    try:
        # Clear memory before loading
        clear_memory()
        
        # Use default if not specified
        if model_name is None:
            model_name = DEFAULT_EMBEDDING_MODEL
        
        # Monitor memory before loading
        gpu_info = get_gpu_memory_info()
        initial_memory = gpu_info['free_mb'] if gpu_info['available'] else get_available_memory_mb()
        
        # Determine device with memory awareness
        device = 'cuda' if gpu_info['available'] and gpu_info['free_mb'] > 500 else 'cpu'
        
        # Load model with optimized settings
        model = SentenceTransformer(model_name, device=device)
        
        # Configure for efficiency
        if hasattr(model, 'tokenizer'):
            if hasattr(model.tokenizer, 'use_fast'):
                model.tokenizer.use_fast = True
        
        if hasattr(model, 'max_seq_length'):
            model.max_seq_length = min(model.max_seq_length, 256)
        
        # Monitor memory after loading
        post_memory = get_gpu_memory_info()['free_mb'] if device == 'cuda' else get_available_memory_mb()
        memory_used = initial_memory - post_memory
        log_error(f"Model {model_name} loaded on {device}, using {memory_used:.0f}MB")
        
        # Clear memory after loading
        clear_memory()
        
        return model
    
    except Exception as e:
        log_error(f"Failed to load embedding model: {str(e)}")
        # Try fallback model if loading fails
        if model_name != DEFAULT_EMBEDDING_MODEL:
            log_error("Attempting to load fallback model...")
            return load_embedding_model(DEFAULT_EMBEDDING_MODEL)
        return None

def calculate_optimal_batch_size(texts: List[str], available_memory_mb: float) -> int:
    """Calculate optimal batch size based on text length and available memory."""
    if not texts:
        return 1
    
    avg_length = sum(len(t) for t in texts) / len(texts)
    estimated_tokens = avg_length / 4  # rough estimate
    memory_per_sample = 0.05 * estimated_tokens  # MB per sample (approximate)
    
    # Calculate optimal batch size with safety margin
    optimal_size = max(1, min(64, int(available_memory_mb / memory_per_sample / 2)))
    
    # Adjust based on text count
    if len(texts) < optimal_size:
        return len(texts)
    
    return optimal_size

def encode_texts(embedding_model, texts: List[str], batch_size: int = None, show_progress: bool = True) -> List[List[float]]:
    """
    Encode texts using the embedding model with adaptive batching and memory optimization.
    """
    try:
        if not texts:
            return []
        
        # Calculate optimal batch size if not provided
        if batch_size is None:
            available_memory = get_available_gpu_memory_mb() if torch.cuda.is_available() else get_available_memory_mb()
            batch_size = calculate_optimal_batch_size(texts, available_memory)
            log_error(f"Using adaptive batch size: {batch_size}")
        
        # Process in batches with memory monitoring
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            # Check memory before each batch
            memory_monitor.check()
            
            batch = texts[i:i+batch_size]
            
            try:
                batch_embeddings = embedding_model.encode(
                    batch,
                    batch_size=min(batch_size, len(batch)),
                    show_progress_bar=show_progress and total_batches > 1,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                embeddings.extend(batch_embeddings.tolist())
                
                # Clear cache if memory pressure is high
                gpu_info = get_gpu_memory_info()
                if gpu_info['available'] and gpu_info['used_mb'] / gpu_info['total_mb'] > 0.8:
                    clear_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log_error(f"GPU OOM at batch size {batch_size}, retrying with smaller batch")
                    clear_memory()
                    
                    # Retry with smaller batch size
                    smaller_batch_size = max(1, batch_size // 2)
                    for j in range(0, len(batch), smaller_batch_size):
                        small_batch = batch[j:j+smaller_batch_size]
                        small_embeddings = embedding_model.encode(
                            small_batch,
                            batch_size=1,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        embeddings.extend(small_embeddings.tolist())
                else:
                    raise e
        
        return embeddings
    
    except Exception as e:
        log_error(f"Error encoding texts: {str(e)}")
        
        # Fallback to one-by-one encoding with memory clearing
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
            
            # Clear memory after each text for safety
            if len(embeddings) % 10 == 0:
                clear_memory()
        
        return embeddings

def get_embedding_dimensions(embedding_model) -> int:
    """Get the dimensions of the embedding model."""
    try:
        # Use minimal text to save memory
        test_text = "test"
        
        if hasattr(embedding_model, 'encode'):
            test_embedding = embedding_model.encode(
                [test_text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        else:
            test_embedding = embedding_model([test_text])
        
        if hasattr(test_embedding, 'shape'):
            return test_embedding.shape[1] if len(test_embedding.shape) > 1 else len(test_embedding[0])
        else:
            return len(test_embedding[0])
    
    except Exception as e:
        log_error(f"Error getting embedding dimensions: {str(e)}")
        return 384  # Default dimension