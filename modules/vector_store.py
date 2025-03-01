"""
Vector Store Module - Handles ChromaDB initialization and operations.
"""
import os
import shutil
import time
import streamlit as st
import chromadb
import numpy as np
from typing import List, Dict, Any

# Vector DB path
VECTOR_DB_PATH = "chroma_vector_db"

def initialize_vector_db() -> bool:
    """
    Initialize the vector database and ensure the directory exists.
    Returns success status.
    """
    try:
        # Ensure directory exists
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        # Try to initialize ChromaDB client
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        
        # Create collection if it doesn't exist
        collection = client.get_or_create_collection(name="report_vectors")
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {str(e)}")
        return None

def reset_vector_db() -> tuple:
    """
    Completely reset the vector database by deleting the directory.
    Returns: (success, message)
    """
    try:
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
        # Recreate empty directory
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        # Reinitialize the collection
        st.cache_resource.clear()
        # Try to create a new collection to validate
        initialize_vector_db()
        return True, "Vector database completely reset."
    except Exception as e:
        return False, f"Failed to reset vector database: {str(e)}"

def add_chunks_to_collection(chunks: List[str], embedding_model, collection):
    """
    Embeds chunks and adds them to ChromaDB with optimized batching.
    """
    if not chunks:
        return
    
    try:
        # Determine appropriate batch size
        batch_size = 16  # Default batch size
        if len(chunks) > 1000:
            batch_size = 32
        elif len(chunks) > 100:
            batch_size = 24
        elif len(chunks) < 10:
            batch_size = 4
        
        total_chunks = len(chunks)
        db_progress = st.progress(0)
        
        # Process in batches to optimize memory usage
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            current_batch = chunks[batch_start:batch_end]
            
            # Embed batch
            embeddings = embedding_model.encode(current_batch, convert_to_numpy=True)
            
            # Add to database
            collection.add(
                ids=[f"{os.urandom(4).hex()}" for _ in range(len(current_batch))],
                embeddings=[emb.tolist() for emb in embeddings],
                metadatas=[{"chunk_id": i + batch_start, "text": chunk} for i, chunk in enumerate(current_batch)]
            )
            
            # Update progress
            db_progress.progress(batch_end / total_chunks)
            time.sleep(0.005)  # Small delay to allow UI updates
        
        st.success(f"Stored {len(chunks)} text chunks in the vector database.")
    
    except Exception as e:
        from modules.utils import log_error
        log_error(f"Error adding chunks to database: {str(e)}")
        st.error(f"Error adding chunks to database: {str(e)}")
        st.error(f"Vector DB initialization error: {str(e)}")
        return False

@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    """Get the ChromaDB collection for document chunks."""
    try:
        # Ensure directory exists first
        if not os.path.exists(VECTOR_DB_PATH):
            if not initialize_vector_db():
                return None
                
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        return client.get_or_create_collection(name="report_vectors")
    except Exception as e:
        st.error(f"Failed to get ChromaDB collection: {str(e)}")
        return None