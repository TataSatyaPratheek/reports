"""
Vector Store Module - Handles ChromaDB initialization and operations (Separated Cache).
Caches only the client instance. Initialization ensures collection exists using the cached client.
Reset uses rename workaround.
"""
import os
import shutil
import time
import streamlit as st
import chromadb
from chromadb.config import Settings # Import Settings
from chromadb.errors import InvalidDimensionException
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from streamlit.delta_generator import DeltaGenerator
from modules.utils import log_error, create_directory_if_not_exists

# --- Constants ---
VECTOR_DB_PATH = "chroma_vector_db"
COLLECTION_NAME = "report_vectors"
DEFAULT_METADATA = {"hnsw:space": "cosine"}

# --- Client Management (Cached Client Only) ---
@st.cache_resource(show_spinner="Initializing Vector Database Client...")
def get_chroma_client() -> Optional[chromadb.Client]:
    """
    Gets a cached persistent ChromaDB client instance using Streamlit's cache.
    Ensures the base directory exists and uses explicit Settings.
    Returns the client instance or None on failure.
    """
    func_name = "get_chroma_client"
    log_error(f"{func_name}: Attempting to get or create client instance...")
    try:
        # 1. Ensure base directory exists
        if not create_directory_if_not_exists(VECTOR_DB_PATH):
            st.error(f"Fatal Error: Could not create/access ChromaDB directory: {VECTOR_DB_PATH}")
            log_error(f"{func_name}: Failed - Directory creation/access failed.")
            return None

        # 2. Define explicit settings
        # Note: is_persistent=True is the default for PersistentClient but explicit is fine.
        settings = Settings(
            persist_directory=VECTOR_DB_PATH,
            is_persistent=True,
            # Add other settings if needed, e.g., anonymized_telemetry=False
            anonymized_telemetry=False
        )
        log_error(f"{func_name}: Using settings: {settings}")

        # 3. Create the client instance (cached by Streamlit)
        client = chromadb.PersistentClient(settings=settings)
        log_error(f"{func_name}: chromadb.PersistentClient created successfully (or retrieved from cache).")

        # 4. Minimal verification (optional, can be removed if still causing issues)
        # Let's try listing collections again as it's less likely to fail than heartbeat
        client.list_collections()
        log_error(f"{func_name}: Client verified via list_collections().")

        return client

    except Exception as e:
        # Catch errors during PersistentClient() instantiation or list_collections
        err_msg = f"Failed to initialize/verify ChromaDB client: {str(e)}"
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        return None

# --- Initialization Step (Uses Cached Client, Creates Collection) ---
def initialize_vector_db() -> bool:
    """
    Ensures the vector database collection exists using the cached client.
    Should be called once during system setup.
    Returns True on success, False on failure.
    """
    func_name = "initialize_vector_db"
    log_error(f"{func_name}: Called.")
    client = get_chroma_client() # Get cached client
    if not client:
        log_error(f"{func_name}: Failed - Could not get Chroma client.")
        # Error already shown by get_chroma_client
        return False

    try:
        # Ensure the collection exists using the retrieved client
        log_error(f"{func_name}: Calling client.get_or_create_collection('{COLLECTION_NAME}')...")
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=DEFAULT_METADATA
        )
        log_error(f"{func_name}: Collection '{COLLECTION_NAME}' ensured successfully.")
        return True

    except Exception as e:
        # Catch ANY exception during get_or_create_collection
        err_msg = f"Failed to create/ensure ChromaDB collection '{COLLECTION_NAME}': {str(e)}"
        # This is where the "tenant" error would likely manifest if it happens at this stage
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        return False

# --- Reset Step (Unchanged - Rename + Cache Clear) ---
def reset_vector_db() -> tuple:
    """
    Resets the vector database: RENAMES old directory, creates new one, clears ALL Streamlit caches.
    Returns: (success: bool, message: str)
    """
    func_name = "reset_vector_db"
    log_error(f"{func_name}: Called.")
    backup_dir_name = None
    try:
        if os.path.exists(VECTOR_DB_PATH):
            try:
                timestamp = int(time.time())
                backup_dir_name = f"{VECTOR_DB_PATH}_backup_{timestamp}"
                log_error(f"{func_name}: Renaming '{VECTOR_DB_PATH}' to '{backup_dir_name}'...")
                os.rename(VECTOR_DB_PATH, backup_dir_name)
                time.sleep(0.1)
                log_error(f"{func_name}: Directory renamed successfully.")
            except OSError as rn_err:
                err_msg = f"Failed to rename existing directory '{VECTOR_DB_PATH}': {str(rn_err)}"
                st.error(err_msg); log_error(f"{func_name}: Failed - {err_msg}"); return False, err_msg
        log_error(f"{func_name}: Creating fresh directory: {VECTOR_DB_PATH}")
        if not create_directory_if_not_exists(VECTOR_DB_PATH):
            err_msg = f"Failed to create fresh directory '{VECTOR_DB_PATH}' after renaming old one."
            st.error(err_msg); log_error(f"{func_name}: Failed - {err_msg}"); return False, err_msg
        time.sleep(0.1)
        log_error(f"{func_name}: Fresh directory created.")
        st.cache_resource.clear()
        log_error(f"{func_name}: Cleared st.cache_resource.")
        success_msg = "Vector database reset (old data backed up). Please re-initialize the system."
        if backup_dir_name: success_msg += f" Backup: {backup_dir_name}"
        log_error(f"{func_name}: {success_msg}")
        return True, success_msg
    except Exception as e:
        err_msg = f"Unexpected error during vector database reset: {str(e)}"
        st.error(err_msg); log_error(f"{func_name}: Failed - {err_msg}")
        try: st.cache_resource.clear()
        except Exception as clear_err: log_error(f"{func_name}: Error during cache clear on reset failure: {clear_err}")
        return False, err_msg

# --- Runtime Collection Retrieval (Uses Cached Client, Gets Collection) ---
def get_chroma_collection() -> Optional[chromadb.Collection]:
    """
    Gets the existing ChromaDB collection using the cached client. Fails if not found.
    Returns the collection object or None if retrieval fails.
    """
    func_name = "get_chroma_collection"
    client = get_chroma_client() # Get cached client
    if not client:
        log_error(f"{func_name}: Failed - Client not available.")
        return None
    try:
        # Use get_collection - expects collection to exist post-initialization
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        # Catch ANY exception (e.g., collection doesn't exist, client disconnected)
        err_msg = f"Failed to get ChromaDB collection '{COLLECTION_NAME}': {str(e)}"
        st.error(err_msg + " System may need re-initialization.")
        log_error(f"{func_name}: Failed - {err_msg}")
        return None

# --- Data Addition (Unchanged) ---
def add_chunks_to_collection(
    chunks: List[str],
    embedding_model,
    collection,
    status: Optional[DeltaGenerator] = None
) -> bool:
    """ Embeds chunks and adds them to the provided ChromaDB collection... """
    func_name = "add_chunks_to_collection"
    if not chunks: return True
    if not embedding_model: err_msg = "Cannot add chunks: Embedding model is missing."; st.error(err_msg); log_error(f"{func_name}: Failed - {err_msg}"); return False
    if not collection: err_msg = "Cannot add chunks: DB collection object is missing."; st.error(err_msg); log_error(f"{func_name}: Failed - {err_msg}"); return False

    def _update_status(label: str):
        if status: status.update(label=label)

    try:
        batch_size = 64
        total_chunks = len(chunks); total_batches = (total_chunks + batch_size - 1) // batch_size
        _update_status(f"Embedding and storing {total_chunks} chunks...")
        log_error(f"{func_name}: Starting add process for {total_chunks} chunks in {total_batches} batches.")
        for i, batch_start in enumerate(range(0, total_chunks, batch_size)):
            batch_end = min(batch_start + batch_size, total_chunks)
            current_batch_texts = chunks[batch_start:batch_end]; num_in_batch = len(current_batch_texts)
            _update_status(f"Processing batch {i + 1}/{total_batches} ({num_in_batch} chunks)...")
            try: embeddings = embedding_model.encode(current_batch_texts, convert_to_numpy=True).tolist()
            except Exception as embed_err: err_msg = f"Error during embedding batch {i+1}: {str(embed_err)}"; st.error(err_msg); log_error(f"{func_name}: Failed - {err_msg}"); return False
            timestamp_ms = int(time.time() * 1000); ids = [f"chunk_{timestamp_ms}_{batch_start + j}" for j in range(num_in_batch)]
            metadatas = [{"text": text} for text in current_batch_texts]
            try: collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            except InvalidDimensionException as ide: err_msg = f"ChromaDB dimension error adding batch {i+1}: {str(ide)}."; st.error(err_msg); log_error(f"{func_name}: Failed - {err_msg}"); return False
            except Exception as add_err: err_msg = f"Error adding batch {i+1} to ChromaDB: {str(add_err)}"; st.error(err_msg); log_error(f"{func_name}: Failed - {err_msg}"); return False
        final_msg = f"Stored {total_chunks} chunks successfully."
        _update_status(final_msg); log_error(f"{func_name}: {final_msg}")
        return True
    except Exception as e:
        err_msg = f"Unexpected error during chunk storage loop: {str(e)}"
        st.error(err_msg); log_error(f"{func_name}: Failed - {err_msg}")
        if status: status.update(label=err_msg, state="error")
        return False
