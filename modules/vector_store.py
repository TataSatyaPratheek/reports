"""
Enhanced Vector Store Module - Handles ChromaDB initialization and operations with modern features.
- Hybrid Retrieval: Vector similarity + BM25 keyword search
- Hierarchical NSW Indexing
- Neural Caching
- Async operation support
- Flash Attention optimization
"""
import os
NLTK_DATA_PATH = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
import nltk
nltk.data.path = [NLTK_DATA_PATH]  # Override all other paths

import os
import shutil
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Awaitable
import asyncio

import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidDimensionException
import numpy as np
from rank_bm25 import BM25Okapi
import torch # Added for CUDA cache management
from cachetools import TTLCache, cached
from FlagEmbedding import FlagReranker  # BGE reranker for superior performance

from modules.utils import log_error, create_directory_if_not_exists

# --- Constants ---
VECTOR_DB_PATH = "chroma_vector_db"
COLLECTION_NAME = "tourism_vectors"
# Enhanced DB metadata including HNSW params for NSW indexing
DEFAULT_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,  # Higher values = better index quality but slower construction
    "hnsw:search_ef": 128,        # Higher values = better recall but slower search
    "hnsw:M": 16,                 # Number of bi-directional links created for each element
}
# Caching configuration
EMBEDDING_CACHE_SIZE = 1000       # Number of text->vector pairs to cache
EMBEDDING_CACHE_TTL = 3600        # Cache TTL in seconds (1 hour)
RERANKER_CACHE_SIZE = 100         # Number of query->reranking results to cache
RERANKER_CACHE_TTL = 600          # Cache TTL in seconds (10 minutes)

# Initialize caches
_embedding_cache = TTLCache(maxsize=EMBEDDING_CACHE_SIZE, ttl=EMBEDDING_CACHE_TTL)
_reranker_cache = TTLCache(maxsize=RERANKER_CACHE_SIZE, ttl=RERANKER_CACHE_TTL)

# --- BM25 Index Store - Used for hybrid search ---
_bm25_index = None

# --- Reranker Instance ---
_reranker = None

def get_reranker():
    """Get or initialize the BGE Reranker."""
    global _reranker
    if _reranker is None:
        try:
            # Clear CUDA cache before loading a potentially large model
            if torch.cuda.is_available():
                log_error("Clearing CUDA cache before loading reranker...")
                torch.cuda.empty_cache()
                log_error("CUDA cache cleared.")

            # Initialize the BGE Reranker
            # Using fp16 for faster inference if GPU is available
            _reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
            log_error("BGE Reranker initialized successfully.")
        except Exception as e:
            log_error(f"Failed to initialize BGE Reranker: {str(e)}")
            # Return None to indicate failure
            return None
    return _reranker

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

        # 2. Define explicit settings with enhanced HNSW configuration
        settings = Settings(
            persist_directory=VECTOR_DB_PATH,
            is_persistent=True,
            anonymized_telemetry=False,
            # Add other settings if needed
        )
        log_error(f"{func_name}: Using settings: {settings}")

        # 3. Create the client instance (cached by Streamlit)
        client = chromadb.PersistentClient(settings=settings)
        log_error(f"{func_name}: chromadb.PersistentClient created successfully (or retrieved from cache).")

        # 4. Minimal verification
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

        # Initialize BM25 index if we have existing documents
        try:
            if collection.count() > 0:
                _initialize_bm25_index(collection)
        except Exception as bm25_err:
            log_error(f"{func_name}: Warning - Failed to initialize BM25 index: {str(bm25_err)}")
            # Don't fail initialization due to BM25 problems - it's an enhancement

        return True

    except Exception as e:
        # Catch ANY exception during get_or_create_collection
        err_msg = f"Failed to create/ensure ChromaDB collection '{COLLECTION_NAME}': {str(e)}"
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        return False

def _initialize_bm25_index(collection):
    """Initialize BM25 index from existing ChromaDB collection."""
    global _bm25_index
    try:
        # Get all documents from collection
        results = collection.get(include=["metadatas"])
        
        if not results or not results.get("metadatas") or len(results["metadatas"]) == 0:
            # No documents in collection, create empty index
            _bm25_index = BM25Okapi([])
            return
            
        # Extract text from metadatas
        texts = []
        for meta in results["metadatas"]:
            if meta and "text" in meta:
                # Tokenize text for BM25 index
                texts.append(meta["text"].split())
        
        if texts:
            _bm25_index = BM25Okapi(texts)
            log_error(f"BM25 index initialized with {len(texts)} documents.")
    except Exception as e:
        log_error(f"Error initializing BM25 index: {str(e)}")
        _bm25_index = None

# --- Reset Step (Enhanced with BM25 cleanup) ---
def reset_vector_db() -> tuple:
    """
    Resets the vector database: RENAMES old directory, creates new one, clears ALL Streamlit caches.
    Returns: (success: bool, message: str)
    """
    func_name = "reset_vector_db"
    log_error(f"{func_name}: Called.")
    backup_dir_name = None
    
    # Reset our caches and global objects
    global _embedding_cache, _reranker_cache, _bm25_index, _reranker
    _embedding_cache.clear()
    _reranker_cache.clear()
    _bm25_index = None
    _reranker = None

    # Explicitly clear CUDA cache during reset
    if torch.cuda.is_available():
        log_error(f"{func_name}: Clearing CUDA cache...")
        torch.cuda.empty_cache()
        log_error(f"{func_name}: CUDA cache cleared.")

    
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

# --- Cached Embedding Generator ---
@cached(cache=_embedding_cache)
def get_cached_embedding(text: str, embedding_model) -> Optional[List[float]]:
    """
    Generates and caches embeddings for text to avoid repeated encoding.
    """
    
    try:
        # Check if flash-attn is available and use it for faster embedding
        try:
            from flash_attn import flash_attn_func
            has_flash_attn = True
        except ImportError:
            has_flash_attn = False
            
        # Use accelerated attention if available
        if has_flash_attn and hasattr(embedding_model, 'forward_with_flash_attn'):
            # Note: Flash attention integration might not directly support prompt_name.
                embedding_output = embedding_model.forward_with_flash_attn(text)
        else:
                embedding_output = embedding_model.encode(text)

        if isinstance(embedding_output, dict):
            return embedding_output.get('embeddings', []) # Return empty list if key missing
        else:
            return embedding_output.tolist()
        
    except Exception as e:
        log_error(f"Error generating embedding for cached text: {str(e)}")
        return None

# --- Hybrid Retrieval Functions ---
def hybrid_retrieval(
    query: str, 
    embedding_model, 
    collection, 
    top_n: int = 10, 
    alpha: float = 0.5,  # Weight between vector and BM25 (1.0 = all vector, 0.0 = all BM25)
    use_reranker: bool = True,
) -> List[Dict]:
    """
    Perform hybrid retrieval combining vector similarity and BM25 keyword retrieval.
    
    Args:
        query: User query text
        embedding_model: Sentence embedding model
        collection: ChromaDB collection
        top_n: Number of results to return
        alpha: Weighting between vector and BM25 retrieval (1.0 = all vector, 0.0 = all BM25)
        use_reranker: Whether to use BGE reranker on results
        
    Returns:
        List of dictionaries with text and metadata
    """
    func_name = "hybrid_retrieval"
    
    # Fail fast if essential components are missing
    if not embedding_model:
        log_error(f"{func_name}: Failed - Embedding model not available.")
        return []
    
    if not collection:
        log_error(f"{func_name}: Failed - Collection not available.")
        return []
    
    try:
        # 1. Get vector results
        # Use cached embedding if available
        query_embedding = get_cached_embedding(query, embedding_model)
        if not query_embedding:
            log_error(f"{func_name}: Warning - Failed to generate cached embedding, using direct encode.")
            # Apply embedding output fix
            embedding_output = embedding_model.encode(query)
            if isinstance(embedding_output, dict):
                query_embedding = embedding_output['embeddings'].tolist()
            else:
                query_embedding = embedding_output.tolist()
        
        vector_results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_n * 2, 100)  # Get more results for hybrid merging
        )
        
        # Extract vector results into standard format for later processing
        if not vector_results or not vector_results.get("ids") or not vector_results["ids"][0]:
            log_error(f"{func_name}: Vector search returned no results.")
            vector_items = []
        else:
            vector_items = []
            for i, item_id in enumerate(vector_results["ids"][0]):
                # Ensure metadata exists
                if (vector_results.get("metadatas") and 
                    vector_results["metadatas"][0] and 
                    i < len(vector_results["metadatas"][0])):
                    
                    metadata = vector_results["metadatas"][0][i]
                    if metadata and "text" in metadata:
                        vector_items.append({
                            "id": item_id,
                            "text": metadata["text"],
                            "metadata": metadata,
                            "score": vector_results["distances"][0][i] if "distances" in vector_results else 1.0
                        })
        
        # 2. Get BM25 results if available
        global _bm25_index
        bm25_items = []
        
        if _bm25_index is None:
            # Try to initialize BM25 index if it doesn't exist
            try:
                _initialize_bm25_index(collection)
            except Exception as bm25_init_err:
                log_error(f"{func_name}: Warning - Could not initialize BM25 index: {str(bm25_init_err)}")
        
        if _bm25_index is not None:
            try:
                # Tokenize query for BM25
                tokenized_query = query.split()
                
                # Get BM25 scores
                bm25_scores = _bm25_index.get_scores(tokenized_query)
                
                # Get top scores and their indices
                top_indices = np.argsort(bm25_scores)[::-1][:top_n * 2]
                
                # Get all documents from collection to map back to texts
                all_docs = collection.get(include=["metadatas", "ids"])
                
                # Create bm25_items from top indices
                for idx in top_indices:
                    if idx < len(all_docs["metadatas"]):
                        metadata = all_docs["metadatas"][idx]
                        if metadata and "text" in metadata:
                            bm25_items.append({
                                "id": all_docs["ids"][idx],
                                "text": metadata["text"],
                                "metadata": metadata,
                                "score": float(bm25_scores[idx])
                            })
            except Exception as bm25_err:
                log_error(f"{func_name}: Warning - Error in BM25 retrieval: {str(bm25_err)}")
        
        # 3. Combine results with weighting
        # Normalize scores within each method
        if vector_items:
            max_vector_score = max(item["score"] for item in vector_items) or 1.0
            min_vector_score = min(item["score"] for item in vector_items) or 0.0
            score_range = max_vector_score - min_vector_score
            
            for item in vector_items:
                if score_range > 0:
                    # Normalize and invert (lower distance = higher similarity)
                    item["normalized_score"] = 1.0 - ((item["score"] - min_vector_score) / score_range)
                else:
                    item["normalized_score"] = 1.0
        
        if bm25_items:
            max_bm25_score = max(item["score"] for item in bm25_items) or 1.0
            min_bm25_score = min(item["score"] for item in bm25_items) or 0.0
            score_range = max_bm25_score - min_bm25_score
            
            for item in bm25_items:
                if score_range > 0:
                    # Normalize (higher score = higher similarity)
                    item["normalized_score"] = (item["score"] - min_bm25_score) / score_range
                else:
                    item["normalized_score"] = 1.0 if item["score"] > 0 else 0.0
        
        # Merge results by ID
        merged_results = {}
        
        # Add vector results with alpha weight
        for item in vector_items:
            merged_results[item["id"]] = {
                "id": item["id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "score": item["normalized_score"] * alpha
            }
        
        # Add BM25 results with (1-alpha) weight
        for item in bm25_items:
            if item["id"] in merged_results:
                # Add BM25 score to existing item
                merged_results[item["id"]]["score"] += item["normalized_score"] * (1.0 - alpha)
            else:
                # Create new item with BM25 score
                merged_results[item["id"]] = {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": item["normalized_score"] * (1.0 - alpha)
                }
        
        # Convert to list and sort by combined score
        results = list(merged_results.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top_n results
        results = results[:top_n]
        
        # 4. Apply reranking if enabled and available
        if use_reranker and results:
            try:
                reranker = get_reranker()
                if reranker:
                    # Check if we have cached results for this query
                    cache_key = (query, tuple((r["id"], r["text"][:20]) for r in results))
                    
                    if cache_key in _reranker_cache:
                        # Use cached reranking results
                        log_error(f"{func_name}: Using cached reranking for query.")
                        results = _reranker_cache[cache_key]
                    else:
                        # Perform reranking with updated Jina reranker client
                        # Perform reranking with BGE reranker
                        corpus = [r["text"] for r in results]
                        pairs = [[query, doc] for doc in corpus]
                        scores = reranker.compute_score(pairs)
                        
                        # Create new results list with reranked order
                        reranked_results = []
                        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                        for idx in ranked_indices:
                            if idx < len(results):
                                # Update score with reranker score
                                results[idx]["score"] = scores[idx]
                                reranked_results.append(results[idx])
                        
                        # Cache the reranked results
                        _reranker_cache[cache_key] = reranked_results
                        results = reranked_results
            except Exception as rerank_err:
                log_error(f"{func_name}: Warning - Reranking failed: {str(rerank_err)}")
                # Continue with non-reranked results
        
        return results
    
    except Exception as e:
        err_msg = f"Error in hybrid retrieval: {str(e)}"
        log_error(f"{func_name}: Failed - {err_msg}")
        return []

# --- Async Hybrid Retrieval ---
async def async_hybrid_retrieval(
    query: str, 
    embedding_model, 
    collection, 
    top_n: int = 10, 
    alpha: float = 0.5,
    use_reranker: bool = True,
) -> List[Dict]:
    """
    Async version of hybrid retrieval with optimized performance.
    """
    # Use asyncio to run the CPU-bound operations in a thread pool
    return await asyncio.to_thread(
        hybrid_retrieval,
        query=query,
        embedding_model=embedding_model,
        collection=collection,
        top_n=top_n,
        alpha=alpha,
        use_reranker=use_reranker,
    )

# --- Data Addition (Enhanced with BM25 updates) ---
def add_chunks_to_collection(
    chunks: List[str],
    embedding_model,
    collection,
    status = None
) -> bool:
    """ 
    Embeds chunks and adds them to the provided ChromaDB collection.
    Also updates BM25 index.
    """
    func_name = "add_chunks_to_collection"
    if not chunks: return True
    if not embedding_model: 
        err_msg = "Cannot add chunks: Embedding model is missing."
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        return False
    if not collection: 
        err_msg = "Cannot add chunks: DB collection object is missing."
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        return False

    def _update_status(label: str):
        if status: status.update(label=label)

    try:
        # Reduced batch size significantly to prevent CUDA OOM with large models like bge-m3
        batch_size = 8 # Try 16 first, reduce further to 8 or 4 if OOM persists
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        _update_status(f"Embedding and storing {total_chunks} chunks...")
        log_error(f"{func_name}: Starting add process for {total_chunks} chunks in {total_batches} batches.")
        
        # Collect BM25 tokenized texts for later index update
        bm25_tokenized_texts = []
        
        # Try to use flash attention if available
        try:
            from flash_attn import flash_attn_func
            has_flash_attn = True
            log_error(f"{func_name}: Flash Attention detected, using accelerated embeddings.")
        except ImportError:
            has_flash_attn = False
            log_error(f"{func_name}: Flash Attention not available, using standard embeddings.")
        
        for i, batch_start in enumerate(range(0, total_chunks, batch_size)):
            batch_end = min(batch_start + batch_size, total_chunks)
            current_batch_texts = chunks[batch_start:batch_end]
            num_in_batch = len(current_batch_texts)
            _update_status(f"Processing batch {i + 1}/{total_batches} ({num_in_batch} chunks)...")
            
            try:
                # Use cached embeddings where available
                embeddings = []
                for text in current_batch_texts:
                    cached_embedding = get_cached_embedding(text, embedding_model)
                    if cached_embedding:
                        embeddings.append(cached_embedding)
                    else:
                        # If not cached, fall back to direct embedding (this will also cache it)
                        # Apply embedding output fix
                        if has_flash_attn and hasattr(embedding_model, 'forward_with_flash_attn'):
                            embedding_output = embedding_model.forward_with_flash_attn(text)
                        else:
                            embedding_output = embedding_model.encode(text)
                        if isinstance(embedding_output, dict):
                            embeddings.append(embedding_output['embeddings'].tolist())
                        else:
                            embeddings.append(embedding_output.tolist())
                        
                # Also prepare data for BM25 index
                for text in current_batch_texts:
                    bm25_tokenized_texts.append(text.split())
            except Exception as embed_err:
                err_msg = f"Error during embedding batch {i+1}: {str(embed_err)}"
                st.error(err_msg)
                log_error(f"{func_name}: Failed - {err_msg}")
                return False
                
            timestamp_ms = int(time.time() * 1000)
            ids = [f"chunk_{timestamp_ms}_{batch_start + j}" for j in range(num_in_batch)]
            metadatas = [{"text": text} for text in current_batch_texts]
            
            try:
                collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            except InvalidDimensionException as ide:
                err_msg = f"ChromaDB dimension error adding batch {i+1}: {str(ide)}."
                st.error(err_msg)
                log_error(f"{func_name}: Failed - {err_msg}")
                return False
            except Exception as add_err:
                err_msg = f"Error adding batch {i+1} to ChromaDB: {str(add_err)}"
                st.error(err_msg)
                log_error(f"{func_name}: Failed - {err_msg}")
                return False
        
        # Update BM25 index with new texts
        if bm25_tokenized_texts:
            try:
                global _bm25_index
                if _bm25_index is None:
                    # Initialize new index with these texts
                    _bm25_index = BM25Okapi(bm25_tokenized_texts)
                    log_error(f"{func_name}: Created new BM25 index with {len(bm25_tokenized_texts)} documents.")
                else:
                    # Need to recreate the index with all docs
                    # Get existing texts
                    try:
                        # Get all documents from collection
                        results = collection.get(include=["metadatas"])
                        all_texts = []
                        
                        if results and results.get("metadatas"):
                            for meta in results["metadatas"]:
                                if meta and "text" in meta:
                                    all_texts.append(meta["text"].split())
                            
                            _bm25_index = BM25Okapi(all_texts)
                            log_error(f"{func_name}: Updated BM25 index with all {len(all_texts)} documents.")
                    except Exception as bm25_update_err:
                        # Don't fail the whole operation if BM25 update fails
                        log_error(f"{func_name}: Warning - Failed to update BM25 index: {str(bm25_update_err)}")
            except Exception as bm25_err:
                log_error(f"{func_name}: Warning - Error creating/updating BM25 index: {str(bm25_err)}")
            
        final_msg = f"Stored {total_chunks} chunks successfully."
        _update_status(final_msg)
        log_error(f"{func_name}: {final_msg}")

        # Clear CUDA cache after potentially large embedding operation
        if torch.cuda.is_available():
            log_error(f"{func_name}: Clearing CUDA cache after embedding...")
            torch.cuda.empty_cache()
            log_error(f"{func_name}: CUDA cache cleared after embedding.")

        return True
    except Exception as e:
        err_msg = f"Unexpected error during chunk storage loop: {str(e)}"
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        if status: status.update(label=err_msg, state="error")
        return False