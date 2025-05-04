# Optimized vector_store.py with better memory management

import os
import shutil
import time
import gc
from typing import List, Dict, Any, Optional, Tuple, Union, Awaitable
import asyncio

import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidDimensionException
import numpy as np
from rank_bm25 import BM25Okapi
import torch
from cachetools import TTLCache, cached
from FlagEmbedding import FlagReranker

from modules.utils import log_error, create_directory_if_not_exists

# --- Constants ---
VECTOR_DB_PATH = "chroma_vector_db"
COLLECTION_NAME = "tourism_vectors"
DEFAULT_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 128,
    "hnsw:M": 16,
}

# Reduced cache sizes for better memory management
EMBEDDING_CACHE_SIZE = 500  # Reduced from 1000
EMBEDDING_CACHE_TTL = 1800  # Reduced from 3600
RERANKER_CACHE_SIZE = 50    # Reduced from 100
RERANKER_CACHE_TTL = 300    # Reduced from 600

# Initialize caches
_embedding_cache = TTLCache(maxsize=EMBEDDING_CACHE_SIZE, ttl=EMBEDDING_CACHE_TTL)
_reranker_cache = TTLCache(maxsize=RERANKER_CACHE_SIZE, ttl=RERANKER_CACHE_TTL)

# --- BM25 Index Store ---
_bm25_index = None

# --- Reranker Instance ---
_reranker = None

def clear_gpu_memory():
    """Helper function to clear GPU memory and garbage collect."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def get_reranker():
    """Get or initialize the BGE Reranker with memory management."""
    global _reranker
    if _reranker is None:
        try:
            clear_gpu_memory()
            log_error("Initializing BGE Reranker...")
            
            # Use CPU if GPU memory is constrained
            device = 'cuda' if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8*1024*1024*1024 else 'cpu'
            _reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True, device=device)
            
            log_error(f"BGE Reranker initialized successfully on {device}.")
        except Exception as e:
            log_error(f"Failed to initialize BGE Reranker: {str(e)}")
            return None
    return _reranker

@st.cache_resource(show_spinner="Initializing Vector Database Client...")
def get_chroma_client() -> Optional[chromadb.Client]:
    """Gets a cached persistent ChromaDB client instance."""
    func_name = "get_chroma_client"
    log_error(f"{func_name}: Attempting to get or create client instance...")
    try:
        if not create_directory_if_not_exists(VECTOR_DB_PATH):
            st.error(f"Fatal Error: Could not create/access ChromaDB directory: {VECTOR_DB_PATH}")
            return None

        settings = Settings(
            persist_directory=VECTOR_DB_PATH,
            is_persistent=True,
            anonymized_telemetry=False,
        )
        
        client = chromadb.PersistentClient(settings=settings)
        client.list_collections()  # Verification
        
        return client

    except Exception as e:
        err_msg = f"Failed to initialize/verify ChromaDB client: {str(e)}"
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        return None

def initialize_vector_db() -> bool:
    """Ensures the vector database collection exists."""
    func_name = "initialize_vector_db"
    client = get_chroma_client()
    if not client:
        return False

    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=DEFAULT_METADATA
        )
        
        # Initialize BM25 if needed
        if collection.count() > 0:
            _initialize_bm25_index(collection)
            
        return True

    except Exception as e:
        err_msg = f"Failed to create/ensure ChromaDB collection: {str(e)}"
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        return False

def _initialize_bm25_index(collection):
    """Initialize BM25 index from existing ChromaDB collection."""
    global _bm25_index
    try:
        results = collection.get(include=["metadatas"])
        
        if not results or not results.get("metadatas") or len(results["metadatas"]) == 0:
            _bm25_index = BM25Okapi([])
            return
            
        texts = []
        for meta in results["metadatas"]:
            if meta and "text" in meta:
                texts.append(meta["text"].split())
        
        if texts:
            _bm25_index = BM25Okapi(texts)
            log_error(f"BM25 index initialized with {len(texts)} documents.")
    except Exception as e:
        log_error(f"Error initializing BM25 index: {str(e)}")
        _bm25_index = None

def get_embedding_dimensions(embedding_model) -> int:
    """Get the dimensions of the embedding model."""
    try:
        test_output = embedding_model.encode(["test"])
        if isinstance(test_output, dict):
            if 'dense_vecs' in test_output:
                return len(test_output['dense_vecs'][0])
            elif 'embeddings' in test_output:
                return len(test_output['embeddings'][0])
            elif 'sentence_embedding' in test_output:
                return len(test_output['sentence_embedding'][0])
        else:
            # For regular numpy array or list output
            return len(test_output[0])
    except Exception as e:
        log_error(f"Error getting embedding dimensions: {str(e)}")
        return 384  # Default to common dimension

def safe_embed_batch(embedding_model, texts: List[str], batch_size: int = 4) -> List[List[float]]:
    """Safely embed a batch of texts with proper error handling and memory management."""
    embeddings = []
    dimensions = get_embedding_dimensions(embedding_model)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            # Clear memory before each mini-batch
            if i > 0:
                clear_gpu_memory()
            
            # Encode the batch
            with torch.no_grad():  # Disable gradient computation
                if hasattr(embedding_model, 'encode'):
                    output = embedding_model.encode(batch_texts)
                    
                    # Handle different output formats
                    if isinstance(output, dict):
                        if 'dense_vecs' in output:
                            batch_embeddings = output['dense_vecs'].tolist()
                        elif 'embeddings' in output:
                            batch_embeddings = output['embeddings'].tolist()
                        elif 'sentence_embedding' in output:
                            batch_embeddings = output['sentence_embedding'].tolist()
                        else:
                            log_error(f"Unknown embedding dictionary format: {output.keys()}")
                            # Try to find any tensor-like value in the dict
                            for key, value in output.items():
                                if hasattr(value, 'shape') and len(value.shape) >= 2:
                                    batch_embeddings = value.tolist()
                                    break
                            else:
                                raise ValueError(f"No suitable embeddings found in output dict")
                    else:
                        # Handle numpy array, tensor, or list output
                        if hasattr(output, 'tolist'):
                            batch_embeddings = output.tolist()
                        else:
                            batch_embeddings = output
                else:
                    raise ValueError("Embedding model does not have encode method")
                
                embeddings.extend(batch_embeddings)
                
        except Exception as e:
            log_error(f"Error in batch embedding: {str(e)}")
            # Fallback to individual embedding
            for text in batch_texts:
                try:
                    single_output = embedding_model.encode([text])
                    if isinstance(single_output, dict):
                        if 'dense_vecs' in single_output:
                            embeddings.append(single_output['dense_vecs'][0].tolist())
                        elif 'embeddings' in single_output:
                            embeddings.append(single_output['embeddings'][0].tolist())
                        elif 'sentence_embedding' in single_output:
                            embeddings.append(single_output['sentence_embedding'][0].tolist())
                    else:
                        if hasattr(single_output, 'tolist'):
                            embeddings.append(single_output[0].tolist())
                        else:
                            embeddings.append(single_output[0])
                except Exception as inner_e:
                    log_error(f"Error embedding single text: {str(inner_e)}")
                    # Return a zero vector as fallback
                    embeddings.append([0.0] * dimensions)
    
    return embeddings

@cached(cache=_embedding_cache)
def get_cached_embedding(text: str, embedding_model) -> Optional[List[float]]:
    """Generates and caches embeddings for text to avoid repeated encoding."""
    try:
        # Try to use the model's encode method
        with torch.no_grad():
            embedding_output = embedding_model.encode([text])
            
            # Handle BGE-M3 output format
            if isinstance(embedding_output, dict):
                if 'dense_vecs' in embedding_output:
                    return embedding_output['dense_vecs'][0].tolist()
                elif 'embeddings' in embedding_output:
                    return embedding_output['embeddings'][0].tolist()
                else:
                    raise ValueError(f"Unknown embedding dictionary format: {embedding_output.keys()}")
            else:
                # Handle numpy array or tensor output
                return embedding_output[0].tolist()
        
    except Exception as e:
        log_error(f"Error generating embedding for cached text: {str(e)}")
        return None

def hybrid_retrieval(
    query: str, 
    embedding_model, 
    collection, 
    top_n: int = 10, 
    alpha: float = 0.5,
    use_reranker: bool = True,
) -> List[Dict]:
    """Perform hybrid retrieval combining vector similarity and BM25 keyword retrieval."""
    func_name = "hybrid_retrieval"
    
    if not embedding_model:
        log_error(f"{func_name}: Failed - Embedding model not available.")
        return []
    
    if not collection:
        log_error(f"{func_name}: Failed - Collection not available.")
        return []
    
    try:
        # 1. Get vector results
        query_embedding = get_cached_embedding(query, embedding_model)
        if not query_embedding:
            log_error(f"{func_name}: Warning - Failed to generate cached embedding, using direct encode.")
            with torch.no_grad():
                embedding_output = embedding_model.encode([query])
                if isinstance(embedding_output, dict):
                    if 'dense_vecs' in embedding_output:
                        query_embedding = embedding_output['dense_vecs'][0].tolist()
                    elif 'embeddings' in embedding_output:
                        query_embedding = embedding_output['embeddings'][0].tolist()
                else:
                    query_embedding = embedding_output[0].tolist()
        
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_n * 2, 100)
        )
        
        # Extract vector results
        vector_items = []
        if vector_results and vector_results.get("ids") and vector_results["ids"][0]:
            for i, item_id in enumerate(vector_results["ids"][0]):
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
            try:
                _initialize_bm25_index(collection)
            except Exception as bm25_init_err:
                log_error(f"{func_name}: Warning - Could not initialize BM25 index: {str(bm25_init_err)}")
        
        if _bm25_index is not None:
            try:
                tokenized_query = query.split()
                bm25_scores = _bm25_index.get_scores(tokenized_query)
                top_indices = np.argsort(bm25_scores)[::-1][:top_n * 2]
                
                all_docs = collection.get(include=["metadatas", "ids"])
                
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
        if vector_items:
            max_vector_score = max(item["score"] for item in vector_items) or 1.0
            min_vector_score = min(item["score"] for item in vector_items) or 0.0
            score_range = max_vector_score - min_vector_score
            
            for item in vector_items:
                if score_range > 0:
                    item["normalized_score"] = 1.0 - ((item["score"] - min_vector_score) / score_range)
                else:
                    item["normalized_score"] = 1.0
        
        if bm25_items:
            max_bm25_score = max(item["score"] for item in bm25_items) or 1.0
            min_bm25_score = min(item["score"] for item in bm25_items) or 0.0
            score_range = max_bm25_score - min_bm25_score
            
            for item in bm25_items:
                if score_range > 0:
                    item["normalized_score"] = (item["score"] - min_bm25_score) / score_range
                else:
                    item["normalized_score"] = 1.0 if item["score"] > 0 else 0.0
        
        # Merge results by ID
        merged_results = {}
        
        for item in vector_items:
            merged_results[item["id"]] = {
                "id": item["id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "score": item["normalized_score"] * alpha
            }
        
        for item in bm25_items:
            if item["id"] in merged_results:
                merged_results[item["id"]]["score"] += item["normalized_score"] * (1.0 - alpha)
            else:
                merged_results[item["id"]] = {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": item["normalized_score"] * (1.0 - alpha)
                }
        
        # Convert to list and sort by combined score
        results = list(merged_results.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_n]
        
        # 4. Apply reranking if enabled
        if use_reranker and results:
            try:
                reranker = get_reranker()
                if reranker:
                    cache_key = (query, tuple((r["id"], r["text"][:20]) for r in results))
                    
                    if cache_key in _reranker_cache:
                        log_error(f"{func_name}: Using cached reranking for query.")
                        results = _reranker_cache[cache_key]
                    else:
                        corpus = [r["text"] for r in results]
                        pairs = [[query, doc] for doc in corpus]
                        scores = reranker.compute_score(pairs)
                        
                        reranked_results = []
                        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                        for idx in ranked_indices:
                            if idx < len(results):
                                results[idx]["score"] = scores[idx]
                                reranked_results.append(results[idx])
                        
                        _reranker_cache[cache_key] = reranked_results
                        results = reranked_results
            except Exception as rerank_err:
                log_error(f"{func_name}: Warning - Reranking failed: {str(rerank_err)}")
        
        return results
    
    except Exception as e:
        err_msg = f"Error in hybrid retrieval: {str(e)}"
        log_error(f"{func_name}: Failed - {err_msg}")
        return []

async def async_hybrid_retrieval(
    query: str, 
    embedding_model, 
    collection, 
    top_n: int = 10, 
    alpha: float = 0.5,
    use_reranker: bool = True,
) -> List[Dict]:
    """Async version of hybrid retrieval with optimized performance."""
    return await asyncio.to_thread(
        hybrid_retrieval,
        query=query,
        embedding_model=embedding_model,
        collection=collection,
        top_n=top_n,
        alpha=alpha,
        use_reranker=use_reranker,
    )

def add_chunks_to_collection(
    chunks: List[str],
    embedding_model,
    collection,
    status = None
) -> bool:
    """Embeds chunks and adds them to the ChromaDB collection with better memory management."""
    func_name = "add_chunks_to_collection"
    
    if not chunks: 
        return True
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
        if status: 
            status.update(label=label)

    try:
        # Adaptive batch sizing based on available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = gpu_memory - torch.cuda.memory_allocated(0)
            # Use smaller batches if memory is constrained
            batch_size = 4 if free_memory < 4*1024*1024*1024 else 8
        else:
            batch_size = 8
        
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        _update_status(f"Embedding and storing {total_chunks} chunks in {total_batches} batches...")
        log_error(f"{func_name}: Starting with batch size {batch_size}")
        
        bm25_tokenized_texts = []
        
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            current_batch_texts = chunks[i:batch_end]
            batch_num = i // batch_size + 1
            
            _update_status(f"Processing batch {batch_num}/{total_batches}...")
            
            try:
                # Clear GPU memory before each batch
                clear_gpu_memory()
                
                # Use the safe embedding function
                embeddings = safe_embed_batch(embedding_model, current_batch_texts, batch_size=4)
                
                # Prepare BM25 data
                for text in current_batch_texts:
                    bm25_tokenized_texts.append(text.split())
                
                # Generate IDs and metadata
                timestamp_ms = int(time.time() * 1000)
                ids = [f"chunk_{timestamp_ms}_{i + j}" for j in range(len(current_batch_texts))]
                metadatas = [{"text": text} for text in current_batch_texts]
                
                # Add to collection
                collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                
                # Clear memory after each batch
                clear_gpu_memory()
                
            except Exception as batch_err:
                err_msg = f"Error processing batch {batch_num}: {str(batch_err)}"
                log_error(f"{func_name}: {err_msg}")
                st.error(err_msg)
                return False
        
        # Update BM25 index
        if bm25_tokenized_texts:
            try:
                global _bm25_index
                if _bm25_index is None:
                    _bm25_index = BM25Okapi(bm25_tokenized_texts)
                else:
                    # Recreate index with all documents
                    results = collection.get(include=["metadatas"])
                    all_texts = []
                    
                    if results and results.get("metadatas"):
                        for meta in results["metadatas"]:
                            if meta and "text" in meta:
                                all_texts.append(meta["text"].split())
                        
                        _bm25_index = BM25Okapi(all_texts)
                        log_error(f"{func_name}: Updated BM25 index with {len(all_texts)} documents.")
            except Exception as bm25_err:
                log_error(f"{func_name}: Warning - BM25 index update failed: {str(bm25_err)}")
        
        final_msg = f"Successfully stored {total_chunks} chunks."
        _update_status(final_msg)
        log_error(f"{func_name}: {final_msg}")
        
        # Final memory cleanup
        clear_gpu_memory()
        
        return True
        
    except Exception as e:
        err_msg = f"Unexpected error during chunk storage: {str(e)}"
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        if status: 
            status.update(label=err_msg, state="error")
        return False

def get_chroma_collection() -> Optional[chromadb.Collection]:
    """Gets the existing ChromaDB collection."""
    func_name = "get_chroma_collection"
    client = get_chroma_client()
    if not client:
        return None
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        err_msg = f"Failed to get ChromaDB collection: {str(e)}"
        st.error(err_msg)
        log_error(f"{func_name}: Failed - {err_msg}")
        return None

def reset_vector_db() -> tuple:
    """Resets the vector database with proper cleanup."""
    func_name = "reset_vector_db"
    backup_dir_name = None
    
    # Clear caches and global objects
    global _embedding_cache, _reranker_cache, _bm25_index, _reranker
    _embedding_cache.clear()
    _reranker_cache.clear()
    _bm25_index = None
    _reranker = None
    
    clear_gpu_memory()
    
    try:
        if os.path.exists(VECTOR_DB_PATH):
            timestamp = int(time.time())
            backup_dir_name = f"{VECTOR_DB_PATH}_backup_{timestamp}"
            os.rename(VECTOR_DB_PATH, backup_dir_name)
            
        if not create_directory_if_not_exists(VECTOR_DB_PATH):
            err_msg = f"Failed to create fresh directory '{VECTOR_DB_PATH}'"
            return False, err_msg
            
        st.cache_resource.clear()
        
        success_msg = "Vector database reset successfully."
        if backup_dir_name:
            success_msg += f" Backup: {backup_dir_name}"
            
        return True, success_msg
        
    except Exception as e:
        err_msg = f"Error during vector database reset: {str(e)}"
        return False, err_msg