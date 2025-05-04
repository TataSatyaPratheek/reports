# Optimized vector_store.py with hybrid search and reranking

import os
import time
import streamlit as st
import chromadb
from chromadb.config import Settings
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional
from FlagEmbedding import FlagReranker

from modules.utils import log_error, create_directory_if_not_exists

# Constants
VECTOR_DB_PATH = "chroma_vector_db"
COLLECTION_NAME = "tourism_vectors"

# BM25 Index Store
_bm25_index = None

# Reranker Instance
_reranker = None

@st.cache_resource(show_spinner="Initializing Vector Database...")
def get_chroma_client() -> Optional[chromadb.Client]:
    """Get ChromaDB client instance."""
    try:
        if not create_directory_if_not_exists(VECTOR_DB_PATH):
            st.error(f"Could not create directory: {VECTOR_DB_PATH}")
            return None

        settings = Settings(
            persist_directory=VECTOR_DB_PATH,
            is_persistent=True,
            anonymized_telemetry=False,
        )
        
        client = chromadb.PersistentClient(settings=settings)
        return client

    except Exception as e:
        log_error(f"Failed to initialize ChromaDB client: {str(e)}")
        return None

def get_reranker():
    """Get or initialize the BGE Reranker."""
    global _reranker
    if _reranker is None:
        try:
            _reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
            log_error("BGE Reranker initialized successfully.")
        except Exception as e:
            log_error(f"Failed to initialize BGE Reranker: {str(e)}")
            return None
    return _reranker

def initialize_vector_db(dimensions: int = None) -> bool:
    """Initialize the vector database collection."""
    client = get_chroma_client()
    if not client:
        return False

    try:
        # Get or create collection
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            
            # If collection exists and has data, initialize BM25
            if collection.count() > 0:
                _initialize_bm25_index(collection)
                
        except:
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        
        return True

    except Exception as e:
        log_error(f"Failed to initialize vector database: {str(e)}")
        return False

def _initialize_bm25_index(collection):
    """Initialize BM25 index from existing ChromaDB collection."""
    global _bm25_index
    try:
        results = collection.get(include=["metadatas"])
        
        if not results or not results.get("metadatas"):
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

def get_chroma_collection():
    """Get the ChromaDB collection."""
    client = get_chroma_client()
    if not client:
        return None
    
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        log_error(f"Failed to get collection: {str(e)}")
        return None

def add_chunks_to_collection(
    chunks: List[str],
    embedding_model,
    collection,
    status = None
) -> bool:
    """Add text chunks to the collection with BM25 indexing."""
    if not chunks or not embedding_model or not collection:
        return False

    def _update_status(label: str):
        if status: 
            status.update(label=label)

    try:
        # Generate embeddings
        _update_status("Generating embeddings...")
        embeddings = embedding_model.encode(chunks)
        
        # Generate IDs
        timestamp = int(time.time() * 1000)
        ids = [f"chunk_{timestamp}_{i}" for i in range(len(chunks))]
        
        # Create metadata
        metadatas = [{"text": chunk} for chunk in chunks]
        
        # Add to collection
        _update_status("Storing in vector database...")
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        # Update BM25 index
        _update_status("Updating BM25 index...")
        bm25_tokenized_texts = [text.split() for text in chunks]
        
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
                log_error(f"Updated BM25 index with {len(all_texts)} documents.")
        
        _update_status(f"Successfully stored {len(chunks)} chunks.")
        return True
        
    except Exception as e:
        log_error(f"Error adding chunks to collection: {str(e)}")
        if status:
            status.update(label=f"Error: {str(e)}", state="error")
        return False

def hybrid_retrieval(
    query: str, 
    embedding_model, 
    collection, 
    top_n: int = 5, 
    alpha: float = 0.7,
    use_reranker: bool = True,
) -> List[Dict]:
    """Perform hybrid retrieval combining vector similarity and BM25."""
    if not embedding_model or not collection:
        return []
    
    try:
        # 1. Get vector results
        query_embedding = embedding_model.encode([query])[0].tolist()
        
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
        
        # 2. Get BM25 results
        global _bm25_index
        bm25_items = []
        
        if _bm25_index is None:
            _initialize_bm25_index(collection)
        
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
                log_error(f"Error in BM25 retrieval: {str(bm25_err)}")
        
        # 3. Combine results
        if vector_items:
            # Normalize vector scores
            max_vector_score = max(item["score"] for item in vector_items)
            min_vector_score = min(item["score"] for item in vector_items)
            score_range = max_vector_score - min_vector_score
            
            for item in vector_items:
                if score_range > 0:
                    item["normalized_score"] = 1.0 - ((item["score"] - min_vector_score) / score_range)
                else:
                    item["normalized_score"] = 1.0
        
        if bm25_items:
            # Normalize BM25 scores
            max_bm25_score = max(item["score"] for item in bm25_items)
            min_bm25_score = min(item["score"] for item in bm25_items)
            score_range = max_bm25_score - min_bm25_score
            
            for item in bm25_items:
                if score_range > 0:
                    item["normalized_score"] = (item["score"] - min_bm25_score) / score_range
                else:
                    item["normalized_score"] = 1.0 if item["score"] > 0 else 0.0
        
        # Merge results
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
        
        # Convert to list and sort
        results = list(merged_results.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_n]
        
        # 4. Apply reranking if enabled
        if use_reranker and results:
            try:
                reranker = get_reranker()
                if reranker:
                    corpus = [r["text"] for r in results]
                    pairs = [[query, doc] for doc in corpus]
                    scores = reranker.compute_score(pairs)
                    
                    # Reorder results based on reranker scores
                    reranked_results = []
                    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                    for idx in ranked_indices:
                        results[idx]["score"] = scores[idx]
                        reranked_results.append(results[idx])
                    
                    results = reranked_results
            except Exception as rerank_err:
                log_error(f"Reranking failed: {str(rerank_err)}")
        
        return results
    
    except Exception as e:
        log_error(f"Error in hybrid retrieval: {str(e)}")
        return []

def reset_vector_db() -> tuple:
    """Reset the vector database."""
    global _bm25_index, _reranker
    _bm25_index = None
    _reranker = None
    
    try:
        # Rename existing database
        if os.path.exists(VECTOR_DB_PATH):
            backup_dir = f"{VECTOR_DB_PATH}_backup_{int(time.time())}"
            os.rename(VECTOR_DB_PATH, backup_dir)
        
        # Create fresh directory
        if not create_directory_if_not_exists(VECTOR_DB_PATH):
            return False, "Failed to create fresh directory"
        
        # Clear cache
        st.cache_resource.clear()
        
        return True, "Vector database reset successfully"
        
    except Exception as e:
        return False, f"Error resetting database: {str(e)}"