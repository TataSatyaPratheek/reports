"""
LLM Interface Module - Handles interactions with the local LLM.
"""
import streamlit as st
import ollama
import time
from typing import List, Dict, Any

def query_llm(
    user_query: str, 
    top_n: int, 
    local_llm_model: str, 
    embedding_model, 
    collection, 
    conversation_memory: str = "", 
    system_prompt: str = None
) -> str:
    """
    Process a user query through the LLM.
    
    1) Encode user_query
    2) Retrieve top_n chunks from DB
    3) Build a prompt with system_prompt and conversation_memory
    4) Call local LLM with Ollama
    5) Return LLM answer
    """
    query_progress = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Encode
        status_text.write("Encoding query...")
        if not embedding_model:
            return "Error: Embedding model not available."
        
        query_vector = embedding_model.encode([user_query]).tolist()
        query_progress.progress(0.3)
        
        # Step 2: Retrieve from DB
        status_text.write("Retrieving relevant information...")
        if not collection:
            return "Error: Vector database not available."
        
        results = collection.query(query_embeddings=query_vector, n_results=top_n)
        query_progress.progress(0.6)
        
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return "I don't have enough context to answer that question. Please upload relevant PDF documents first."
        
        retrieved_texts = [doc["text"] for doc in results["metadatas"][0]]
        context = "\n\n".join(retrieved_texts)
        
        # Step 3: Build prompt
        if conversation_memory.strip():
            conversation_prompt = f"Previous conversation:\n{conversation_memory}\n\n"
        else:
            conversation_prompt = ""
        
        # Use provided system prompt or fallback to default
        if not system_prompt:
            system_prompt = "You are a helpful assistant. Provide clear and accurate information based on the document context."
        
        query_prompt = f"""
{system_prompt}

Here is relevant information from the documents:

{context}

{conversation_prompt}Based on this information, please answer the following question: {user_query}
"""
        
        # Step 4: Local LLM call
        status_text.write("Running local LLM...")
        
        response = ollama.chat(
            model=local_llm_model,
            messages=[{"role": "user", "content": query_prompt}]
        )
        query_progress.progress(1.0)
        status_text.empty()
        
        return response["message"]["content"]
    
    except Exception as e:
        error_msg = str(e)
        from modules.utils import log_error
        log_error(f"Error querying LLM: {error_msg}")
        query_progress.progress(1.0)
        status_text.empty()
        
        if "connection refused" in error_msg.lower():
            return "Error: Could not connect to Ollama. Please ensure Ollama is running."
        elif "model not found" in error_msg.lower():
            return f"Error: Model '{local_llm_model}' not found. Please select a different model or download it first."
        else:
            return f"Error querying the LLM: {error_msg}"