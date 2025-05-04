# Optimized llm_interface.py with hybrid retrieval support

import streamlit as st
import ollama
from typing import List, Dict, Any, Optional

from modules.utils import log_error
from modules.vector_store import hybrid_retrieval

class SlidingWindowMemory:
    """Maintains a sliding window of conversation history."""
    
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.memory_items = []
        self.current_token_count = 0
    
    def add(self, role: str, content: str):
        """Add a conversation item to memory."""
        estimated_tokens = len(content.split()) + 10
        
        self.memory_items.append({
            "role": role,
            "content": content,
            "tokens": estimated_tokens
        })
        self.current_token_count += estimated_tokens
        
        self._trim_to_max_tokens()
    
    def _trim_to_max_tokens(self):
        """Remove oldest items until under token limit."""
        while self.current_token_count > self.max_tokens and len(self.memory_items) > 0:
            removed = self.memory_items.pop(0)
            self.current_token_count -= removed["tokens"]
    
    def get_formatted_history(self) -> str:
        """Get formatted conversation history for prompt context."""
        formatted = []
        for item in self.memory_items:
            formatted.append(f"{item['role']}: {item['content']}")
        return "\n\n".join(formatted)
    
    def clear(self):
        """Clear all memory."""
        self.memory_items.clear()
        self.current_token_count = 0

def query_llm(
    user_query: str,
    top_n: int,
    local_llm_model: str,
    embedding_model,
    collection,
    conversation_memory: str = "",
    system_prompt: str = None,
    use_hybrid_retrieval: bool = True,
    hybrid_alpha: float = 0.7,
    use_reranker: bool = True
) -> str:
    """Query the LLM with hybrid retrieved context."""
    try:
        # Retrieve relevant context
        if use_hybrid_retrieval:
            results = hybrid_retrieval(
                query=user_query,
                embedding_model=embedding_model,
                collection=collection,
                top_n=top_n,
                alpha=hybrid_alpha,
                use_reranker=use_reranker
            )
            
            retrieved_texts = [item["text"] for item in results if "text" in item]
        else:
            # Standard vector search
            query_vector = embedding_model.encode([user_query])[0].tolist()
            
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_n
            )
            
            retrieved_texts = []
            if results.get("metadatas") and results["metadatas"][0]:
                for meta in results["metadatas"][0]:
                    if meta and "text" in meta:
                        retrieved_texts.append(meta["text"])
        
        if not retrieved_texts:
            return "I couldn't find relevant information in the documents."
        
        context = "\n\n---\n\n".join(retrieved_texts)
        
        # Build prompt
        final_system_prompt = system_prompt or "You are a helpful assistant. Answer based on the document context provided."
        
        conversation_prompt = f"Previous conversation:\n{conversation_memory}\n\n" if conversation_memory else ""
        
        query_prompt = f"""{final_system_prompt}

<DOCUMENT_CONTEXT>
{context}
</DOCUMENT_CONTEXT>

{conversation_prompt}Question: {user_query}

Answer based on the document context:"""
        
        # Call LLM
        response = ollama.chat(
            model=local_llm_model,
            messages=[{"role": "user", "content": query_prompt}]
        )
        
        if not response or "message" not in response or "content" not in response["message"]:
            return "Error: Invalid response from the LLM."
        
        return response["message"]["content"]
        
    except Exception as e:
        log_error(f"Error in query_llm: {str(e)}")
        return f"Error processing query: {str(e)}"