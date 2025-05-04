# Optimized llm_interface.py with better memory management

import streamlit as st
import ollama
import time
import asyncio
import torch
from typing import List, Dict, Any, Optional, Tuple, Union, Awaitable
from modules.utils import log_error
from modules.vector_store import hybrid_retrieval, async_hybrid_retrieval

import os
NLTK_DATA_PATH = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
import nltk
nltk.data.path = [NLTK_DATA_PATH]

# Try to import flash_attn for optimization
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

class SlidingWindowMemory:
    """Maintains a sliding window of conversation history with dynamic sizing based on token counts."""
    
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.memory_items = []
        self.current_token_count = 0
    
    def add(self, role: str, content: str, estimated_tokens: Optional[int] = None):
        """Add a conversation item to memory."""
        if estimated_tokens is None:
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
    
    def get_messages_list(self) -> List[Dict[str, str]]:
        """Get history as a list of message dicts for Ollama chat API."""
        return [{"role": item["role"], "content": item["content"]} for item in self.memory_items]
    
    def clear(self):
        """Clear all memory."""
        self.memory_items.clear()
        self.current_token_count = 0

async def async_reformulate_query(user_query: str, conversation_memory: str = "", local_llm_model: str = "llama3.2:latest") -> str:
    """Asynchronous version of query reformulation."""
    if len(user_query.split()) <= 3 or not conversation_memory.strip():
        return user_query
    
    try:
        reformulation_prompt = f"""Given the conversation history and the current question, reformulate the question to be fully self-contained, 
explicit, and clear for retrieving relevant information. The reformulated question should make sense on its own without requiring
the conversation context.

CONVERSATION HISTORY:
{conversation_memory}

CURRENT QUESTION: {user_query}

REFORMULATED QUESTION:"""

        response = await ollama.agenerate(
            model=local_llm_model,
            prompt=reformulation_prompt,
            options={
                'temperature': 0.1,
                'num_predict': 200,
            }
        )
        
        reformulated = response.get('response', '').strip()
        
        if not reformulated or len(reformulated) < len(user_query) / 2:
            return user_query
        
        log_error(f"async_reformulate_query: Original: '{user_query}' → Reformulated: '{reformulated}'")
        return reformulated
        
    except Exception as e:
        log_error(f"Failed to reformulate query asynchronously: {str(e)}")
        return user_query

def reformulate_query(user_query: str, conversation_memory: str = "", local_llm_model: str = "llama3.2:latest") -> str:
    """Use the LLM to reformulate the query for better retrieval, accounting for conversation context."""
    func_name = "reformulate_query"
    
    if len(user_query.split()) <= 3 or not conversation_memory.strip():
        return user_query
    
    try:
        reformulation_prompt = f"""Given the conversation history and the current question, reformulate the question to be fully self-contained, 
explicit, and clear for retrieving relevant information. The reformulated question should make sense on its own without requiring
the conversation context.

CONVERSATION HISTORY:
{conversation_memory}

CURRENT QUESTION: {user_query}

REFORMULATED QUESTION:"""

        response = ollama.generate(
            model=local_llm_model,
            prompt=reformulation_prompt,
            options={
                'temperature': 0.1,
                'num_predict': 200,
            }
        )
        
        reformulated = response.get('response', '').strip()
        
        if not reformulated or len(reformulated) < len(user_query) / 2:
            return user_query
        
        log_error(f"{func_name}: Original: '{user_query}' → Reformulated: '{reformulated}'")
        return reformulated
        
    except Exception as e:
        log_error(f"{func_name}: Failed to reformulate query: {str(e)}")
        return user_query

async def async_query_llm(
    user_query: str,
    top_n: int,
    local_llm_model: str,
    embedding_model,
    collection,
    conversation_memory: str = "",
    system_prompt: str = None,
    use_hybrid_retrieval: bool = True,
    use_query_reformulation: bool = True,
    hybrid_alpha: float = 0.7,
    use_reranker: bool = True
) -> str:
    """Asynchronous version of query_llm with optimized performance."""
    try:
        # Step 1: Query reformulation if enabled
        effective_query = user_query
        if use_query_reformulation and conversation_memory:
            reformulation_task = asyncio.create_task(
                async_reformulate_query(
                    user_query=user_query,
                    conversation_memory=conversation_memory,
                    local_llm_model=local_llm_model
                )
            )
        else:
            reformulation_task = None
        
        # Step 2: Start retrieval
        retrieval_task = asyncio.create_task(
            async_hybrid_retrieval(
                query=user_query,
                embedding_model=embedding_model,
                collection=collection,
                top_n=top_n,
                alpha=hybrid_alpha,
                use_reranker=use_reranker
            ) if use_hybrid_retrieval else asyncio.to_thread(
                lambda: collection.query(
                    query_embeddings=[embedding_model.encode([user_query])[0].tolist()],
                    n_results=top_n,
                    include=["metadatas", "distances"]
                )
            )
        )
        
        # Wait for reformulation if requested
        if reformulation_task:
            effective_query = await reformulation_task
            
            if effective_query != user_query:
                if not retrieval_task.done():
                    retrieval_task.cancel()
                
                retrieval_task = asyncio.create_task(
                    async_hybrid_retrieval(
                        query=effective_query,
                        embedding_model=embedding_model,
                        collection=collection,
                        top_n=top_n,
                        alpha=hybrid_alpha,
                        use_reranker=use_reranker
                    ) if use_hybrid_retrieval else asyncio.to_thread(
                        lambda: collection.query(
                            query_embeddings=[embedding_model.encode([effective_query])[0].tolist()],
                            n_results=top_n,
                            include=["metadatas", "distances"]
                        )
                    )
                )
        
        # Wait for retrieval results
        results = await retrieval_task
        
        # Extract texts
        if use_hybrid_retrieval:
            retrieved_texts = [item["text"] for item in results if "text" in item]
        else:
            if not results or not results.get("ids") or not results["ids"][0]:
                return "I couldn't find specific information related to your query in the uploaded documents."
            
            if results.get("metadatas") and results["metadatas"] and results["metadatas"][0]:
                retrieved_texts = [meta["text"] for meta in results["metadatas"][0] 
                                 if meta and "text" in meta]
            else:
                return "Retrieved information format issue."
        
        if not retrieved_texts:
            return "I couldn't find specific information related to your query in the uploaded documents."
        
        context = "\n\n---\n\n".join(retrieved_texts)
        
        # Build prompt
        conversation_prompt = f"Previous conversation:\n{conversation_memory}\n\n" if conversation_memory.strip() else ""
        final_system_prompt = system_prompt if system_prompt else "You are a helpful assistant. Provide clear and accurate information based *only* on the document context provided below. Do not make assumptions or use external knowledge."
        
        query_prompt = f"""{final_system_prompt}

<DOCUMENT_CONTEXT>
{context}
</DOCUMENT_CONTEXT>

{conversation_prompt}Question: {user_query}

Answer based *only* on the document context:"""
        
        # Clear CUDA cache before LLM call
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Call Ollama asynchronously
        response = await ollama.achat(
            model=local_llm_model,
            messages=[{"role": "user", "content": query_prompt}]
        )
        
        if not response or "message" not in response or "content" not in response["message"]:
            return "Error: Received an invalid or empty response from the LLM."
        
        return response["message"]["content"]
        
    except Exception as e:
        return f"Unexpected error in async query processing: {str(e)}"

# Updated query_llm function in llm_interface.py

def query_llm(
    user_query: str,
    top_n: int,
    local_llm_model: str,
    embedding_model,
    collection,
    conversation_memory: str = "",
    system_prompt: str = None,
    use_hybrid_retrieval: bool = True,
    use_query_reformulation: bool = True,
    hybrid_alpha: float = 0.7,
    use_reranker: bool = True
) -> str:
    """Enhanced process for querying the LLM with improved retrieval and context management."""
    try:
        # Step 1: Query reformulation if enabled
        effective_query = user_query
        
        if use_query_reformulation and conversation_memory:
            try:
                effective_query = reformulate_query(
                    user_query=user_query,
                    conversation_memory=conversation_memory,
                    local_llm_model=local_llm_model
                )
            except Exception as reform_err:
                log_error(f"Query reformulation error (non-critical): {str(reform_err)}")
        
        # Step 2: Retrieve from DB
        if not embedding_model:
            log_error("LLM Query Error: Embedding model not available.")
            return "Error: Embedding model not available."
        
        if not collection:
            log_error("LLM Query Error: Vector database collection not available.")
            return "Error: Vector database collection not available."
        
        try:
            if use_hybrid_retrieval:
                results = hybrid_retrieval(
                    query=effective_query,
                    embedding_model=embedding_model,
                    collection=collection,
                    top_n=top_n,
                    alpha=hybrid_alpha,
                    use_reranker=use_reranker
                )
                
                retrieved_texts = [item["text"] for item in results if item.get("text")]
            else:
                # Handle embedding output format for BGE-M3
                with torch.no_grad():
                    embedding_output = embedding_model.encode([effective_query])
                    
                    if isinstance(embedding_output, dict):
                        if 'dense_vecs' in embedding_output:
                            query_vector = embedding_output['dense_vecs'][0].tolist()
                        elif 'embeddings' in embedding_output:
                            query_vector = embedding_output['embeddings'][0].tolist()
                        else:
                            raise ValueError(f"Unknown embedding format: {embedding_output.keys()}")
                    else:
                        query_vector = embedding_output[0].tolist()
                
                db_results = collection.query(query_embeddings=[query_vector], n_results=top_n)
                
                if not db_results or not db_results.get("ids") or not db_results["ids"] or not db_results["ids"][0]:
                    if collection.count() == 0:
                        return "The document database is currently empty. Please upload and process documents first."
                    else:
                        return "I couldn't find specific information related to your query in the uploaded documents."
                
                retrieved_texts = []
                if db_results.get("metadatas") and db_results["metadatas"] and db_results["metadatas"][0]:
                    retrieved_texts = [meta["text"] for meta in db_results["metadatas"][0] 
                                      if meta and "text" in meta]
        except Exception as retrieval_err:
            err_msg = f"Error during retrieval: {str(retrieval_err)}"
            log_error(err_msg)
            return err_msg
        
        if not retrieved_texts:
            if collection.count() == 0:
                return "The document database is currently empty. Please upload and process documents first."
            else:
                return "I couldn't find specific information related to your query in the uploaded documents."
        
        context = "\n\n---\n\n".join(retrieved_texts)
        
        # Step 3: Build prompt
        conversation_prompt = f"Previous conversation:\n{conversation_memory}\n\n" if conversation_memory.strip() else ""
        final_system_prompt = system_prompt if system_prompt else "You are a helpful assistant. Provide clear and accurate information based *only* on the document context provided below. Do not make assumptions or use external knowledge."
        
        query_prompt = f"""{final_system_prompt}

<DOCUMENT_CONTEXT>
{context}
</DOCUMENT_CONTEXT>

{conversation_prompt}Question: {user_query}

Answer based *only* on the document context:"""

        # Clear CUDA cache before LLM call
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 4: Call LLM
        try:
            ollama_options = {
                'temperature': 0.7,
            }
            
            if HAS_FLASH_ATTN:
                ollama_options['flash_attn'] = True
            
            response = ollama.chat(
                model=local_llm_model,
                messages=[{"role": "user", "content": query_prompt}],
                options=ollama_options
            )

            if not response or "message" not in response or "content" not in response["message"]:
                err_msg = "Error: Received an invalid or empty response from the LLM."
                log_error(f"Invalid LLM response structure: {response}")
                return err_msg

            llm_answer = response["message"]["content"]
            return llm_answer

        except ollama.ResponseError as ollama_err:
            error_body = str(ollama_err)
            err_msg = f"Ollama API Error: {error_body}"
            log_error(f"Ollama ResponseError: {err_msg}")
            return err_msg
            
        except Exception as llm_err:
            err_msg = f"Unexpected error during LLM call: {str(llm_err)}"
            log_error(err_msg)
            return err_msg

    except Exception as outer_e:
        err_msg = f"Unexpected error in query processing: {str(outer_e)}"
        log_error(err_msg)
        return err_msg