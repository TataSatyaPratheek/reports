"""
Enhanced LLM Interface Module - Handles interactions with the local LLM.
Added features:
- Dynamic query reformulation
- Context-aware memory with sliding window
- Flash Attention optimization
- Asynchronous query processing
- Integration with hybrid retrieval and reranking
"""
import streamlit as st
import ollama
import time
import asyncio
import torch # Added for CUDA cache management
from typing import List, Dict, Any, Optional, Tuple, Union, Awaitable
from modules.utils import log_error
from modules.vector_store import hybrid_retrieval, async_hybrid_retrieval

import os
NLTK_DATA_PATH = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
import nltk
nltk.data.path = [NLTK_DATA_PATH]  # Override all other paths


# Try to import flash_attn for optimization
try:
    try:
        from flash_attn import flash_attn_func
        HAS_FLASH_ATTN = True
    except ImportError:
        HAS_FLASH_ATTN = False
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

class SlidingWindowMemory:
    """
    Maintains a sliding window of conversation history with dynamic sizing based on token counts.
    """
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.memory_items = []
        self.current_token_count = 0
    
    def add(self, role: str, content: str, estimated_tokens: Optional[int] = None):
        """Add a conversation item to memory."""
        # Estimate tokens if not provided (very rough heuristic)
        if estimated_tokens is None:
            estimated_tokens = len(content.split()) + 10  # Add overhead for role, etc.
        
        # Add new item
        self.memory_items.append({
            "role": role,
            "content": content,
            "tokens": estimated_tokens
        })
        self.current_token_count += estimated_tokens
        
        # Trim if needed
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
    """
    Asynchronous version of query reformulation.
    """
    # Skip reformulation for simple queries or if no conversation context
    if len(user_query.split()) <= 3 or not conversation_memory.strip():
        return user_query
        
    try:
        # Prepare prompt for query reformulation
        reformulation_prompt = f"""Given the conversation history and the current question, reformulate the question to be fully self-contained, 
explicit, and clear for retrieving relevant information. The reformulated question should make sense on its own without requiring
the conversation context.

CONVERSATION HISTORY:
{conversation_memory}

CURRENT QUESTION: {user_query}

REFORMULATED QUESTION:"""

        # Make light-weight LLM call using new async Ollama interface
        response = await ollama.agenerate(
            model=local_llm_model,
            prompt=reformulation_prompt,
            options={
                'temperature': 0.1,  # Low temperature for more deterministic output
                'num_predict': 200,  # Limit output length
            }
        )
        
        reformulated = response.get('response', '').strip()
        
        # Validate reformulation (ensure it's not too different or too short)
        if not reformulated or len(reformulated) < len(user_query) / 2:
            return user_query
            
        log_error(f"async_reformulate_query: Original: '{user_query}' → Reformulated: '{reformulated}'")
        return reformulated
        
    except Exception as e:
        log_error(f"Failed to reformulate query asynchronously: {str(e)}")
        return user_query  # Fall back to original query

def reformulate_query(user_query: str, conversation_memory: str = "", local_llm_model: str = "llama3.2:latest") -> str:
    """
    Use the LLM to reformulate the query for better retrieval, accounting for conversation context.
    
    Args:
        user_query: The original user query
        conversation_memory: Previous conversation history
        local_llm_model: The LLM model to use
        
    Returns:
        Reformulated query or original query if reformulation fails
    """
    func_name = "reformulate_query"
    
    # Skip reformulation for simple queries or if no conversation context
    if len(user_query.split()) <= 3 or not conversation_memory.strip():
        return user_query
    
    try:
        # Prepare prompt for query reformulation
        reformulation_prompt = f"""Given the conversation history and the current question, reformulate the question to be fully self-contained, 
explicit, and clear for retrieving relevant information. The reformulated question should make sense on its own without requiring
the conversation context.

CONVERSATION HISTORY:
{conversation_memory}

CURRENT QUESTION: {user_query}

REFORMULATED QUESTION:"""

        # Make light-weight LLM call
        response = ollama.generate(
            model=local_llm_model,
            prompt=reformulation_prompt,
            options={
                'temperature': 0.1,  # Low temperature for more deterministic output
                'num_predict': 200,  # Limit output length
            }
        )
        
        reformulated = response.get('response', '').strip()
        
        # Validate reformulation (ensure it's not too different or too short)
        if not reformulated or len(reformulated) < len(user_query) / 2:
            return user_query
            
        log_error(f"{func_name}: Original: '{user_query}' → Reformulated: '{reformulated}'")
        return reformulated
        
    except Exception as e:
        log_error(f"{func_name}: Failed to reformulate query: {str(e)}")
        return user_query  # Fall back to original query

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
    """
    Asynchronous version of query_llm with optimized performance.
    """
    # Use asyncio to manage the overall flow
    try:
        # Step 1: Query reformulation if enabled (in parallel with other operations)
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
            
        # Step 2: Start retrieval (can be run in parallel with reformulation)
        # We'll use the original query for now and potentially update it
        retrieval_task = asyncio.create_task(
            async_hybrid_retrieval( # Initial call
                query=user_query,
                embedding_model=embedding_model,
                collection=collection,
                top_n=top_n,
                alpha=hybrid_alpha,
                # Note: use_hybrid_retrieval check is outside, this assumes hybrid is used
                use_reranker=use_reranker
            ) if use_hybrid_retrieval else asyncio.to_thread(
                lambda: collection.query( # Standard query fallback
                    query_embeddings=(
                        output['embeddings'].tolist()
                        if isinstance((output := embedding_model.encode([user_query])), dict)
                        else output.tolist()
                    ),
                    n_results=top_n,
                    include=["metadatas", "distances"] # Ensure metadatas are included
                )
            )
        )
        
        # Wait for reformulation if requested
        if reformulation_task:
            effective_query = await reformulation_task
            
            # If reformulation succeeded and is different, start a new retrieval with the better query
            if effective_query != user_query:
                # Cancel the original retrieval task if it's still running
                if not retrieval_task.done():
                    retrieval_task.cancel()
                    
                # Start a new retrieval with the reformulated query
                retrieval_task = asyncio.create_task(
                    async_hybrid_retrieval( # Call after reformulation
                        query=effective_query,
                        embedding_model=embedding_model,
                        collection=collection,
                        top_n=top_n,
                        alpha=hybrid_alpha,
                        # Note: use_hybrid_retrieval check is outside, this assumes hybrid is used
                        use_reranker=use_reranker
                    ) if use_hybrid_retrieval else asyncio.to_thread(
                        lambda: collection.query(
                            query_embeddings=(
                                output['embeddings'].tolist()
                                if isinstance((output := embedding_model.encode([effective_query])), dict)
                                else output.tolist()
                            ),
                            n_results=top_n,
                            include=["metadatas", "distances"] # Ensure metadatas are included
                        )
                    )
                )
        
        # Wait for retrieval results
        results = await retrieval_task
        
        # Extract texts
        if use_hybrid_retrieval:
            # Hybrid retrieval returns a list of dicts with 'text' key
            retrieved_texts = [item["text"] for item in results if "text" in item]
        else:
            # Standard query returns a different structure
            if not results or not results.get("ids") or not results["ids"][0]:
                return "I couldn't find specific information related to your query in the uploaded documents."
                
            if results.get("metadatas") and results["metadatas"] and results["metadatas"][0]:
                retrieved_texts = [meta["text"] for meta in results["metadatas"][0] 
                                 if meta and "text" in meta]
            else:
                return "Retrieved information format issue."
        
        # Check if we have any texts
        if not retrieved_texts:
            return "I couldn't find specific information related to your query in the uploaded documents."
            
        # Create context from retrieved texts
        context = "\n\n---\n\n".join(retrieved_texts)
                
        # Build prompt
        conversation_prompt = f"Previous conversation:\n{conversation_memory}\n\n" if conversation_memory.strip() else ""
        
        # Use provided system prompt or fallback
        final_system_prompt = system_prompt if system_prompt else "You are a helpful assistant. Provide clear and accurate information based *only* on the document context provided below. Do not make assumptions or use external knowledge."
        
        # Clearer prompt structure
        query_prompt = f"""{final_system_prompt}

        # --- Clear CUDA cache before LLM call ---
        if torch.cuda.is_available():
            log_error("async_query_llm: Clearing CUDA cache before Ollama call...")
            torch.cuda.empty_cache()
            log_error("async_query_llm: CUDA cache cleared.")

<DOCUMENT_CONTEXT>
{context}
</DOCUMENT_CONTEXT>

{conversation_prompt}Question: {user_query}

Answer based *only* on the document context:"""
        
        # Call Ollama asynchronously
        response = await ollama.achat(
            model=local_llm_model,
            messages=[{"role": "user", "content": query_prompt}]
            # Add options if needed, e.g., temperature, top_p
        )
        
        # Check response
        if not response or "message" not in response or "content" not in response["message"]:
            return "Error: Received an invalid or empty response from the LLM."
            
        return response["message"]["content"]
        
    except Exception as e:
        return f"Unexpected error in async query processing: {str(e)}"

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
    """
    Enhanced process for querying the LLM with improved retrieval and context management.
    
    Args:
        user_query: The user's query text
        top_n: Number of document chunks to retrieve
        local_llm_model: Name of the local LLM model to use
        embedding_model: Sentence embedding model
        collection: ChromaDB collection
        conversation_memory: Previous conversation history
        system_prompt: Custom system prompt
        use_hybrid_retrieval: Whether to use hybrid vector+BM25 retrieval
        use_query_reformulation: Whether to reformulate the query for better retrieval
        hybrid_alpha: Weight between vector and BM25 (1.0 = all vector, 0.0 = all BM25)
        use_reranker: Whether to use reranker on results
        
    Returns:
        LLM answer string
    """
    # Use st.status for combined progress/status
    with st.status("Processing query...", expanded=False) as status:
        try:
            # Step 1: Query reformulation if enabled
            status.update(label="Analyzing query...")
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
                    # Continue with original query
            
            # Step 2: Retrieve from DB using hybrid retrieval if enabled
            status.update(label="Retrieving relevant information...")
            if not embedding_model:
                log_error("LLM Query Error: Embedding model not available.")
                return "Error: Embedding model not available."
                
            if not collection:
                log_error("LLM Query Error: Vector database collection not available.")
                return "Error: Vector database collection not available."
            
            try:
                # Check if we can use Flash Attention for improved performance
                use_flash_attn = HAS_FLASH_ATTN and hasattr(embedding_model, 'forward_with_flash_attn')
                
                if use_hybrid_retrieval:
                    results = hybrid_retrieval( # Synchronous call
                        query=effective_query,
                        embedding_model=embedding_model,
                        collection=collection,
                        top_n=top_n,
                        alpha=hybrid_alpha,
                        # Note: use_hybrid_retrieval check is outside, this assumes hybrid is used
                        use_reranker=use_reranker
                    )
                    
                    # Extract texts from results
                    retrieved_texts = [item["text"] for item in results if item.get("text")]
                else:
                    # Fall back to simple vector search
                    if use_flash_attn:
                        query_vector = embedding_model.forward_with_flash_attn(effective_query).tolist()
                    else:
                        # Apply embedding output fix
                        embedding_output = embedding_model.encode([effective_query])
                        if isinstance(embedding_output, dict):
                            query_vector = embedding_output['embeddings'].tolist()
                        else:
                            query_vector = embedding_output.tolist()
                        
                    db_results = collection.query(query_embeddings=query_vector, n_results=top_n)
                    
                    # Check if 'ids' exists and is not empty, and the first list within 'ids' is not empty
                    if not db_results or not db_results.get("ids") or not db_results["ids"] or not db_results["ids"][0]:
                        # No results found, but DB query itself didn't fail
                        status.update(label="No relevant information found in documents.", state="complete")
                        if collection.count() == 0:
                            return "The document database is currently empty. Please upload and process documents first."
                        else:
                            return "I couldn't find specific information related to your query in the uploaded documents."
                    
                    # Extract texts using metadatas
                    retrieved_texts = []
                    if db_results.get("metadatas") and db_results["metadatas"] and db_results["metadatas"][0]:
                        retrieved_texts = [meta["text"] for meta in db_results["metadatas"][0] 
                                          if meta and "text" in meta]
            except Exception as retrieval_err:
                err_msg = f"Error during retrieval: {str(retrieval_err)}"
                log_error(err_msg)
                status.update(label=err_msg, state="error")
                return err_msg
            
            # Check if we retrieved any texts
            if not retrieved_texts:
                status.update(label="No relevant information found in documents.", state="complete")
                if collection.count() == 0:
                    return "The document database is currently empty. Please upload and process documents first."
                else:
                    return "I couldn't find specific information related to your query in the uploaded documents."
            
            # Create context from retrieved texts
            context = "\n\n---\n\n".join(retrieved_texts)
                    
            # Step 3: Build prompt with clearer structure
            status.update(label="Building prompt for LLM...")
            conversation_prompt = f"Previous conversation:\n{conversation_memory}\n\n" if conversation_memory.strip() else ""

            # Use provided system prompt or fallback
            final_system_prompt = system_prompt if system_prompt else "You are a helpful assistant. Provide clear and accurate information based *only* on the document context provided below. Do not make assumptions or use external knowledge."

            # Clearer prompt structure
            query_prompt = f"""{final_system_prompt}

            # --- Clear CUDA cache before LLM call ---
            if torch.cuda.is_available():
                log_error("query_llm: Clearing CUDA cache before Ollama call...")
                torch.cuda.empty_cache()
                log_error("query_llm: CUDA cache cleared.")

<DOCUMENT_CONTEXT>
{context}
</DOCUMENT_CONTEXT>

{conversation_prompt}Question: {user_query}

Answer based *only* on the document context:"""

            # Step 4: Local LLM call
            status.update(label=f"Querying LLM ({local_llm_model})...")

            try:
                # Check for Ollama options for improved performance
                ollama_options = {
                    'temperature': 0.7,  # Standard temperature
                }
                
                # Add Flash Attention if available
                if HAS_FLASH_ATTN:
                    ollama_options['flash_attn'] = True
                    
                response = ollama.chat(
                    model=local_llm_model,
                    messages=[{"role": "user", "content": query_prompt}],
                    options=ollama_options
                )

                # Check response structure before accessing content
                if not response or "message" not in response or "content" not in response["message"]:
                     err_msg = "Error: Received an invalid or empty response from the LLM."
                     log_error(f"Invalid LLM response structure: {response}")
                     status.update(label=err_msg, state="error")
                     return err_msg

                llm_answer = response["message"]["content"]
                status.update(label="Query complete!", state="complete")
                return llm_answer

            except ollama.ResponseError as ollama_err:
                # Handle specific Ollama errors
                error_body = str(ollama_err)
                if "connection refused" in error_body.lower():
                    err_msg = "Error: Could not connect to Ollama. Please ensure the Ollama service is running."
                elif "model" in error_body.lower() and "not found" in error_body.lower():
                    err_msg = f"Error: Ollama model '{local_llm_model}' not found. It might need to be pulled or is misspelled."
                elif "context window" in error_body.lower():
                     err_msg = f"Error: The prompt context is too large for the model '{local_llm_model}'. Try reducing 'Top Results' or 'Conversation Memory'."
                else:
                    # Try to get more details if available (structure might vary)
                    try:
                         details = ollama_err.response.json()
                         err_msg = f"Ollama API Error: {details.get('error', str(ollama_err))}"
                    except: # Fallback to string representation
                         err_msg = f"Ollama API Error: {error_body}"

                log_error(f"Ollama ResponseError: {err_msg}")
                status.update(label=err_msg, state="error")
                return err_msg
            except Exception as llm_err: # Catch other unexpected errors during chat call
                 err_msg = f"Unexpected error during LLM call: {str(llm_err)}"
                 log_error(err_msg)
                 status.update(label=err_msg, state="error")
                 return err_msg

        except Exception as outer_e: # Catch errors in the overall try block logic
            err_msg = f"Unexpected error in query processing: {str(outer_e)}"
            log_error(err_msg)
            # Ensure status is updated even if error is before status object creation
            if 'status' in locals():
                 status.update(label=err_msg, state="error")
            else:
                 st.error(err_msg) # Fallback display
            return err_msg