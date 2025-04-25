"""
LLM Interface Module - Handles interactions with the local LLM.
"""
import streamlit as st
import ollama
import time
from typing import List, Dict, Any
from modules.utils import log_error # Import log_error
# REMOVE THIS LINE: from chromadb.errors import NotEnoughElementsException # Import specific DB error

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
    # Use st.status for combined progress/status
    with st.status("Processing query...", expanded=False) as status:
        try:
            # Step 1: Encode
            status.update(label="Encoding query...")
            if not embedding_model:
                log_error("LLM Query Error: Embedding model not available.")
                return "Error: Embedding model not available."

            try:
                 query_vector = embedding_model.encode([user_query]).tolist()
            except Exception as embed_err:
                 err_msg = f"Error encoding query: {str(embed_err)}"
                 log_error(err_msg)
                 status.update(label=err_msg, state="error")
                 return err_msg

            # Step 2: Retrieve from DB
            status.update(label="Retrieving relevant information...")
            if not collection:
                log_error("LLM Query Error: Vector database collection not available.")
                return "Error: Vector database collection not available."

            try:
                # ChromaDB 0.5.x handles n_results > count automatically
                results = collection.query(query_embeddings=query_vector, n_results=top_n)
            # REMOVE THIS ENTIRE BLOCK - ChromaDB 0.5.x handles this case
            # except NotEnoughElementsException:
            #      # Handle case where collection has fewer elements than top_n
            #      st.info(f"Collection has fewer than {top_n} elements. Retrieving all available.")
            #      try:
            #           results = collection.query(query_embeddings=query_vector, n_results=collection.count())
            #      except Exception as query_err_fallback:
            #           err_msg = f"Error querying vector database (fallback): {str(query_err_fallback)}"
            #           log_error(err_msg)
            #           status.update(label=err_msg, state="error")
            #           return err_msg
            # END REMOVED BLOCK
            except Exception as query_err: # Catch other, genuine query errors
                 err_msg = f"Error querying vector database: {str(query_err)}"
                 log_error(err_msg)
                 status.update(label=err_msg, state="error")
                 return err_msg


            # Check results structure carefully
            # Check if 'ids' exists and is not empty, and the first list within 'ids' is not empty
            if not results or not results.get("ids") or not results["ids"] or not results["ids"][0]:
                # No results found, but DB query itself didn't fail
                status.update(label="No relevant information found in documents.", state="complete")
                # Give a slightly more informative message
                if collection.count() == 0:
                    return "The document database is currently empty. Please upload and process documents first."
                else:
                    return "I couldn't find specific information related to your query in the uploaded documents."


            # Extract texts using metadatas if available, otherwise handle potential missing keys
            try:
                # Ensure 'metadatas' exists and has the expected structure
                if not results.get("metadatas") or not results["metadatas"] or not results["metadatas"][0]:
                     log_error("LLM Query Warning: Database results missing 'metadatas' structure.")
                     status.update(label="Retrieved information format issue.", state="warning")
                     return "Error: Retrieved relevant document sections, but couldn't extract text content (missing metadata)."

                retrieved_texts = [meta["text"] for meta in results["metadatas"][0] if meta and "text" in meta]
                if not retrieved_texts:
                     # Results found but 'text' missing in metadata
                     log_error("LLM Query Warning: Retrieved documents have no 'text' in metadata.")
                     status.update(label="Retrieved information format issue.", state="warning")
                     return "Error: Retrieved relevant document sections, but couldn't extract text content (missing 'text' field)."
                context = "\n\n---\n\n".join(retrieved_texts) # Use separator
            except (TypeError, IndexError, KeyError) as result_parse_err:
                 err_msg = f"Error parsing database results structure: {str(result_parse_err)}"
                 log_error(err_msg)
                 status.update(label=err_msg, state="error")
                 return err_msg

            # Step 3: Build prompt
            status.update(label="Building prompt for LLM...")
            conversation_prompt = f"Previous conversation:\n{conversation_memory}\n\n" if conversation_memory.strip() else ""

            # Use provided system prompt or fallback
            final_system_prompt = system_prompt if system_prompt else "You are a helpful assistant. Provide clear and accurate information based *only* on the document context provided below. Do not make assumptions or use external knowledge."

            # Clearer prompt structure
            query_prompt = f"""{final_system_prompt}

<DOCUMENT_CONTEXT>
{context}
</DOCUMENT_CONTEXT>

{conversation_prompt}Question: {user_query}

Answer based *only* on the document context:"""

            # Step 4: Local LLM call
            status.update(label=f"Querying LLM ({local_llm_model})...")

            try:
                response = ollama.chat(
                    model=local_llm_model,
                    messages=[{"role": "user", "content": query_prompt}]
                    # Add options if needed, e.g., temperature, top_p
                    # options={'temperature': 0.7}
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
