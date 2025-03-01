"""
PDF Processor Module - Handles PDF parsing and text chunking.
"""
import os
import tempfile
import time
import streamlit as st
import nltk
import openparse
from typing import List

def smart_chunking(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    """
    Splits text into sentences and groups them into overlapping chunks.
    Each chunk has at most 'chunk_size' words; consecutive chunks share 'overlap' sentences.
    
    Optimized for performance with large texts.
    """
    if not text or not text.strip():
        return []
    
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            length = len(words)
            
            if current_length + length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep last 'overlap' sentences for next chunk
                offset = min(overlap, len(current_chunk))
                current_chunk = current_chunk[-offset:]
                current_length = sum(len(sent.split()) for sent in current_chunk)
            
            current_chunk.append(sentence)
            current_length += length

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    except Exception as e:
        from modules.utils import log_error
        log_error(f"Error in chunking: {str(e)}")
        return [text] if text else []  # Fall back to returning the whole text as one chunk

def process_uploaded_pdf(uploaded_file, chunk_size: int, overlap: int) -> List[str]:
    """
    Saves an uploaded PDF temporarily, parses it with openparse,
    and returns a list of text chunks.
    
    Optimized for memory efficiency and error handling.
    """
    chunks = []
    tmp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        parser = openparse.DocumentParser()
        parsed_doc = parser.parse(tmp_file_path)
        
        # Clean up temporary file immediately after parsing
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            tmp_file_path = None
            
        # Extract text nodes in batches to avoid memory issues
        text_data = []
        for node in parsed_doc.nodes:
            if node.text and node.text.strip():
                text_data.append(node.text)
        
        if not text_data:
            st.warning(f"No text content found in {uploaded_file.name}")
            return []

        # Process in batches with progress indicator
        section_progress = st.progress(0)
        total_sections = len(text_data)
        
        for idx, text in enumerate(text_data):
            section_progress.progress((idx + 1) / total_sections)
            batch_chunks = smart_chunking(text, chunk_size=chunk_size, overlap=overlap)
            chunks.extend(batch_chunks)
            time.sleep(0.01)  # Small delay to allow UI updates
        
        return chunks
    
    except Exception as e:
        from modules.utils import log_error
        log_error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return []
    
    finally:
        # Ensure temporary file is cleaned up
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception as e:
                from modules.utils import log_error
                log_error(f"Error removing temporary file: {str(e)}")