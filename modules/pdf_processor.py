"""
PDF Processor Module - Handles PDF parsing and text chunking.
"""
import os
import tempfile
import time
import streamlit as st
import nltk
import openparse
from typing import List, Optional # Import Optional
from streamlit.delta_generator import DeltaGenerator # Import for type hinting status
from modules.utils import log_error

def smart_chunking(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    """
    Splits text into sentences and groups them into overlapping chunks.
    (Implementation remains the same as previous version - error handling included)
    """
    if not text or not text.strip():
        return []
    try:
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
             err_msg = "NLTK 'punkt' tokenizer not found. Please ensure it's downloaded during initialization."
             st.error(err_msg) # Keep error visible if NLTK fails critically
             log_error(err_msg)
             return [text]
        except Exception as nltk_e:
             err_msg = f"NLTK sentence tokenization error: {str(nltk_e)}"
             st.warning(err_msg)
             log_error(err_msg)
             return [text]

        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)
            if current_word_count + sentence_word_count > chunk_size and current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                overlap_sentence_count = min(overlap, len(current_chunk_sentences))
                current_chunk_sentences = current_chunk_sentences[-overlap_sentence_count:]
                current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
            elif not current_chunk_sentences and sentence_word_count > chunk_size:
                 chunks.append(sentence)
                 current_chunk_sentences = []
                 current_word_count = 0
                 continue
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_word_count
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        return chunks
    except Exception as e:
        err_msg = f"Unexpected error during text chunking: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        return [text] if text else []


# MODIFIED: Added 'status' parameter and updated feedback
def process_uploaded_pdf(
    uploaded_file,
    chunk_size: int,
    overlap: int,
    status: Optional[DeltaGenerator] = None # Accept optional status object
) -> List[str]:
    """
    Saves an uploaded PDF temporarily, parses it with openparse,
    and returns a list of text chunks. Updates status object.
    """
    chunks = []
    tmp_file_path = None
    parser = None # Define parser outside try block for cleanup

    def _update_status(label: str):
        """Helper to update status if provided."""
        if status:
            status.update(label=label)

    try:
        # Create temporary file
        _update_status("Preparing temporary file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Initialize parser
        _update_status("Initializing document parser...")
        try:
             parser = openparse.DocumentParser()
        except Exception as parser_init_e:
             err_msg = f"Failed to initialize DocumentParser: {str(parser_init_e)}"
             st.error(err_msg) # Show critical error
             log_error(err_msg)
             if status: status.update(label=err_msg, state="error")
             return []

        # Parse the document
        _update_status(f"Parsing {uploaded_file.name}...")
        # parse_start_time = time.time() # Removed timing for less verbosity
        parsed_doc = parser.parse(tmp_file_path)
        # parse_duration = time.time() - parse_start_time
        # st.write(f"Parsing completed in {parse_duration:.2f} seconds.") # Removed

        # Clean up temporary file immediately after parsing
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                tmp_file_path = None
            except OSError as rm_err:
                 log_error(f"Warning: Could not remove temporary file {tmp_file_path}: {str(rm_err)}")

        # Extract text nodes
        _update_status("Extracting text content...")
        text_data = [node.text for node in parsed_doc.nodes if node.text and node.text.strip()]

        if not text_data:
            warn_msg = f"No text content extracted from {uploaded_file.name}."
            st.warning(warn_msg) # Keep warning visible
            log_error(warn_msg)
            _update_status(warn_msg) # Update status as well
            return []

        # Chunk the extracted text
        _update_status(f"Chunking text ({len(text_data)} sections)...")
        # chunk_progress = st.progress(0) # Removed progress bar here
        total_sections = len(text_data)

        for idx, text_section in enumerate(text_data):
            # Update status periodically for large documents
            if idx % 20 == 0 and total_sections > 20: # Update every 20 sections
                 _update_status(f"Chunking text (section {idx+1}/{total_sections})...")

            section_chunks = smart_chunking(text_section, chunk_size=chunk_size, overlap=overlap)
            chunks.extend(section_chunks)

        # chunk_progress.empty() # Removed
        # st.write(f"Generated {len(chunks)} chunks.") # Removed

        _update_status(f"Generated {len(chunks)} chunks.") # Final update before returning
        return chunks

    except ImportError as e:
         err_msg = f"Import error during PDF processing: {str(e)}"
         st.error(err_msg)
         log_error(err_msg)
         if status: status.update(label=err_msg, state="error")
         return []
    except FileNotFoundError as e:
         err_msg = f"File not found during PDF processing: {str(e)}"
         st.error(err_msg)
         log_error(err_msg)
         if status: status.update(label=err_msg, state="error")
         return []
    except openparse.errors.ParsingError as parse_err:
         err_msg = f"Failed to parse PDF {uploaded_file.name}: {str(parse_err)}"
         st.error(err_msg) # Show parsing error
         log_error(err_msg)
         if status: status.update(label=err_msg, state="error")
         return []
    except Exception as e:
        err_msg = f"Unexpected error processing PDF {uploaded_file.name}: {str(e)}"
        st.error(err_msg) # Show unexpected error
        log_error(err_msg)
        if status: status.update(label=err_msg, state="error")
        return []

    finally:
        # Ensure temporary file is cleaned up
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except OSError as final_rm_err:
                log_error(f"Error removing temporary file in finally block: {str(final_rm_err)}")
        # Clean up parser resources if necessary (depends on openparse implementation)
        # if parser: del parser # Example if explicit cleanup needed

