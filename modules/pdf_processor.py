"""
Enhanced PDF Processor Module - Handles PDF parsing and text chunking.
Optimized for tourism and travel document analysis.
"""
import os
import tempfile
import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union

import streamlit as st
import nltk
import openparse
from streamlit.delta_generator import DeltaGenerator
import fitz  # PyMuPDF for enhanced PDF handling

from modules.utils import log_error, create_directory_if_not_exists
from modules.nlp_models import extract_tourism_entities, calculate_text_complexity

# Tourism-specific section identifiers to help with intelligent chunking
TOURISM_SECTION_MARKERS = [
    r"destination(?:s)?",
    r"accommodation(?:s)?",
    r"transport(?:ation)?",
    r"activities",
    r"attraction(?:s)?",
    r"sight(?:-)?seeing",
    r"tour(?:s)?",
    r"travel(?:ing)?",
    r"flight(?:s)?",
    r"hotel(?:s)?",
    r"resort(?:s)?",
    r"restaurant(?:s)?",
    r"cuisine",
    r"food and drink",
    r"shopping",
    r"budget",
    r"cost(?:s)?",
    r"price(?:s)?",
    r"seasonal",
    r"weather",
    r"climate",
    r"culture",
    r"local customs",
    r"travel tips",
    r"itinerary",
    r"day trip(?:s)?",
    r"excursion(?:s)?",
    r"package(?:s)?",
    r"booking",
    r"reservation(?:s)?",
    r"travel insurance",
    r"health and safety",
    r"travel advisory",
    r"visa(?:s)?",
    r"passport(?:s)?",
    r"currency",
    r"exchange rate(?:s)?",
    r"language(?:s)?",
    r"phrase(?:s)?",
    r"communication",
    r"internet",
    r"wifi",
    r"transportation",
    r"getting around",
    r"map(?:s)?",
    r"direction(?:s)?",
    r"sustainability",
    r"eco(?:-)?friendly",
    r"green travel",
    r"responsible tourism",
    r"luxury travel",
    r"budget travel",
    r"family travel",
    r"solo travel",
    r"group travel",
    r"adventure travel",
    r"cruise(?:s)?",
    r"all(?:-)?inclusive",
    r"review(?:s)?",
    r"rating(?:s)?",
    r"recommendation(?:s)?",
    r"travel guide(?:s)?",
]

# Compile regex patterns for performance
TOURISM_SECTION_PATTERNS = [re.compile(fr"(?i)(?:^|\s|\n)({marker})(?::|\s)", re.IGNORECASE) for marker in TOURISM_SECTION_MARKERS]

def is_tourism_section_start(text: str) -> bool:
    """
    Check if text starts with a tourism-related section marker.
    
    Args:
        text: Text to check for section markers
        
    Returns:
        True if text appears to start a tourism section
    """
    text_lower = text.lower()
    return any(pattern.search(text_lower) for pattern in TOURISM_SECTION_PATTERNS)

def smart_chunking(
    text: str, 
    chunk_size: int = 250, 
    overlap: int = 50,
    respect_sections: bool = True,
    min_chunk_size: int = 50
) -> List[Dict[str, Any]]:
    """
    Enhanced text chunking optimized for tourism documents.
    Splits text into chunks with metadata about tourism entities.
    
    Args:
        text: Text to split into chunks
        chunk_size: Target size of chunks in words
        overlap: Number of words to overlap between chunks
        respect_sections: Whether to avoid breaking across detected tourism sections
        min_chunk_size: Minimum chunk size to keep (smaller chunks are merged)
        
    Returns:
        List of dictionaries with text chunks and metadata
    """
    if not text or not text.strip():
        return []
    
    try:
        # Tokenize into sentences
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            err_msg = "NLTK 'punkt' tokenizer not found. Please ensure it's downloaded during initialization."
            st.error(err_msg)
            log_error(err_msg)
            return [{"text": text, "metadata": {}}]
        except Exception as nltk_e:
            err_msg = f"NLTK sentence tokenization error: {str(nltk_e)}"
            st.warning(err_msg)
            log_error(err_msg)
            return [{"text": text, "metadata": {}}]

        # Process chunks with section awareness
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        current_section = "general"
        
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            sentence_word_count = len(words)
            
            # Check if this sentence starts a new tourism section
            if respect_sections and is_tourism_section_start(sentence):
                # If we have accumulated sentences, save the current chunk before starting new section
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "section": current_section,
                            "word_count": current_word_count,
                            "sentence_count": len(current_chunk_sentences),
                            "complexity": calculate_text_complexity(chunk_text)
                        }
                    })
                    # No overlap when section changes
                    current_chunk_sentences = [sentence]
                    current_word_count = sentence_word_count
                    # Update current section
                    for pattern in TOURISM_SECTION_PATTERNS:
                        match = pattern.search(sentence.lower())
                        if match:
                            current_section = match.group(1)
                            break
                    continue
            
            # Handle normal chunking logic
            if current_word_count + sentence_word_count > chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section": current_section,
                        "word_count": current_word_count,
                        "sentence_count": len(current_chunk_sentences),
                        "complexity": calculate_text_complexity(chunk_text)
                    }
                })
                # Keep overlap sentences for the next chunk
                overlap_sentence_count = min(overlap, len(current_chunk_sentences))
                current_chunk_sentences = current_chunk_sentences[-overlap_sentence_count:]
                current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
            
            # Special case for very long single sentences
            elif not current_chunk_sentences and sentence_word_count > chunk_size:
                chunks.append({
                    "text": sentence,
                    "metadata": {
                        "section": current_section,
                        "word_count": sentence_word_count,
                        "sentence_count": 1,
                        "complexity": calculate_text_complexity(sentence)
                    }
                })
                current_chunk_sentences = []
                current_word_count = 0
                continue
            
            # Add the current sentence to our chunk
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_word_count
        
        # Add any remaining sentences as a final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "section": current_section,
                    "word_count": current_word_count,
                    "sentence_count": len(current_chunk_sentences),
                    "complexity": calculate_text_complexity(chunk_text)
                }
            })
        
        # Post-processing: merge small chunks
        if min_chunk_size > 0:
            merged_chunks = []
            temp_chunk = None
            
            for chunk in chunks:
                if temp_chunk is None:
                    temp_chunk = chunk
                elif chunk["metadata"]["word_count"] < min_chunk_size or temp_chunk["metadata"]["word_count"] < min_chunk_size:
                    # Merge with previous chunk
                    combined_text = temp_chunk["text"] + " " + chunk["text"]
                    temp_chunk = {
                        "text": combined_text,
                        "metadata": {
                            "section": temp_chunk["metadata"]["section"],
                            "word_count": temp_chunk["metadata"]["word_count"] + chunk["metadata"]["word_count"],
                            "sentence_count": temp_chunk["metadata"]["sentence_count"] + chunk["metadata"]["sentence_count"],
                            "complexity": calculate_text_complexity(combined_text)
                        }
                    }
                else:
                    merged_chunks.append(temp_chunk)
                    temp_chunk = chunk
            
            # Add the last temp chunk if it exists
            if temp_chunk:
                merged_chunks.append(temp_chunk)
            
            chunks = merged_chunks
        
        # Extract tourism entities for each chunk
        for chunk in chunks:
            tourism_entities = extract_tourism_entities(chunk["text"])
            chunk["metadata"]["tourism_entities"] = tourism_entities
            
            # Add payment method detection if present
            payment_keywords = ["payment", "pay", "credit card", "debit card", "cash", "transaction", 
                              "wallet", "banking", "money", "currency", "exchange", "fee", "transfer"]
            
            chunk["metadata"]["has_payment_info"] = any(keyword in chunk["text"].lower() for keyword in payment_keywords)
            
            # Add segment detection
            segment_keywords = {
                "demographic": ["age", "gender", "family", "children", "senior", "young", "generation", "gen z", "millennial", "boomer"],
                "luxury": ["luxury", "premium", "exclusive", "high-end", "upscale", "elite", "vip", "deluxe", "5-star", "first class"],
                "budget": ["budget", "affordable", "cheap", "economic", "low-cost", "value", "bargain", "discount", "deal", "saving"],
                "sustainability": ["sustainable", "eco", "green", "environment", "carbon", "footprint", "responsible", "ethical", "conservation"]
            }
            
            segment_matches = {}
            for segment, keywords in segment_keywords.items():
                segment_matches[segment] = any(keyword in chunk["text"].lower() for keyword in keywords)
            
            chunk["metadata"]["segment_matches"] = segment_matches
        
        return chunks
        
    except Exception as e:
        err_msg = f"Unexpected error during text chunking: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        return [{"text": text, "metadata": {}}] if text else []

def process_uploaded_pdf(
    uploaded_file,
    chunk_size: int,
    overlap: int,
    status: Optional[DeltaGenerator] = None,
    extract_images: bool = False,
    image_output_dir: str = "extracted_images"
) -> List[Dict[str, Any]]:
    """
    Enhanced PDF processing with tourism-focused chunking and optional image extraction.
    
    Args:
        uploaded_file: Streamlit uploaded file
        chunk_size: Target size of chunks in words
        overlap: Number of words to overlap between chunks
        status: Optional Streamlit status object for progress updates
        extract_images: Whether to extract images from PDF
        image_output_dir: Directory to save extracted images
        
    Returns:
        List of dictionaries with text chunks and metadata
    """
    chunks = []
    tmp_file_path = None
    parser = None
    
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
             st.error(err_msg)
             log_error(err_msg)
             if status: status.update(label=err_msg, state="error")
             return []
        
        # Parse the document
        _update_status(f"Parsing {uploaded_file.name}...")
        parsed_doc = parser.parse(tmp_file_path)
        
        # Handle image extraction if requested
        if extract_images:
            _update_status("Extracting images from PDF...")
            image_paths = extract_images_from_pdf(tmp_file_path, image_output_dir, uploaded_file.name)
            if image_paths:
                _update_status(f"Extracted {len(image_paths)} images.")
            else:
                _update_status("No images extracted.")
        
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
            st.warning(warn_msg)
            log_error(warn_msg)
            _update_status(warn_msg)
            return []
        
        # Chunk the extracted text
        _update_status(f"Chunking text ({len(text_data)} sections)...")
        total_sections = len(text_data)
        
        for idx, text_section in enumerate(text_data):
            # Update status periodically for large documents
            if idx % 20 == 0 and total_sections > 20:
                 _update_status(f"Chunking text (section {idx+1}/{total_sections})...")
            
            # Use tourism-optimized chunking
            section_chunks = smart_chunking(
                text_section, 
                chunk_size=chunk_size, 
                overlap=overlap, 
                respect_sections=True
            )
            
            # Add file metadata to each chunk
            for chunk in section_chunks:
                chunk["metadata"]["filename"] = uploaded_file.name
                chunk["metadata"]["section_index"] = idx
                chunks.extend([chunk])
        
        _update_status(f"Generated {len(chunks)} chunks with tourism metadata.")
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
         st.error(err_msg)
         log_error(err_msg)
         if status: status.update(label=err_msg, state="error")
         return []
    except Exception as e:
        err_msg = f"Unexpected error processing PDF {uploaded_file.name}: {str(e)}"
        st.error(err_msg)
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

def extract_images_from_pdf(
    pdf_path: str, 
    output_dir: str, 
    base_filename: str
) -> List[str]:
    """
    Extract images from PDF for additional processing.
    Useful for tourism brochures with maps, attractions, etc.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images
        base_filename: Base name for extracted images
        
    Returns:
        List of paths to extracted images
    """
    image_paths = []
    
    try:
        # Create output directory if it doesn't exist
        if not create_directory_if_not_exists(output_dir):
            log_error(f"Failed to create directory for extracted images: {output_dir}")
            return []
        
        # Clean filename for use in image names
        clean_name = re.sub(r'[^\w\-_]', '_', os.path.splitext(base_filename)[0])
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        image_count = 0
        
        # Iterate through pages
        for page_num, page in enumerate(doc):
            # Get images from page
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                
                try:
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_data = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Only process if the image is large enough (avoid small icons)
                    if len(image_data) > 10000:  # Skip tiny images (adjust threshold as needed)
                        image_count += 1
                        image_name = f"{clean_name}_page{page_num+1}_img{image_count}.{image_ext}"
                        image_path = os.path.join(output_dir, image_name)
                        
                        # Save image
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        
                        image_paths.append(image_path)
                except Exception as img_err:
                    log_error(f"Error extracting image {img_index} from page {page_num+1}: {str(img_err)}")
                    continue
        
        return image_paths
        
    except ImportError as e:
        log_error(f"PyMuPDF (fitz) import error for image extraction: {str(e)}")
        return []
    except Exception as e:
        log_error(f"Error extracting images from PDF: {str(e)}")
        return []