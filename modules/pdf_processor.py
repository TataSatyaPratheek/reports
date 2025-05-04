# modules/pdf_processor.py
"""
Optimized PDF Processor Module with memory-efficient processing.
"""
import tempfile
import time
import os
import re
import gc
from typing import List, Dict, Any, Optional

import streamlit as st
import openparse
import fitz  # PyMuPDF for image extraction
from streamlit.delta_generator import DeltaGenerator

from modules.utils import log_error, create_directory_if_not_exists
from modules.memory_utils import memory_monitor, get_available_memory_mb

@st.cache_resource
def get_openparse_parser():
    """Get a cached OpenParse parser instance."""
    try:
        return openparse.DocumentParser()
    except Exception as e:
        log_error(f"Failed to initialize DocumentParser: {str(e)}")
        return None

def chunk_text_stream(text: str, chunk_size: int = 512, overlap: int = 64, callback=None) -> List[str]:
    """Stream-based text chunking with memory efficiency."""
    if not text or not text.strip():
        return []
    
    text = text.strip()
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
            
            # Callback for progress updates
            if callback and len(chunks) % 10 == 0:
                callback(len(chunks))
            
            # Memory management for very large documents
            if len(chunks) % 100 == 0:
                memory_monitor.check()
                gc.collect()
    
    return chunks

def process_uploaded_pdf(
    uploaded_file,
    chunk_size: int,
    overlap: int,
    status: Optional[DeltaGenerator] = None,
    extract_images: bool = True,
    image_output_dir: str = "extracted_images"
) -> List[Dict[str, Any]]:
    """Process PDF with memory-efficient parsing and optional image extraction."""
    chunks = []
    tmp_file_path = None
    
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
        
        # Get parser from pool
        _update_status("Initializing document parser...")
        parser = get_openparse_parser()
        
        if not parser:
            # Fallback to PyMuPDF if OpenParse fails
            _update_status("Using fallback PDF parser...")
            return fallback_pdf_processing(tmp_file_path, uploaded_file.name, chunk_size, overlap, extract_images, image_output_dir)
        
        # Parse the document with OpenParse
        _update_status(f"Parsing {uploaded_file.name}...")
        parsed_doc = parser.parse(tmp_file_path)
        
        # Handle image extraction if requested
        if extract_images:
            _update_status("Extracting images from PDF...")
            image_paths = extract_images_paginated(tmp_file_path, image_output_dir, uploaded_file.name)
            if image_paths:
                _update_status(f"Extracted {len(image_paths)} images.")
        
        # Extract text nodes from OpenParse
        _update_status("Extracting text content...")
        text_data = [node.text for node in parsed_doc.nodes if node.text and node.text.strip()]
        
        if not text_data:
            _update_status("No text content found, trying fallback method...")
            return fallback_pdf_processing(tmp_file_path, uploaded_file.name, chunk_size, overlap, extract_images, image_output_dir)
        
        # Stream-based chunking
        _update_status(f"Chunking text ({len(text_data)} sections)...")
        
        for idx, text_section in enumerate(text_data):
            def chunk_callback(count):
                _update_status(f"Section {idx+1}/{len(text_data)}: {count} chunks")
            
            section_chunks = chunk_text_stream(text_section, chunk_size, overlap, chunk_callback)
            
            for chunk_idx, chunk in enumerate(section_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "filename": uploaded_file.name,
                        "section_index": idx,
                        "chunk_index": chunk_idx,
                        "source": "openparse"
                    }
                })
            
            # Memory management between sections
            memory_monitor.check()
        
        _update_status(f"Generated {len(chunks)} chunks.")
        return chunks
        
    except Exception as e:
        log_error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
        return []
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception as e:
                log_error(f"Error removing temporary file: {str(e)}")

def fallback_pdf_processing(
    pdf_path: str,
    filename: str,
    chunk_size: int,
    overlap: int,
    extract_images: bool = True,
    image_output_dir: str = "extracted_images"
) -> List[Dict[str, Any]]:
    """Fallback PDF processing using PyMuPDF with memory optimization."""
    chunks = []
    
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Process in pages to avoid memory overload
        pages_per_batch = 10
        full_text = ""
        
        for page_batch_start in range(0, len(doc), pages_per_batch):
            batch_text = ""
            batch_end = min(page_batch_start + pages_per_batch, len(doc))
            
            # Extract text from batch of pages
            for page_num in range(page_batch_start, batch_end):
                page = doc[page_num]
                text = page.get_text()
                if text:
                    batch_text += text + "\n"
            
            full_text += batch_text
            
            # Memory management between batches
            memory_monitor.check()
        
        # Extract images if requested
        if extract_images:
            extract_images_paginated(pdf_path, image_output_dir, filename)
        
        doc.close()
        
        # Chunk the text
        if full_text.strip():
            text_chunks = chunk_text_stream(full_text, chunk_size, overlap)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": i,
                        "source": "pymupdf"
                    }
                })
        
        return chunks
        
    except Exception as e:
        log_error(f"Fallback PDF processing failed: {str(e)}")
        return []

def extract_images_paginated(
    pdf_path: str, 
    output_dir: str, 
    base_filename: str,
    max_pages_per_batch: int = 5
) -> List[str]:
    """Extract images from PDF with memory-efficient pagination."""
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
        total_pages = len(doc)
        image_count = 0
        
        # Process in batches
        for batch_start in range(0, total_pages, max_pages_per_batch):
            batch_end = min(batch_start + max_pages_per_batch, total_pages)
            
            # Extract images from batch
            for page_num in range(batch_start, batch_end):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    
                    try:
                        # Extract image
                        base_image = doc.extract_image(xref)
                        image_data = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Only process if the image is large enough (avoid small icons)
                        if len(image_data) > 10000:  # Skip tiny images
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
            
            # Memory management between batches
            memory_monitor.check()
            gc.collect()
        
        doc.close()
        return image_paths
        
    except Exception as e:
        log_error(f"Error extracting images from PDF: {str(e)}")
        return []