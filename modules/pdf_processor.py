"""
Optimized PDF Processor Module - Handles PDF parsing with OpenParse and image extraction.
"""
import tempfile
import time
import os
import re
from typing import List, Dict, Any, Optional

import streamlit as st
import openparse
import fitz  # PyMuPDF for image extraction
from streamlit.delta_generator import DeltaGenerator

from modules.utils import log_error, create_directory_if_not_exists

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Simple text chunking by character count with overlap."""
    if not text or not text.strip():
        return []
    
    text = text.strip()
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    
    return chunks

def process_uploaded_pdf(
    uploaded_file,
    chunk_size: int,
    overlap: int,
    status: Optional[DeltaGenerator] = None,
    extract_images: bool = True,
    image_output_dir: str = "extracted_images"
) -> List[Dict[str, Any]]:
    """Process PDF with OpenParse and optional image extraction."""
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
        
        # Initialize OpenParse parser
        _update_status("Initializing document parser...")
        try:
            parser = openparse.DocumentParser()
        except Exception as parser_init_e:
            log_error(f"Failed to initialize DocumentParser: {str(parser_init_e)}")
            # Fallback to PyMuPDF if OpenParse fails
            _update_status("Using fallback PDF parser...")
            return fallback_pdf_processing(tmp_file_path, uploaded_file.name, chunk_size, overlap, extract_images, image_output_dir)
        
        # Parse the document with OpenParse
        _update_status(f"Parsing {uploaded_file.name}...")
        parsed_doc = parser.parse(tmp_file_path)
        
        # Handle image extraction if requested
        if extract_images:
            _update_status("Extracting images from PDF...")
            image_paths = extract_images_from_pdf(tmp_file_path, image_output_dir, uploaded_file.name)
            if image_paths:
                _update_status(f"Extracted {len(image_paths)} images.")
        
        # Extract text nodes from OpenParse
        _update_status("Extracting text content...")
        text_data = [node.text for node in parsed_doc.nodes if node.text and node.text.strip()]
        
        if not text_data:
            _update_status("No text content found, trying fallback method...")
            # Fallback to PyMuPDF for text extraction
            return fallback_pdf_processing(tmp_file_path, uploaded_file.name, chunk_size, overlap, extract_images, image_output_dir)
        
        # Chunk the extracted text
        _update_status(f"Chunking text ({len(text_data)} sections)...")
        
        for idx, text_section in enumerate(text_data):
            section_chunks = chunk_text(text_section, chunk_size, overlap)
            
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
    """Fallback PDF processing using PyMuPDF when OpenParse fails."""
    chunks = []
    
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = ""
        
        # Extract text from all pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                full_text += text + "\n"
        
        # Extract images if requested
        if extract_images:
            extract_images_from_pdf(pdf_path, image_output_dir, filename)
        
        doc.close()
        
        # Chunk the text
        if full_text.strip():
            text_chunks = chunk_text(full_text, chunk_size, overlap)
            
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

def extract_images_from_pdf(
    pdf_path: str, 
    output_dir: str, 
    base_filename: str
) -> List[str]:
    """Extract images from PDF for tourism content analysis."""
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
        
        doc.close()
        return image_paths
        
    except Exception as e:
        log_error(f"Error extracting images from PDF: {str(e)}")
        return []