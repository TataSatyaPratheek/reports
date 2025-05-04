"""
Enhanced NLP Models Module - Handles loading and managing NLP models.
Optimized for tourism and travel document analysis.
"""
import streamlit as st

import os
NLTK_DATA_PATH = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
import nltk
nltk.data.path = [NLTK_DATA_PATH]  # Override all other paths

import spacy
from FlagEmbedding import BGEM3FlagModel # Changed import
import subprocess
import sys
import asyncio
from nltk.tokenize import sent_tokenize, word_tokenize # Import specific functions
import torch # Import torch for CUDA operations
from typing import Optional, Dict, List, Union, Tuple, Any # Added Any
from modules.utils import log_error, PerformanceMonitor

# Define model names as constants
SPACY_MODEL_NAME = "en_core_web_sm"
# Using a more powerful embedding model for better travel domain understanding
EMBEDDING_MODEL_NAME = "dunzhang/stella_en_400M_v5"  # Stella 400M for superior embeddings and retrieval
EMBEDDING_MODEL_NAME = "BAAI/bge-m3" # Changed to BGE-M3
TRAVEL_ENTITIES = ["DESTINATION", "ACCOMMODATION", "TRANSPORTATION", "ACTIVITY", "ATTRACTION"]

# Define tourism-related keywords for entity recognition enhancements
TOURISM_KEYWORDS = {
    "DESTINATION": ["country", "city", "island", "resort", "destination", "region", "town", "village"],
    "ACCOMMODATION": ["hotel", "hostel", "airbnb", "resort", "villa", "apartment", "camping", "glamping", "lodge"],
    "TRANSPORTATION": ["flight", "train", "bus", "car rental", "taxi", "ferry", "cruise", "uber", "subway", "transit"],
    "ACTIVITY": ["tour", "excursion", "safari", "hike", "swim", "dive", "surf", "ski", "adventure", "experience"],
    "ATTRACTION": ["museum", "monument", "landmark", "beach", "mountain", "national park", "temple", "castle", "cathedral"]
}

@st.cache_resource(show_spinner="Loading NLTK resources...")
def load_nltk_resources():
    """Load required NLTK resources (punkt, wordnet, stopwords) with proper verification."""
    # Ensure NLTK data path is set (redundant with top-level but safe)
    nltk.data.path = [os.environ['NLTK_DATA']]
    if NLTK_DATA_PATH not in nltk.data.path:
        # If somehow it wasn't set by the top-level code, set it now.
        nltk.data.path = [NLTK_DATA_PATH]
        log_error(f"Re-set {NLTK_DATA_PATH} in nltk.data.path within load_nltk_resources")
        log_error(f"{NLTK_DATA_PATH} already in nltk.data.path")

    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]

    for path, name in resources:
        try:
            # First check if already present
            nltk.data.find(path)
            log_error(f"NLTK resource '{name}' found at {path}")
            continue
        except LookupError:
            log_error(f"Resource '{name}' not found. Attempting download...")

        # Try downloading twice
        for attempt in range(1, 3):
            try:
                log_error(f"Download attempt {attempt}/2 for {name}")
                # Ensure the download directory exists
                os.makedirs(NLTK_DATA_PATH, exist_ok=True)
                nltk.download(name, download_dir=NLTK_DATA_PATH, quiet=False) # Added quiet=False for visibility

                # Verify after download
                try:
                    nltk.data.find(path)
                    log_error(f"Successfully verified {name} after download")
                    break # Exit inner loop on success
                except LookupError:
                    log_error(f"Verification failed for {name} immediately after download attempt {attempt}.")
                    if attempt == 2:
                        log_error(f"Critical: {name} verification failed after 2 download attempts")
                        st.error(f"Failed to download and verify NLTK resource '{name}'. Chunking/analysis may fail.")
                        return False
                    # Continue to next attempt if not the last one

            except Exception as download_error:
                log_error(f"Download failed for {name} (attempt {attempt}): {str(download_error)}")
                if attempt == 2:
                    st.error(f"Failed to download NLTK resource '{name}' after 2 attempts: {str(download_error)}")
                    return False
                # Wait a bit before retrying? (Optional)
                # time.sleep(1)

    # Final verification pass after attempting downloads for all resources
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            log_error(f"Final verification failed for {name}. It should have been downloaded.")
            st.error(f"NLTK resource '{name}' is missing after download attempts. Please check network and permissions for {NLTK_DATA_PATH}.")
            return False

    return True

@st.cache_resource(show_spinner=f"Loading SpaCy model ({SPACY_MODEL_NAME})...")
def load_spacy_model():
    """Load spaCy model with error handling and download attempt."""
    try:
        # Load the model
        nlp = spacy.load(SPACY_MODEL_NAME)

        # Add tourism-specific entity types
        if "ner" in nlp.pipe_names:
            # Only modify if model has NER component
            try:
                for entity_type in TRAVEL_ENTITIES:
                    if entity_type not in nlp.pipe("").ents:
                        nlp.get_pipe("ner").add_label(entity_type)
                st.success(f"Enhanced NER with tourism-specific entity types.")
            except Exception as ner_err:
                log_error(f"Note: Could not enhance NER component: {str(ner_err)}")
                # Not critical, so continue with base model
        
        return nlp
    except OSError:
        st.warning(f"SpaCy model '{SPACY_MODEL_NAME}' not found locally.")
        st.info(f"Attempting to download '{SPACY_MODEL_NAME}'...")
        try:
            result = subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL_NAME],
                                 check=True, capture_output=True, text=True, timeout=300)
            st.success(f"Successfully downloaded '{SPACY_MODEL_NAME}'.")
            return spacy.load(SPACY_MODEL_NAME)
        except subprocess.CalledProcessError as e:
            err_msg = f"Failed to download SpaCy model '{SPACY_MODEL_NAME}'. Error: {e.stderr}"
            st.error(err_msg)
            log_error(err_msg)
            return None
        except Exception as e_inner:
            err_msg = f"Failed during SpaCy model download/load: {str(e_inner)}"
            st.error(err_msg)
            log_error(err_msg)
            return None
    except Exception as e_outer:
        err_msg = f"Unexpected error loading SpaCy model '{SPACY_MODEL_NAME}': {str(e_outer)}"
        st.error(err_msg)
        log_error(err_msg)
        return None

@st.cache_resource(show_spinner=f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
def load_embedding_model():
    """Load sentence transformer model with error handling."""
    func_name = "load_embedding_model"
    try:
        # --- Immediate Fix 2: Clear CUDA cache before loading ---
        if torch.cuda.is_available():
            log_error(f"{func_name}: Clearing CUDA cache...")
            torch.cuda.empty_cache()
            log_error(f"{func_name}: CUDA cache cleared.")

        log_error(f"{func_name}: Loading SentenceTransformer '{EMBEDDING_MODEL_NAME}'...")
        # Load the model, specifying device if CUDA is available
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        use_fp16: bool = (device == 'cuda') # Use FP16 only if on GPU
        model: Any = BGEM3FlagModel(EMBEDDING_MODEL_NAME, device=device, use_fp16=use_fp16) # Changed loader
        log_error(f"{func_name}: Loaded {EMBEDDING_MODEL_NAME} on {device} with use_fp16={use_fp16}")
        st.success(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")

        # --- Critical Check: Print memory summary ---
        if device == 'cuda':
            log_error(f"{func_name}: CUDA Memory Summary after loading:")
            log_error(torch.cuda.memory_summary())

        return model
    except ImportError:
        err_msg = "Sentence Transformers library not installed. Cannot load embedding model."
        st.error(err_msg)
        log_error(err_msg)
        return None
    except OSError as e:
         err_msg = f"OS error loading embedding model '{EMBEDDING_MODEL_NAME}': {str(e)}. Check network connection and cache permissions."
         st.error(err_msg)
         log_error(err_msg)
         return None
    except Exception as e:
        err_msg = f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        return None

async def async_load_embedding_model():
    """Async wrapper for loading embedding model."""
    # This is a simple wrapper for now but could be enhanced for true async loading
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_embedding_model)

def extract_tourism_entities(text: str, nlp=None) -> Dict[str, List[str]]:
    """
    Extract tourism-related entities from text using spaCy and keyword matching.
    
    Args:
        text: Input text to analyze
        nlp: Optional SpaCy model (will load if not provided)
        
    Returns:
        Dictionary of entities by category
    """
    if not nlp:
        nlp = load_spacy_model()
        if not nlp:
            return {entity_type: [] for entity_type in TRAVEL_ENTITIES}
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize results dictionary
    results = {entity_type: [] for entity_type in TRAVEL_ENTITIES}
    
    # Extract entities from spaCy NER
    for ent in doc.ents:
        # Map spaCy entities to our tourism categories
        if ent.label_ == "GPE" or ent.label_ == "LOC":
            results["DESTINATION"].append(ent.text)
        elif ent.label_ == "FAC" or ent.label_ == "ORG":
            # Could be accommodation, attraction or transportation
            # We'll use keyword matching to disambiguate
            entity_text = ent.text.lower()
            if any(keyword in entity_text for keyword in TOURISM_KEYWORDS["ACCOMMODATION"]):
                results["ACCOMMODATION"].append(ent.text)
            elif any(keyword in entity_text for keyword in TOURISM_KEYWORDS["ATTRACTION"]):
                results["ATTRACTION"].append(ent.text)
            elif any(keyword in entity_text for keyword in TOURISM_KEYWORDS["TRANSPORTATION"]):
                results["TRANSPORTATION"].append(ent.text)
    
    # Additional keyword-based extraction
    for category, keywords in TOURISM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text.lower():
                # Find the full term containing the keyword
                # This is a simple approach - could be enhanced with more sophisticated extraction
                start = text.lower().find(keyword)
                if start >= 0:
                    # Get surrounding words for context
                    context_start = max(0, start - 20)
                    context_end = min(len(text), start + len(keyword) + 20)
                    context = text[context_start:context_end]
                    
                    # Extract the term using NLP (could be improved)
                    term = keyword
                    for token in nlp(context):
                        if keyword in token.text.lower() and token.text not in results[category]:
                            results[category].append(token.text)
    
    # Remove duplicates and sort
    for category in results:
        results[category] = sorted(list(set(results[category])))
    
    return results

def calculate_text_complexity(text: str) -> Dict[str, float]:
    """
    Calculate text complexity metrics useful for tourism content.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary of complexity metrics
    """
    if not text:
        return {
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "readability_score": 0
        }
    
    try: # Start of the main try block
        try: # Inner try for NLTK check/download
            sent_tokenize("Test sentence.") # Force punkt check
        except LookupError:
            log_error("NLTK 'punkt' not found during complexity calc, attempting download...")
            nltk.download('punkt', download_dir=os.environ['NLTK_DATA'])
            nltk.data.path = [os.environ['NLTK_DATA']] # Re-set path
            log_error("NLTK 'punkt' downloaded, proceeding with complexity calc.")

        # Now use tokenizers after the check
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
        except LookupError as nltk_err:
            log_error(f"NLTK resource missing during complexity calculation: {nltk_err}. Returning default complexity.")
            return {"avg_word_length": 0, "avg_sentence_length": 0, "readability_score": 0}

        # Filter out punctuation
        words = [word for word in words if word.isalpha()]
        
        if not words or not sentences:
            return {
                "avg_word_length": 0,
                "avg_sentence_length": 0,
                "readability_score": 0
            }
        
        # Calculate metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability score (approximation of Flesch Reading Ease)
        # Higher score = easier to read
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 5)
        readability_score = max(0, min(100, readability_score))  # Clamp between 0-100
        
        return {
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "readability_score": round(readability_score, 2)
        }
    except Exception as e: # This except block is now correctly indented within the main try block
        log_error(f"Error calculating text complexity: {str(e)}")
        return {
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "readability_score": 0
        }