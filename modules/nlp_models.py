# Adaptive nlp_models.py with intelligent model selection

import streamlit as st
import os
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import subprocess
import sys
import asyncio
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import gc
from typing import Optional, Dict, List, Union, Tuple, Any
from modules.utils import log_error, PerformanceMonitor

# Set NLTK data path
NLTK_DATA_PATH = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
nltk.data.path = [NLTK_DATA_PATH]

# Define model names as constants
SPACY_MODEL_NAME = "en_core_web_sm"
TRAVEL_ENTITIES = ["DESTINATION", "ACCOMMODATION", "TRANSPORTATION", "ACTIVITY", "ATTRACTION"]

# Embedding models hierarchy with memory requirements
EMBEDDING_MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "memory_mb": 80,
        "loader": "SentenceTransformer",
        "performance": "58.9 MTEB",
        "description": "Extremely lightweight, good for basic tasks"
    },
    {
        "name": "all-MiniLM-L12-v2",
        "dimensions": 384,
        "memory_mb": 120,
        "loader": "SentenceTransformer",
        "performance": "59.8 MTEB",
        "description": "Slightly better than L6, still very light"
    },
    {
        "name": "paraphrase-MiniLM-L6-v2",
        "dimensions": 384,
        "memory_mb": 90,
        "loader": "SentenceTransformer",
        "performance": "60.2 MTEB",
        "description": "Optimized for semantic similarity"
    },
    {
        "name": "all-mpnet-base-v2",
        "dimensions": 768,
        "memory_mb": 420,
        "loader": "SentenceTransformer",
        "performance": "63.3 MTEB",
        "description": "Good balance of performance and size"
    },
    {
        "name": "sentence-transformers/all-distilroberta-v1",
        "dimensions": 768,
        "memory_mb": 350,
        "loader": "SentenceTransformer",
        "performance": "61.0 MTEB",
        "description": "DistilRoBERTa-based, efficient"
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "memory_mb": 130,
        "loader": "SentenceTransformer",
        "performance": "62.2 MTEB",
        "description": "Small BGE model, good performance"
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "dimensions": 768,
        "memory_mb": 450,
        "loader": "SentenceTransformer",
        "performance": "64.2 MTEB",
        "description": "Base BGE model, better performance"
    },
    {
        "name": "dunzhang/stella_en_400M_v5",
        "dimensions": 1024,
        "memory_mb": 800,
        "loader": "SentenceTransformer",
        "performance": "65.1 MTEB",
        "description": "High performance Stella model"
    },
    {
        "name": "BAAI/bge-m3",
        "dimensions": 1024,
        "memory_mb": 1200,
        "loader": "BGEM3FlagModel",
        "performance": "66.0 MTEB",
        "description": "Most powerful, multi-lingual BGE"
    }
]

# Tourism keywords dictionary
TOURISM_KEYWORDS = {
    "DESTINATION": ["country", "city", "island", "resort", "destination", "region", "town", "village"],
    "ACCOMMODATION": ["hotel", "hostel", "airbnb", "resort", "villa", "apartment", "camping", "glamping", "lodge"],
    "TRANSPORTATION": ["flight", "train", "bus", "car rental", "taxi", "ferry", "cruise", "uber", "subway", "transit"],
    "ACTIVITY": ["tour", "excursion", "safari", "hike", "swim", "dive", "surf", "ski", "adventure", "experience"],
    "ATTRACTION": ["museum", "monument", "landmark", "beach", "mountain", "national park", "temple", "castle", "cathedral"]
}

def clear_memory():
    """Helper function to clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information if available."""
    if not torch.cuda.is_available():
        return {"available": False, "total_mb": 0, "free_mb": 0, "used_mb": 0}
    
    try:
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory / (1024 * 1024)  # Convert to MB
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 * 1024)
        free_memory = total_memory - reserved_memory
        
        return {
            "available": True,
            "device_name": gpu_properties.name,
            "total_mb": total_memory,
            "free_mb": free_memory,
            "used_mb": allocated_memory,
            "reserved_mb": reserved_memory
        }
    except Exception as e:
        log_error(f"Error getting GPU memory info: {str(e)}")
        return {"available": False, "total_mb": 0, "free_mb": 0, "used_mb": 0}

def get_embedding_dimensions(embedding_model) -> int:
    """Get the dimensions of the embedding model dynamically."""
    try:
        # Check if this is a known model with predefined dimensions
        model_name = getattr(embedding_model, 'model_name', None)
        if model_name:
            for model_info in EMBEDDING_MODELS:
                if model_info["name"] == model_name:
                    return model_info["dimensions"]
        
        # Fallback to dynamic detection
        test_text = "test sentence for dimension check"
        
        # Handle different model types
        if hasattr(embedding_model, 'encode'):
            output = embedding_model.encode([test_text])
            
            # Handle different output formats
            if isinstance(output, dict):
                # BGE-M3 style output
                if 'dense_vecs' in output:
                    return len(output['dense_vecs'][0])
                elif 'embeddings' in output:
                    return len(output['embeddings'][0])
                elif 'sentence_embedding' in output:
                    return len(output['sentence_embedding'][0])
                else:
                    # Try to find any tensor-like value in the dict
                    for key, value in output.items():
                        if hasattr(value, 'shape') and len(value.shape) >= 2:
                            return value.shape[1]
                    raise ValueError(f"Unknown embedding dictionary format: {output.keys()}")
            else:
                # Standard numpy array or tensor output
                if hasattr(output, 'shape'):
                    return output.shape[1] if len(output.shape) > 1 else len(output[0])
                else:
                    return len(output[0])
        
        # If we can't determine dimensions, return default
        log_error(f"Could not determine embedding dimensions for model type: {type(embedding_model)}")
        return 384  # Default to common dimension
        
    except Exception as e:
        log_error(f"Error getting embedding dimensions: {str(e)}")
        return 384  # Default to common dimension

def select_best_embedding_model(gpu_info: Dict[str, float], preferred_model: Optional[str] = None) -> Dict[str, Any]:
    """Select the best embedding model based on available resources."""
    # If user specified a model, try to use it if possible
    if preferred_model:
        for model in EMBEDDING_MODELS:
            if model["name"] == preferred_model:
                if not gpu_info["available"] or gpu_info["free_mb"] >= model["memory_mb"] * 1.2:  # 20% buffer
                    return model
                else:
                    log_error(f"Preferred model {preferred_model} requires {model['memory_mb']}MB but only {gpu_info['free_mb']}MB available")
                    break
    
    # If no GPU or in CPU mode, use the lightest model
    if not gpu_info["available"]:
        return EMBEDDING_MODELS[0]  # all-MiniLM-L6-v2
    
    # Select the best model that fits in available memory (with 20% buffer)
    selected_model = EMBEDDING_MODELS[0]  # Default to lightest
    
    for model in EMBEDDING_MODELS:
        required_memory = model["memory_mb"] * 1.2  # Add 20% buffer
        if gpu_info["free_mb"] >= required_memory:
            selected_model = model
        else:
            break  # Models are ordered by size, so stop if one doesn't fit
    
    return selected_model

@st.cache_resource(show_spinner="Loading NLTK resources...")
def load_nltk_resources():
    """Load required NLTK resources with proper verification."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
            log_error(f"NLTK resource '{name}' found at {path}")
        except LookupError:
            log_error(f"Resource '{name}' not found. Attempting download...")
            for attempt in range(1, 3):
                try:
                    os.makedirs(NLTK_DATA_PATH, exist_ok=True)
                    nltk.download(name, download_dir=NLTK_DATA_PATH, quiet=False)
                    
                    # Verify after download
                    nltk.data.find(path)
                    log_error(f"Successfully verified {name} after download")
                    break
                except Exception as e:
                    log_error(f"Download attempt {attempt} failed for {name}: {str(e)}")
                    if attempt == 2:
                        st.error(f"Failed to download NLTK resource '{name}'")
                        return False
    return True

@st.cache_resource(show_spinner=f"Loading SpaCy model ({SPACY_MODEL_NAME})...")
def load_spacy_model():
    """Load spaCy model with memory-efficient settings."""
    try:
        # Load with memory optimizations
        nlp = spacy.load(SPACY_MODEL_NAME, 
                        exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        
        # Add custom components if needed
        if "ner" in nlp.pipe_names:
            try:
                for entity_type in TRAVEL_ENTITIES:
                    nlp.get_pipe("ner").add_label(entity_type)
            except Exception as ner_err:
                log_error(f"Could not enhance NER component: {str(ner_err)}")
        
        return nlp
    
    except OSError:
        st.warning(f"SpaCy model '{SPACY_MODEL_NAME}' not found locally.")
        st.info(f"Attempting to download '{SPACY_MODEL_NAME}'...")
        try:
            result = subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL_NAME],
                                 check=True, capture_output=True, text=True, timeout=300)
            st.success(f"Successfully downloaded '{SPACY_MODEL_NAME}'.")
            return spacy.load(SPACY_MODEL_NAME)
        except Exception as e:
            err_msg = f"Failed to download SpaCy model: {str(e)}"
            st.error(err_msg)
            log_error(err_msg)
            return None

@st.cache_resource(show_spinner="Selecting best embedding model...")
def get_model_selection_info():
    """Get information about available models and system resources."""
    gpu_info = get_gpu_memory_info()
    return gpu_info, EMBEDDING_MODELS

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(preferred_model: Optional[str] = None):
    """Load embedding model with intelligent selection based on available resources."""
    func_name = "load_embedding_model"
    
    try:
        # Clear memory before loading
        clear_memory()
        
        # Get GPU information
        gpu_info = get_gpu_memory_info()
        
        # Select the best model for available resources
        selected_model = select_best_embedding_model(gpu_info, preferred_model)
        model_name = selected_model["name"]
        loader_type = selected_model["loader"]
        
        log_error(f"{func_name}: GPU Info - Available: {gpu_info['available']}, Free: {gpu_info['free_mb']}MB")
        log_error(f"{func_name}: Selected model: {model_name} (requires {selected_model['memory_mb']}MB)")
        
        # Show selection info to user
        if gpu_info["available"]:
            st.info(f"🔍 GPU detected: {gpu_info['device_name']} with {gpu_info['free_mb']:.0f}MB free memory")
        else:
            st.info("🔍 No GPU detected, using CPU mode")
        
        st.info(f"📊 Selected embedding model: **{model_name}**")
        st.info(f"   • Performance: {selected_model['performance']}")
        st.info(f"   • Description: {selected_model['description']}")
        st.info(f"   • Memory requirement: {selected_model['memory_mb']}MB")
        
        # Load the model based on loader type
        if loader_type == "BGEM3FlagModel":
            # Use BGE-M3 specific loader
            device = 'cuda' if gpu_info["available"] and gpu_info["free_mb"] > selected_model["memory_mb"] * 1.5 else 'cpu'
            use_fp16 = device == 'cuda'
            
            model = BGEM3FlagModel(model_name, 
                                 device=device, 
                                 use_fp16=use_fp16,
                                 normalize_embeddings=True)
        else:
            # Use SentenceTransformer loader
            device = 'cuda' if gpu_info["available"] and gpu_info["free_mb"] > selected_model["memory_mb"] * 1.5 else 'cpu'
            model = SentenceTransformer(model_name, device=device)
        
        # Store model name in the model object
        model.model_name = model_name
        
        log_error(f"{func_name}: Successfully loaded {model_name} on {device}")
        st.success(f"✅ Embedding model '{model_name}' loaded successfully on {device}")
        
        # Store the model info in session state
        if 'selected_embedding_model' not in st.session_state:
            st.session_state.selected_embedding_model = model_name
        
        # Clear memory after loading
        clear_memory()
        
        return model
    
    except Exception as e:
        err_msg = f"Failed to load embedding model: {str(e)}"
        st.error(err_msg)
        log_error(err_msg)
        
        # Try fallback to lightest model
        if 'tried_fallback' not in st.session_state:
            st.session_state.tried_fallback = True
            st.warning("Attempting to load fallback model (all-MiniLM-L6-v2)...")
            return load_embedding_model("all-MiniLM-L6-v2")
        
        return None

def extract_tourism_entities(text: str, nlp=None) -> Dict[str, List[str]]:
    """Extract tourism-related entities from text with memory efficiency."""
    if not nlp:
        nlp = load_spacy_model()
        if not nlp:
            return {entity_type: [] for entity_type in TRAVEL_ENTITIES}
    
    # Process in smaller chunks for memory efficiency
    MAX_TEXT_LENGTH = 100000  # Limit text length
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    doc = nlp(text)
    results = {entity_type: [] for entity_type in TRAVEL_ENTITIES}
    
    # Extract entities from spaCy NER
    for ent in doc.ents:
        if ent.label_ == "GPE" or ent.label_ == "LOC":
            results["DESTINATION"].append(ent.text)
        elif ent.label_ == "FAC" or ent.label_ == "ORG":
            entity_text = ent.text.lower()
            for category, keywords in TOURISM_KEYWORDS.items():
                if any(keyword in entity_text for keyword in keywords):
                    results[category].append(ent.text)
                    break
    
    # Remove duplicates and sort
    for category in results:
        results[category] = sorted(list(set(results[category])))
    
    return results

def calculate_text_complexity(text: str) -> Dict[str, float]:
    """Calculate text complexity metrics with error handling."""
    if not text:
        return {
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "readability_score": 0
        }
    
    try:
        # Tokenize text
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
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
        
        # Simple readability score
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 5)
        readability_score = max(0, min(100, readability_score))
        
        return {
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "readability_score": round(readability_score, 2)
        }
    except Exception as e:
        log_error(f"Error calculating text complexity: {str(e)}")
        return {
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "readability_score": 0
        }

async def async_load_embedding_model(preferred_model: Optional[str] = None):
    """Async wrapper for loading embedding model."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_embedding_model, preferred_model)