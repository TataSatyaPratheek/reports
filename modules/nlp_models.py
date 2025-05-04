# modules/nlp_models.py
"""
Optimized NLP Models module with enhanced model selection and memory management.
"""
import streamlit as st
import os
import spacy
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import subprocess
import sys
import asyncio
import torch
import gc
import re
from typing import Optional, Dict, List, Union, Tuple, Any
from modules.utils import log_error
from modules.memory_utils import (
    get_available_memory_mb, 
    get_available_gpu_memory_mb,
    memory_monitor,
    get_gpu_memory_info,
    optimize_tensor_precision
)

# Define model names as constants
SPACY_MODEL_NAME = "en_core_web_sm"
TRAVEL_ENTITIES = ["DESTINATION", "ACCOMMODATION", "TRANSPORTATION", "ACTIVITY", "ATTRACTION"]

# Enhanced embedding models hierarchy with memory requirements
EMBEDDING_MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "memory_mb": 80,
        "loader": "SentenceTransformer",
        "performance": "58.9 MTEB",
        "speed_score": 95,
        "description": "Extremely lightweight, good for basic tasks"
    },
    {
        "name": "all-MiniLM-L12-v2",
        "dimensions": 384,
        "memory_mb": 120,
        "loader": "SentenceTransformer",
        "performance": "59.8 MTEB",
        "speed_score": 90,
        "description": "Slightly better than L6, still very light"
    },
    {
        "name": "paraphrase-MiniLM-L6-v2",
        "dimensions": 384,
        "memory_mb": 90,
        "loader": "SentenceTransformer",
        "performance": "60.2 MTEB",
        "speed_score": 92,
        "description": "Optimized for semantic similarity"
    },
    {
        "name": "all-mpnet-base-v2",
        "dimensions": 768,
        "memory_mb": 420,
        "loader": "SentenceTransformer",
        "performance": "63.3 MTEB",
        "speed_score": 75,
        "description": "Good balance of performance and size"
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "memory_mb": 130,
        "loader": "SentenceTransformer",
        "performance": "62.2 MTEB",
        "speed_score": 88,
        "description": "Small BGE model, good performance"
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "dimensions": 768,
        "memory_mb": 450,
        "loader": "SentenceTransformer",
        "performance": "64.2 MTEB",
        "speed_score": 70,
        "description": "Base BGE model, better performance"
    },
    {
        "name": "dunzhang/stella_en_400M_v5",
        "dimensions": 1024,
        "memory_mb": 800,
        "loader": "SentenceTransformer",
        "performance": "65.1 MTEB",
        "speed_score": 60,
        "description": "High performance Stella model"
    },
    {
        "name": "BAAI/bge-m3",
        "dimensions": 1024,
        "memory_mb": 1200,
        "loader": "BGEM3FlagModel",
        "performance": "66.0 MTEB",
        "speed_score": 50,
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

def select_best_embedding_model(gpu_info: Dict[str, float], preferred_model: Optional[str] = None, performance_target: str = "balanced") -> Dict[str, Any]:
    """Enhanced model selection with performance profiling."""
    # Performance weights for different optimization targets
    performance_weights = {
        "low_latency": {"speed": 0.7, "memory": 0.2, "accuracy": 0.1},
        "high_accuracy": {"speed": 0.1, "memory": 0.2, "accuracy": 0.7},
        "balanced": {"speed": 0.3, "memory": 0.3, "accuracy": 0.4}
    }
    
    weights = performance_weights.get(performance_target, performance_weights["balanced"])
    
    # If user specified a model, try to use it if possible
    if preferred_model:
        for model in EMBEDDING_MODELS:
            if model["name"] == preferred_model:
                if not gpu_info["available"] or gpu_info["free_mb"] >= model["memory_mb"] * 1.2:  # 20% buffer
                    return model
                else:
                    log_error(f"Preferred model {preferred_model} requires {model['memory_mb']}MB but only {gpu_info['free_mb']}MB available")
                    break
    
    # Score and select models based on target
    scored_models = []
    for model in EMBEDDING_MODELS:
        # Skip models that don't fit in memory
        if gpu_info["available"] and model["memory_mb"] * 1.2 > gpu_info["free_mb"]:
            continue
        
        # Extract accuracy score
        accuracy_score = float(model["performance"].split()[0])
        
        # Calculate composite score
        score = (
            model["speed_score"] / 100.0 * weights["speed"] +
            (1 - model["memory_mb"] / 1500.0) * weights["memory"] +  # Normalize by max expected memory
            accuracy_score / 70.0 * weights["accuracy"]  # Normalize by max expected accuracy
        )
        
        scored_models.append((model, score))
    
    # Return best model or fallback to lightest
    if scored_models:
        return max(scored_models, key=lambda x: x[1])[0]
    
    return EMBEDDING_MODELS[0]  # Fallback to lightest model

@st.cache_resource(show_spinner=f"Loading SpaCy model ({SPACY_MODEL_NAME})...")
def load_spacy_model(components: List[str] = None):
    """Load only required SpaCy components for memory efficiency."""
    if components is None:
        components = ["ner"]  # Default to NER only
    
    try:
        # Determine which components to exclude
        all_components = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        exclude = [comp for comp in all_components if comp not in components]
        
        # Load with specific components only
        nlp = spacy.load(SPACY_MODEL_NAME, exclude=exclude)
        
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
            return spacy.load(SPACY_MODEL_NAME, exclude=exclude)
        except Exception as e:
            err_msg = f"Failed to download SpaCy model: {str(e)}"
            st.error(err_msg)
            log_error(err_msg)
            return None

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(preferred_model: Optional[str] = None, performance_target: str = "balanced"):
    """Load embedding model with enhanced intelligent selection based on available resources."""
    func_name = "load_embedding_model"
    
    try:
        # Clear memory before loading
        clear_memory()
        
        # Get GPU information
        gpu_info = get_gpu_memory_info()
        
        # Select the best model for available resources and target
        selected_model = select_best_embedding_model(gpu_info, preferred_model, performance_target)
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
        
        # Determine device and precision
        device = 'cuda' if gpu_info["available"] and gpu_info["free_mb"] > selected_model["memory_mb"] * 1.5 else 'cpu'
        use_fp16 = device == 'cuda' and gpu_info["free_mb"] > selected_model["memory_mb"] * 2
        
        # Load the model based on loader type
        if loader_type == "BGEM3FlagModel":
            model = BGEM3FlagModel(model_name, 
                                 device=device, 
                                 use_fp16=use_fp16,
                                 normalize_embeddings=True)
        else:
            model = SentenceTransformer(model_name, device=device)
            
            # Apply precision optimization
            if use_fp16:
                autocast_ctx, model = optimize_tensor_precision(model, "fp16")
            else:
                autocast_ctx, model = optimize_tensor_precision(model, "fp32")
        
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
            return load_embedding_model("all-MiniLM-L6-v2", "low_latency")
        
        return None

def extract_tourism_entities_streaming(text: str, nlp=None, chunk_size: int = 5000, overlap: int = 100) -> Dict[str, List[str]]:
    """Extract entities using a streaming approach for memory efficiency."""
    if not nlp:
        nlp = load_spacy_model(components=["ner"])
        if not nlp:
            return {entity_type: [] for entity_type in TRAVEL_ENTITIES}
    
    results = {entity_type: [] for entity_type in TRAVEL_ENTITIES}
    
    # Process text in overlapping chunks
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:min(i + chunk_size, len(text))]
        doc = nlp(chunk)
        
        # Extract entities from this chunk
        for ent in doc.ents:
            if ent.label_ == "GPE" or ent.label_ == "LOC":
                results["DESTINATION"].append(ent.text)
            elif ent.label_ == "FAC" or ent.label_ == "ORG":
                entity_text = ent.text.lower()
                for category, keywords in TOURISM_KEYWORDS.items():
                    if any(keyword in entity_text for keyword in keywords):
                        results[category].append(ent.text)
                        break
        
        # Clear spaCy document from memory
        del doc
        
        # Check memory periodically
        if i % (chunk_size * 5) == 0:
            memory_monitor.check()
    
    # Remove duplicates and sort
    for category in results:
        results[category] = sorted(list(set(results[category])))
    
    return results

def get_embedding_dimensions(embedding_model) -> int:
    """Get the dimensions of the embedding model dynamically with memory efficiency."""
    try:
        # Check if this is a known model with predefined dimensions
        model_name = getattr(embedding_model, 'model_name', None)
        if model_name:
            for model_info in EMBEDDING_MODELS:
                if model_info["name"] == model_name:
                    return model_info["dimensions"]
        
        # Fallback to dynamic detection with minimal text
        test_text = "test"
        
        if hasattr(embedding_model, 'encode'):
            output = embedding_model.encode([test_text], batch_size=1, show_progress_bar=False)
            
            if isinstance(output, dict):
                if 'dense_vecs' in output:
                    return len(output['dense_vecs'][0])
                elif 'embeddings' in output:
                    return len(output['embeddings'][0])
                elif 'sentence_embedding' in output:
                    return len(output['sentence_embedding'][0])
                else:
                    for key, value in output.items():
                        if hasattr(value, 'shape') and len(value.shape) >= 2:
                            return value.shape[1]
                    raise ValueError(f"Unknown embedding dictionary format: {output.keys()}")
            else:
                if hasattr(output, 'shape'):
                    return output.shape[1] if len(output.shape) > 1 else len(output[0])
                else:
                    return len(output[0])
        
        # Default fallback
        return 384
        
    except Exception as e:
        log_error(f"Error getting embedding dimensions: {str(e)}")
        return 384  # Default to common dimension

def calculate_text_complexity(text: str) -> Dict[str, float]:
    """Calculate text complexity metrics with memory efficiency."""
    if not text:
        return {
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "readability_score": 0
        }
    
    try:
        # Process in chunks for very large texts
        max_chars = 10000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        # Simple sentence splitting by common punctuation
        sentences = re.split(r'[.!?]+\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text)
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

@st.cache_resource(show_spinner="Selecting best embedding model...")
def get_model_selection_info():
    """Get information about available models and system resources."""
    gpu_info = get_gpu_memory_info()
    return gpu_info, EMBEDDING_MODELS

async def async_load_embedding_model(preferred_model: Optional[str] = None, performance_target: str = "balanced"):
    """Async wrapper for loading embedding model."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_embedding_model, preferred_model, performance_target)