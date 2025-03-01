"""
NLP Models Module - Handles loading and managing NLP models.
"""
import streamlit as st
import nltk
import spacy
from sentence_transformers import SentenceTransformer
import subprocess
import sys

@st.cache_resource(show_spinner=False)
def load_nltk_resources():
    """Load required NLTK resources."""
    try:
        nltk.download("punkt", quiet=True)
    except Exception as e:
        st.warning(f"Failed to download NLTK resources: {str(e)}")

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    """Load spaCy model with error handling."""
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Failed to load spaCy model: {str(e)}")
        st.info("Attempting to download spaCy model...")
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return spacy.load("en_core_web_sm")
        except Exception as e2:
            st.error(f"Failed to download spaCy model: {str(e2)}")
            return None

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load sentence transformer model with error handling."""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        return None