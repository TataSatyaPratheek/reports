import streamlit as st
import time
import sys
from typing import Tuple, Dict, Any, List, Optional

# Import modules
from modules.system_setup import (
    ensure_dependencies, setup_ollama, refresh_available_models,
    install_package, download_model, DEFAULT_MODEL_NAME, TOURISM_RECOMMENDED_MODELS
)
from modules.vector_store import initialize_vector_db, reset_vector_db, get_chroma_collection, hybrid_retrieval
from modules.embedding_service import load_embedding_model, get_embedding_dimensions
from modules.pdf_processor import process_uploaded_pdf
from modules.vector_store import add_chunks_to_collection
from modules.llm_interface import query_llm, SlidingWindowMemory
from modules.ui_components import display_chat, show_system_resources, apply_tourism_theme, display_tourism_entities, render_tourism_dashboard
from modules.utils import log_error, TourismLogger, extract_tourism_metrics_from_text

# Initialize logger
logger = TourismLogger()

# --- Constants and AGENT_ROLES ---
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64
DEFAULT_TOP_N = 5
DEFAULT_CONVERSATION_MEMORY = 3
DEFAULT_HYBRID_ALPHA = 0.7  # Weight balance between vector and BM25 search

# Tourism agent roles with specialized system prompts
AGENT_ROLES = {
    "Travel Trends Analyst": "You are an expert travel trends analyst. Focus on identifying macro trends in the travel industry, emerging destinations, changing consumer preferences, and industry forecasts. Provide data-driven insights when available and contextualize trends within broader economic and social patterns. Reference specific metrics, percentages, and growth figures when present in the documents.",
    
    "Payment Specialist": "You are a payment systems specialist focused on the tourism sector. Your expertise is in analyzing how different payment methods are used across various travel segments. Highlight differences in payment preferences between demographics, regions, and travel types. Provide specific details on adoption rates, transaction volumes, and emerging payment technologies in the travel space.",
    
    "Market Segmentation Expert": "You are a tourism market segmentation expert. Your role is to help analyze different customer segments in the travel industry based on demographics, attitudes, motivations, destinations, and other factors. Identify distinct characteristics of each segment, their preferences, spending patterns, and how they can be effectively targeted. Provide strategic insights for positioning tourism offerings to specific segments.",
    
    "Sustainability Tourism Advisor": "You are a sustainability tourism advisor. Focus on ecological and social sustainability practices in the travel industry. Analyze trends in sustainable tourism, consumer demand for eco-friendly options, certification standards, and the business case for sustainability. Highlight innovations, best practices, and the impact of sustainability initiatives on different travel segments.",
    
    "Gen Z Travel Specialist": "You are a Gen Z travel specialist. Your expertise is in understanding the unique travel preferences, behaviors, and expectations of Generation Z travelers (born 1997-2012). Analyze their digital behaviors, spending patterns, destination preferences, and how they differ from other generations. Provide insights on effectively engaging with this demographic through appropriate channels and experiences.",
    
    "Luxury Tourism Consultant": "You are a luxury tourism consultant. Focus on the high-end travel market, analyzing trends, consumer expectations, and service standards in luxury travel. Provide insights on spending patterns, exclusive experiences, personalization expectations, and how luxury travel is evolving. Highlight distinctions between traditional luxury and emerging premium concepts in the travel space.",
    
    "Tourism Analytics Expert": "You are a tourism analytics expert with deep knowledge of travel industry data and metrics. Analyze visitor statistics, revenue figures, booking patterns, customer journey data, and market trends. Present quantitative insights clearly and provide context for interpreting tourism performance indicators. Help translate data into actionable business recommendations for tourism stakeholders.",
    
    "General Tourism Assistant": "You are a helpful tourism information assistant. Provide clear, accurate, and balanced information about travel topics based on the provided document context. Offer insights on destinations, travel planning, industry trends, and tourism services. Present information in an accessible manner suitable for both industry professionals and travelers. Focus on presenting factual information rather than personal opinions."
}

def initialize_session_state():
    """Initialize or reset session state variables with tourism focus."""
    st.session_state.setdefault("processed_files", set())
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("sliding_window_memory", SlidingWindowMemory(max_tokens=2048))
    st.session_state.setdefault("system_initialized", False)
    st.session_state.setdefault("initialization_complete", False)
    st.session_state.setdefault("initialization_status", "Pending")
    st.session_state.setdefault("error_log", [])
    st.session_state.setdefault("available_models", [DEFAULT_MODEL_NAME])
    st.session_state.setdefault("current_agent_role", "Travel Trends Analyst")
    st.session_state.setdefault("custom_prompt", AGENT_ROLES.get("Travel Trends Analyst", ""))
    st.session_state.setdefault("permissions", {
        "allow_system_check": True,
        "allow_package_install": False,
        "allow_ollama_install": False,
        "allow_model_download": False
    })
    
    # Feature flags
    st.session_state.setdefault("use_hybrid_retrieval", True)
    st.session_state.setdefault("use_reranker", True)
    st.session_state.setdefault("hybrid_alpha", DEFAULT_HYBRID_ALPHA)
    
    # Tourism-specific session state
    st.session_state.setdefault("extracted_tourism_entities", {})
    st.session_state.setdefault("tourism_metrics", {})
    st.session_state.setdefault("show_tourism_dashboard", False)

def main():
    """Main application entry point for Tourism RAG Chatbot."""
    # Set page config
    st.set_page_config(
        page_title="Tourism Insights ChatBot",
        page_icon="🏝️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply tourism-themed styling
    apply_tourism_theme()
    
    # Page header with tourism branding
    st.markdown('<p class="main-header">🌍 Tourism Insights Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze travel trends, market segments, payment methods, and sustainability with AI-powered document analysis</p>', unsafe_allow_html=True)

    # Initialize session state
    if "system_initialized" not in st.session_state:
        initialize_session_state()

    # Render sidebar with tourism focus
    sidebar_params = render_tourism_sidebar()

    # Main content area
    if not sidebar_params["system_initialized"]:
        st.warning("System not ready. Please initialize the system via the sidebar.")
    else:
        # Process tourism documents
        process_tourism_documents(
            sidebar_params["uploaded_files"],
            sidebar_params["chunk_size"],
            sidebar_params["overlap"]
        )
        
        # Show tourism dashboard if enabled
        if st.session_state.get("show_tourism_dashboard", False):
            display_tourism_insights()
        
        # Render chat interface with tourism analysis
        render_tourism_chat_interface(sidebar_params)

def render_tourism_sidebar():
    """Render the sidebar UI with tourism-specific elements."""
    with st.sidebar:
        st.markdown("## 🏝️ Tourism Explorer Settings")
        st.markdown("---")
        
        # --- Initialization Block ---
        if not st.session_state.initialization_complete:
            st.markdown("### 🚀 System Setup")
            st.info("System requires initialization before analyzing tourism documents.")
                
            st.markdown("**Permissions:**")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.permissions["allow_package_install"] = st.checkbox(
                    "Allow Package Install",
                    value=st.session_state.permissions["allow_package_install"]
                )
            with col2:
                st.session_state.permissions["allow_ollama_install"] = st.checkbox(
                    "Allow Ollama Install",
                    value=st.session_state.permissions["allow_ollama_install"]
                )
                st.session_state.permissions["allow_model_download"] = st.checkbox(
                    "Allow Model Download",
                    value=st.session_state.permissions["allow_model_download"]
                )
                
            if st.button("Initialize Tourism Explorer", key="init_button", use_container_width=True, type="primary"):
                with st.status("Initializing tourism analysis system...", expanded=True) as status:
                    success, message = initialize_system()
                    if success:
                        status.update(label=message, state="complete", expanded=False)
                        st.success("Tourism Explorer initialized successfully!")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        status.update(label="Initialization Failed", state="error", expanded=True)
                        st.error(f"Details: {message}")
            
            st.markdown("---")

        # --- Initialized Sections ---
        if st.session_state.system_initialized:
            with st.expander("📊 System Resources", expanded=False):
                show_system_resources()
                
            st.markdown("---")
            st.markdown("### 📑 Tourism Documents")
            uploaded_files = st.file_uploader(
                "Upload Tourism Documents",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload travel brochures, market reports, research papers, or other tourism-related PDFs."
            )
            
            # --- Tourism Analysis Options ---
            st.markdown("---")
            st.markdown("### 🔍 Tourism Analysis")
            
            # Toggle dashboard view
            st.session_state.show_tourism_dashboard = st.toggle(
                "Show Tourism Dashboard",
                value=st.session_state.get("show_tourism_dashboard", False),
                help="Display visual dashboard with tourism insights from analyzed documents."
            )
            
            # --- Tourism Expertise Selection ---
            selected_role = st.selectbox(
                "Tourism Expertise",
                options=list(AGENT_ROLES.keys()),
                index=list(AGENT_ROLES.keys()).index(st.session_state.current_agent_role)
            )
            
            if selected_role != st.session_state.current_agent_role:
                st.session_state.current_agent_role = selected_role
                st.session_state.custom_prompt = AGENT_ROLES.get(selected_role, "")
                st.rerun()
            
            # --- Tourism LLM Model ---
            with st.expander("🤖 AI Model", expanded=False):
                st.session_state.local_llm_model = st.selectbox(
                    "Select Tourism AI Model",
                    options=st.session_state.available_models,
                    index=0
                )
            
            # --- Advanced Settings ---
            with st.expander("⚙️ Analysis Settings", expanded=False):
                st.session_state.chunk_size = st.slider(
                    "Document Chunk Size",
                    256, 1024, DEFAULT_CHUNK_SIZE, 128
                )
                
                st.session_state.overlap = st.slider(
                    "Context Overlap",
                    0, 128, DEFAULT_OVERLAP, 16
                )
                
                st.session_state.top_n = st.slider(
                    "Search Results",
                    1, 20, DEFAULT_TOP_N
                )
                
            # --- Advanced Retrieval Features ---
            with st.expander("🧠 Advanced Features", expanded=False):
                st.session_state.use_hybrid_retrieval = st.toggle(
                    "Hybrid Search",
                    value=st.session_state.get("use_hybrid_retrieval", True),
                    help="Combine vector and keyword search for better tourism content retrieval."
                )
                
                if st.session_state.use_hybrid_retrieval:
                    st.session_state.hybrid_alpha = st.slider(
                        "Semantic/Keyword Balance",
                        0.0, 1.0, DEFAULT_HYBRID_ALPHA, 0.1
                    )
                
                st.session_state.use_reranker = st.toggle(
                    "Neural Reranking",
                    value=st.session_state.get("use_reranker", True),
                    help="Use AI to improve the ranking of tourism search results."
                )
            
            # --- Reset Database Section ---
            with st.expander("🗑️ Reset Database", expanded=False):
                if st.button("Reset Database", key="reset_db", use_container_width=True):
                    success, message = reset_vector_db()
                    if success:
                        st.success(message)
                        st.session_state.processed_files.clear()
                        st.session_state.messages = []
                        st.session_state.extracted_tourism_entities = {}
                        st.session_state.tourism_metrics = {}
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)

    # --- Return Params ---
    return {
        "uploaded_files": uploaded_files if 'uploaded_files' in locals() else None,
        "chunk_size": st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
        "overlap": st.session_state.get("overlap", DEFAULT_OVERLAP),
        "top_n": st.session_state.get("top_n", DEFAULT_TOP_N),
        "local_llm_model": st.session_state.get("local_llm_model", DEFAULT_MODEL_NAME),
        "system_prompt": st.session_state.get("custom_prompt", ""),
        "system_initialized": st.session_state.get("system_initialized", False),
        "use_hybrid_retrieval": st.session_state.get("use_hybrid_retrieval", True),
        "use_reranker": st.session_state.get("use_reranker", True),
        "hybrid_alpha": st.session_state.get("hybrid_alpha", DEFAULT_HYBRID_ALPHA)
    }

def process_tourism_documents(uploaded_files, chunk_size, overlap):
    """Process uploaded tourism documents with enhanced analysis."""
    if not uploaded_files:
        return

    files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if not files_to_process:
        return

    st.markdown("---")
    st.subheader("📝 Tourism Document Processing")

    embedding_model = load_embedding_model()
    collection = get_chroma_collection()

    if not embedding_model or not collection:
        st.error("Core components not available for processing.")
        return

    all_tourism_entities = {
        "DESTINATION": set(),
        "ACCOMMODATION": set(),
        "TRANSPORTATION": set(),
        "ACTIVITY": set(),
        "ATTRACTION": set()
    }
    
    tourism_metrics = {
        "segments": {},
        "payment_methods": {},
        "sustainability": {},
        "trends": {}
    }
    
    for pdf_file in files_to_process:
        with st.status(f"Analyzing tourism document: {pdf_file.name}...", expanded=True) as status:
            try:
                # Process with image extraction
                chunks = process_uploaded_pdf(
                    pdf_file, 
                    chunk_size, 
                    overlap, 
                    status=status,
                    extract_images=True
                )
                
                if chunks:
                    # Extract tourism metrics from chunks
                    for chunk in chunks:
                        metrics = extract_tourism_metrics_from_text(chunk["text"])
                        
                        # Process payment information
                        payment_keywords = ["credit card", "debit card", "cash", "digital wallet", 
                                          "mobile payment", "cryptocurrency"]
                        
                        for payment in payment_keywords:
                            if payment in chunk["text"].lower():
                                tourism_metrics["payment_methods"][payment] = tourism_metrics["payment_methods"].get(payment, 0) + 1
                    
                    # Add to collection
                    success = add_chunks_to_collection(
                        [c["text"] for c in chunks], 
                        embedding_model, 
                        collection, 
                        status=status
                    )
                    
                    if success:
                        st.session_state.processed_files.add(pdf_file.name)
                        status.update(label=f"Successfully analyzed: {pdf_file.name}", state="complete")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                status.update(label=f"Error processing {pdf_file.name}", state="error")
    
    # Save metrics to session state
    st.session_state.tourism_metrics = tourism_metrics

def display_tourism_insights():
    """Display tourism insights dashboard."""
    st.markdown("---")
    st.subheader("📊 Tourism Insights Dashboard")
    
    if st.session_state.get("extracted_tourism_entities"):
        st.markdown("### 🔎 Extracted Tourism Entities")
        display_tourism_entities(st.session_state.extracted_tourism_entities)
    
    if st.session_state.get("tourism_metrics"):
        st.markdown("### 📈 Tourism Market Analysis")
        render_tourism_dashboard(st.session_state.tourism_metrics)
    else:
        st.info("Process tourism documents to generate insights.")

def render_tourism_chat_interface(params):
    """Render the tourism-focused chat interface."""
    st.markdown("---")
    st.subheader("💬 Tourism Insights Chat")
    
    display_chat(
        st.session_state.messages,
        current_role=st.session_state.get("current_agent_role", "Tourism Assistant")
    )
    
    user_query = st.chat_input(
        "Ask about travel trends, market segments, payment methods...",
        disabled=not bool(st.session_state.processed_files)
    )
    
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        embedding_model = load_embedding_model()
        collection = get_chroma_collection()
        
        if embedding_model and collection:
            try:
                answer = query_llm(
                    user_query=user_query,
                    top_n=params["top_n"],
                    local_llm_model=params["local_llm_model"],
                    embedding_model=embedding_model,
                    collection=collection,
                    system_prompt=params["system_prompt"],
                    use_hybrid_retrieval=params["use_hybrid_retrieval"],
                    hybrid_alpha=params["hybrid_alpha"],
                    use_reranker=params["use_reranker"]
                )
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
        
        st.rerun()

def initialize_system():
    """Initialize system components for tourism analysis."""
    logger.info("Starting tourism system initialization...")
    
    try:
        # Check dependencies
        mismatched = ensure_dependencies()
        if mismatched and st.session_state.permissions["allow_package_install"]:
            for pkg, _, _ in mismatched:
                if not install_package(pkg):
                    return False, f"Failed to install {pkg}"
        
        # Check Ollama
        if not setup_ollama(install=st.session_state.permissions["allow_ollama_install"]):
            return False, "Ollama setup failed"
        
        # Load embedding model
        embedding_model = load_embedding_model()
        if not embedding_model:
            return False, "Failed to load embedding model"
        
        # Initialize vector DB
        dimensions = get_embedding_dimensions(embedding_model)
        if not initialize_vector_db(dimensions=dimensions):
            return False, "Failed to initialize vector database"
        
        # Refresh models
        st.session_state.available_models = refresh_available_models()
        
        st.session_state.system_initialized = True
        st.session_state.initialization_complete = True
        
        return True, "Tourism Explorer initialized successfully!"
        
    except Exception as e:
        return False, f"Initialization error: {str(e)}"

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")