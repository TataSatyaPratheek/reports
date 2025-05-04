# app.py - Enhanced version with dynamic model selection and actual insights
import streamlit as st
import time
import sys
from typing import Dict, Any, List, Optional

# Import existing modules
from modules.system_setup import (
    ensure_dependencies, setup_ollama, refresh_available_models,
    install_package, download_model, DEFAULT_MODEL_NAME, TOURISM_RECOMMENDED_MODELS
)
from modules.vector_store import initialize_vector_db, reset_vector_db, get_chroma_collection, hybrid_retrieval
from modules.embedding_service import load_embedding_model, get_embedding_dimensions
from modules.pdf_processor import process_uploaded_pdf
from modules.vector_store import add_chunks_to_collection
from modules.llm_interface import query_llm, SlidingWindowMemory
from modules.ui_components import display_chat, show_system_resources, apply_tourism_theme
from modules.utils import log_error, TourismLogger

# Import new enhanced modules
from modules.model_selection import render_model_selection_dashboard, ModelSelector
from modules.insights_generator import render_insights_dashboard, TourismInsightsGenerator

# Initialize logger
logger = TourismLogger()

# Constants
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64
DEFAULT_TOP_N = 5
DEFAULT_CONVERSATION_MEMORY = 3
DEFAULT_HYBRID_ALPHA = 0.7

# Enhanced agent roles
AGENT_ROLES = {
    "Travel Trends Analyst": "You are an expert travel trends analyst...",
    "Payment Specialist": "You are a payment systems specialist...",
    "Market Segmentation Expert": "You are a tourism market segmentation expert...",
    "Sustainability Tourism Advisor": "You are a sustainability tourism advisor...",
    "Gen Z Travel Specialist": "You are a Gen Z travel specialist...",
    "Luxury Tourism Consultant": "You are a luxury tourism consultant...",
    "Tourism Analytics Expert": "You are a tourism analytics expert...",
    "General Tourism Assistant": "You are a helpful tourism information assistant..."
}

def initialize_session_state():
    """Initialize session state with enhanced features"""
    st.session_state.setdefault("processed_files", set())
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("sliding_window_memory", SlidingWindowMemory(max_tokens=2048))
    st.session_state.setdefault("system_initialized", False)
    st.session_state.setdefault("initialization_complete", False)
    st.session_state.setdefault("error_log", [])
    st.session_state.setdefault("available_models", [DEFAULT_MODEL_NAME])
    st.session_state.setdefault("current_agent_role", "Travel Trends Analyst")
    st.session_state.setdefault("custom_prompt", AGENT_ROLES.get("Travel Trends Analyst", ""))
    
    # Enhanced session state for new features
    st.session_state.setdefault("selected_embedding_model", "all-MiniLM-L6-v2")
    st.session_state.setdefault("model_selection_params", {
        "priority": "balanced",
        "max_latency": 100,
        "min_accuracy": 60.0
    })
    st.session_state.setdefault("document_chunks", [])
    st.session_state.setdefault("tourism_insights", {})
    st.session_state.setdefault("show_insights_dashboard", True)
    
    # Feature flags
    st.session_state.setdefault("use_hybrid_retrieval", True)
    st.session_state.setdefault("use_reranker", True)
    st.session_state.setdefault("hybrid_alpha", DEFAULT_HYBRID_ALPHA)

def main():
    """Enhanced main application entry point"""
    st.set_page_config(
        page_title="Tourism Insights Explorer Pro",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_tourism_theme()
    
    st.markdown('<p class="main-header">🌍 Tourism Insights Explorer Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered travel industry analysis with dynamic model selection</p>', unsafe_allow_html=True)

    if "system_initialized" not in st.session_state:
        initialize_session_state()

    sidebar_params = render_enhanced_sidebar()

    if not sidebar_params["system_initialized"]:
        st.warning("System not ready. Please initialize the system via the sidebar.")
    else:
        # Main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Selection", "📄 Document Processing", "🔍 Insights Dashboard", "💬 Chat Interface"])
        
        with tab1:
            selected_model = render_model_selection_dashboard(st.container())
            st.session_state.selected_embedding_model = selected_model
        
        with tab2:
            process_tourism_documents_enhanced(
                sidebar_params["uploaded_files"],
                sidebar_params["chunk_size"],
                sidebar_params["overlap"]
            )
        
        with tab3:
            if st.session_state.get("document_chunks"):
                render_insights_dashboard(
                    st.session_state.document_chunks,
                    sidebar_params["local_llm_model"]
                )
            else:
                st.info("Please process documents first to generate insights.")
        
        with tab4:
            render_tourism_chat_interface(sidebar_params)

def render_enhanced_sidebar():
    """Enhanced sidebar with model selection parameters"""
    with st.sidebar:
        st.markdown("## 🌍 Tourism Explorer Pro")
        st.markdown("---")
        
        # System initialization (similar to original)
        if not st.session_state.get("initialization_complete", False):
            # ... (keep existing initialization code)
            pass
        
        # Model selection parameters
        if st.session_state.get("system_initialized", False):
            with st.expander("🎯 Model Selection", expanded=True):
                st.session_state.model_selection_params["priority"] = st.select_slider(
                    "Performance Priority",
                    options=["speed", "balanced", "accuracy"],
                    value=st.session_state.model_selection_params.get("priority", "balanced")
                )
                
                st.session_state.model_selection_params["max_latency"] = st.slider(
                    "Max Latency (ms)",
                    10, 500, 
                    st.session_state.model_selection_params.get("max_latency", 100),
                    step=10
                )
                
                st.session_state.model_selection_params["min_accuracy"] = st.slider(
                    "Min Accuracy Score",
                    58.0, 66.0,
                    st.session_state.model_selection_params.get("min_accuracy", 60.0),
                    step=0.5
                )
            
            # Document upload
            st.markdown("### 📑 Tourism Documents")
            uploaded_files = st.file_uploader(
                "Upload Tourism Documents",
                type=["pdf"],
                accept_multiple_files=True
            )
            
            # Analysis settings
            with st.expander("⚙️ Analysis Settings", expanded=False):
                chunk_size = st.slider("Document Chunk Size", 256, 1024, DEFAULT_CHUNK_SIZE, 128)
                overlap = st.slider("Context Overlap", 0, 128, DEFAULT_OVERLAP, 16)
                top_n = st.slider("Search Results", 1, 20, DEFAULT_TOP_N)
            
            # Tourism expertise selection
            st.markdown("### 🔍 Tourism Analysis")
            selected_role = st.selectbox(
                "Tourism Expertise",
                options=list(AGENT_ROLES.keys()),
                index=list(AGENT_ROLES.keys()).index(st.session_state.get("current_agent_role", "Travel Trends Analyst"))
            )
            
            if selected_role != st.session_state.get("current_agent_role"):
                st.session_state.current_agent_role = selected_role
                st.session_state.custom_prompt = AGENT_ROLES.get(selected_role, "")
            
            # LLM Model selection
            local_llm_model = st.selectbox(
                "LLM Model",
                options=st.session_state.get("available_models", [DEFAULT_MODEL_NAME]),
                index=0
            )
            
            # Advanced features
            with st.expander("🧠 Advanced Features", expanded=False):
                st.session_state.use_hybrid_retrieval = st.toggle(
                    "Hybrid Search",
                    value=st.session_state.get("use_hybrid_retrieval", True)
                )
                
                if st.session_state.use_hybrid_retrieval:
                    st.session_state.hybrid_alpha = st.slider(
                        "Semantic/Keyword Balance",
                        0.0, 1.0, DEFAULT_HYBRID_ALPHA, 0.1
                    )
                
                st.session_state.use_reranker = st.toggle(
                    "Neural Reranking",
                    value=st.session_state.get("use_reranker", True)
                )
            
            # Reset database
            with st.expander("🗑️ Reset Database", expanded=False):
                if st.button("Reset All Data", type="secondary"):
                    success, message = reset_vector_db()
                    if success:
                        st.session_state.processed_files.clear()
                        st.session_state.messages = []
                        st.session_state.document_chunks = []
                        st.session_state.tourism_insights = {}
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    return {
        "uploaded_files": uploaded_files if 'uploaded_files' in locals() else None,
        "chunk_size": chunk_size if 'chunk_size' in locals() else DEFAULT_CHUNK_SIZE,
        "overlap": overlap if 'overlap' in locals() else DEFAULT_OVERLAP,
        "top_n": top_n if 'top_n' in locals() else DEFAULT_TOP_N,
        "local_llm_model": local_llm_model if 'local_llm_model' in locals() else DEFAULT_MODEL_NAME,
        "system_prompt": st.session_state.get("custom_prompt", ""),
        "system_initialized": st.session_state.get("system_initialized", False),
        "use_hybrid_retrieval": st.session_state.get("use_hybrid_retrieval", True),
        "use_reranker": st.session_state.get("use_reranker", True),
        "hybrid_alpha": st.session_state.get("hybrid_alpha", DEFAULT_HYBRID_ALPHA)
    }

def process_tourism_documents_enhanced(uploaded_files, chunk_size, overlap):
    """Enhanced document processing with dynamic model selection"""
    if not uploaded_files:
        return
    
    files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if not files_to_process:
        return
    
    st.markdown("### 📝 Document Processing with Dynamic Model Selection")
    
    # Use the selected embedding model
    embedding_model = load_embedding_model(st.session_state.get("selected_embedding_model", "all-MiniLM-L6-v2"))
    collection = get_chroma_collection()
    
    if not embedding_model or not collection:
        st.error("Core components not available.")
        return
    
    all_chunks = []
    
    for pdf_file in files_to_process:
        with st.status(f"Processing: {pdf_file.name}...", expanded=True) as status:
            try:
                chunks = process_uploaded_pdf(
                    pdf_file, 
                    chunk_size, 
                    overlap, 
                    status=status,
                    extract_images=True
                )
                
                if chunks:
                    all_chunks.extend([c["text"] for c in chunks])
                    
                    success = add_chunks_to_collection(
                        [c["text"] for c in chunks], 
                        embedding_model, 
                        collection, 
                        status=status
                    )
                    
                    if success:
                        st.session_state.processed_files.add(pdf_file.name)
                        status.update(label=f"✅ Processed: {pdf_file.name}", state="complete")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                status.update(label=f"❌ Error: {pdf_file.name}", state="error")
    
    # Store chunks for insights generation
    st.session_state.document_chunks = all_chunks

def render_tourism_chat_interface(params):
    """Enhanced chat interface with resource monitoring"""
    st.markdown("### 💬 Tourism Insights Chat")
    
    # Add resource monitoring widget
    with st.container():
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.markdown("#### System Resources")
            show_system_resources()
        
        with col3:
            current_model = st.session_state.get("selected_embedding_model", "all-MiniLM-L6-v2")
            st.markdown(f"**Active Model:**\n{current_model.split('/')[-1]}")
    
    st.markdown("---")
    
    # Display chat history
    display_chat(
        st.session_state.messages,
        current_role=st.session_state.get("current_agent_role", "Tourism Assistant")
    )
    
    # Chat input
    user_query = st.chat_input(
        "Ask about travel trends, market segments, sustainability...",
        disabled=not bool(st.session_state.processed_files)
    )
    
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        embedding_model = load_embedding_model(st.session_state.get("selected_embedding_model"))
        collection = get_chroma_collection()
        
        if embedding_model and collection:
            try:
                # Show inference timing
                start_time = time.time()
                
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
                
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Add timing info to response
                answer_with_timing = f"{answer}\n\n---\n*Response generated in {inference_time:.0f}ms using {st.session_state.get('selected_embedding_model', 'default model')}*"
                
                st.session_state.messages.append({"role": "assistant", "content": answer_with_timing})
                
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
        
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        log_error(f"Critical error: {str(e)}")