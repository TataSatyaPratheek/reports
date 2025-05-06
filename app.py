# app.py - Fully optimized version with improved UI based on CEO recommendations
import streamlit as st
import time
import sys
import gc
import torch
from typing import Dict, Any, List, Optional

# Import optimized modules
from modules.system_setup import (
    ensure_dependencies, setup_ollama, refresh_available_models,
    install_package, download_model, DEFAULT_MODEL_NAME, TOURISM_RECOMMENDED_MODELS
)
from modules.vector_store import initialize_vector_db, reset_vector_db, get_chroma_collection, hybrid_retrieval
from modules.pdf_processor import process_uploaded_pdf
from modules.nlp_models import load_embedding_model, get_embedding_dimensions
from modules.vector_store import add_chunks_to_collection
from modules.llm_interface import query_llm, SlidingWindowMemory
from modules.ui_components import show_system_resources, render_tourism_dashboard_lazy, display_chat
from modules.utils import log_error, TourismLogger
from modules.model_selection import render_model_selection_dashboard, ModelSelector
from modules.insights_generator import render_insights_dashboard, TourismInsightsGenerator

# Import memory management utilities
from modules.memory_utils import memory_monitor, get_available_memory_mb, get_available_gpu_memory_mb

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
    "Travel Trends Analyst": "You are an expert travel trends analyst specializing in identifying emerging patterns and shifts in tourism behavior.",
    "Payment Specialist": "You are a payment systems specialist focused on financial technology trends in the travel and tourism sector.",
    "Market Segmentation Expert": "You are a tourism market segmentation expert who can analyze different traveler demographics and preferences.",
    "Sustainability Tourism Advisor": "You are a sustainability tourism advisor helping organizations implement eco-friendly practices.",
    "Gen Z Travel Specialist": "You are a Gen Z travel specialist with deep knowledge of youth travel preferences and digital engagement.",
    "Luxury Tourism Consultant": "You are a luxury tourism consultant specializing in high-end travel experiences and premium markets.",
    "Tourism Analytics Expert": "You are a tourism analytics expert who excels at interpreting data trends and providing actionable insights.",
    "General Tourism Assistant": "You are a helpful tourism information assistant with broad knowledge of the travel industry."
}

# Key insight questions
KEY_TOURISM_QUESTIONS = [
    "What are the current macro travel trends?",
    "How do payment methods vary across market segments?",
    "What are the key market segmentation types in tourism?",
    "Tell me about sustainability initiatives in tourism"
]

def initialize_session_state():
    """Initialize session state with enhanced features and memory monitoring"""
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
    
    # Memory monitoring
    st.session_state.setdefault("memory_monitoring_enabled", True)
    st.session_state.setdefault("performance_target", "balanced")
    
    # UI state
    st.session_state.setdefault("welcome_message_shown", False)
    st.session_state.setdefault("sidebar_section", "setup")

@st.cache_resource
def get_memory_monitor():
    """Get global memory monitor instance"""
    return memory_monitor

def apply_tourism_theme():
    """Apply enhanced tourism-themed CSS with strong light theme overrides"""
    st.markdown("""
    <style>
        /* Force light mode and override dark theme */
        html, body, [class*="css"] {
            color: #212121 !important;
            background-color: white !important;
        }
        
        /* Dark text on all elements to ensure visibility */
        div, span, p, h1, h2, h3, h4, h5, h6, li, label, a {
            color: #212121 !important;
        }
        
        /* Force light background on all containers */
        .stApp, .main .block-container, div[data-testid="stAppViewContainer"] {
            background-color: white !important;
        }
        
        /* Sidebar force light theme */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0 !important;
        }
        
        /* Sidebar text and elements */
        [data-testid="stSidebar"] div, [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] h4, [data-testid="stSidebar"] button {
            color: #212121 !important;
        }
        
        /* Main container styling */
        .main-container {
            background-color: white !important;
            padding: 20px !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
            margin: 20px 0 !important;
        }
        
        /* Header styling */
        .main-header { 
            color: #1976D2 !important;
            font-size: 2.4rem !important; 
            font-weight: 600 !important;
            text-align: center !important;
            margin-bottom: 0.5rem !important;
        }
        
        .sub-header { 
            color: #616161 !important;
            font-size: 1.2rem !important;
            text-align: center !important;
            margin-bottom: 2rem !important;
        }
        
        /* Chat container with auto-scroll support */
        .chat-container {
            display: flex !important;
            flex-direction: column !important;
            height: 65vh !important;
            background-color: #f8f9fa !important;
            border-radius: 12px !important;
            padding: 20px !important;
            margin-top: 20px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            overflow-y: auto !important;
        }
        
        /* Messages container to allow scroll */
        .messages-container {
            flex: 1 !important;
            overflow-y: auto !important;
            padding-right: 10px !important;
            margin-bottom: 15px !important;
        }
        
        /* User message styling */
        .stChatMessage[data-testid="user-message"] {
            background-color: #e3f2fd !important;
            border-radius: 18px 18px 0 18px !important;
            padding: 12px 18px !important;
            margin-bottom: 15px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
            border: none !important;
            color: #0d47a1 !important;
        }
        
        /* Assistant message styling */
        .stChatMessage[data-testid="assistant-message"] {
            background-color: white !important;
            border-radius: 18px 18px 18px 0 !important;
            padding: 12px 18px !important;
            margin-bottom: 15px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
            border: 1px solid #e0e0e0 !important;
            color: #212121 !important;
        }
        
        /* Chat input area - fixed at bottom */
        .chat-input-container {
            position: sticky !important;
            bottom: 0 !important;
            background-color: #f8f9fa !important;
            padding: 10px 0 !important;
            border-top: 1px solid #e0e0e0 !important;
        }
        
        /* Question buttons styling */
        .question-btn {
            background-color: #f3f7fa !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 10px !important;
            padding: 10px !important;
            margin-bottom: 10px !important;
            transition: all 0.2s !important;
            color: #1976D2 !important;
            font-weight: 500 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        }
        
        .question-btn:hover {
            background-color: #e3f2fd !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 3px 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #1976D2 !important;
            color: white !important;
            border: none !important;
            font-weight: 500 !important;
        }
        
        /* Secondary button */
        button[kind="secondary"] {
            background-color: #f0f0f0 !important;
            color: #212121 !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        /* Chat message avatars */
        .stChatMessageAvatar {
            background-color: white !important;
        }
        
        /* Status elements */
        .stStatusWidget {
            background-color: white !important;
            color: #212121 !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        /* Metrics */
        [data-testid="stMetric"] {
            background-color: white !important;
            color: #212121 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #1976D2 !important;
        }
        
        [data-testid="stMetricValue"] {
            color: #212121 !important;
        }
        
        /* Form elements */
        .stTextInput, .stSelectbox select, .stSlider {
            background-color: white !important;
            color: #212121 !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        /* Welcome message styling */
        .welcome-message {
            background-color: #f9f9f9 !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 10px !important;
            padding: 20px !important;
            text-align: center !important;
            margin: 20px 0 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.08) !important;
        }
        
        .welcome-message h3 {
            color: #1976D2 !important;
            font-size: 1.5rem !important;
            margin-bottom: 10px !important;
        }
        
        /* Make sure file uploader is visible */
        .stFileUploader {
            background-color: white !important;
        }
        
        .stFileUploader label {
            color: #212121 !important;
        }
        
        .stFileUploader button {
            background-color: #1976D2 !important;
            color: white !important;
        }
        
        /* Fix for any other dark elements */
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            background-color: white !important;
            color: #212121 !important;
        }
    </style>
    
    <script>
        // Auto-scroll chat function
        function autoScrollChat() {
            const chatContainer = document.querySelector('.chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        
        // Call scroll function when content changes
        const observer = new MutationObserver(autoScrollChat);
        
        // Start observing when DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            const chatContainer = document.querySelector('.chat-container');
            if (chatContainer) {
                observer.observe(chatContainer, { childList: true, subtree: true });
                autoScrollChat();
            }
        });
    </script>
    """, unsafe_allow_html=True)

def main():
    """Memory-optimized main application entry point with improved UI"""
    st.set_page_config(
        page_title="Tourism Insights Explorer Pro",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",  # Start with expanded sidebar initially for visibility
    )
    
    # Apply custom light-themed CSS
    apply_tourism_theme()
    
    if "system_initialized" not in st.session_state:
        initialize_session_state()
    
    # Get memory monitor instance
    monitor = get_memory_monitor()
    
    # Check memory before rendering
    if st.session_state.get("memory_monitoring_enabled", True):
        monitor.check()

    # Render sidebar with all functionality
    sidebar_params = render_enhanced_sidebar()
    
    # Main content area with centered chat experience
    if not sidebar_params["system_initialized"]:
        # High visibility initialization message
        st.markdown("""
        <div style="background-color: #e8f4f8; padding: 20px; border-radius: 10px; 
                   text-align: center; border: 1px solid #b3e5fc; margin: 30px 0;">
            <h2 style="color: #0277bd; margin-bottom: 15px;">Welcome to Tourism Insights Explorer Pro</h2>
            <p style="font-size: 18px; color: #01579b; margin-bottom: 20px;">
                System needs initialization before you can start exploring tourism insights.
            </p>
            <p style="font-size: 16px; color: #0288d1;">
                Please click the "Initialize System" button in the sidebar to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Large arrow pointing to sidebar
        st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <span style="font-size: 40px; color: #0277bd;">👈</span>
            <p style="font-size: 18px; color: #0277bd; font-weight: bold;">Click "Initialize System" in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Clean, central container for the main chat interface with clear styling
        with st.container():
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            
            # Bold, visible header
            st.markdown('<h1 class="main-header">🌍 Tourism Insights Explorer Pro</h1>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Your AI-powered guide to travel industry analysis</p>', unsafe_allow_html=True)
            
            # Add prominent welcome message
            if not st.session_state.get("welcome_message_shown", False):
                st.markdown("""
                <div class="welcome-message">
                    <h3>Welcome to Tourism Insights Explorer Pro!</h3>
                    <p>How can I help you with travel trends today? Choose a question below or type your own.</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.welcome_message_shown = True
            
            # Quick access prompt buttons with high visibility
            st.markdown("<p style='font-weight:500; color:#1976D2; margin-bottom:10px;'>Explore key tourism questions:</p>", unsafe_allow_html=True)
            prompt_cols = st.columns(2)
            for i, question in enumerate(KEY_TOURISM_QUESTIONS):
                col = prompt_cols[i % 2]
                with col:
                    if st.button(question, key=f"prompt_{i}", use_container_width=True, 
                               help=f"Click to explore {question.lower()}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
            
            # Enhanced chat interface with auto-scroll
            st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
            
            # Messages area (scrollable)
            st.markdown('<div class="messages-container">', unsafe_allow_html=True)
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Fixed chat input at bottom
            st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
            
            user_query = st.chat_input(
                "Ask about travel trends, market segments, payment methods, or deep dive topics...",
                disabled=not bool(st.session_state.processed_files)
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if user_query:
                process_user_query(user_query, sidebar_params)
    
    # Periodic memory cleanup
    if time.time() % 60 < 1:  # Every minute
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def render_enhanced_sidebar():
    """Enhanced sidebar with clean organization (no nested expanders)"""
    with st.sidebar:
        st.markdown("## 🌍 Tourism Explorer Pro")
        
        # Sidebar navigation
        sections = ["Setup", "Documents", "Models", "Analytics", "Settings"]
        cols = st.columns(len(sections))
        
        for i, section in enumerate(sections):
            if cols[i].button(section, key=f"nav_{section.lower()}"):
                st.session_state.sidebar_section = section.lower()
        
        st.divider()
        
        # Main system initialization section (always visible if not initialized)
        if not st.session_state.get("initialization_complete", False):
            st.markdown("### 🚀 System Initialization")
            
            if st.button("Initialize System", type="primary", use_container_width=True):
                with st.status("Setting up tourism system...", expanded=True) as status:
                    try:
                        # Check dependencies
                        status.update(label="Checking dependencies...")
                        mismatched = ensure_dependencies()
                        
                        if mismatched:
                            for pkg, required_ver, current_ver in mismatched:
                                st.warning(f"{pkg}: required={required_ver}, current={current_ver}")
                                if st.button(f"Install {pkg}=={required_ver}"):
                                    if install_package(f"{pkg}=={required_ver}"):
                                        st.success(f"Installed {pkg}")
                                        st.rerun()
                        
                        # Setup Ollama
                        status.update(label="Setting up Ollama...")
                        if not setup_ollama():
                            st.error("Ollama setup failed")
                            return {
                                "system_initialized": False,
                                "uploaded_files": None,
                                "chunk_size": DEFAULT_CHUNK_SIZE,
                                "overlap": DEFAULT_OVERLAP,
                                "top_n": DEFAULT_TOP_N,
                                "local_llm_model": DEFAULT_MODEL_NAME,
                                "system_prompt": "",
                                "use_hybrid_retrieval": True,
                                "use_reranker": True,
                                "hybrid_alpha": DEFAULT_HYBRID_ALPHA
                            }
                        
                        # Check models
                        status.update(label="Checking available models...")
                        available_models = refresh_available_models()
                        st.session_state.available_models = available_models
                        
                        if not available_models:
                            st.warning("No models found")
                            if st.button(f"Download {DEFAULT_MODEL_NAME}"):
                                success, message = download_model(DEFAULT_MODEL_NAME)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        # Initialize vector DB
                        status.update(label="Initializing vector database...")
                        if initialize_vector_db():
                            st.session_state.system_initialized = True
                            st.session_state.initialization_complete = True
                            status.update(label="✅ System ready!", state="complete")
                            st.rerun()
                        else:
                            st.error("Vector DB initialization failed")
                    
                    except Exception as e:
                        st.error(f"Initialization error: {str(e)}")
                        log_error(f"System init error: {str(e)}")
        
        # Display appropriate section based on selection
        if st.session_state.get("system_initialized", False):
            current_section = st.session_state.get("sidebar_section", "setup")
            
            # SETUP SECTION
            if current_section == "setup":
                st.markdown("### 🚀 System Status")
                
                # Show resource monitoring
                show_system_resources()
                
                st.markdown("##### System Options")
                st.toggle("Memory Monitoring", value=st.session_state.get("memory_monitoring_enabled", True), 
                         key="memory_monitoring_toggle")
                
                performance_target = st.select_slider(
                    "Performance Target",
                    options=["low_latency", "balanced", "high_accuracy"],
                    value=st.session_state.get("performance_target", "balanced")
                )
                if performance_target != st.session_state.get("performance_target"):
                    st.session_state.performance_target = performance_target
            
            # DOCUMENTS SECTION
            elif current_section == "documents":
                st.markdown("### 📑 Document Management")
                
                # Document upload
                uploaded_files = st.file_uploader(
                    "Upload Tourism Documents",
                    type=["pdf"],
                    accept_multiple_files=True
                )
                
                # Processing settings
                st.markdown("##### Processing Settings")
                chunk_size = st.slider("Document Chunk Size", 256, 1024, DEFAULT_CHUNK_SIZE, 128)
                overlap = st.slider("Context Overlap", 0, 128, DEFAULT_OVERLAP, 16)
                top_n = st.slider("Search Results", 1, 20, DEFAULT_TOP_N)
                
                # Process documents button
                if uploaded_files and st.button("Process Documents", type="primary", use_container_width=True):
                    process_documents(uploaded_files, chunk_size, overlap)
                
                # Show processed files
                if st.session_state.processed_files:
                    st.markdown("##### Processed Documents")
                    for file in st.session_state.processed_files:
                        st.markdown(f"✅ {file}")
                
                # Reset database option
                with st.container():
                    st.markdown("##### Database Management")
                    if st.button("Reset Database", type="secondary", use_container_width=True):
                        success, message = reset_vector_db()
                        if success:
                            st.session_state.processed_files.clear()
                            st.session_state.messages = []
                            st.session_state.document_chunks = []
                            st.session_state.tourism_insights = {}
                            gc.collect()
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            
            # MODELS SECTION
            elif current_section == "models":
                st.markdown("### 🎯 Model Selection")
                
                # Performance settings
                st.markdown("##### Performance Priority")
                priority = st.select_slider(
                    "Optimization Priority",
                    options=["speed", "balanced", "accuracy"],
                    value=st.session_state.model_selection_params.get("priority", "balanced")
                )
                
                if priority != st.session_state.model_selection_params.get("priority"):
                    st.session_state.model_selection_params["priority"] = priority
                
                # Latency and accuracy sliders
                max_latency = st.slider(
                    "Max Latency (ms)",
                    min_value=10,
                    max_value=500,
                    value=st.session_state.model_selection_params.get("max_latency", 100),
                    step=10
                )
                
                if max_latency != st.session_state.model_selection_params.get("max_latency"):
                    st.session_state.model_selection_params["max_latency"] = max_latency
                
                min_accuracy = st.slider(
                    "Min Accuracy Score",
                    min_value=58.0,
                    max_value=66.0,
                    value=st.session_state.model_selection_params.get("min_accuracy", 60.0),
                    step=0.5
                )
                
                if min_accuracy != st.session_state.model_selection_params.get("min_accuracy"):
                    st.session_state.model_selection_params["min_accuracy"] = min_accuracy
                
                # Embedding model selection (compact mode)
                st.markdown("##### Embedding Model")
                selector = ModelSelector()
                selected_model = selector.select_model(
                    max_latency_ms=max_latency,
                    priority=priority,
                    min_accuracy=min_accuracy
                )
                
                # Show recommended model
                st.info(f"Recommended model: **{selected_model}**")
                
                if st.button("Use Recommended Model", key="use_recommended", use_container_width=True):
                    st.session_state.selected_embedding_model = selected_model
                    st.success(f"Selected model: {selected_model}")
                
                # LLM Model selection
                st.markdown("##### LLM Model")
                local_llm_model = st.selectbox(
                    "Select Model",
                    options=st.session_state.get("available_models", [DEFAULT_MODEL_NAME]),
                    index=0
                )
            
            # ANALYTICS SECTION
            elif current_section == "analytics":
                st.markdown("### 💡 Tourism Expertise")
                
                # Tourism expertise selection
                selected_role = st.selectbox(
                    "Select Expertise",
                    options=list(AGENT_ROLES.keys()),
                    index=list(AGENT_ROLES.keys()).index(st.session_state.get("current_agent_role", "Travel Trends Analyst"))
                )
                
                if selected_role != st.session_state.get("current_agent_role"):
                    st.session_state.current_agent_role = selected_role
                    st.session_state.custom_prompt = AGENT_ROLES.get(selected_role, "")
                    st.success(f"Role changed to: {selected_role}")
                
                # Show role description
                st.markdown(f"**Description**: {AGENT_ROLES.get(selected_role, '')}")
                
                # Insights Dashboard
                st.markdown("##### Insights Dashboard")
                if st.button("Open Full Insights Dashboard", use_container_width=True):
                    st.session_state.show_insights_dashboard = True
                    st.session_state.active_tab = 2  # Switch to insights tab
                    st.rerun()
                    
                # Feature toggles
                st.markdown("##### Features")
                
                # Hybrid retrieval
                use_hybrid = st.toggle(
                    "Hybrid Search",
                    value=st.session_state.get("use_hybrid_retrieval", True),
                    key="use_hybrid_toggle"
                )
                
                if use_hybrid != st.session_state.get("use_hybrid_retrieval"):
                    st.session_state.use_hybrid_retrieval = use_hybrid
                
                # Show hybrid slider only if hybrid is enabled
                if use_hybrid:
                    hybrid_alpha = st.slider(
                        "Vector/Keyword Balance",
                        0.0, 1.0, DEFAULT_HYBRID_ALPHA, 0.1,
                        help="1.0 = vector only, 0.0 = keywords only"
                    )
                    
                    if hybrid_alpha != st.session_state.get("hybrid_alpha"):
                        st.session_state.hybrid_alpha = hybrid_alpha
                
                # Reranker toggle
                use_reranker = st.toggle(
                    "Neural Reranking",
                    value=st.session_state.get("use_reranker", True),
                    key="use_reranker_toggle"
                )
                
                if use_reranker != st.session_state.get("use_reranker"):
                    st.session_state.use_reranker = use_reranker
            
            # SETTINGS SECTION
            elif current_section == "settings":
                st.markdown("### ⚙️ System Settings")
                
                # Memory Management
                st.markdown("##### Memory Management")
                
                # GPU memory
                if torch.cuda.is_available():
                    if st.button("Clear GPU Memory", use_container_width=True):
                        torch.cuda.empty_cache()
                        st.success("GPU memory cleared")
                
                # Garbage collection
                if st.button("Run Garbage Collection", use_container_width=True):
                    gc.collect()
                    st.success("Garbage collection completed")
                
                # Cache clearing
                if st.button("Clear All Caches", use_container_width=True):
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    gc.collect()
                    st.success("All caches cleared")
                
                # Status logs
                st.markdown("##### Error Logs")
                if st.session_state.get("error_log"):
                    log_count = len(st.session_state.error_log)
                    st.markdown(f"**{log_count} log entries**")
                    
                    if st.button("View Logs"):
                        for log in st.session_state.error_log[-10:]:  # Show last 10 logs
                            st.text(log)
                    
                    if st.button("Clear Logs"):
                        st.session_state.error_log = []
                        st.success("Logs cleared")
                else:
                    st.info("No error logs")
        
        # Initialize potentially undefined variables with defaults before returning
        uploaded_files = None
        chunk_size = DEFAULT_CHUNK_SIZE
        overlap = DEFAULT_OVERLAP
        top_n = DEFAULT_TOP_N
        local_llm_model = DEFAULT_MODEL_NAME

        # Determine parameters to return based on current section
        if current_section == "documents":
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
        else:
            return {
                "uploaded_files": None,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "overlap": DEFAULT_OVERLAP,
                "top_n": DEFAULT_TOP_N,
                "local_llm_model": local_llm_model if 'local_llm_model' in locals() else DEFAULT_MODEL_NAME,
                "system_prompt": st.session_state.get("custom_prompt", ""),
                "system_initialized": st.session_state.get("system_initialized", False),
                "use_hybrid_retrieval": st.session_state.get("use_hybrid_retrieval", True),
                "use_reranker": st.session_state.get("use_reranker", True),
                "hybrid_alpha": st.session_state.get("hybrid_alpha", DEFAULT_HYBRID_ALPHA)
            }

def process_documents(uploaded_files, chunk_size, overlap):
    """Memory-optimized document processing"""
    if not uploaded_files:
        return
    
    files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if not files_to_process:
        st.success("All files already processed!")
        return
    
    with st.status("Processing Documents", expanded=True) as status:
        # Check memory before processing
        available_memory = get_available_memory_mb()
        if available_memory < 1000:  # Less than 1GB available
            st.warning(f"Low memory warning: {available_memory:.0f}MB available. Processing may be slower.")
        
        # Use the selected embedding model with performance target
        embedding_model = load_embedding_model(
            st.session_state.get("selected_embedding_model", "all-MiniLM-L6-v2"),
            st.session_state.get("performance_target", "balanced")
        )
        collection = get_chroma_collection()
        
        if not embedding_model or not collection:
            st.error("Core components not available.")
            return
        
        all_chunks = []
        
        for pdf_file in files_to_process:
            status.update(label=f"Processing: {pdf_file.name}...")
            try:
                # Check memory before each file
                memory_monitor.check()
                
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
                    
                    # Clear memory after each file
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                status.update(label=f"❌ Error: {pdf_file.name}", state="error")
        
        # Store chunks for insights generation
        st.session_state.document_chunks = all_chunks
        status.update(label=f"✅ Processed {len(files_to_process)} documents", state="complete")

def process_user_query(user_query, params):
    """Process a user query and generate response"""
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        embedding_model = load_embedding_model(
            st.session_state.get("selected_embedding_model"),
            st.session_state.get("performance_target", "balanced")
        )
        collection = get_chroma_collection()
        
        if embedding_model and collection:
            try:
                # Check memory before inference
                memory_monitor.check()
                
                # Show typing indicator
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
                
                # Add timing info subtly
                answer_with_info = f"{answer}\n\n<small>*Generated in {inference_time:.0f}ms*</small>"
                
                message_placeholder.markdown(answer_with_info, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": answer_with_info})
                
                # Clear memory after response
                gc.collect()
                
            except Exception as e:
                message_placeholder.markdown(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        log_error(f"Critical error: {str(e)}")
        
        # Try to recover by clearing memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()