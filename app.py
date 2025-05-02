import streamlit as st
import time
import os
import sys
from typing import Tuple, Dict, Any, List, Optional

# Import modules
from modules.system_setup import (
    ensure_dependencies, setup_ollama, refresh_available_models,
    install_package, download_model, DEFAULT_MODEL_NAME, TOURISM_RECOMMENDED_MODELS
)
from modules.vector_store import initialize_vector_db, reset_vector_db, get_chroma_collection, hybrid_retrieval
from modules.nlp_models import load_nltk_resources, load_spacy_model, load_embedding_model, extract_tourism_entities
from modules.pdf_processor import process_uploaded_pdf
from modules.vector_store import add_chunks_to_collection
from modules.llm_interface import query_llm, SlidingWindowMemory
from modules.ui_components import display_chat, show_system_resources, apply_tourism_theme, display_tourism_entities, render_tourism_dashboard
from modules.utils import log_error, PerformanceMonitor, TourismLogger, extract_tourism_metrics_from_text

# Initialize logger
logger = TourismLogger()

# --- Constants and AGENT_ROLES ---
DEFAULT_CHUNK_SIZE = 250
DEFAULT_OVERLAP = 50
DEFAULT_TOP_N = 10
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
    
    "General Tourism Assistant": "You are a helpful tourism information assistant. Provide clear, accurate, and balanced information about travel topics based on the provided document context. Offer insights on destinations, travel planning, industry trends, and tourism services. Present information in an accessible manner suitable for both industry professionals and travelers. Focus on presenting factual information rather than personal opinions.",
    
    "Custom": ""  # Placeholder for custom prompt
}

# --- initialize_session_state with tourism-specific defaults ---
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
    st.session_state.setdefault("current_agent_role", "Travel Trends Analyst")  # Changed default role
    st.session_state.setdefault("custom_prompt", AGENT_ROLES.get("Travel Trends Analyst", ""))
    st.session_state.setdefault("permissions", {
        "allow_system_check": True,
        "allow_package_install": False,
        "allow_ollama_install": False,
        "allow_model_download": False
    })
    
    # Feature flags
    st.session_state.setdefault("use_hybrid_retrieval", True)
    st.session_state.setdefault("use_query_reformulation", True)
    st.session_state.setdefault("use_reranker", True)
    st.session_state.setdefault("hybrid_alpha", DEFAULT_HYBRID_ALPHA)
    
    # Tourism-specific session state
    st.session_state.setdefault("extracted_tourism_entities", {})
    st.session_state.setdefault("tourism_metrics", {})
    st.session_state.setdefault("show_tourism_dashboard", False)
    st.session_state.setdefault("show_tourism_toolbar", True)

# --- Main UI function for tourism app ---
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
        if st.session_state.initialization_status in ["Pending", "Failed"]:
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

# --- Tourism-specific sidebar ---
def render_tourism_sidebar():
    """Render the sidebar UI with tourism-specific elements."""
    with st.sidebar:
        st.markdown("## 🏝️ Tourism Explorer Settings")
        st.markdown("---")
        
        # --- Initialization Block ---
        show_init_block = not st.session_state.initialization_complete or st.session_state.initialization_status == "Failed"
        if show_init_block:
            st.markdown("### 🚀 System Setup")
            if st.session_state.initialization_status == "Failed":
                 st.error("Previous initialization failed. Please check permissions and retry.")
            else:
                st.info("System requires initialization before analyzing tourism documents.")
                
            st.markdown("**Permissions:**")
            st.caption("Grant permissions for automated setup.")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.permissions["allow_package_install"] = st.checkbox(
                    "Allow Package Install",
                    value=st.session_state.permissions["allow_package_install"],
                    help="Allow the system to install required packages for tourism analysis."
                )
            with col2:
                st.session_state.permissions["allow_ollama_install"] = st.checkbox(
                    "Allow Ollama Install",
                    value=st.session_state.permissions["allow_ollama_install"],
                    help="Allow the system to install Ollama for local AI processing."
                )
                st.session_state.permissions["allow_model_download"] = st.checkbox(
                    "Allow Model Download",
                    value=st.session_state.permissions["allow_model_download"],
                    help="Allow downloading AI models optimized for tourism analysis."
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
                        
            st.markdown(f"**Status:** `{st.session_state.initialization_status}`")
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
            
            if st.session_state.processed_files:
                with st.expander(f"📋 Processed Documents ({len(st.session_state.processed_files)})", expanded=False):
                    for filename in sorted(list(st.session_state.processed_files)):
                        st.caption(f"• {filename}")

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
            current_role = st.session_state.get("current_agent_role", "Travel Trends Analyst")
            role_options = list(AGENT_ROLES.keys())
            try:
                current_role_index = role_options.index(current_role)
            except ValueError:
                current_role_index = role_options.index("Travel Trends Analyst")
                
            selected_role = st.selectbox(
                "Tourism Expertise",
                options=role_options,
                index=current_role_index,
                key="agent_role_selector",
                help="Select the specialized tourism expertise for your analysis."
            )
            
            if selected_role != st.session_state.current_agent_role:
                st.session_state.current_agent_role = selected_role
                st.session_state.custom_prompt = AGENT_ROLES.get(selected_role, "") if selected_role != "Custom" else st.session_state.custom_prompt
                st.rerun()
            
            # --- Custom Prompt ---
            is_custom_role = (st.session_state.current_agent_role == "Custom")
            with st.expander("✏️ Custom Tourism Prompt", expanded=is_custom_role):
                custom_prompt_input = st.text_area(
                    "Custom Tourism Expertise Instructions",
                    value=st.session_state.custom_prompt,
                    height=150,
                    key="custom_prompt_text_area",
                    help="Create custom instructions for specialized tourism analysis."
                )
                if is_custom_role and custom_prompt_input != st.session_state.custom_prompt:
                    st.session_state.custom_prompt = custom_prompt_input
            
            # --- Tourism LLM Model ---
            with st.expander("🤖 AI Model", expanded=False):
                model_options = st.session_state.available_models or [DEFAULT_MODEL_NAME]
                current_model = st.session_state.get("local_llm_model", DEFAULT_MODEL_NAME)
                
                if DEFAULT_MODEL_NAME not in model_options:
                    model_options.insert(0, DEFAULT_MODEL_NAME)
                    
                try:
                    current_model_index = model_options.index(current_model)
                except ValueError:
                    current_model_index = 0
                    
                selected_model = st.selectbox(
                    "Select Tourism AI Model",
                    options=model_options,
                    index=current_model_index,
                    key="model_selector",
                    help="Choose the AI model specialized for tourism analysis."
                )
                st.session_state.local_llm_model = selected_model
                
                # Show model recommendation info if available
                if selected_model in TOURISM_RECOMMENDED_MODELS:
                    info = TOURISM_RECOMMENDED_MODELS[selected_model]
                    st.info(f"📝 {info['description']}")
                    st.info(f"✅ Recommended for: {', '.join(info['recommend_for'])}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Refresh Models", key="refresh_models", use_container_width=True):
                        with st.spinner("Checking available tourism AI models..."):
                            st.session_state.available_models = refresh_available_models()
                            st.rerun()
                            
                with col2:
                    if (st.session_state.permissions["allow_model_download"] and 
                        DEFAULT_MODEL_NAME not in st.session_state.available_models):
                        if st.button(f"Download {DEFAULT_MODEL_NAME}", key="download_default", use_container_width=True):
                            with st.status(f"Downloading tourism-optimized model...", expanded=True) as dl_status:
                                success, msg = download_model(DEFAULT_MODEL_NAME)
                                if success:
                                    dl_status.update(label=f"Tourism model downloaded!", state="complete", expanded=False)
                                    st.session_state.available_models = refresh_available_models()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    dl_status.update(label="Download Failed", state="error", expanded=True)
                                    st.error(msg)
                    elif not st.session_state.permissions["allow_model_download"]:
                        st.caption("Model download permission required.")
            
            # --- Advanced Tourism Analysis Settings ---
            with st.expander("⚙️ Analysis Settings", expanded=False):
                st.slider(
                    "Document Chunk Size",
                    50, 1000,
                    st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
                    50,
                    help="Size of document chunks for tourism content analysis (in words).",
                    key="chunk_size"
                )
                
                st.slider(
                    "Context Overlap",
                    0, 200,
                    st.session_state.get("overlap", DEFAULT_OVERLAP),
                    10,
                    help="Overlap between adjacent chunks to maintain context between sections.",
                    key="overlap"
                )
                
                st.slider(
                    "Search Results",
                    1, 20,
                    st.session_state.get("top_n", DEFAULT_TOP_N),
                    help="Number of relevant tourism document chunks to retrieve for each query.",
                    key="top_n"
                )
                
                st.slider(
                    "Conversation Memory",
                    0, 10,
                    st.session_state.get("conversation_memory_count", DEFAULT_CONVERSATION_MEMORY),
                    help="Number of previous conversation turns to include for context.",
                    key="conversation_memory_count"
                )
            
            # --- Advanced Retrieval Features ---
            with st.expander("🧠 Advanced Features", expanded=False):
                st.toggle(
                    "Hybrid Search",
                    value=st.session_state.get("use_hybrid_retrieval", True),
                    help="Combine vector and keyword search for better tourism content retrieval.",
                    key="use_hybrid_retrieval"
                )
                
                if st.session_state.use_hybrid_retrieval:
                    st.slider(
                        "Semantic/Keyword Balance",
                        0.0, 1.0,
                        st.session_state.get("hybrid_alpha", DEFAULT_HYBRID_ALPHA),
                        0.1,
                        help="Balance between semantic understanding (1.0) and keyword matching (0.0).",
                        key="hybrid_alpha"
                    )
                
                st.toggle(
                    "Neural Reranking",
                    value=st.session_state.get("use_reranker", True),
                    help="Use AI to improve the ranking of tourism search results.",
                    key="use_reranker"
                )
                
                st.toggle(
                    "Query Enhancement",
                    value=st.session_state.get("use_query_reformulation", True),
                    help="Automatically enhance queries with tourism context and terminology.",
                    key="use_query_reformulation"
                )
                
                if st.button("Clear Conversation", key="clear_conversation", use_container_width=True):
                    st.session_state.messages = []
                    if "sliding_window_memory" in st.session_state:
                        st.session_state.sliding_window_memory.clear()
                    st.success("Tourism conversation cleared.")
            
            # --- Reset Database Section ---
            with st.expander("🗑️ Reset Database", expanded=False):
                st.warning("Permanently deletes analyzed tourism data & requires re-initialization.")
                reset_placeholder = st.empty()
                if st.button("Confirm Reset", key="reset_vector_db", use_container_width=True, type="secondary"):
                    try:
                        with st.spinner("Resetting tourism database and clearing cache..."):
                            reset_success, reset_message = reset_vector_db()
                        if reset_success:
                            reset_placeholder.success(reset_message)
                            st.session_state.processed_files.clear()
                            st.session_state.messages = []
                            st.session_state.extracted_tourism_entities = {}
                            st.session_state.tourism_metrics = {}
                            if "sliding_window_memory" in st.session_state:
                                st.session_state.sliding_window_memory.clear()
                            # Reset flags to force manual re-init via UI
                            st.session_state.system_initialized = False
                            st.session_state.initialization_complete = False
                            st.session_state.initialization_status = "Pending"
                            logger.info("Tourism database reset successful.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            reset_placeholder.error(f"Database reset failed: {reset_message}")
                    except Exception as e:
                         logger.error(f"Critical error during reset: {str(e)}")
                         reset_placeholder.error(f"An unexpected error occurred during reset: {str(e)}")
            
            # --- Error Log ---
            with st.expander("📝 Logs", expanded=False):
                if st.button("Clear Error Log", key="clear_errors", use_container_width=True):
                    st.session_state.error_log = []
                    st.success("Error log cleared.")
                    
                if st.session_state.error_log:
                    st.markdown("**Error Log:**")
                    st.text_area(
                        "Log Messages:",
                        value="\n".join(st.session_state.error_log),
                        height=200,
                        disabled=True
                    )

        # --- Return Params ---
        return {
            "uploaded_files": uploaded_files if 'uploaded_files' in locals() else None,
            "chunk_size": st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
            "overlap": st.session_state.get("overlap", DEFAULT_OVERLAP),
            "top_n": st.session_state.get("top_n", DEFAULT_TOP_N),
            "conversation_memory_count": st.session_state.get("conversation_memory_count", DEFAULT_CONVERSATION_MEMORY),
            "local_llm_model": st.session_state.get("local_llm_model", DEFAULT_MODEL_NAME),
            "system_prompt": st.session_state.get("custom_prompt", AGENT_ROLES.get("Travel Trends Analyst", "")),
            "system_initialized": st.session_state.get("system_initialized", False),
            "use_hybrid_retrieval": st.session_state.get("use_hybrid_retrieval", True),
            "use_reranker": st.session_state.get("use_reranker", True), 
            "use_query_reformulation": st.session_state.get("use_query_reformulation", True),
            "hybrid_alpha": st.session_state.get("hybrid_alpha", DEFAULT_HYBRID_ALPHA)
        }

# --- Tourism document processing ---
def process_tourism_documents(uploaded_files, chunk_size, overlap):
    """Process uploaded tourism documents with enhanced analysis and entity extraction."""
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        return
    if not uploaded_files:
        return

    st.markdown("---")
    st.subheader("📝 Tourism Document Processing")
    col1, col2 = st.columns([3, 1])
    batch_status_text = col1.empty()
    batch_progress = col2.empty()
    
    files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    total_new_files = len(files_to_process)
    
    if total_new_files == 0:
        batch_status_text.info("No new tourism documents to process.")
        time.sleep(1.5)
        batch_status_text.empty()
        return

    embedding_model = load_embedding_model()
    collection = get_chroma_collection()

    if not embedding_model or not collection:
        st.error("Core components (embedding model or DB collection) not available for processing.")
        logger.error("Tourism document processing failed: Missing embedding model or collection.")
        return

    files_processed_count = 0
    errors_occurred = False
    batch_progress.progress(0)
    
    # Initialize aggregated tourism entities and metrics
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
    
    for i, pdf_file in enumerate(files_to_process):
        with st.status(f"Analyzing tourism document: {pdf_file.name}...", expanded=True) as file_status:
            try:
                # Process with tourism-focused extraction
                chunks = process_uploaded_pdf(
                    pdf_file, 
                    chunk_size, 
                    overlap, 
                    status=file_status,
                    extract_images=True
                )
                
                if chunks:
                    # Extract tourism entities from each chunk
                    for chunk in chunks:
                        # Extract entities if not already in chunk metadata
                        if "tourism_entities" not in chunk.get("metadata", {}):
                            entities = extract_tourism_entities(chunk["text"])
                            chunk["metadata"]["tourism_entities"] = entities
                        else:
                            entities = chunk["metadata"]["tourism_entities"]
                        
                        # Aggregate entities
                        for entity_type, items in entities.items():
                            if entity_type in all_tourism_entities:
                                all_tourism_entities[entity_type].update(items)
                        
                        # Extract tourism metrics
                        metrics = extract_tourism_metrics_from_text(chunk["text"])
                        
                        # Process segment information
                        if "segment_matches" in chunk["metadata"]:
                            for segment, has_match in chunk["metadata"]["segment_matches"].items():
                                if has_match:
                                    tourism_metrics["segments"][segment] = tourism_metrics["segments"].get(segment, 0) + 1
                        
                        # Process payment information
                        if chunk["metadata"].get("has_payment_info", False):
                            # Simple increment for now - could be enhanced with actual payment method extraction
                            payment_keywords = ["credit card", "debit card", "cash", "digital wallet", 
                                            "mobile payment", "cryptocurrency"]
                            
                            for payment in payment_keywords:
                                if payment in chunk["text"].lower():
                                    tourism_metrics["payment_methods"][payment] = tourism_metrics["payment_methods"].get(payment, 0) + 1
                    
                    # Add to collection
                    add_success = add_chunks_to_collection([c["text"] for c in chunks], embedding_model, collection, status=file_status)
                    
                    if add_success:
                        st.session_state.processed_files.add(pdf_file.name)
                        files_processed_count += 1
                        file_status.update(label=f"Successfully analyzed tourism document: {pdf_file.name}", state="complete", expanded=False)
                    else:
                        errors_occurred = True
                else:
                    errors_occurred = True
                    if file_status._label.startswith("Generated"):
                        file_status.update(label=f"No content extracted from {pdf_file.name}", state="warning", expanded=False)
            except Exception as e:
                logger.error(f"Critical error processing tourism document {pdf_file.name}: {str(e)}")
                errors_occurred = True
                file_status.update(label=f"Unexpected error processing {pdf_file.name}", state="error", expanded=True)
                st.error(f"Details: {str(e)}")
                
        batch_progress.progress((i + 1) / total_new_files)

    # Save extracted entities to session state
    if all_tourism_entities:
        # Convert sets to lists for storage
        for entity_type in all_tourism_entities:
            all_tourism_entities[entity_type] = list(all_tourism_entities[entity_type])
        
        st.session_state.extracted_tourism_entities = all_tourism_entities
    
    # Process and save tourism metrics
    if tourism_metrics["segments"]:
        # Convert raw counts to percentages
        total_segments = sum(tourism_metrics["segments"].values())
        for segment in tourism_metrics["segments"]:
            tourism_metrics["segments"][segment] = round((tourism_metrics["segments"][segment] / total_segments) * 100, 1)
    
    if tourism_metrics["payment_methods"]:
        # Convert raw counts to percentages
        total_payments = sum(tourism_metrics["payment_methods"].values())
        for payment in tourism_metrics["payment_methods"]:
            tourism_metrics["payment_methods"][payment] = round((tourism_metrics["payment_methods"][payment] / total_payments) * 100, 1)
    
    # Save metrics to session state
    st.session_state.tourism_metrics = tourism_metrics
    
    # Display completion message
    if files_processed_count > 0 and not errors_occurred:
        batch_status_text.success(f"Successfully analyzed {files_processed_count} tourism document(s).")
    elif files_processed_count > 0 and errors_occurred:
        batch_status_text.warning(f"Processed {files_processed_count} document(s), but some analysis was incomplete.")
    elif errors_occurred:
        batch_status_text.error("Failed to process tourism documents. Check status messages.")
        
    time.sleep(3)
    batch_status_text.empty()
    batch_progress.empty()

# --- Tourism insights dashboard ---
def display_tourism_insights():
    """Display tourism insights dashboard with visualizations of extracted data."""
    st.markdown("---")
    st.subheader("📊 Tourism Insights Dashboard")
    
    # Show extracted tourism entities
    if st.session_state.get("extracted_tourism_entities"):
        st.markdown("### 🔎 Extracted Tourism Entities")
        display_tourism_entities(st.session_state.extracted_tourism_entities)
    
    # Show tourism metrics if available
    if st.session_state.get("tourism_metrics"):
        st.markdown("### 📈 Tourism Market Analysis")
        render_tourism_dashboard(st.session_state.tourism_metrics)
    else:
        st.info("Process tourism documents to generate insights. The dashboard will show visualizations of market segments, payment methods, and travel trends extracted from your documents.")

# --- Tourism chat interface ---
def render_tourism_chat_interface(params):
    """Render the tourism-focused chat interface with enhanced features."""
    st.markdown("---")
    st.subheader("💬 Tourism Insights Chat")
    
    # Create a container for the chat display
    chat_display_area = st.container()
    
    # Display tourism toolbar if enabled
    if st.session_state.get("show_tourism_toolbar", True):
        with st.container():
            cols = st.columns([1, 1, 1, 1])
            with cols[0]:
                if st.button("📊 Travel Trends", use_container_width=True):
                    st.session_state.current_agent_role = "Travel Trends Analyst"
                    st.session_state.custom_prompt = AGENT_ROLES.get("Travel Trends Analyst", "")
                    if "messages" in st.session_state and st.session_state.messages:
                        st.rerun()
            with cols[1]:
                if st.button("💳 Payment Analysis", use_container_width=True):
                    st.session_state.current_agent_role = "Payment Specialist"
                    st.session_state.custom_prompt = AGENT_ROLES.get("Payment Specialist", "")
                    if "messages" in st.session_state and st.session_state.messages:
                        st.rerun()
            with cols[2]:
                if st.button("👥 Market Segments", use_container_width=True):
                    st.session_state.current_agent_role = "Market Segmentation Expert"
                    st.session_state.custom_prompt = AGENT_ROLES.get("Market Segmentation Expert", "")
                    if "messages" in st.session_state and st.session_state.messages:
                        st.rerun()
            with cols[3]:
                if st.button("🌱 Sustainability", use_container_width=True):
                    st.session_state.current_agent_role = "Sustainability Tourism Advisor"
                    st.session_state.custom_prompt = AGENT_ROLES.get("Sustainability Tourism Advisor", "")
                    if "messages" in st.session_state and st.session_state.messages:
                        st.rerun()
    
    # Display the chat history
    with chat_display_area:
        display_chat(
            st.session_state.messages,
            current_role=st.session_state.get("current_agent_role", "Tourism Assistant")
        )
        
    st.markdown("---")
    
    # Chat input area with tourism-specific placeholders
    chat_input_disabled = not bool(st.session_state.processed_files)
    input_placeholder = "Ask about travel trends, market segments, payment methods, or sustainability..." if not chat_input_disabled else "Please analyze tourism documents first"
    
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area(
            "Your tourism query:",
            height=100,
            placeholder=input_placeholder,
            disabled=chat_input_disabled,
            key="chat_input"
        )
        
        # Sample questions feature
        if not st.session_state.messages:
            st.caption("Try asking: 'What are the macro trends in travel for 2025?' or 'How do payment methods differ across segments?'")
        
        submitted = st.form_submit_button(
            "Ask Tourism Assistant",
            use_container_width=True,
            disabled=chat_input_disabled
        )

    # Handle form submission or sample question selection
    if "selected_sample_question" in st.session_state:
        user_query = st.session_state.selected_sample_question
        del st.session_state.selected_sample_question
        submitted = True
        
    if submitted and user_query.strip() and not chat_input_disabled:
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Add to sliding window memory
        if "sliding_window_memory" in st.session_state:
            st.session_state.sliding_window_memory.add("User", user_query)
        st.rerun()

    # Process the last user message if not already processed
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_user_query = st.session_state.messages[-1]["content"]
        
        # Get conversation memory
        if "sliding_window_memory" in st.session_state:
            conversation_memory = st.session_state.sliding_window_memory.get_formatted_history()
        else:
            memory_limit = params["conversation_memory_count"] * 2
            memory_slice = st.session_state.messages[-(memory_limit + 1) : -1] if memory_limit > 0 else []
            conversation_memory = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in memory_slice)

        embedding_model = load_embedding_model()
        collection = get_chroma_collection()

        if not embedding_model or not collection:
             answer = "Error: Cannot access tourism knowledge base. Core components are missing."
             logger.error(answer)
             st.session_state.messages.append({"role": "assistant", "content": answer})
             st.rerun()
        else:
            # Use enhanced tourism-focused LLM query
            with st.status("Analyzing tourism data...", expanded=False) as status:
                answer = query_llm(
                    user_query=last_user_query,
                    top_n=params["top_n"],
                    local_llm_model=params["local_llm_model"],
                    embedding_model=embedding_model,
                    collection=collection,
                    conversation_memory=conversation_memory,
                    system_prompt=params["system_prompt"],
                    use_hybrid_retrieval=params["use_hybrid_retrieval"],
                    use_query_reformulation=params["use_query_reformulation"],
                    hybrid_alpha=params["hybrid_alpha"],
                    use_reranker=params["use_reranker"]
                )
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            # Add to sliding window memory
            if "sliding_window_memory" in st.session_state:
                st.session_state.sliding_window_memory.add("Assistant", answer)
            st.rerun()

def initialize_system():
    """Initialize system components for tourism analysis."""
    func_name = "initialize_system"
    logger.info(f"Starting tourism system initialization...")
    st.session_state.system_initialized = False
    st.session_state.initialization_status = "In Progress..."
    overall_success = True
    error_messages = []

    # --- Dependency Check ---
    logger.info("Checking tourism analysis dependencies...")
    mismatched = ensure_dependencies()
    if mismatched:
        missing_required = [f"{p}=={r} (Found: {i})" if i != "Missing" else f"{p}=={r} (Missing)" for p, r, i in mismatched if p != 'en_core_web_sm']
        missing_spacy_model = any(p == 'en_core_web_sm' for p, r, i in mismatched)
        if missing_required:
            if st.session_state.permissions["allow_package_install"]:
                if not all(install_package(f"{pkg}=={req_v}") for pkg, req_v, _ in mismatched if pkg != 'en_core_web_sm'):
                    overall_success = False
                    error_messages.append("Tourism package installation failed.")
            else:
                overall_success = False
                error_messages.append("Tourism dependencies missing. Grant installation permission.")
        if missing_spacy_model:
            if st.session_state.permissions["allow_package_install"]:
                if not install_package("en_core_web_sm"):
                    overall_success = False
                    error_messages.append("Tourism language model download failed.")
            else:
                overall_success = False
                error_messages.append("Tourism language model missing. Grant permission.")
    if not overall_success:
        logger.error(f"Tourism dependency check failed.")
        return False, "\n".join(error_messages)
    logger.info("Tourism dependencies OK.")

    # --- Ollama Check ---
    logger.info("Checking Ollama for tourism AI models...")
    ollama_ready = setup_ollama(install=st.session_state.permissions["allow_ollama_install"])
    if not ollama_ready:
        overall_success = False
        error_messages.append("Tourism AI engine (Ollama) setup failed.")
    if not overall_success:
        logger.error("Ollama setup failed.")
        return False, "\n".join(error_messages)
    logger.info("Tourism AI engine OK.")

    # --- Load NLP Resources ---
    logger.info("Loading tourism analysis models...")
    load_nltk_resources()
    nlp_model = load_spacy_model()
    embedding_model = load_embedding_model()
    if not nlp_model or not embedding_model:
        overall_success = False
        error_messages.append("Tourism NLP model initialization failed.")
    if not overall_success:
        logger.error("Tourism NLP model load failed.")
        return False, "\n".join(error_messages)
    logger.info("Tourism NLP models OK.")

    # --- Vector DB Initialization ---
    logger.info("Initializing tourism knowledge base...")
    db_success = initialize_vector_db()
    if not db_success:
        overall_success = False
        error_messages.append("Tourism knowledge base initialization failed.")
    if not overall_success:
        logger.error("Tourism knowledge base initialization failed.")
        return False, "\n".join(error_messages)
    logger.info("Tourism knowledge base OK.")

    # --- Refresh Models ---
    logger.info("Checking available tourism AI models...")
    st.session_state.available_models = refresh_available_models()
    logger.info(f"Found {len(st.session_state.available_models)} tourism AI models.")

    # --- Final ---
    if overall_success:
        st.session_state.initialization_complete = True
        st.session_state.system_initialized = True
        st.session_state.initialization_status = "Completed Successfully"
        logger.info("Tourism Explorer initialized successfully!")
        return True, "Tourism Explorer initialized successfully!"
    else:
        st.session_state.initialization_status = "Failed"
        logger.error(f"Tourism system initialization failed. Errors: {error_messages}")
        return False, "Tourism Explorer initialization failed. See errors above or check logs."

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_time = time.strftime("%Y-%m-%d %H:%M:%S")
        error_details = f"Unhandled tourism application error at {error_time}: {str(e)}"
        print(f"ERROR: {error_details}", file=sys.stderr)
        
        if "error_log" in st.session_state:
            log_error(error_details)
            st.error(f"A critical error occurred. Check logs or restart the Tourism Explorer. Error: {str(e)}")
        else:
            st.error(f"A critical Tourism Explorer error occurred: {str(e)}")