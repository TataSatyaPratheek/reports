import streamlit as st
import time
import os
import sys
import asyncio
from typing import Tuple, Dict, Any, List, Optional

# Import modules
from modules.system_setup import (
    ensure_dependencies, setup_ollama, refresh_available_models,
    install_package, download_model, DEFAULT_MODEL_NAME
)
from modules.vector_store import initialize_vector_db, reset_vector_db, get_chroma_collection
from modules.nlp_models import load_nltk_resources, load_spacy_model, load_embedding_model
from modules.pdf_processor import process_uploaded_pdf
from modules.vector_store import add_chunks_to_collection
from modules.llm_interface import query_llm, SlidingWindowMemory
from modules.ui_components import display_chat, show_system_resources
from modules.utils import log_error, PerformanceMonitor

# --- Constants and AGENT_ROLES ---
DEFAULT_CHUNK_SIZE = 250
DEFAULT_OVERLAP = 50
DEFAULT_TOP_N = 10
DEFAULT_CONVERSATION_MEMORY = 3
DEFAULT_HYBRID_ALPHA = 0.7  # Weight balance between vector and BM25 search

# Agent roles with system prompts
AGENT_ROLES = {
    "Financial Analyst": "You are an expert financial analyst assistant. Your goal is to analyze financial documents, reports, and data to provide clear insights and explanations. Focus on identifying key financial metrics, trends, and potential implications from the provided document context. Use precise terminology but explain concepts in an accessible way.",
    
    "Academic Research Assistant": "You are an academic research assistant with expertise across multiple disciplines. Your role is to help analyze academic papers, research findings, and scholarly content. Focus on accurately representing the research methodology, findings, and implications from the provided document context. Maintain academic rigor while making concepts accessible.",
    
    "Technical Documentation Expert": "You are a technical documentation expert specializing in explaining complex technical concepts. Your goal is to help users understand technical documents, code, APIs, and systems. Focus on providing clear, structured explanations of technical content from the provided document context. Break down complex ideas into understandable components.",
    
    "Legal Document Analyzer": "You are a legal document analyzer with expertise in interpreting legal texts. Your role is to help users understand legal documents, contracts, regulations, and terminology. Focus on explaining the key provisions, implications, and potential considerations from the provided document context. Provide clear explanations while maintaining accuracy.",
    
    "Medical Literature Assistant": "You are a medical literature assistant with knowledge of medical terminology and research. Your goal is to help analyze medical publications, studies, and health information. Focus on accurately explaining medical concepts, research findings, and health information from the provided document context. Present information clearly while maintaining scientific accuracy.",
    
    "General Assistant": "You are a helpful assistant specialized in analyzing documents and providing accurate information. Your goal is to help users understand the content and extract relevant information from documents. Focus on presenting key information, insights, and explanations from the provided document context in a clear, accessible manner.",
    
    "Custom": ""  # Placeholder for custom prompt
}

# --- initialize_session_state with new features ---
def initialize_session_state():
    """Initialize or reset session state variables."""
    st.session_state.setdefault("processed_files", set())
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("sliding_window_memory", SlidingWindowMemory(max_tokens=2048))
    st.session_state.setdefault("system_initialized", False)
    st.session_state.setdefault("initialization_complete", False)
    st.session_state.setdefault("initialization_status", "Pending")
    st.session_state.setdefault("error_log", [])
    st.session_state.setdefault("available_models", [DEFAULT_MODEL_NAME])
    st.session_state.setdefault("current_agent_role", "General Assistant")
    st.session_state.setdefault("custom_prompt", AGENT_ROLES.get("General Assistant", ""))
    st.session_state.setdefault("permissions", {
        "allow_system_check": True,
        "allow_package_install": False,
        "allow_ollama_install": False,
        "allow_model_download": False
    })
    
    # New feature flags
    st.session_state.setdefault("use_hybrid_retrieval", True)
    st.session_state.setdefault("use_query_reformulation", True)
    st.session_state.setdefault("use_reranker", True)
    st.session_state.setdefault("hybrid_alpha", DEFAULT_HYBRID_ALPHA)

# --- initialize_system (Uses REVISED vector_store.initialize_vector_db) ---
def initialize_system() -> Tuple[bool, str]:
    """ Initialize system components. Returns (success, message). """
    func_name = "initialize_system"
    log_error(f"{func_name}: Starting...")
    st.session_state.system_initialized = False
    st.session_state.initialization_status = "In Progress..."
    overall_success = True
    error_messages = []

    # --- Dependency Check ---
    log_error(f"{func_name}: Checking dependencies...")
    mismatched = ensure_dependencies()
    if mismatched:
        missing_required = [f"{p}=={r} (Found: {i})" if i != "Missing" else f"{p}=={r} (Missing)" for p, r, i in mismatched if p != 'en_core_web_sm']
        missing_spacy_model = any(p == 'en_core_web_sm' for p, r, i in mismatched)
        if missing_required:
            if st.session_state.permissions["allow_package_install"]:
                if not all(install_package(f"{pkg}=={req_v}") for pkg, req_v, _ in mismatched if pkg != 'en_core_web_sm'):
                    overall_success = False
                    error_messages.append("Package install failed.")
            else:
                overall_success = False
                error_messages.append("Dependency issues. Grant permission.")
        if missing_spacy_model:
            if st.session_state.permissions["allow_package_install"]:
                if not install_package("en_core_web_sm"):
                    overall_success = False
                    error_messages.append("SpaCy model download failed.")
            else:
                overall_success = False
                error_messages.append("SpaCy model missing. Grant permission.")
    if not overall_success:
        log_error(f"{func_name}: Failed - Dependency check.")
        return False, "\n".join(error_messages)
    log_error(f"{func_name}: Dependencies OK.")

    # --- Ollama Check ---
    log_error(f"{func_name}: Checking Ollama...")
    ollama_ready = setup_ollama(install=st.session_state.permissions["allow_ollama_install"])
    if not ollama_ready:
        overall_success = False
        error_messages.append("Ollama setup failed.")
    if not overall_success:
        log_error(f"{func_name}: Failed - Ollama setup.")
        return False, "\n".join(error_messages)
    log_error(f"{func_name}: Ollama OK.")

    # --- Load Resources (Models are cached via @st.cache_resource) ---
    log_error(f"{func_name}: Loading NLTK...")
    load_nltk_resources()
    log_error(f"{func_name}: Loading SpaCy model...")
    nlp_model = load_spacy_model()
    log_error(f"{func_name}: Loading Embedding model...")
    embedding_model = load_embedding_model()
    if not nlp_model or not embedding_model:
        overall_success = False
        error_messages.append("NLP model load failed.")
    if not overall_success:
        log_error(f"{func_name}: Failed - NLP model load.")
        return False, "\n".join(error_messages)
    log_error(f"{func_name}: NLP Models OK.")

    # --- Vector DB Initialization (Uses REVISED initialize_vector_db) ---
    log_error(f"{func_name}: Initializing Vector DB...")
    # This now just calls the function that ensures the cached client/collection exists
    db_success = initialize_vector_db()
    if not db_success:
        overall_success = False
        error_messages.append("Vector DB initialization failed.")  # Message from initialize_vector_db shown in UI
    if not overall_success:
        log_error(f"{func_name}: Failed - Vector DB init.")
        return False, "\n".join(error_messages)
    log_error(f"{func_name}: Vector DB OK.")

    # --- Refresh Models ---
    log_error(f"{func_name}: Refreshing Ollama models...")
    st.session_state.available_models = refresh_available_models()
    log_error(f"{func_name}: Found {len(st.session_state.available_models)} models.")

    # --- Final ---
    if overall_success:
        st.session_state.initialization_complete = True
        st.session_state.system_initialized = True
        st.session_state.initialization_status = "Completed Successfully"
        log_error(f"{func_name}: System initialization completed successfully.")
        return True, "System initialization complete!"
    else:
        st.session_state.initialization_status = "Failed"
        log_error(f"{func_name}: System initialization failed. Errors: {error_messages}")
        return False, "System initialization failed. See errors above or check logs."


# --- render_sidebar (Enhanced with new feature controls) ---
def render_sidebar():
    """Render the sidebar UI with enhanced features."""
    with st.sidebar:
        st.markdown("## 📄 PDF Analyzer Settings")
        st.markdown("---")
        # --- Initialization Block ---
        show_init_block = not st.session_state.initialization_complete or st.session_state.initialization_status == "Failed"
        if show_init_block:
            st.markdown("### 🚀 System Setup")
            if st.session_state.initialization_status == "Failed":
                 st.error("Previous initialization failed. Please check permissions and retry.")
            else:
                st.info("System requires initialization.")
                
            st.markdown("**Permissions:**")
            st.caption("Grant permissions for automated setup.")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.permissions["allow_package_install"] = st.checkbox(
                    "Allow Package Install/Update",
                    value=st.session_state.permissions["allow_package_install"],
                    help="Allow the system to install or update required Python packages."
                )
            with col2:
                st.session_state.permissions["allow_ollama_install"] = st.checkbox(
                    "Allow Ollama Install",
                    value=st.session_state.permissions["allow_ollama_install"],
                    help="Allow the system to install Ollama if not already installed."
                )
                st.session_state.permissions["allow_model_download"] = st.checkbox(
                    "Allow Model Download",
                    value=st.session_state.permissions["allow_model_download"],
                    help="Allow the system to download LLM models via Ollama."
                )
                
            if st.button("Initialize System", key="init_button", use_container_width=True, type="primary"):
                 with st.status("Initializing system...", expanded=True) as status:
                     success, message = initialize_system()
                     if success:
                        status.update(label=message, state="complete", expanded=False)
                        st.success("Initialization successful!")
                        time.sleep(1.5)
                        st.rerun()
                     else:
                        status.update(label="Initialization Failed", state="error", expanded=True)
                        st.error(f"Details: {message}")
                        
            st.markdown(f"**Status:** `{st.session_state.initialization_status}`")
            st.markdown("---")

        # --- Initialized Sections ---
        if st.session_state.system_initialized:
            with st.expander("📊 System Status", expanded=False):
                show_system_resources()
                
            st.markdown("---")
            st.markdown("### 📄 Document Management")
            uploaded_files = st.file_uploader(
                "Upload PDF Files",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload PDF documents to analyze and chat with."
            )
            
            if st.session_state.processed_files:
                with st.expander(f"📋 Processed Files ({len(st.session_state.processed_files)})", expanded=False):
                    for filename in sorted(list(st.session_state.processed_files)):
                        st.caption(f"• {filename}")

            # --- Reset Database Section (Forces Manual Re-init) ---
            with st.expander("🗑️ Reset Database", expanded=False):
                st.warning("Permanently deletes analyzed data & requires re-initialization.")
                reset_placeholder = st.empty()
                if st.button("Confirm Reset", key="reset_vector_db", use_container_width=True, type="secondary"):
                    try:
                        with st.spinner("Resetting database and clearing cache..."):
                            reset_success, reset_message = reset_vector_db()  # Calls new reset
                        if reset_success:
                            reset_placeholder.success(reset_message)
                            st.session_state.processed_files.clear()
                            st.session_state.messages = []
                            # Reset sliding window memory
                            if "sliding_window_memory" in st.session_state:
                                st.session_state.sliding_window_memory.clear()
                            # Reset flags to force manual re-init via UI
                            st.session_state.system_initialized = False
                            st.session_state.initialization_complete = False
                            st.session_state.initialization_status = "Pending"
                            log_error("Reset successful. Flags set for user re-initialization.")
                            time.sleep(2)
                            st.rerun()  # Rerun shows init block
                        else:
                            reset_placeholder.error(f"Database reset failed: {reset_message}")
                    except Exception as e:
                         log_error(f"Critical error during reset button action: {str(e)}")
                         reset_placeholder.error(f"An unexpected error occurred during reset: {str(e)}")
            # --- End Reset Database Section ---

            st.markdown("---")
            st.markdown("### ⚙️ Configuration")
            
            # --- Role Selector ---
            current_role = st.session_state.get("current_agent_role", "General Assistant")
            role_options = list(AGENT_ROLES.keys())
            try:
                current_role_index = role_options.index(current_role)
            except ValueError:
                current_role_index = role_options.index("General Assistant")
                
            selected_role = st.selectbox(
                "Assistant Role",
                options=role_options,
                index=current_role_index,
                key="agent_role_selector",
                help="Select the role that best matches your document type."
            )
            
            if selected_role != st.session_state.current_agent_role:
                st.session_state.current_agent_role = selected_role
                st.session_state.custom_prompt = AGENT_ROLES.get(selected_role, "") if selected_role != "Custom" else st.session_state.custom_prompt
                st.rerun()
                
            # --- Custom Prompt ---
            is_custom_role = (st.session_state.current_agent_role == "Custom")
            with st.expander("🔍 Custom System Prompt", expanded=is_custom_role):
                custom_prompt_input = st.text_area(
                    "System Prompt",
                    value=st.session_state.custom_prompt,
                    height=150,
                    key="custom_prompt_text_area",
                    help="Customize the system instructions for the LLM."
                )
                if is_custom_role and custom_prompt_input != st.session_state.custom_prompt:
                    st.session_state.custom_prompt = custom_prompt_input
                    
            # --- LLM Model ---
            with st.expander("🤖 LLM Model", expanded=False):
                model_options = st.session_state.available_models or [DEFAULT_MODEL_NAME]
                current_model = st.session_state.get("local_llm_model", DEFAULT_MODEL_NAME)
                
                if DEFAULT_MODEL_NAME not in model_options:
                    model_options.insert(0, DEFAULT_MODEL_NAME)
                    
                try:
                    current_model_index = model_options.index(current_model)
                except ValueError:
                    current_model_index = 0
                    
                selected_model = st.selectbox(
                    "Select LLM Model",
                    options=model_options,
                    index=current_model_index,
                    key="model_selector",
                    help="Choose the local LLM model to use for generating responses."
                )
                st.session_state.local_llm_model = selected_model
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Refresh List", key="refresh_models", use_container_width=True):
                        with st.spinner("Checking Ollama models..."):
                            st.session_state.available_models = refresh_available_models()
                            st.rerun()
                            
                with col2:
                    if (st.session_state.permissions["allow_model_download"] and 
                        DEFAULT_MODEL_NAME not in st.session_state.available_models):
                        if st.button(f"Download {DEFAULT_MODEL_NAME}", key="download_default", use_container_width=True):
                            with st.status(f"Downloading {DEFAULT_MODEL_NAME}...", expanded=True) as dl_status:
                                success, msg = download_model(DEFAULT_MODEL_NAME)
                                if success:
                                    dl_status.update(label=f"{DEFAULT_MODEL_NAME} Downloaded!", state="complete", expanded=False)
                                    st.session_state.available_models = refresh_available_models()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    dl_status.update(label="Download Failed", state="error", expanded=True)
                                    st.error(msg)
                    elif not st.session_state.permissions["allow_model_download"]:
                        st.caption("Model download disabled.")
                        
            # --- Processing Settings ---
            with st.expander("⚡ Processing", expanded=False):
                st.session_state.chunk_size = st.slider(
                    "Chunk Size",
                    50, 1000,
                    st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
                    50,
                    help="Size of document chunks in words. Smaller chunks provide more focused context, larger chunks preserve more context."
                )
                
                st.session_state.overlap = st.slider(
                    "Overlap",
                    0, 200,
                    st.session_state.get("overlap", DEFAULT_OVERLAP),
                    10,
                    help="Overlap between adjacent chunks in words. Higher overlap helps maintain context between chunks."
                )
                
                st.session_state.top_n = st.slider(
                    "Top Results",
                    1, 20,
                    st.session_state.get("top_n", DEFAULT_TOP_N),
                    help="Number of relevant document chunks to retrieve for each query."
                )
                
                st.session_state.conversation_memory_count = st.slider(
                    "Memory Turns",
                    0, 10,
                    st.session_state.get("conversation_memory_count", DEFAULT_CONVERSATION_MEMORY),
                    help="Number of previous conversation turns to include for context."
                )
                
            # --- Advanced Features (New) ---
            with st.expander("🧠 Advanced Features", expanded=False):
                st.session_state.use_hybrid_retrieval = st.toggle(
                    "Hybrid Retrieval",
                    value=st.session_state.get("use_hybrid_retrieval", True),
                    help="Combine vector similarity and keyword search for better results."
                )
                
                if st.session_state.use_hybrid_retrieval:
                    st.session_state.hybrid_alpha = st.slider(
                        "Vector/Keyword Balance",
                        0.0, 1.0,
                        st.session_state.get("hybrid_alpha", DEFAULT_HYBRID_ALPHA),
                        0.1,
                        help="Balance between vector similarity (1.0) and keyword search (0.0)."
                    )
                
                st.session_state.use_reranker = st.toggle(
                    "Reranker",
                    value=st.session_state.get("use_reranker", True),
                    help="Use neural reranker to improve search result order."
                )
                
                st.session_state.use_query_reformulation = st.toggle(
                    "Query Reformulation",
                    value=st.session_state.get("use_query_reformulation", True),
                    help="Automatically improve queries based on conversation context."
                )
                
                if st.button("Clear Conversation", key="clear_conversation", use_container_width=True):
                    st.session_state.messages = []
                    if "sliding_window_memory" in st.session_state:
                        st.session_state.sliding_window_memory.clear()
                    st.success("Conversation cleared.")
                
            # --- Advanced Options ---
            with st.expander("🛠️ Advanced", expanded=False):
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
            "system_prompt": st.session_state.get("custom_prompt", AGENT_ROLES.get("General Assistant", "")),
            "system_initialized": st.session_state.get("system_initialized", False),
            "use_hybrid_retrieval": st.session_state.get("use_hybrid_retrieval", True),
            "use_reranker": st.session_state.get("use_reranker", True), 
            "use_query_reformulation": st.session_state.get("use_query_reformulation", True),
            "hybrid_alpha": st.session_state.get("hybrid_alpha", DEFAULT_HYBRID_ALPHA)
        }


# --- process_documents (Uses REVISED get_chroma_collection) ---
def process_documents(uploaded_files, chunk_size, overlap):
    """Process uploaded PDF documents using st.status for feedback."""
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        return
    if not uploaded_files:
        return

    st.markdown("---")
    st.subheader("📝 Document Processing")
    col1, col2 = st.columns([3, 1])
    batch_status_text = col1.empty()
    batch_progress = col2.empty()
    files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    total_new_files = len(files_to_process)
    
    if total_new_files == 0:
        batch_status_text.info("No new files to process.")
        time.sleep(1.5)
        batch_status_text.empty()
        return

    embedding_model = load_embedding_model()
    collection = get_chroma_collection()  # Uses the new logic

    if not embedding_model or not collection:
        st.error("Core components (embedding model or DB collection) not available for processing.")
        log_error("Processing failed: Missing embedding model or collection.")
        return

    files_processed_count = 0
    errors_occurred = False
    batch_progress.progress(0)
    
    for i, pdf_file in enumerate(files_to_process):
        with st.status(f"Processing {pdf_file.name}...", expanded=True) as file_status:
            try:
                new_chunks = process_uploaded_pdf(pdf_file, chunk_size, overlap, status=file_status)
                if new_chunks:
                    add_success = add_chunks_to_collection(new_chunks, embedding_model, collection, status=file_status)
                    if add_success:
                        st.session_state.processed_files.add(pdf_file.name)
                        files_processed_count += 1
                        file_status.update(label=f"Successfully processed {pdf_file.name}", state="complete", expanded=False)
                    else:
                        errors_occurred = True
                else:
                    errors_occurred = True
                    if file_status._label.startswith("Generated"):
                        file_status.update(label=f"No text chunks extracted from {pdf_file.name}", state="warning", expanded=False)
            except Exception as e:
                log_error(f"Critical error processing {pdf_file.name} in main loop: {str(e)}")
                errors_occurred = True
                file_status.update(label=f"Unexpected error processing {pdf_file.name}", state="error", expanded=True)
                st.error(f"Details: {str(e)}")
                
        batch_progress.progress((i + 1) / total_new_files)

    if files_processed_count > 0 and not errors_occurred:
        batch_status_text.success(f"Successfully processed {files_processed_count} new file(s).")
    elif files_processed_count > 0 and errors_occurred:
        batch_status_text.warning(f"Processed {files_processed_count} file(s), but errors occurred.")
    elif errors_occurred:
        batch_status_text.error("Failed to process new files. Check status messages.")
        
    time.sleep(3)
    batch_status_text.empty()
    batch_progress.empty()


# --- render_chat_interface (Enhanced with new features) ---
def render_chat_interface(params):
    """Render the chat interface with enhanced features."""
    st.markdown("---")
    st.subheader("💬 Chat with Your Documents")
    chat_display_area = st.container()
    
    with chat_display_area:
        display_chat(
            st.session_state.messages,
            current_role=st.session_state.get("current_agent_role", "Assistant")
        )
        
    st.markdown("---")
    chat_input_disabled = not bool(st.session_state.processed_files)
    input_placeholder = "Ask a question..." if not chat_input_disabled else "Please process documents first"
    
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area(
            "Your question:",
            height=100,
            placeholder=input_placeholder,
            disabled=chat_input_disabled,
            key="chat_input"
        )
        submitted = st.form_submit_button(
            "Send",
            use_container_width=True,
            disabled=chat_input_disabled
        )

    if submitted and user_query.strip() and not chat_input_disabled:
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Add to sliding window memory
        if "sliding_window_memory" in st.session_state:
            st.session_state.sliding_window_memory.add("User", user_query)
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_user_query = st.session_state.messages[-1]["content"]
        
        # Get conversation memory - either from sliding window or fallback to old method
        if "sliding_window_memory" in st.session_state:
            conversation_memory = st.session_state.sliding_window_memory.get_formatted_history()
        else:
            memory_limit = params["conversation_memory_count"] * 2
            memory_slice = st.session_state.messages[-(memory_limit + 1) : -1] if memory_limit > 0 else []
            conversation_memory = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in memory_slice)

        embedding_model = load_embedding_model()
        collection = get_chroma_collection()  # Uses the new logic

        if not embedding_model or not collection:
             answer = "Error: Cannot query LLM - core components missing (model or DB collection)."
             log_error(answer)
             st.session_state.messages.append({"role": "assistant", "content": answer})
             st.rerun()
        else:
            # Use enhanced query_llm with new parameters
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


# --- main (Enhanced with new feature params) ---
def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="PDF Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
        /* Scrollbar styles */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: #888; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
        
        /* Header styles */
        .main-header { font-size: 2.5rem; font-weight: 600; color: #1E88E5; margin-bottom: 0.2rem; text-align: center; }
        .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; text-align: center; }
        
        /* Message styles */
        .stChatMessage { border-radius: 10px; padding: 0.75rem; margin-bottom: 0.5rem; }
        
        /* Button styles */
        .stButton>button { border-radius: 8px; }
        
        /* Status widget styles */
        .stStatusWidget-content { padding-top: 0.5rem; padding-bottom: 0.5rem; overflow-wrap: break-word; word-wrap: break-word; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">Enhanced PDF Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with your documents using local AI models with advanced RAG features</p>', unsafe_allow_html=True)

    if "system_initialized" not in st.session_state:
        initialize_session_state()

    sidebar_params = render_sidebar()

    if not sidebar_params["system_initialized"]:
        if st.session_state.initialization_status in ["Pending", "Failed"]:
             st.warning("System not ready. Please initialize the system via the sidebar.")
    else:
        process_documents(
            sidebar_params["uploaded_files"],
            sidebar_params["chunk_size"],
            sidebar_params["overlap"]
        )
        render_chat_interface(sidebar_params)

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_time = time.strftime("%Y-%m-%d %H:%M:%S")
        error_details = f"Unhandled application error at {error_time}: {str(e)}"
        print(f"ERROR: {error_details}", file=sys.stderr)
        
        if "error_log" in st.session_state:
            log_error(error_details)
            st.error(f"A critical error occurred. Check logs or restart. Error: {str(e)}")
        else:
            st.error(f"A critical application error occurred: {str(e)}")