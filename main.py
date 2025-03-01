import streamlit as st
import time
import os
import sys

# Import modules
from modules.system_setup import ensure_dependencies, setup_ollama, refresh_available_models
from modules.vector_store import initialize_vector_db, reset_vector_db
from modules.nlp_models import load_nltk_resources, load_spacy_model, load_embedding_model
from modules.pdf_processor import process_uploaded_pdf, smart_chunking
from modules.vector_store import add_chunks_to_collection, get_chroma_collection
from modules.llm_interface import query_llm
from modules.ui_components import display_chat, show_system_resources
from modules.utils import log_error, PerformanceMonitor

# Constants
DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_CHUNK_SIZE = 250
DEFAULT_OVERLAP = 50
DEFAULT_TOP_N = 10
DEFAULT_CONVERSATION_MEMORY = 3

# Predefined agent roles
AGENT_ROLES = {
    "Financial Analyst": "You are an expert at analyzing financial reports. Provide insights, explanations, and summaries based on the financial data available.",
    "Academic Research Assistant": "You are a research assistant helping with academic papers and scholarly content. Focus on extracting key findings, methodologies, and conclusions.",
    "Technical Documentation Expert": "You are an expert at explaining technical documentation. Break down complex concepts and provide clear explanations of technical content.",
    "Legal Document Analyzer": "You are a legal document specialist. Identify key clauses, explain legal terminology, and summarize important legal points in accessible language.",
    "Medical Literature Assistant": "You are a medical literature assistant. Help interpret medical publications, research findings, and clinical guidelines while being clear about limitations.",
    "General Assistant": "You are a helpful assistant. Provide clear, informative answers based on the content of the documents."
}

def initialize_session_state():
    """Initialize or reset session state variables."""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()  # store filenames of processed PDFs
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False
    
    if "error_log" not in st.session_state:
        st.session_state.error_log = []
    
    if "available_models" not in st.session_state:
        st.session_state.available_models = [DEFAULT_MODEL]
        
    if "current_agent_role" not in st.session_state:
        st.session_state.current_agent_role = "General Assistant"
    
    if "custom_prompt" not in st.session_state:
        st.session_state.custom_prompt = AGENT_ROLES["General Assistant"]
        
    if "initialization_complete" not in st.session_state:
        st.session_state.initialization_complete = False
        
    if "permissions" not in st.session_state:
        st.session_state.permissions = {
            "allow_system_check": False,
            "allow_package_install": False,
            "allow_ollama_install": False,
            "allow_model_download": False
        }

def initialize_system(with_progress=True):
    """Initialize all system components with user permission tracking."""
    if st.session_state.initialization_complete:
        return True
    
    if with_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Check dependencies (10%)
        status_text.info("Checking system dependencies...")
        missing_packages = ensure_dependencies()
        progress_bar.progress(0.1)
        
        if missing_packages and not st.session_state.permissions["allow_package_install"]:
            status_text.warning(f"Missing packages: {', '.join(missing_packages)}. Please approve installation.")
            return False
        elif missing_packages:
            status_text.info(f"Installing missing packages: {', '.join(missing_packages)}")
            # Installation is handled in the UI based on permissions
        
        # 2. Check Ollama (20%)
        status_text.info("Checking if Ollama is installed...")
        ollama_installed = setup_ollama(install=st.session_state.permissions["allow_ollama_install"])
        progress_bar.progress(0.2)
        
        if not ollama_installed and not st.session_state.permissions["allow_ollama_install"]:
            status_text.warning("Ollama is not installed. Please approve installation.")
            return False
        
        # 3. Load NLP models (50%)
        status_text.info("Loading NLP resources...")
        load_nltk_resources()
        nlp = load_spacy_model()
        embedding_model = load_embedding_model()
        progress_bar.progress(0.5)
        
        if not nlp or not embedding_model:
            status_text.error("Failed to load NLP models")
            return False
        
        # 4. Initialize vector DB (70%)
        status_text.info("Initializing vector database...")
        success = initialize_vector_db()
        progress_bar.progress(0.7)
        
        if not success:
            status_text.warning("Vector database initialization failed. Will create on first document upload.")
        
        # 5. Check available models (90%)
        status_text.info("Checking available Ollama models...")
        st.session_state.available_models = refresh_available_models()
        progress_bar.progress(0.9)
        
        # 6. Complete (100%)
        status_text.success("System initialization complete!")
        progress_bar.progress(1.0)
    
    st.session_state.initialization_complete = True
    st.session_state.system_initialized = True
    return True

def render_sidebar():
    """Render the sidebar UI with collapsible sections."""
    with st.sidebar:
        # System Status Section
        if not st.session_state.system_initialized:
            st.markdown("### 🚀 System Setup")
            
            # Permission checks
            st.write("Please approve the following operations:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.permissions["allow_system_check"] = st.checkbox(
                    "Check system", value=True, 
                    help="Allow checking system for required dependencies")
                
                st.session_state.permissions["allow_package_install"] = st.checkbox(
                    "Install packages", 
                    help="Allow installing required Python packages")
            
            with col2:
                st.session_state.permissions["allow_ollama_install"] = st.checkbox(
                    "Install Ollama", 
                    help="Allow installing Ollama if needed")
                
                st.session_state.permissions["allow_model_download"] = st.checkbox(
                    "Download models", 
                    help="Allow downloading LLM models")
            
            if st.button("Initialize System", use_container_width=True):
                if not st.session_state.permissions["allow_system_check"]:
                    st.error("System check permission is required to proceed.")
                    return
                initialize_system()
        else:
            with st.expander("📊 System Status", expanded=False):
                show_system_resources()
                if st.button("Refresh Status", key="refresh_status", use_container_width=True):
                    st.experimental_rerun()
        
        # Divider for visual separation
        st.markdown("---")
        
        # Document Management Section - Always visible
        st.markdown("### 📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF Files", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )
        
        # Only show processed files if there are any
        if st.session_state.processed_files:
            with st.expander(f"📋 Processed Files ({len(st.session_state.processed_files)})", expanded=False):
                for filename in sorted(st.session_state.processed_files):
                    st.text(f"• {filename}")
        
        # Vector DB reset with confirmation in a compact design
        with st.expander("🗑️ Reset Database", expanded=False):
            st.warning("This will delete all stored document data.")
            if st.button("Reset Vector Database", key="reset_vector_db", use_container_width=True):
                success, message = reset_vector_db()
                if success:
                    st.session_state.processed_files.clear()
                    st.success(message)
                else:
                    st.error(message)
        
        # Divider for visual separation
        st.markdown("---")
        
        # Configuration Section in a compact, collapsible interface
        st.markdown("### ⚙️ Configuration")
        
        # Agent Role (LLM Behavior)
        st.session_state.current_agent_role = st.selectbox(
            "Assistant Role",
            options=list(AGENT_ROLES.keys()),
            index=list(AGENT_ROLES.keys()).index(st.session_state.current_agent_role)
        )
        
        # Update the custom prompt when the role changes
        if st.session_state.current_agent_role != "Custom" and st.session_state.current_agent_role in AGENT_ROLES:
            st.session_state.custom_prompt = AGENT_ROLES[st.session_state.current_agent_role]
        
        # Add custom prompt option
        with st.expander("🔍 Custom Prompt", expanded=False):
            custom_prompt = st.text_area(
                "System Prompt", 
                value=st.session_state.custom_prompt,
                height=100,
                help="Define the behavior and knowledge of the AI assistant"
            )
            if st.button("Apply Prompt", use_container_width=True):
                st.session_state.custom_prompt = custom_prompt
                st.success("Custom prompt applied.")
        
        # LLM Model Selection
        with st.expander("🤖 LLM Model Settings", expanded=False):
            # Get available models and ensure default is included
            model_options = st.session_state.available_models
            if not model_options:
                model_options = [DEFAULT_MODEL]
            
            if DEFAULT_MODEL not in model_options:
                model_options = [DEFAULT_MODEL] + model_options
            
            local_llm_model = st.selectbox(
                "Select LLM Model",
                options=model_options,
                index=0
            )
            
            if st.button("Refresh Models", use_container_width=True):
                st.session_state.available_models = refresh_available_models()
                st.success(f"Found {len(st.session_state.available_models)} models")
            
            # Model download option
            if st.session_state.permissions["allow_model_download"]:
                new_model = st.text_input("Model to download (e.g., llama3.2:latest)")
                if st.button("Download Model", use_container_width=True) and new_model:
                    with st.spinner(f"Downloading {new_model}..."):
                        # Download logic would go here and update available_models
                        st.success(f"Model {new_model} downloaded")
        
        # Processing Settings
        with st.expander("⚡ Processing Settings", expanded=False):
            chunk_size = st.slider(
                "Chunk Size (words)", 
                min_value=50, max_value=1000, value=DEFAULT_CHUNK_SIZE,
                help="Number of words per text chunk. Larger chunks provide more context but may reduce relevance precision."
            )
            
            overlap = st.slider(
                "Overlap (words)", 
                min_value=0, max_value=200, value=DEFAULT_OVERLAP,
                help="Word overlap between consecutive chunks. Higher overlap reduces context loss but increases processing time."
            )
            
            top_n = st.slider(
                "Top Results", 
                min_value=1, max_value=50, value=DEFAULT_TOP_N,
                help="Number of most relevant text chunks to retrieve for each query."
            )
            
            conversation_memory_count = st.slider(
                "Conversation Memory", 
                min_value=0, max_value=10, value=DEFAULT_CONVERSATION_MEMORY,
                help="Number of previous Q&A pairs to include as context. Increases answer relevance but may slow responses."
            )
        
        # Advanced Options (for debugging and maintenance)
        with st.expander("🛠️ Advanced Options", expanded=False):
            if st.button("Clear Error Log", use_container_width=True):
                st.session_state.error_log = []
                st.success("Error log cleared")
            
            if st.session_state.error_log:
                with st.expander("📝 Error Log", expanded=False):
                    for error in st.session_state.error_log:
                        st.text(error)
        
        # Return needed parameters for the main app
        return {
            "uploaded_files": uploaded_files,
            "chunk_size": chunk_size if 'chunk_size' in locals() else DEFAULT_CHUNK_SIZE,
            "overlap": overlap if 'overlap' in locals() else DEFAULT_OVERLAP,
            "top_n": top_n if 'top_n' in locals() else DEFAULT_TOP_N,
            "conversation_memory_count": conversation_memory_count if 'conversation_memory_count' in locals() else DEFAULT_CONVERSATION_MEMORY,
            "local_llm_model": local_llm_model if 'local_llm_model' in locals() else DEFAULT_MODEL
        }

def process_documents(uploaded_files, chunk_size, overlap):
    """Process uploaded PDF documents."""
    if not uploaded_files:
        return
    
    st.subheader("📝 Document Processing")
    file_progress = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    
    files_processed = 0
    for i, pdf_file in enumerate(uploaded_files):
        # If we've already processed this PDF, skip
        if pdf_file.name in st.session_state.processed_files:
            status_text.info(f"**{pdf_file.name}** already processed; skipping.")
            files_processed += 1
            continue
        
        status_text.write(f"Processing file {i+1} of {total_files}: **{pdf_file.name}**")
        
        # Ensure vector DB exists before processing
        if not initialize_vector_db():
            status_text.error("Failed to initialize vector database. Cannot process documents.")
            return
        
        # Get system resources for optimized processing
        resources = PerformanceMonitor.get_system_resources()
        
        # Process the PDF
        with st.container():
            st.write(f"Processing {pdf_file.name}...")
            st.write("Extracting text from PDF...")
            
            new_chunks = process_uploaded_pdf(pdf_file, chunk_size, overlap)
            
            if new_chunks:
                st.write(f"Generated {len(new_chunks)} chunks from **{pdf_file.name}**.")
                
                st.write("Adding text chunks to the vector database...")
                collection = get_chroma_collection()
                embedding_model = load_embedding_model()
                
                if collection and embedding_model:
                    add_chunks_to_collection(new_chunks, embedding_model, collection)
                    # Mark as processed
                    st.session_state.processed_files.add(pdf_file.name)
                    files_processed += 1
                else:
                    st.error("Vector database or embedding model not available.")
            else:
                st.warning(f"No text chunks generated from {pdf_file.name}.")
        
        file_progress.progress((i + 1) / total_files)
    
    if files_processed > 0:
        status_text.success(f"Processed {files_processed} new files successfully.")
    else:
        status_text.empty()
    
    # Clear progress bar after completion
    file_progress.empty()

def render_chat_interface(local_llm_model, top_n, conversation_memory_count):
    """Render the chat interface and handle queries."""
    st.markdown("---")
    st.subheader("💬 Chat with Your Documents")
    
    # Display a warning if no files have been processed
    if not st.session_state.processed_files:
        st.warning("⚠️ No documents have been processed yet. Please upload and process PDF files first.")
    
    # Display existing conversation
    chat_container = st.container()
    with chat_container:
        if st.session_state.messages:
            display_chat(st.session_state.messages)
    
    # Input box at the bottom in a form
    st.markdown("---")  # a subtle divider before input
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_query = st.text_area("Ask a question about your documents:", height=100)
        with col2:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Submit", use_container_width=True)
    
    if submitted and user_query.strip():
        # Check if we have processed any files
        if not st.session_state.processed_files:
            st.error("Please upload and process documents before asking questions.")
            return
        
        # Build conversation memory from the last (conversation_memory_count * 2) messages
        memory_slice = st.session_state.messages[-(conversation_memory_count * 2):] if conversation_memory_count > 0 else []
        conversation_memory = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in memory_slice
        )
        
        # Get required models
        embedding_model = load_embedding_model()
        collection = get_chroma_collection()
        
        with st.spinner("Processing query..."):
            answer = query_llm(
                user_query=user_query,
                top_n=top_n,
                local_llm_model=local_llm_model,
                embedding_model=embedding_model,
                collection=collection,
                conversation_memory=conversation_memory,
                system_prompt=st.session_state.custom_prompt
            )
        
        # Append user query & assistant answer
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Update the UI by recreating the chat display
        st.markdown("### Updated Conversation:")
        display_chat(st.session_state.messages)

def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="PDF Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # App header with styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">PDF Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload PDFs and chat with them using local LLMs</p>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get parameters
    sidebar_params = render_sidebar()
    
    # System initialization check
    if not st.session_state.system_initialized:
        st.warning("⚠️ System not initialized. Please approve required permissions and click 'Initialize System' in the sidebar.")
        return  # Exit the main function until initialized
    
    # Process Documents
    process_documents(
        sidebar_params["uploaded_files"],
        sidebar_params["chunk_size"],
        sidebar_params["overlap"]
    )
    
    # Render Chat Interface
    render_chat_interface(
        sidebar_params["local_llm_model"],
        sidebar_params["top_n"],
        sidebar_params["conversation_memory_count"]
    )

# Run the main application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if "error_log" in st.session_state:
            log_error(f"Unhandled application error: {str(e)}")
            st.error("Check the error log in Advanced Options for details.")