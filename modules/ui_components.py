"""
UI Components Module - Handles UI rendering and components.
"""
import streamlit as st
from typing import List, Dict, Any
import psutil

def display_chat(messages: List[Dict[str, str]], current_role: str = "Assistant"):
    """
    Render chat messages with improved styling and uniformity.
    User messages show question first, then "You".
    Assistant messages show role name first, then response.
    """
    if not messages:
        st.info("No messages yet. Ask a question to start the conversation.")
        return
    
    for msg in messages:
        if msg["role"] == "assistant":
            # Assistant message
            left_col, right_col = st.columns([1, 3])
            with left_col:
                st.markdown(f"**{current_role}:**")
            with right_col:
                st.markdown(f"{msg['content']}")
        else:
            # User message - question first, then "You"
            left_col, right_col = st.columns([3, 1])
            with left_col:
                st.markdown(f"{msg['content']}")
            with right_col:
                st.markdown("**:blue[You]**")
        
        # Add subtle divider between messages
        st.markdown("<hr style='margin: 5px 0; opacity: 0.3;'>", unsafe_allow_html=True)

def show_system_resources():
    """Display system resource usage."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        st.progress(cpu_percent / 100, text=f"CPU: {cpu_percent:.1f}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        st.progress(memory_percent / 100, text=f"Memory: {memory_percent:.1f}%")
        
        # Available memory
        available_memory_gb = memory.available / (1024 ** 3)
        st.text(f"Available Memory: {available_memory_gb:.2f} GB")
        
        # Disk usage for the current directory
        disk = psutil.disk_usage('.')
        disk_percent = disk.percent
        st.text(f"Disk Usage: {disk_percent:.1f}%")
        
    except Exception as e:
        st.warning(f"Could not retrieve system resources: {str(e)}")

def render_file_uploader():
    """Render the file uploader component."""
    return st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )

def render_agent_role_selector(current_role: str, roles: Dict[str, str]):
    """Render the agent role selector dropdown."""
    return st.selectbox(
        "Assistant Role",
        options=list(roles.keys()),
        index=list(roles.keys()).index(current_role) if current_role in roles else 0
    )

def render_model_selector(available_models: List[str], default_model: str = "llama3.2:latest"):
    """Render the model selector dropdown."""
    model_options = available_models if available_models else [default_model]
    
    # Ensure default model is included
    if default_model not in model_options:
        model_options.insert(0, default_model)
    
    return st.selectbox(
        "Select LLM Model",
        options=model_options,
        index=0
    )