"""
Enhanced UI Components Module - Handles UI rendering and components.
Optimized for tourism RAG chatbot application.
"""
import streamlit as st
import os
NLTK_DATA_PATH = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = NLTK_DATA_PATH
import nltk
nltk.data.path = [NLTK_DATA_PATH]  # Override all other paths

from typing import List, Dict, Any, Optional, Tuple, Union
import psutil
import random
import pandas as pd
import numpy as np
import altair as alt
import datetime
import time
from modules.utils import log_error, create_directory_if_not_exists

# Theme colors for tourism UI
TOURISM_COLORS = {
    "primary": "#1E88E5",
    "secondary": "#26A69A",
    "accent": "#FFC107",
    "success": "#4CAF50",
    "warning": "#FF9800",
    "error": "#F44336",
    "ocean": "#03A9F4",
    "forest": "#4CAF50",
    "desert": "#FF9800",
    "mountain": "#795548",
    "city": "#607D8B",
}

# Tourism segment colors
SEGMENT_COLORS = {
    "luxury": "#B71C1C",
    "budget": "#004D40",
    "family": "#1565C0",
    "solo": "#6A1B9A",
    "adventure": "#EF6C00",
    "cultural": "#4527A0",
    "sustainability": "#2E7D32",
    "food": "#D84315",
    "romantic": "#AD1457",
    "wellness": "#00838F",
}

def apply_tourism_theme():
    """Apply custom tourism-themed CSS to the application."""
    st.markdown(f"""
    <style>
        /* Tourism theme colors */
        :root {{
            --primary-color: {TOURISM_COLORS["primary"]};
            --secondary-color: {TOURISM_COLORS["secondary"]};
            --accent-color: {TOURISM_COLORS["accent"]};
            --success-color: {TOURISM_COLORS["success"]};
            --warning-color: {TOURISM_COLORS["warning"]};
            --error-color: {TOURISM_COLORS["error"]};
        }}
        
        /* Scrollbar styles */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: #f1f1f1; border-radius: 10px; }}
        ::-webkit-scrollbar-thumb {{ background: #888; border-radius: 10px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #555; }}
        
        /* Header styles */
        .main-header {{ 
            font-size: 2.5rem; 
            font-weight: 600; 
            color: var(--primary-color); 
            margin-bottom: 0.2rem; 
            text-align: center; 
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }}
        
        .sub-header {{ 
            font-size: 1.1rem; 
            color: #555; 
            margin-bottom: 1.5rem; 
            text-align: center; 
        }}
        
        /* Card styles for chunks */
        .tourism-card {{
            border: 1px solid #eee;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }}
        
        .tourism-card:hover {{
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-color: var(--secondary-color);
        }}
        
        /* Message styles */
        .stChatMessage {{ 
            border-radius: 10px; 
            padding: 0.9rem; 
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        /* Chat user message specific */
        .stChatMessage [data-testid="StChatMessageContent"] div[data-testid="chatAvatarIcon-user"] p {{
            background-color: #E3F2FD !important;
            padding: 10px 15px;
            border-radius: 18px 18px 0px 18px;
        }}
        
        /* Chat assistant message specific */
        .stChatMessage [data-testid="StChatMessageContent"] div[data-testid="chatAvatarIcon-assistant"] p {{
            background-color: #E8F5E9 !important;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 0px;
        }}
        
        /* Button styles */
        .stButton>button {{ 
            border-radius: 8px; 
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{ 
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Status widget styles */
        .stStatusWidget-content {{ 
            padding-top: 0.5rem; 
            padding-bottom: 0.5rem; 
            overflow-wrap: break-word; 
            word-wrap: break-word; 
        }}
        
        /* Tourism-specific badges */
        .tourism-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        
        .tourism-badge-luxury {{
            background-color: {SEGMENT_COLORS["luxury"]};
            color: white;
        }}
        
        .tourism-badge-budget {{
            background-color: {SEGMENT_COLORS["budget"]};
            color: white;
        }}
        
        .tourism-badge-family {{
            background-color: {SEGMENT_COLORS["family"]};
            color: white;
        }}
        
        .tourism-badge-solo {{
            background-color: {SEGMENT_COLORS["solo"]};
            color: white;
        }}
        
        .tourism-badge-adventure {{
            background-color: {SEGMENT_COLORS["adventure"]};
            color: white;
        }}
        
        .tourism-badge-cultural {{
            background-color: {SEGMENT_COLORS["cultural"]};
            color: white;
        }}
        
        .tourism-badge-sustainability {{
            background-color: {SEGMENT_COLORS["sustainability"]};
            color: white;
        }}
    </style>
    """, unsafe_allow_html=True)

def display_chat(messages: List[Dict[str, str]], current_role: str = "Tourism Assistant"):
    """
    Render chat messages with improved styling and tourism-specific features.
    
    Args:
        messages: List of message dictionaries with "role" and "content"
        current_role: Role name for the assistant
    """
    if not messages:
        st.info("No messages yet. Ask a question about travel trends, payment methods, market segments, or specific tourism topics.")
        
        # Show sample questions when chat is empty
        with st.expander("Sample Questions", expanded=True):
            st.markdown("### Try asking about:")
            sample_questions = [
                "What are the main trends in travel for 2025?",
                "How do payment methods differ between luxury and budget travelers?",
                "What are the key market segments in tourism?",
                "Tell me about sustainability trends in tourism.",
                "How are Gen Z travelers different from other generations?",
                "What are the unique characteristics of luxury travel?",
                "What payment methods are popular for international travel?",
                "How can tourism businesses better target different segments?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(sample_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"sample_q_{i}", use_container_width=True):
                        # Add selected question to session state for parent to handle
                        if "selected_sample_question" not in st.session_state:
                            st.session_state.selected_sample_question = question
                            # Will be handled by parent component on next rerun
        return
    
    # Display messages
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            # Process assistant message for tourism-specific formatting
            message_content = process_tourism_message(msg["content"])
            
            # Assistant message
            left_col, right_col = st.columns([1, 5])
            with left_col:
                st.markdown(f"**{current_role}:**")
            with right_col:
                st.markdown(message_content)
        else:
            # User message - question first, then "You"
            left_col, right_col = st.columns([5, 1])
            with left_col:
                st.markdown(f"{msg['content']}")
            with right_col:
                st.markdown("**:blue[You]**")
        
        # Add subtle divider between messages
        st.markdown("<hr style='margin: 8px 0; opacity: 0.2;'>", unsafe_allow_html=True)

def process_tourism_message(content: str) -> str:
    """
    Process assistant message to highlight tourism-specific content.
    
    Args:
        content: Original message content
        
    Returns:
        Formatted message with highlighted terms
    """
    # Highlight tourism segments with badges
    for segment, color in SEGMENT_COLORS.items():
        # Only add badges for segments that actually appear in the content
        if segment.lower() in content.lower():
            # Create badge HTML
            badge = f'<span class="tourism-badge tourism-badge-{segment}">{segment.capitalize()}</span>'
            
            # Selective replacement (only entire words, not substrings)
            # We're not using simple replace to avoid replacing parts of words
            content = replace_word_with_badge(content, segment, badge)
    
    # Highlight payment methods
    payment_methods = ["credit card", "debit card", "cash", "digital wallet", "mobile payment", 
                     "cryptocurrency", "bank transfer", "prepaid card", "traveler's cheque"]
    
    for method in payment_methods:
        if method.lower() in content.lower():
            # Use a special formatting for payment methods
            highlight = f'<span style="background-color: #E3F2FD; padding: 2px 5px; border-radius: 4px; font-weight: 500;">{method}</span>'
            content = replace_word_with_badge(content, method, highlight)
    
    return content

def replace_word_with_badge(text: str, word: str, badge: str) -> str:
    """
    Replace a word with a badge, ensuring only whole words are replaced.
    
    Args:
        text: Original text
        word: Word to replace
        badge: HTML badge to insert
    
    Returns:
        Text with word replaced by badge
    """
    import re
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, badge, text, flags=re.IGNORECASE)

def show_system_resources():
    """Display system resource usage with tourism-optimized thresholds."""
    try:
        # Get current time for display
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        cpu_label = "CPU Usage"
        if cpu_percent > 80:
            cpu_label += " (⚠️ High)"
        
        st.progress(cpu_percent / 100, text=f"{cpu_label}: {cpu_percent:.1f}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_label = "Memory Usage"
        if memory_percent > 85:
            memory_label += " (⚠️ High)"
        
        st.progress(memory_percent / 100, text=f"{memory_label}: {memory_percent:.1f}%")
        
        # Available memory with tourism document processing estimate
        available_memory_gb = memory.available / (1024 ** 3)
        
        # Estimate how many tourism PDFs can be processed
        est_pdf_capacity = int(available_memory_gb * 5)  # rough estimate: 5 docs per GB
        
        memory_msg = f"Available Memory: {available_memory_gb:.2f} GB"
        if est_pdf_capacity > 0:
            memory_msg += f" (≈ {est_pdf_capacity} tourism docs)"
        
        st.text(memory_msg)
        
        # Disk usage for the current directory
        disk = psutil.disk_usage('.')
        disk_percent = disk.percent
        disk_label = "Disk Usage"
        if disk_percent > 90:
            disk_label += " (⚠️ High)"
        
        st.text(f"{disk_label}: {disk_percent:.1f}%")
        
        # Last updated timestamp
        st.caption(f"Last updated: {current_time}")
        
    except Exception as e:
        st.warning(f"Could not retrieve system resources: {str(e)}")

def display_tourism_stats(metrics: Dict[str, Any]):
    """
    Display tourism statistics in a visually appealing way.
    
    Args:
        metrics: Dictionary of metrics to display
    """
    # Use columns to arrange metrics
    cols = st.columns(len(metrics) if len(metrics) <= 4 else 4)
    
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i % 4]:
            st.metric(
                label=label,
                value=value,
                delta=None
            )

def display_tourism_segments(segments_data: Dict[str, float]):
    """
    Display tourism market segments with visualization.
    
    Args:
        segments_data: Dictionary mapping segment names to values/percentages
    """
    # Convert to DataFrame for Altair
    df = pd.DataFrame({
        'Segment': list(segments_data.keys()),
        'Value': list(segments_data.values())
    })
    
    # Create bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Value:Q', title='Percentage'),
        y=alt.Y('Segment:N', title=None, sort='-x'),
        color=alt.Color('Segment:N', scale=alt.Scale(
            domain=list(segments_data.keys()),
            range=[SEGMENT_COLORS.get(s.lower(), "#1E88E5") for s in segments_data.keys()]
        )),
        tooltip=['Segment:N', 'Value:Q']
    ).properties(
        title='Tourism Market Segments',
        height=min(300, len(segments_data) * 40)
    )
    
    st.altair_chart(chart, use_container_width=True)

def display_tourism_entities(entities: Dict[str, List[str]]):
    """
    Display tourism entities extracted from documents.
    
    Args:
        entities: Dictionary mapping entity types to lists of entities
    """
    if not entities or all(len(v) == 0 for v in entities.values()):
        st.info("No tourism entities found in the document.")
        return
    
    # Create tabs for different entity types
    tabs = st.tabs(list(entities.keys()))
    
    for i, (entity_type, items) in enumerate(entities.items()):
        with tabs[i]:
            if not items:
                st.info(f"No {entity_type.lower()} entities found.")
                continue
                
            # Display as a tag cloud
            html_content = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;">'
            
            for item in items:
                # Generate a consistent color based on the item
                color_seed = sum(ord(c) for c in item) % len(list(TOURISM_COLORS.values()))
                color = list(TOURISM_COLORS.values())[color_seed]
                
                html_content += f'''
                <div style="background-color: {color}; color: white; padding: 5px 10px;
                         border-radius: 15px; font-size: 0.9rem; font-weight: 500;">
                    {item}
                </div>
                '''
            
            html_content += '</div>'
            st.markdown(html_content, unsafe_allow_html=True)

def render_file_uploader():
    """Render the tourism-focused file uploader component."""
    return st.file_uploader(
        "Upload Tourism Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload travel brochures, tourism reports, market research, or other travel-related PDFs"
    )

def render_agent_role_selector(current_role: str, roles: Dict[str, str]):
    """Render the tourism-specific agent role selector dropdown."""
    return st.selectbox(
        "Tourism Assistant Role",
        options=list(roles.keys()),
        index=list(roles.keys()).index(current_role) if current_role in roles else 0,
        help="Select the role that best fits your tourism query needs"
    )

def render_model_selector(available_models: List[str], default_model: str = "llama3.2:latest"):
    """Render the model selector dropdown with tourism-focused description."""
    model_options = available_models if available_models else [default_model]
    
    # Ensure default model is included
    if default_model not in model_options:
        model_options.insert(0, default_model)
    
    return st.selectbox(
        "Select LLM Model",
        options=model_options,
        index=0,
        help="Choose the AI model powering your tourism assistant"
    )

def render_tourism_dashboard(data: Dict[str, Any]):
    """
    Render a dashboard with tourism insights.
    
    Args:
        data: Dictionary with tourism analytics data
    """
    st.subheader("📊 Tourism Insights Dashboard")
    
    tabs = st.tabs(["Market Segments", "Payment Methods", "Travel Trends", "Sustainability"])
    
    with tabs[0]:
        if "segments" in data:
            display_tourism_segments(data["segments"])
        else:
            sample_segments = {
                "Luxury": 22.5,
                "Budget": 29.8,
                "Family": 18.2,
                "Solo": 15.5,
                "Adventure": 14.0
            }
            display_tourism_segments(sample_segments)
            st.caption("Sample data - Process tourism documents to see actual insights")
    
    with tabs[1]:
        if "payment_methods" in data:
            payment_data = data["payment_methods"]
            
            # Convert to DataFrame for chart
            df = pd.DataFrame({
                'Method': list(payment_data.keys()),
                'Percentage': list(payment_data.values())
            })
            
            chart = alt.Chart(df).mark_arc().encode(
                theta=alt.Theta(field="Percentage", type="quantitative"),
                color=alt.Color(field="Method", type="nominal"),
                tooltip=['Method', 'Percentage']
            ).properties(
                title='Payment Methods in Tourism'
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Process tourism documents with payment information to generate payment insights")
            
            # Show placeholder image
            st.image("https://via.placeholder.com/800x400?text=Payment+Methods+Visualization", 
                     caption="Sample visualization - Upload payment-related documents to see actual data")
    
    with tabs[2]:
        if "trends" in data:
            trends = data["trends"]
            
            # Create line chart for trends
            dates = pd.date_range(end=datetime.datetime.now(), periods=len(trends), freq='M')
            df = pd.DataFrame({
                'Date': dates,
                'Value': list(trends.values())
            })
            
            # Create line chart
            chart = alt.Chart(df).mark_line(point=True).encode(
                x='Date:T',
                y='Value:Q',
                tooltip=['Date:T', 'Value:Q']
            ).properties(
                title='Tourism Trends'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Display key trends
            st.subheader("Key Trends")
            for trend, value in trends.items():
                st.markdown(f"- **{trend}**: {value}")
        else:
            st.info("Process tourism trend documents to generate trend insights")
            
            # Generate sample trend data
            dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='Q')
            values = [50 + i * 5 + random.uniform(-10, 10) for i in range(len(dates))]
            df = pd.DataFrame({'Date': dates, 'Value': values})
            
            chart = alt.Chart(df).mark_line(point=True).encode(
                x='Date:T',
                y='Value:Q',
                tooltip=['Date:T', 'Value:Q']
            ).properties(
                title='Sample Tourism Growth Trend'
            )
            
            st.altair_chart(chart, use_container_width=True)
            st.caption("Sample data - Process tourism documents to see actual insights")
    
    with tabs[3]:
        if "sustainability" in data:
            sustainability = data["sustainability"]
            
            # Display sustainability metrics
            for metric, value in sustainability.items():
                st.metric(
                    label=metric,
                    value=value,
                    delta=random.uniform(-5, 15) if isinstance(value, (int, float)) else None
                )
        else:
            st.info("Process sustainability-focused tourism documents to generate insights")
            
            # Sample sustainability data
            st.metric("Eco-conscious Travelers", "68%", "+12%")
            st.metric("Carbon Offset Adoption", "37%", "+8%")
            st.metric("Sustainable Accommodation Demand", "52%", "+15%")
            st.caption("Sample data - Process tourism documents to see actual insights")