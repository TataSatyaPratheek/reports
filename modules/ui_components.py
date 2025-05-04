"""
Optimized UI Components Module - Maintains tourism dashboard and visualization features.
"""
import streamlit as st
import psutil
import pandas as pd
import altair as alt
import datetime
from typing import List, Dict, Any, Optional

# Theme colors
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
    "city": "#607D8B"
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
    "wellness": "#00838F"
}

def apply_tourism_theme():
    """Apply tourism-themed CSS."""
    st.markdown(f"""
    <style>
        :root {{
            --primary-color: {TOURISM_COLORS["primary"]};
            --secondary-color: {TOURISM_COLORS["secondary"]};
            --accent-color: {TOURISM_COLORS["accent"]};
        }}
        
        .main-header {{ 
            font-size: 2.5rem; 
            font-weight: 600; 
            color: var(--primary-color); 
            margin-bottom: 0.2rem; 
            text-align: center; 
        }}
        
        .sub-header {{ 
            font-size: 1.1rem; 
            color: #555; 
            margin-bottom: 1.5rem; 
            text-align: center; 
        }}
        
        .stButton>button {{ 
            border-radius: 8px; 
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{ 
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .tourism-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
    </style>
    """, unsafe_allow_html=True)

def display_chat(messages: List[Dict[str, str]], current_role: str = "Assistant"):
    """Render chat messages."""
    if not messages:
        st.info("No messages yet. Ask a question about tourism documents.")
        return
    
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def show_system_resources():
    """Display system resource usage."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        st.progress(cpu_percent / 100, text=f"CPU Usage: {cpu_percent:.1f}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        st.progress(memory_percent / 100, text=f"Memory Usage: {memory_percent:.1f}%")
        
        # Available memory
        available_memory_gb = memory.available / (1024 ** 3)
        st.text(f"Available Memory: {available_memory_gb:.2f} GB")
        
    except Exception as e:
        st.warning(f"Could not retrieve system resources: {str(e)}")

def display_tourism_entities(entities: Dict[str, List[str]]):
    """Display tourism entities extracted from documents."""
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

def render_tourism_dashboard(data: Dict[str, Any]):
    """Render a dashboard with tourism insights."""
    st.subheader("📊 Tourism Insights Dashboard")
    
    tabs = st.tabs(["Market Segments", "Payment Methods", "Travel Trends", "Sustainability"])
    
    with tabs[0]:
        if "segments" in data and data["segments"]:
            display_tourism_segments(data["segments"])
        else:
            st.info("Process documents with market segment information to see insights.")
    
    with tabs[1]:
        if "payment_methods" in data and data["payment_methods"]:
            display_payment_methods(data["payment_methods"])
        else:
            st.info("Process documents with payment information to see insights.")
    
    with tabs[2]:
        st.info("Process more documents to generate travel trend insights.")
    
    with tabs[3]:
        st.info("Process sustainability-focused documents to see insights.")

def display_tourism_segments(segments_data: Dict[str, float]):
    """Display tourism market segments with visualization."""
    if not segments_data:
        return
    
    # Convert to DataFrame for Altair
    df = pd.DataFrame({
        'Segment': list(segments_data.keys()),
        'Value': list(segments_data.values())
    })
    
    # Create bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Value:Q', title='Count'),
        y=alt.Y('Segment:N', title=None, sort='-x'),
        color=alt.value(TOURISM_COLORS["primary"])
    ).properties(
        title='Document Coverage by Segment',
        height=min(300, len(segments_data) * 40)
    )
    
    st.altair_chart(chart, use_container_width=True)

def display_payment_methods(payment_data: Dict[str, int]):
    """Display payment methods visualization."""
    if not payment_data:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Method': list(payment_data.keys()),
        'Count': list(payment_data.values())
    })
    
    # Create bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Count:Q', title='Mentions'),
        y=alt.Y('Method:N', title=None, sort='-x'),
        color=alt.value(TOURISM_COLORS["secondary"])
    ).properties(
        title='Payment Methods Mentioned in Documents'
    )
    
    st.altair_chart(chart, use_container_width=True)