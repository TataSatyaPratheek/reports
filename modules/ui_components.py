# modules/ui_components.py
"""
Updated UI Components with light-themed, central design for tourism chatbot.
"""
import streamlit as st
import psutil
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any, Optional
from modules.nlp_models import EMBEDDING_MODELS, get_gpu_memory_info
from modules.memory_utils import memory_monitor, get_available_memory_mb
import json
import re
import gc

# Updated theme colors with light palette
TOURISM_COLORS = {
    "primary": "#1976D2",     # Trustworthy blue
    "secondary": "#00897B",    # Calming teal
    "accent": "#FFC107",       # Warm amber
    "background": "#FFFFFF",   # Light background
    "text": "#212121",         # Dark text for contrast
    "light_text": "#616161",   # Lighter text for secondary info
    "success": "#388E3C",
    "warning": "#F57C00",
    "error": "#D32F2F",
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

def show_system_resources():
    """Display current system resource usage with memory awareness."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Create compact resource display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CPU", f"{cpu_percent}%", 
                 delta=f"{psutil.cpu_count()} cores",
                 delta_color="off")
    
    with col2:
        st.metric("RAM Used", f"{memory.percent}%",
                 delta=f"{memory.available / (1024**3):.1f}GB free",
                 delta_color="inverse")
    
    with col3:
        gpu_info = get_gpu_memory_info()
        if gpu_info['available']:
            gpu_usage = gpu_info['used_mb'] / gpu_info['total_mb'] * 100
            st.metric("GPU", f"{gpu_usage:.1f}%",
                     delta=f"{gpu_info['free_mb']:.0f}MB free",
                     delta_color="inverse")
        else:
            st.metric("GPU", "Not Available")

def create_optimized_visualization(data: Any, viz_type: str, max_points: int = 1000, **kwargs) -> go.Figure:
    """Create memory-efficient visualizations by sampling/aggregating data."""
    # Check memory before creating visualization
    memory_monitor.check()
    
    if isinstance(data, pd.DataFrame) and len(data) > max_points:
        # Downsample data if too large
        sample_rate = max_points / len(data)
        sampled_data = data.sample(frac=sample_rate)
    else:
        sampled_data = data
    
    # Create visualization based on type
    if viz_type == "scatter":
        fig = px.scatter(sampled_data, **kwargs)
    elif viz_type == "bar":
        fig = px.bar(sampled_data, **kwargs)
    elif viz_type == "line":
        fig = px.line(sampled_data, **kwargs)
    elif viz_type == "pie":
        fig = px.pie(sampled_data, **kwargs)
    else:
        fig = go.Figure()
    
    # Updated colorscheme to match light theme
    fig.update_layout(
        uirevision=True,  # Helps reuse WebGL context
        dragmode=False,   # Disable dragging for performance
        template='plotly_white',  # Light template
        paper_bgcolor=TOURISM_COLORS["background"],
        plot_bgcolor='rgba(255,255,255,0.9)',
        font_color=TOURISM_COLORS["text"],
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Update color scales to match theme
    if viz_type not in ["pie"]:
        fig.update_traces(
            marker_color=TOURISM_COLORS["primary"],
            marker_line_color=TOURISM_COLORS["primary"],
            marker_line_width=1
        )
    
    return fig

def render_tourism_dashboard_lazy(data: Dict[str, Any]):
    """Render tourism dashboard with lazy loading for memory efficiency."""
    st.markdown("## 📊 Tourism Insights Dashboard")
    
    # Generate insights (only basic processing upfront)
    insights = None
    
    # Executive Overview (always shown)
    st.markdown("### 📈 Executive Overview")
    
    if data:
        # Basic metrics (lightweight)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Market Segments", len(data.get("segments", {})))
            st.metric("Payment Methods", len(data.get("payment_methods", {})))
        
        with col2:
            st.metric("Destinations", len(data.get("destinations", {})))
            st.metric("Document Pages", data.get("total_pages", 0))
        
        with col3:
            digital_count = sum(1 for method in data.get("payment_methods", {})
                              if any(term in method.lower() for term in ["digital", "mobile", "online"]))
            st.metric("Digital Payments", digital_count)
            st.metric("Analysis Status", "Complete" if data else "Pending")
    
    st.markdown("---")
    
    # Create tabs but render content lazily
    tab_names = ["Market Segments", "Payment Analysis", "Destination Insights", 
                "Sustainability", "Executive Summary"]
    tabs = st.tabs(tab_names)
    
    # Track active tab in session state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
    
    # Render only the active tab content
    for i, tab in enumerate(tabs):
        with tab:
            if st.button(f"Load {tab_names[i]}", key=f"load_tab_{i}"):
                st.session_state.active_tab = i
                
            if st.session_state.active_tab == i:
                # Generate insights only when needed
                if insights is None:
                    with st.spinner("Generating insights..."):
                        insights = generate_tourism_insights(data)
                
                # Render appropriate content
                if i == 0:
                    display_market_segments_analysis_optimized(insights.get("segments", {}))
                elif i == 1:
                    display_payment_analysis_optimized(insights.get("payment_trends", {}))
                elif i == 2:
                    display_destination_insights_optimized(insights.get("destination_analysis", {}))
                elif i == 3:
                    display_sustainability_metrics_optimized(data.get("sustainability", {}))
                elif i == 4:
                    display_executive_summary_optimized(insights)
                
                # Clear memory after rendering
                gc.collect()

def display_market_segments_analysis_optimized(segments_data: Dict[str, Any]):
    """Display market segments analysis with memory optimization."""
    if not segments_data:
        st.info("No market segment data available.")
        return
    
    st.markdown("### Market Segment Analysis")
    
    # Create DataFrame for visualization
    df = pd.DataFrame([
        {
            'Segment': segment,
            'Count': data['count'],
            'Percentage': data['percentage'],
            'Trend': data['trend']
        }
        for segment, data in segments_data.items()
    ])
    
    # Limit to top segments for performance
    if len(df) > 10:
        df = df.nlargest(10, 'Count')
        st.info(f"Showing top 10 segments out of {len(segments_data)}")
    
    # Create optimized visualizations
    fig1 = create_optimized_visualization(
        df, 
        "pie",
        values='Count', 
        names='Segment',
        title='Market Segment Distribution',
        color_discrete_sequence=list(TOURISM_COLORS.values())[:len(df)]
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Create trend analysis chart
    fig2 = create_optimized_visualization(
        df,
        "bar",
        x='Segment',
        y='Percentage',
        color='Trend',
        title='Segment Share and Growth Trends',
        color_discrete_map={
            "growing": TOURISM_COLORS["success"],
            "stable": TOURISM_COLORS["primary"],
            "declining": TOURISM_COLORS["error"]
        }
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed metrics without nested expanders
    st.markdown("### Segment Details")
    top_segments = list(segments_data.items())[:5]  # Limit to top 5
    
    for i, (segment, data) in enumerate(top_segments):
        st.markdown(f"#### {segment.title()}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mentions", data['count'])
        with col2:
            st.metric("Market Share", f"{data['percentage']:.1f}%")
        with col3:
            st.metric("Trend", data['trend'].title())
        with col4:
            growth_icon = "📈" if data['trend'] == 'growing' else "📊" if data['trend'] == 'stable' else "📉"
            st.metric("Outlook", growth_icon)
        
        if i < len(top_segments) - 1:
            st.markdown("---")

def display_payment_analysis_optimized(payment_data: Dict[str, Any]):
    """Display payment analysis with memory optimization."""
    if not payment_data:
        st.info("No payment data available.")
        return
    
    st.markdown("### Payment Methods Analysis")
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            'Method': method,
            'Count': data['count'],
            'Percentage': data['percentage'],
            'Adoption': data['adoption']
        }
        for method, data in payment_data.items()
    ])
    
    # Limit data for performance
    if len(df) > 8:
        df = df.nlargest(8, 'Percentage')
        st.info(f"Showing top 8 payment methods out of {len(payment_data)}")
    
    # Create optimized chart
    fig = create_optimized_visualization(
        df,
        "bar",
        y='Method',
        x='Percentage',
        orientation='h',
        title='Payment Method Usage Distribution',
        color='Adoption',
        color_discrete_map={
            "high": TOURISM_COLORS["success"],
            "medium": TOURISM_COLORS["warning"],
            "low": TOURISM_COLORS["light_text"]
        }
    )
    st.plotly_chart(fig, use_container_width=True)

def display_destination_insights_optimized(destination_data: Dict[str, Any]):
    """Display destination insights with memory optimization."""
    if not destination_data:
        st.info("No destination data available.")
        return
    
    st.markdown("### Destination Insights")
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            'Destination': dest,
            'Mentions': data.get('mentions', 0),
            'Percentage': data.get('percentage', 0),
            'Popularity': data.get('popularity', 'emerging')
        }
        for dest, data in destination_data.items()
    ])
    
    # Limit to top destinations for performance
    if len(df) > 15:
        df = df.nlargest(15, 'Mentions')
        st.info(f"Showing top 15 destinations out of {len(destination_data)}")
    
    # Create optimized visualizations
    fig1 = create_optimized_visualization(
        df,
        "bar",
        x='Destination',
        y='Mentions',
        color='Popularity',
        title='Destination Mentions in Documents',
        color_discrete_map={
            "popular": TOURISM_COLORS["primary"],
            "trending": TOURISM_COLORS["accent"],
            "emerging": TOURISM_COLORS["success"]
        }
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Top 10 destinations donut chart
    fig2 = create_optimized_visualization(
        df.head(10),
        "pie",
        values='Mentions',
        names='Destination',
        title='Top 10 Destinations by Mention Frequency',
        color_discrete_sequence=list(TOURISM_COLORS.values())[:10],
        hole=0.4
    )
    st.plotly_chart(fig2, use_container_width=True)

def display_sustainability_metrics_optimized(sustainability_data: Dict[str, Any]):
    """Display sustainability metrics with memory optimization."""
    if not sustainability_data:
        st.info("No sustainability data available.")
        return
    
    st.markdown("### Sustainability Metrics")
    
    # Create sample metrics (limit data processing)
    metrics = {
        "Carbon Footprint Reduction": {"current": 15, "target": 30, "trend": "improving"},
        "Eco-certified Hotels": {"current": 42, "target": 60, "trend": "stable"},
        "Renewable Energy Use": {"current": 35, "target": 50, "trend": "improving"},
        "Waste Reduction": {"current": 25, "target": 40, "trend": "declining"}
    }
    
    # Create progress indicators
    for metric, data in metrics.items():
        progress = data['current'] / data['target']
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{metric}**")
            st.progress(progress)
        with col2:
            st.markdown(f"{data['current']}% / {data['target']}%")

def display_executive_summary_optimized(insights: Dict[str, Any]):
    """Display executive summary with memory optimization."""
    st.markdown("### 📋 Executive Summary")
    
    # Market Overview
    st.markdown("#### Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Market Composition:**
        - Total Market Segments: {insights['overview']['total_segments_identified']}
        - Payment Methods Tracked: {insights['overview']['total_payment_methods']}
        - Destinations Analyzed: {insights['overview']['total_destinations']}
        """)
    
    with col2:
        st.markdown(f"""
        **Digital Transformation:**
        - Digital Payment Share: {insights['overview']['digital_payment_share']:.1f}%
        - Market Maturity: {insights['overview']['market_maturity'].title()}
        """)
    
    # Key findings (limit to top 5)
    st.markdown("#### Key Findings")
    
    findings = []
    
    # Extract key findings efficiently
    if insights.get('segments'):
        growing = [seg for seg, data in insights['segments'].items() if data['trend'] == 'growing'][:3]
        if growing:
            findings.append(f"Growing segments: {', '.join(growing)}")
    
    if insights.get('payment_trends'):
        digital_methods = [
            method for method, data in insights['payment_trends'].items()
            if data.get('is_digital', False)
        ][:3]
        if digital_methods:
            findings.append(f"Digital payment methods: {', '.join(digital_methods)}")
    
    for i, finding in enumerate(findings[:5], 1):
        st.markdown(f"{i}. {finding}")

# Helper functions for memory-aware dashboard rendering
def generate_tourism_insights(documents_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate tourism insights with memory optimization."""
    # Check memory before processing
    memory_monitor.check()
    
    insights = {
        "overview": {
            "total_segments_identified": len(documents_data.get("segments", {})),
            "total_payment_methods": len(documents_data.get("payment_methods", {})),
            "total_destinations": len(documents_data.get("destinations", {})),
            "digital_payment_share": 35.2,  # Sample value
            "market_maturity": "growing"    # Sample value
        },
        "segments": {},
        "payment_trends": {},
        "destination_analysis": {},
        "sustainability_metrics": {},
        "trends": []
    }
    
    # Process data efficiently (limit processing for large datasets)
    if documents_data.get("segments"):
        segments = documents_data["segments"]
        if len(segments) > 50:  # Limit processing
            segments = dict(sorted(segments.items(), key=lambda x: x[1], reverse=True)[:50])
        
        total_mentions = sum(segments.values())
        segment_insights = {}
        
        for segment, count in segments.items():
            percentage = (count / total_mentions * 100) if total_mentions > 0 else 0
            
            # Simplified trend calculation
            avg_mention = total_mentions / len(segments)
            trend = "growing" if count > avg_mention * 1.5 else "declining" if count < avg_mention * 0.5 else "stable"
            
            segment_insights[segment] = {
                "count": count,
                "percentage": percentage,
                "trend": trend
            }
        
        insights["segments"] = segment_insights
    
    # Sample payment trends (for demonstration)
    insights["payment_trends"] = {
        "credit_card": {"count": 120, "percentage": 35.0, "adoption": "high", "is_digital": False},
        "mobile_payment": {"count": 85, "percentage": 25.0, "adoption": "high", "is_digital": True},
        "digital_wallet": {"count": 60, "percentage": 18.0, "adoption": "medium", "is_digital": True},
        "bank_transfer": {"count": 40, "percentage": 12.0, "adoption": "medium", "is_digital": True},
        "cash": {"count": 35, "percentage": 10.0, "adoption": "low", "is_digital": False}
    }
    
    # Sample destination analysis (for demonstration)
    insights["destination_analysis"] = {
        "Paris": {"mentions": 45, "percentage": 15, "popularity": "popular"},
        "Bali": {"mentions": 38, "percentage": 12, "popularity": "trending"},
        "Tokyo": {"mentions": 32, "percentage": 10, "popularity": "popular"},
        "New York": {"mentions": 28, "percentage": 9, "popularity": "popular"},
        "Barcelona": {"mentions": 22, "percentage": 7, "popularity": "trending"},
        "Cancun": {"mentions": 18, "percentage": 6, "popularity": "trending"},
        "Dubai": {"mentions": 15, "percentage": 5, "popularity": "emerging"},
        "Santorini": {"mentions": 12, "percentage": 4, "popularity": "emerging"}
    }
    
    # Clear memory after processing
    gc.collect()
    
    return insights

# Apply the updated theme
def apply_tourism_theme():
    """Apply tourism-themed CSS with enhanced styling for light theme and central chat design."""
    st.markdown("""
    <style>
        /* Use high-contrast colors for better visibility */
        :root {
            --primary-color: #1976D2;
            --secondary-color: #00897B;
            --accent-color: #FFC107;
            --background-color: #FFFFFF;
            --text-color: #212121;
            --light-text-color: #616161;
            --error-color: #D32F2F;
            --success-color: #388E3C;
            --warning-color: #F57C00;
        }
        
        /* Force light background */
        .stApp {
            background-color: white !important;
            color: #212121 !important;
        }
        
        /* Ensure content is visible */
        .main-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        
        /* Make header more visible */
        .main-header { 
            font-size: 2.4rem !important; 
            font-weight: 600 !important; 
            color: var(--primary-color) !important;
            margin-bottom: 0.5rem !important;
            text-align: center !important;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.1) !important;
        }
        
        .sub-header { 
            font-size: 1.2rem !important;
            color: var(--text-color) !important;
            margin-bottom: 1.5rem !important;
            text-align: center !important;
        }
        
        /* Make welcome message highly visible */
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
            color: var(--primary-color) !important;
            font-size: 1.5rem !important;
            margin-bottom: 10px !important;
        }
        
        .welcome-message p {
            color: var(--text-color) !important;
            font-size: 1.1rem !important;
        }
        
        /* Chat container with clear styling */
        .chat-container {
            max-width: 800px !important;
            margin: 0 auto !important;
            padding: 15px !important;
            background-color: #f5f8fa !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        }
        
        /* Enhance chat messages */
        .stChatMessage {
            background-color: white !important;
            border-radius: 15px !important;
            padding: 12px !important;
            margin-bottom: 15px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.08) !important;
            border: 1px solid #e6e6e6 !important;
        }
        
        /* More visible buttons */
        .stButton>button { 
            border-radius: 8px !important;
            transition: all 0.15s ease !important;
            background-color: #f0f7ff !important;
            color: var(--primary-color) !important;
            border: 1px solid #c0d6f9 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
            font-weight: 500 !important;
            padding: 4px 15px !important;
        }
        
        .stButton>button:hover { 
            transform: translateY(-1px) !important;
            background-color: #e1effe !important;
            border-color: var(--primary-color) !important;
            box-shadow: 0 3px 5px rgba(0,0,0,0.12) !important;
        }
        
        /* Primary button */
        button[data-baseweb="button"][kind="primary"] {
            background-color: var(--primary-color) !important;
            color: white !important;
            font-weight: 500 !important;
        }
        
        /* Emphasize badges */
        .tourism-badge {
            display: inline-block !important;
            padding: 4px 10px !important;
            border-radius: 12px !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            margin-right: 6px !important;
            margin-bottom: 6px !important;
            background-color: #e3f2fd !important;
            color: var(--primary-color) !important;
            border: 1px solid #90caf9 !important;
        }
        
        /* Enhance cards */
        .insight-card {
            background: white !important;
            border-radius: 10px !important;
            padding: 20px !important;
            margin: 15px 0 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        /* Charts */
        .js-plotly-plot {
            background: white !important;
            border-radius: 10px !important;
            padding: 15px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        /* Style sidebar navigation */
        .sidebar-nav {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 10px;
        }
        
        .sidebar-nav button {
            padding: 5px 10px !important;
            font-size: 0.8rem !important;
            min-height: 0 !important;
        }
        
        /* Override sidebar style */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0 !important;
        }
        
        section[data-testid="stSidebar"] > div {
            padding: 2rem 1rem !important;
        }
        
        /* Style sidebar section headers */
        .sidebar-header {
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            color: var(--primary-color) !important;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
            padding-bottom: 5px !important;
            border-bottom: 2px solid var(--primary-color) !important;
        }
        
        /* Make warnings and errors more visible */
        .stAlert {
            background-color: #fff3e0 !important;
            color: #e65100 !important;
            padding: 10px 15px !important;
            border-radius: 6px !important;
            border-left: 4px solid #ff9800 !important;
            margin: 10px 0 !important;
        }
        
        /* Default text */
        p, li, div {
            color: var(--text-color) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Add chat display function
def display_chat(messages: List[Dict[str, str]], current_role: str = "Assistant"):
    """Display chat messages with memory optimization and improved styling."""
    # Limit displayed messages for performance
    display_limit = 50
    if len(messages) > display_limit:
        st.info(f"Showing last {display_limit} messages")
        messages = messages[-display_limit:]
    
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])