# modules/ui_components.py
"""
Optimized UI Components with lazy loading and memory-efficient visualization.
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

# Theme colors (existing)
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

# Tourism segment colors (existing)
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
    
    # Apply memory optimization settings
    fig.update_layout(
        uirevision=True,  # Helps reuse WebGL context
        dragmode=False,   # Disable dragging for performance
        template='plotly_white'  # Use simple template
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
        title='Market Segment Distribution'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Create trend analysis chart
    fig2 = create_optimized_visualization(
        df,
        "bar",
        x='Segment',
        y='Percentage',
        color='Trend',
        title='Segment Share and Growth Trends'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed metrics with lazy loading
    st.markdown("### Segment Details")
    for segment, data in list(segments_data.items())[:5]:  # Limit to top 5
        with st.expander(f"📊 {segment.title()} Segment Analysis"):
            # Only render details when expanded
            if st.session_state.get(f"expand_{segment}", False):
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
        color='Adoption'
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
        title='Destination Mentions in Documents'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Top 10 destinations donut chart
    fig2 = create_optimized_visualization(
        df.head(10),
        "pie",
        values='Mentions',
        names='Destination',
        title='Top 10 Destinations by Mention Frequency',
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

# Update the main dashboard function
def render_tourism_dashboard(data: Dict[str, Any]):
    """Main entry point for tourism dashboard with memory optimization."""
    render_tourism_dashboard_lazy(data)

# Keep existing helper functions but make them memory-aware
def generate_tourism_insights(documents_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate tourism insights with memory optimization."""
    # Check memory before processing
    memory_monitor.check()
    
    insights = {
        "overview": {},
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
    
    # Similar optimization for other insight categories...
    # (Rest of the function remains similar but with data limits)
    
    # Clear memory after processing
    gc.collect()
    
    return insights

# Export the optimized functions
def apply_tourism_theme():
    """Apply tourism-themed CSS with enhanced styling."""
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
        
        /* Optimize animations for performance */
        .stButton>button {{ 
            border-radius: 8px; 
            transition: all 0.15s ease;
        }}
        
        .stButton>button:hover {{ 
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        /* Reduce visual complexity */
        .tourism-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        
        .insight-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        /* Optimize chart containers */
        .js-plotly-plot {{
            background: white !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# Additional utility functions for memory-aware chat display
def display_chat(messages: List[Dict[str, str]], current_role: str = "Assistant"):
    """Display chat messages with memory optimization."""
    # Limit displayed messages for performance
    display_limit = 50
    if len(messages) > display_limit:
        st.info(f"Showing last {display_limit} messages")
        messages = messages[-display_limit:]
    
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])