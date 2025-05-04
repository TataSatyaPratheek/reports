# Enhanced UI Components with Dynamic Model Selection and Tourism Insights

import streamlit as st
import psutil
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from modules.nlp_models import EMBEDDING_MODELS, get_gpu_memory_info
import json
import re

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

def display_embedding_model_selector() -> str:
    """Display embedding model selector with resource visualization."""
    st.markdown("### 🧠 Embedding Model Selection")
    
    # Get system resources
    gpu_info = get_gpu_memory_info()
    memory = psutil.virtual_memory()
    available_memory_mb = memory.available / (1024 ** 2)
    
    # Create resource overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if gpu_info['available']:
            st.metric("GPU", gpu_info['device_name'])
            st.metric("GPU Memory", f"{gpu_info['free_mb']:.0f} MB free")
        else:
            st.warning("No GPU detected")
    
    with col2:
        st.metric("CPU Cores", psutil.cpu_count())
        st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
    
    with col3:
        st.metric("RAM Available", f"{available_memory_mb:.0f} MB")
        st.metric("System Load", f"{memory.percent}%")
    
    # Create model comparison DataFrame
    models_df = pd.DataFrame([
        {
            'Model': model['name'].split('/')[-1],  # Simplify model name for display
            'Memory (MB)': model['memory_mb'],
            'Dimensions': model['dimensions'],
            'Performance': model['performance'],
            'GPU Fit': '✅' if gpu_info['available'] and gpu_info['free_mb'] > model['memory_mb'] * 1.2 else '❌',
            'RAM Fit': '✅' if available_memory_mb > model['memory_mb'] * 1.2 else '❌',
            'Description': model['description']
        }
        for model in EMBEDDING_MODELS
    ])
    
    # Resource fit visualization
    fig = go.Figure()
    
    # Add bars for memory requirements
    fig.add_trace(go.Bar(
        x=models_df['Model'],
        y=models_df['Memory (MB)'],
        name='Memory Required',
        marker_color=TOURISM_COLORS['primary']
    ))
    
    # Add line for available GPU memory
    if gpu_info['available']:
        fig.add_trace(go.Scatter(
            x=models_df['Model'],
            y=[gpu_info['free_mb']] * len(models_df),
            mode='lines',
            name='GPU Memory Available',
            line=dict(color=TOURISM_COLORS['success'], dash='dash')
        ))
    
    # Add line for available RAM
    fig.add_trace(go.Scatter(
        x=models_df['Model'],
        y=[available_memory_mb] * len(models_df),
        mode='lines',
        name='RAM Available',
        line=dict(color=TOURISM_COLORS['secondary'], dash='dot')
    ))
    
    fig.update_layout(
        title='Model Memory Requirements vs Available Resources',
        xaxis_title='Model',
        yaxis_title='Memory (MB)',
        height=400,
        showlegend=True,
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model selection table
    st.markdown("### Model Comparison")
    
    # Highlight recommended models
    def highlight_recommended(row):
        if row['GPU Fit'] == '✅' or row['RAM Fit'] == '✅':
            return ['background-color: #e8f5e9'] * len(row)
        else:
            return ['background-color: #ffebee'] * len(row)
    
    styled_df = models_df.style.apply(highlight_recommended, axis=1)
    st.dataframe(styled_df)
    
    # Model selector
    recommended_models = models_df[(models_df['GPU Fit'] == '✅') | (models_df['RAM Fit'] == '✅')]['Model'].tolist()
    
    if recommended_models:
        st.success(f"Recommended models based on your system: {', '.join(recommended_models)}")
    else:
        st.warning("Your system may struggle with embedding models. Using the lightest model is recommended.")
    
    selected_model = st.selectbox(
        "Select Embedding Model",
        options=[model['name'] for model in EMBEDDING_MODELS],
        format_func=lambda x: f"{x.split('/')[-1]} - {next(m['description'] for m in EMBEDDING_MODELS if m['name'] == x)}",
        index=0  # Default to first (lightest) model
    )
    
    # Performance comparison chart
    perf_df = pd.DataFrame([
        {
            'Model': model['name'].split('/')[-1],
            'Performance Score': float(model['performance'].split()[0]),
            'Memory (MB)': model['memory_mb'],
            'Efficiency': float(model['performance'].split()[0]) / (model['memory_mb'] / 100)
        }
        for model in EMBEDDING_MODELS
    ])
    
    fig2 = px.scatter(
        perf_df,
        x='Memory (MB)',
        y='Performance Score',
        text='Model',
        size='Efficiency',
        color='Efficiency',
        color_continuous_scale='Viridis',
        title='Performance vs Memory Trade-off',
        labels={
            'Performance Score': 'MTEB Score',
            'Memory (MB)': 'Memory Requirements (MB)',
            'Efficiency': 'Performance/Memory Ratio'
        }
    )
    
    fig2.update_traces(textposition='top center')
    fig2.update_layout(height=500)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    return selected_model

def generate_tourism_insights(documents_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive tourism insights from processed documents."""
    insights = {
        "overview": {},
        "segments": {},
        "payment_trends": {},
        "destination_analysis": {},
        "sustainability_metrics": {},
        "trends": {}
    }
    
    # Analyze segments with enhanced insights
    if "segments" in documents_data and documents_data["segments"]:
        total_mentions = sum(documents_data["segments"].values())
        segment_insights = {}
        
        for segment, count in documents_data["segments"].items():
            percentage = (count / total_mentions * 100) if total_mentions > 0 else 0
            
            # Determine trend based on relative frequency
            avg_mention = total_mentions / len(documents_data["segments"])
            if count > avg_mention * 1.5:
                trend = "growing"
            elif count < avg_mention * 0.5:
                trend = "declining"
            else:
                trend = "stable"
            
            segment_insights[segment] = {
                "count": count,
                "percentage": percentage,
                "trend": trend,
                "characteristics": get_segment_characteristics(segment),
                "recommendations": get_segment_recommendations(segment)
            }
        
        insights["segments"] = segment_insights
    
    # Analyze payment methods with deeper insights
    if "payment_methods" in documents_data and documents_data["payment_methods"]:
        total_payments = sum(documents_data["payment_methods"].values())
        payment_insights = {}
        
        for method, count in documents_data["payment_methods"].items():
            percentage = (count / total_payments * 100) if total_payments > 0 else 0
            
            # Determine adoption level
            if percentage > 30:
                adoption = "high"
            elif percentage > 15:
                adoption = "medium"
            else:
                adoption = "low"
            
            # Check if it's a digital payment method
            is_digital = any(term in method.lower() for term in ["digital", "mobile", "crypto", "wallet"])
            
            payment_insights[method] = {
                "count": count,
                "percentage": percentage,
                "adoption": adoption,
                "is_digital": is_digital,
                "growth_potential": "high" if is_digital else "moderate",
                "recommendations": get_payment_recommendations(method, adoption)
            }
        
        insights["payment_trends"] = payment_insights
    
    # Analyze destinations
    if "destinations" in documents_data and documents_data["destinations"]:
        destination_insights = {}
        total_destinations = sum(documents_data["destinations"].values())
        
        for destination, count in documents_data["destinations"].items():
            percentage = (count / total_destinations * 100) if total_destinations > 0 else 0
            destination_insights[destination] = {
                "mentions": count,
                "percentage": percentage,
                "popularity": "high" if percentage > 10 else "medium" if percentage > 5 else "emerging"
            }
        
        insights["destination_analysis"] = destination_insights
    
    # Generate comprehensive overview
    insights["overview"] = {
        "total_segments_identified": len(documents_data.get("segments", {})),
        "total_payment_methods": len(documents_data.get("payment_methods", {})),
        "total_destinations": len(documents_data.get("destinations", {})),
        "document_coverage": get_coverage_assessment(documents_data),
        "primary_segments": sorted(
            documents_data.get("segments", {}).items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3] if documents_data.get("segments") else [],
        "digital_payment_share": calculate_digital_payment_share(documents_data.get("payment_methods", {})),
        "market_maturity": assess_market_maturity(documents_data)
    }
    
    # Extract key trends
    insights["trends"] = extract_key_trends(documents_data, insights)
    
    return insights

def get_segment_characteristics(segment: str) -> List[str]:
    """Get characteristics for a tourism segment."""
    characteristics = {
        "luxury": ["High spending", "Personalized service", "Exclusive experiences", "Premium accommodations"],
        "budget": ["Cost-conscious", "Value-focused", "Basic amenities", "Group travel"],
        "family": ["Child-friendly activities", "Safety priority", "Educational experiences", "All-inclusive options"],
        "adventure": ["Physical activities", "Risk tolerance", "Remote destinations", "Nature-focused"],
        "cultural": ["Heritage sites", "Local experiences", "Learning focus", "Authentic interactions"],
        "wellness": ["Health-focused", "Relaxation", "Spa services", "Mindfulness activities"],
        "business": ["Efficiency focus", "Connectivity needs", "Meeting facilities", "Location priority"]
    }
    return characteristics.get(segment, ["General travel interests"])

def get_segment_recommendations(segment: str) -> List[str]:
    """Get recommendations for targeting a tourism segment."""
    recommendations = {
        "luxury": [
            "Focus on exclusive partnerships",
            "Enhance personalization services",
            "Develop VIP experiences",
            "Invest in premium amenities"
        ],
        "budget": [
            "Create value packages",
            "Implement group discounts",
            "Optimize cost structures",
            "Develop budget-friendly options"
        ],
        "family": [
            "Create family packages",
            "Develop kid-friendly programs",
            "Ensure safety certifications",
            "Offer educational activities"
        ],
        "adventure": [
            "Partner with adventure operators",
            "Develop unique experiences",
            "Ensure safety protocols",
            "Create challenge-based packages"
        ],
        "cultural": [
            "Partner with local communities",
            "Develop cultural programs",
            "Train cultural ambassadors",
            "Create educational materials"
        ],
        "wellness": [
            "Develop wellness programs",
            "Partner with health professionals",
            "Create relaxation spaces",
            "Offer healthy dining options"
        ],
        "business": [
            "Enhance business facilities",
            "Improve connectivity infrastructure",
            "Create corporate packages",
            "Develop loyalty programs"
        ]
    }
    return recommendations.get(segment, ["Develop targeted marketing", "Enhance service quality"])

def get_payment_recommendations(method: str, adoption: str) -> List[str]:
    """Get recommendations for payment methods based on adoption level."""
    if "digital" in method.lower() or "mobile" in method.lower():
        if adoption == "low":
            return [
                "Increase awareness of digital payment options",
                "Provide incentives for digital payment adoption",
                "Ensure robust security measures",
                "Offer user training and support"
            ]
        else:
            return [
                "Expand digital payment infrastructure",
                "Integrate with popular payment platforms",
                "Optimize transaction processing",
                "Enhance security protocols"
            ]
    else:
        if adoption == "high":
            return [
                "Maintain traditional payment support",
                "Gradually introduce digital alternatives",
                "Ensure payment method diversity",
                "Train staff on payment processing"
            ]
        else:
            return [
                "Evaluate payment method relevance",
                "Consider phasing out if declining",
                "Educate customers on alternatives",
                "Monitor usage trends"
            ]

def get_coverage_assessment(data: Dict[str, Any]) -> str:
    """Assess document coverage based on data richness."""
    metrics = 0
    if data.get("segments") and len(data["segments"]) > 0:
        metrics += 1
    if data.get("payment_methods") and len(data["payment_methods"]) > 0:
        metrics += 1
    if data.get("destinations") and len(data["destinations"]) > 0:
        metrics += 1
    if data.get("sustainability") and len(data["sustainability"]) > 0:
        metrics += 1
    
    if metrics >= 3:
        return "comprehensive"
    elif metrics >= 2:
        return "moderate"
    else:
        return "limited"

def calculate_digital_payment_share(payment_methods: Dict[str, int]) -> float:
    """Calculate the share of digital payment methods."""
    if not payment_methods:
        return 0.0
    
    digital_keywords = ["digital", "mobile", "crypto", "wallet", "online", "app"]
    digital_count = sum(
        count for method, count in payment_methods.items()
        if any(keyword in method.lower() for keyword in digital_keywords)
    )
    total_count = sum(payment_methods.values())
    
    return (digital_count / total_count * 100) if total_count > 0 else 0.0

def assess_market_maturity(data: Dict[str, Any]) -> str:
    """Assess market maturity based on data patterns."""
    indicators = []
    
    # Check segment diversity
    if data.get("segments") and len(data["segments"]) >= 5:
        indicators.append("diverse_segments")
    
    # Check digital payment adoption
    digital_share = calculate_digital_payment_share(data.get("payment_methods", {}))
    if digital_share > 30:
        indicators.append("high_digital_adoption")
    
    # Check for sustainability mentions
    if data.get("sustainability") and len(data["sustainability"]) > 0:
        indicators.append("sustainability_focus")
    
    if len(indicators) >= 2:
        return "mature"
    elif len(indicators) == 1:
        return "developing"
    else:
        return "emerging"

def extract_key_trends(data: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract key trends from the analyzed data."""
    trends = []
    
    # Digital transformation trend
    digital_share = insights["overview"].get("digital_payment_share", 0)
    if digital_share > 20:
        trends.append({
            "name": "Digital Transformation",
            "description": f"Digital payment methods account for {digital_share:.1f}% of mentions",
            "importance": "high" if digital_share > 40 else "medium",
            "recommendation": "Accelerate digital infrastructure development"
        })
    
    # Segment evolution trend
    if insights.get("segments"):
        growing_segments = [
            segment for segment, data in insights["segments"].items()
            if data.get("trend") == "growing"
        ]
        if growing_segments:
            trends.append({
                "name": "Segment Growth",
                "description": f"Growing segments: {', '.join(growing_segments)}",
                "importance": "high",
                "recommendation": "Focus resources on growing segments"
            })
    
    # Sustainability trend
    if data.get("sustainability"):
        trends.append({
            "name": "Sustainability Focus",
            "description": "Increasing emphasis on sustainable tourism practices",
            "importance": "medium",
            "recommendation": "Develop comprehensive sustainability strategy"
        })
    
    return trends

def render_tourism_dashboard(data: Dict[str, Any]):
    """Render an enhanced dashboard with tourism insights."""
    st.markdown("## 📊 Tourism Insights Dashboard")
    
    # Generate insights
    insights = generate_tourism_insights(data)
    
    # Overview metrics
    st.markdown("### 📈 Executive Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Market Segments", 
            insights["overview"]["total_segments_identified"],
            "Identified"
        )
        st.metric(
            "Payment Methods", 
            insights["overview"]["total_payment_methods"],
            "Tracked"
        )
    
    with col2:
        st.metric(
            "Destinations", 
            insights["overview"]["total_destinations"],
            "Analyzed"
        )
        st.metric(
            "Market Maturity",
            insights["overview"]["market_maturity"].title()
        )
    
    with col3:
        st.metric(
            "Digital Payments",
            f"{insights['overview']['digital_payment_share']:.1f}%",
            "Share"
        )
        st.metric(
            "Coverage Quality",
            insights["overview"]["document_coverage"].title()
        )
    
    # Key trends section
    if insights.get("trends"):
        st.markdown("### 🔍 Key Tourism Trends")
        
        for trend in insights["trends"]:
            with st.expander(f"📈 {trend['name']}", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Description:** {trend['description']}")
                    st.markdown(f"**Recommendation:** {trend['recommendation']}")
                
                with col2:
                    importance_color = {
                        "high": "🔴",
                        "medium": "🟡",
                        "low": "🟢"
                    }
                    st.markdown(f"**Importance:** {importance_color.get(trend['importance'], '⚪')} {trend['importance'].title()}")
    
    st.markdown("---")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Market Segments", 
        "Payment Analysis", 
        "Destination Insights", 
        "Sustainability",
        "Executive Summary"
    ])
    
    with tab1:
        display_market_segments_analysis(insights["segments"])
    
    with tab2:
        display_payment_analysis(insights["payment_trends"])
    
    with tab3:
        display_destination_insights(insights.get("destination_analysis", {}))
    
    with tab4:
        display_sustainability_metrics(data.get("sustainability", {}))
    
    with tab5:
        display_executive_summary(insights)

def display_market_segments_analysis(segments_data: Dict[str, Any]):
    """Display detailed market segments analysis."""
    if not segments_data:
        st.info("No market segment data available. Process documents with segment information to see insights.")
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
    
    # Sort by count
    df = df.sort_values('Count', ascending=False)
    
    # Create distribution chart
    fig = px.pie(
        df, 
        values='Count', 
        names='Segment',
        title='Market Segment Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create trend analysis
    fig2 = px.bar(
        df,
        x='Segment',
        y='Percentage',
        color='Trend',
        title='Segment Share and Growth Trends',
        color_discrete_map={
            'growing': TOURISM_COLORS['success'],
            'stable': TOURISM_COLORS['primary'],
            'declining': TOURISM_COLORS['error']
        }
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed metrics
    st.markdown("### Segment Details")
    for segment, data in segments_data.items():
        with st.expander(f"📊 {segment.title()} Segment Analysis"):
            # Key metrics
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
            
            st.markdown("---")
            
            # Characteristics and Recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Key Characteristics")
                for char in data.get('characteristics', []):
                    st.markdown(f"• {char}")
            
            with col2:
                st.markdown("#### 💡 Strategic Recommendations")
                for rec in data.get('recommendations', []):
                    st.markdown(f"• {rec}")
            
            # Growth potential visualization
            st.markdown("#### 📊 Growth Potential Analysis")
            
            growth_factors = {
                "Market Size": data['count'] / 10,  # Normalize for visualization
                "Growth Trend": 3 if data['trend'] == 'growing' else 2 if data['trend'] == 'stable' else 1,
                "Digital Readiness": 3 if segment in ['business', 'luxury'] else 2,
                "Sustainability Alignment": 3 if segment in ['wellness', 'cultural'] else 2,
                "Post-Pandemic Recovery": 3 if segment in ['wellness', 'adventure'] else 2
            }
            
            growth_df = pd.DataFrame([
                {"Factor": factor, "Score": score}
                for factor, score in growth_factors.items()
            ])
            
            fig3 = px.bar(
                growth_df,
                x='Factor',
                y='Score',
                title=f'{segment.title()} Segment - Growth Factors',
                color='Score',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig3, use_container_width=True)

def display_payment_analysis(payment_data: Dict[str, Any]):
    """Display payment methods analysis."""
    if not payment_data:
        st.info("No payment data available. Process documents with payment information to see insights.")
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
    
    # Sort by percentage
    df = df.sort_values('Percentage', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        df,
        y='Method',
        x='Percentage',
        orientation='h',
        title='Payment Method Usage Distribution',
        color='Adoption',
        color_discrete_map={
            'high': TOURISM_COLORS['success'],
            'medium': TOURISM_COLORS['primary'],
            'low': TOURISM_COLORS['warning']
        }
    )
    
    fig.update_layout(
        xaxis_title="Usage Percentage (%)",
        yaxis_title="Payment Method",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Payment trends over time (simulated)
    st.markdown("### Payment Adoption Trends")
    
    # Create simulated trend data
    methods = df['Method'].tolist()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    
    trend_data = []
    for method in methods:
        base_value = df[df['Method'] == method]['Percentage'].iloc[0]
        for i, month in enumerate(months):
            # Simulate growth for digital payments
            if 'digital' in method.lower() or 'mobile' in method.lower():
                value = base_value * (1 + 0.05 * i)  # 5% monthly growth
            else:
                value = base_value * (1 - 0.02 * i)  # 2% monthly decline
            
            trend_data.append({
                'Method': method,
                'Month': month,
                'Usage': value
            })
    
    trend_df = pd.DataFrame(trend_data)
    
    fig2 = px.line(
        trend_df,
        x='Month',
        y='Usage',
        color='Method',
        title='Payment Method Adoption Trends (6-month projection)',
        markers=True
    )
    
    st.plotly_chart(fig2, use_container_width=True)

def display_destination_insights(destination_data: Dict[str, Any]):
    """Display destination analysis insights."""
    if not destination_data:
        st.info("No destination data available. Process more documents to see destination insights.")
        return
    
    st.markdown("### Destination Insights")
    
    # Use actual data from processed documents
    if destination_data:
        # Create DataFrame from actual data
        df = pd.DataFrame([
            {
                'Destination': dest,
                'Mentions': data.get('mentions', 0),
                'Percentage': data.get('percentage', 0),
                'Popularity': data.get('popularity', 'emerging')
            }
            for dest, data in destination_data.items()
        ])
        
        # Sort by mentions
        df = df.sort_values('Mentions', ascending=False)
        
        # Create bar chart for destination mentions
        fig1 = px.bar(
            df,
            x='Destination',
            y='Mentions',
            color='Popularity',
            title='Destination Mentions in Documents',
            color_discrete_map={
                'high': TOURISM_COLORS['success'],
                'medium': TOURISM_COLORS['primary'],
                'emerging': TOURISM_COLORS['secondary']
            }
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Create donut chart for destination distribution
        fig2 = px.pie(
            df.head(10),  # Top 10 destinations
            values='Mentions',
            names='Destination',
            title='Top 10 Destinations by Mention Frequency',
            hole=0.4
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display destination rankings
        st.markdown("### Destination Rankings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🏆 Most Mentioned")
            for idx, row in df.head(5).iterrows():
                popularity_icon = "🔥" if row['Popularity'] == 'high' else "⭐" if row['Popularity'] == 'medium' else "🌱"
                st.markdown(f"{popularity_icon} **{row['Destination']}**: {row['Mentions']} mentions")
        
        with col2:
            st.markdown("#### 📈 Emerging Destinations")
            emerging = df[df['Popularity'] == 'emerging'].head(5)
            if not emerging.empty:
                for idx, row in emerging.iterrows():
                    st.markdown(f"🌱 **{row['Destination']}**: {row['Mentions']} mentions")
            else:
                st.info("No emerging destinations identified")
        
        with col3:
            st.markdown("#### 🌟 Popular Destinations")
            popular = df[df['Popularity'].isin(['high', 'medium'])].head(5)
            if not popular.empty:
                for idx, row in popular.iterrows():
                    st.markdown(f"⭐ **{row['Destination']}**: {row['Percentage']:.1f}% share")
            else:
                st.info("No popular destinations identified")
        
        # Destination analysis insights
        st.markdown("### Destination Analysis")
        
        total_destinations = len(df)
        high_popularity = len(df[df['Popularity'] == 'high'])
        emerging_count = len(df[df['Popularity'] == 'emerging'])
        
        insights_text = f"""
        Based on the analyzed documents:
        
        - Total destinations identified: **{total_destinations}**
        - High popularity destinations: **{high_popularity}**
        - Emerging destinations: **{emerging_count}**
        - Geographic diversity: **{'High' if total_destinations > 20 else 'Moderate' if total_destinations > 10 else 'Limited'}**
        
        **Key Insights:**
        - {df.iloc[0]['Destination'] if not df.empty else 'Unknown'} appears to be the most discussed destination
        - {'Multiple emerging destinations suggest market growth opportunities' if emerging_count > 3 else 'Limited emerging destinations identified'}
        - {'Diverse destination portfolio indicates healthy market' if total_destinations > 15 else 'Market appears concentrated on few destinations'}
        """
        
        st.markdown(insights_text)

def display_executive_summary(insights: Dict[str, Any]):
    """Display executive summary of tourism insights."""
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
        - Market Maturity: {insights['overview']['market_maturity'].title()}
        """)
    
    with col2:
        st.markdown(f"""
        **Digital Transformation:**
        - Digital Payment Share: {insights['overview']['digital_payment_share']:.1f}%
        - Document Coverage: {insights['overview']['document_coverage'].title()}
        - Primary Segments: {', '.join([seg[0] for seg in insights['overview']['primary_segments'][:3]])}
        """)
    
    # Key Findings
    st.markdown("#### Key Findings")
    
    findings = []
    
    # Segment findings
    if insights.get('segments'):
        growing = [seg for seg, data in insights['segments'].items() if data['trend'] == 'growing']
        if growing:
            findings.append(f"Growing segments: {', '.join(growing)}")
    
    # Payment findings
    if insights.get('payment_trends'):
        digital_methods = [
            method for method, data in insights['payment_trends'].items()
            if data.get('is_digital', False)
        ]
        if digital_methods:
            findings.append(f"Digital payment methods gaining traction: {', '.join(digital_methods)}")
    
    # Destination findings
    if insights.get('destination_analysis'):
        top_destinations = sorted(
            insights['destination_analysis'].items(),
            key=lambda x: x[1].get('mentions', 0),
            reverse=True
        )[:3]
        if top_destinations:
            findings.append(f"Top destinations: {', '.join([d[0] for d in top_destinations])}")
    
    for i, finding in enumerate(findings, 1):
        st.markdown(f"{i}. {finding}")
    
    # Strategic Recommendations
    st.markdown("#### Strategic Recommendations")
    
    recommendations = []
    
    # Based on digital payment share
    if insights['overview']['digital_payment_share'] > 30:
        recommendations.append("Accelerate digital payment infrastructure development")
    else:
        recommendations.append("Develop digital payment adoption strategy")
    
    # Based on market maturity
    if insights['overview']['market_maturity'] == 'mature':
        recommendations.append("Focus on innovation and differentiation")
    elif insights['overview']['market_maturity'] == 'developing':
        recommendations.append("Expand market presence and capabilities")
    else:
        recommendations.append("Establish foundational infrastructure")
    
    # Based on segment trends
    if insights.get('segments'):
        growing_segments = [seg for seg, data in insights['segments'].items() if data['trend'] == 'growing']
        if growing_segments:
            recommendations.append(f"Prioritize resources for growing segments: {', '.join(growing_segments[:2])}")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Visual Summary
    st.markdown("#### Market Performance Dashboard")
    
    # Create a summary metrics visualization
    metrics_data = {
        'Metric': ['Market Diversity', 'Digital Adoption', 'Segment Growth', 'Destination Appeal'],
        'Score': [
            min(100, insights['overview']['total_segments_identified'] * 10),
            insights['overview']['digital_payment_share'],
            len([s for s in insights.get('segments', {}).values() if s.get('trend') == 'growing']) * 20,
            min(100, insights['overview']['total_destinations'] * 5)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig = px.bar(
        metrics_df,
        x='Metric',
        y='Score',
        title='Tourism Market Health Indicators',
        color='Score',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_sustainability_metrics(sustainability_data: Dict[str, Any]):
    """Display sustainability metrics and analysis."""
    if not sustainability_data:
        st.info("No sustainability data available. Process documents with sustainability information to see insights.")
        return
    
    st.markdown("### Sustainability Metrics")
    
    # Create sample sustainability metrics
    metrics = {
        "Carbon Footprint Reduction": {"current": 15, "target": 30, "trend": "improving"},
        "Eco-certified Hotels": {"current": 42, "target": 60, "trend": "stable"},
        "Renewable Energy Use": {"current": 35, "target": 50, "trend": "improving"},
        "Waste Reduction": {"current": 25, "target": 40, "trend": "declining"},
        "Local Community Engagement": {"current": 65, "target": 80, "trend": "improving"}
    }
    
    # Create progress indicators
    for metric, data in metrics.items():
        progress = data['current'] / data['target']
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{metric}**")
            st.progress(progress)
        with col2:
            trend_color = {
                "improving": TOURISM_COLORS['success'],
                "stable": TOURISM_COLORS['primary'],
                "declining": TOURISM_COLORS['error']
            }[data['trend']]
            
            st.markdown(
                f"<span style='color: {trend_color}'>{data['current']}% / {data['target']}%</span>",
                unsafe_allow_html=True
            )
    
    # Create radar chart for sustainability scores
    categories = list(metrics.keys())
    current_values = [metrics[cat]['current'] for cat in categories]
    target_values = [metrics[cat]['target'] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=current_values,
        theta=categories,
        fill='toself',
        name='Current Performance'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=target_values,
        theta=categories,
        fill='toself',
        name='Target Goals'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title='Sustainability Performance Overview'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Export enhanced functions
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
        
        .insight-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .metric-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            text-align: center;
        }}
        
        .trend-up {{
            color: {TOURISM_COLORS['success']};
            font-weight: bold;
        }}
        
        .trend-down {{
            color: {TOURISM_COLORS['error']};
            font-weight: bold;
        }}
    </style>
    """, unsafe_allow_html=True)