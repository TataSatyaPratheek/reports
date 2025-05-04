# modules/model_selection.py
"""
Dynamic model selection framework with resource-speed-accuracy visualization
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import psutil
import torch
import time
from typing import Dict, List, Any, Optional, Tuple

# Comprehensive model characteristics database
MODEL_DATABASE = {
    "all-MiniLM-L6-v2": {
        "size_mb": 80,
        "dimensions": 384,
        "speed_ms": 15,
        "context_length": 256,
        "accuracy_score": 58.9,
        "best_for": "Rapid prototyping, low-resource environments",
        "hardware": "CPU",
        "description": "Extremely lightweight, good for basic tasks"
    },
    "all-MiniLM-L12-v2": {
        "size_mb": 120,
        "dimensions": 384,
        "speed_ms": 25,
        "context_length": 256,
        "accuracy_score": 59.8,
        "best_for": "Better quality with minimal resources",
        "hardware": "CPU",
        "description": "Slightly better than L6, still very light"
    },
    "paraphrase-MiniLM-L6-v2": {
        "size_mb": 90,
        "dimensions": 384,
        "speed_ms": 18,
        "context_length": 256,
        "accuracy_score": 60.2,
        "best_for": "Semantic similarity tasks, tourism reviews",
        "hardware": "CPU",
        "description": "Optimized for semantic similarity"
    },
    "all-mpnet-base-v2": {
        "size_mb": 420,
        "dimensions": 768,
        "speed_ms": 45,
        "context_length": 384,
        "accuracy_score": 63.3,
        "best_for": "Balanced performance, general tourism content",
        "hardware": "CPU/GPU",
        "description": "Good balance of performance and size"
    },
    "BAAI/bge-small-en-v1.5": {
        "size_mb": 130,
        "dimensions": 384,
        "speed_ms": 22,
        "context_length": 512,
        "accuracy_score": 62.2,
        "best_for": "Document retrieval, tourism guides",
        "hardware": "CPU",
        "description": "Small BGE model, good performance"
    },
    "BAAI/bge-base-en-v1.5": {
        "size_mb": 450,
        "dimensions": 768,
        "speed_ms": 55,
        "context_length": 512,
        "accuracy_score": 64.2,
        "best_for": "High-quality retrieval, market analysis",
        "hardware": "GPU recommended",
        "description": "Base BGE model, better performance"
    },
    "dunzhang/stella_en_400M_v5": {
        "size_mb": 800,
        "dimensions": 1024,
        "speed_ms": 120,
        "context_length": 1024,
        "accuracy_score": 65.1,
        "best_for": "Premium content analysis, research papers",
        "hardware": "GPU required",
        "description": "High performance Stella model"
    },
    "BAAI/bge-m3": {
        "size_mb": 1200,
        "dimensions": 1024,
        "speed_ms": 180,
        "context_length": 8192,
        "accuracy_score": 66.0,
        "best_for": "Multi-lingual tourism content, SOTA results",
        "hardware": "GPU (16GB+ VRAM)",
        "description": "Most powerful, multi-lingual BGE"
    }
}

class ModelSelector:
    """Dynamic model selection with resource analysis"""
    
    def __init__(self):
        self.model_db = MODEL_DATABASE
        self.system_resources = self._get_system_resources()
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource availability"""
        memory = psutil.virtual_memory()
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            device_props = torch.cuda.get_device_properties(0)
            gpu_memory = device_props.total_memory / (1024 * 1024)  # MB
            gpu_name = device_props.name
        else:
            gpu_memory = 0
            gpu_name = "No GPU"
        
        return {
            "cpu_count": psutil.cpu_count(),
            "ram_mb": memory.available / (1024 * 1024),
            "gpu_available": gpu_available,
            "gpu_memory_mb": gpu_memory,
            "gpu_name": gpu_name
        }
    
    def select_model(self, 
                    max_latency_ms: int = 500, 
                    priority: str = "balanced",
                    min_accuracy: float = None) -> str:
        """Select optimal model based on constraints"""
        available_ram = self.system_resources["ram_mb"]
        gpu_available = self.system_resources["gpu_available"]
        
        # Filter compatible models
        compatible_models = []
        for name, specs in self.model_db.items():
            # Check resource constraints
            if specs["size_mb"] * 1.5 > available_ram:  # 1.5x buffer
                continue
            
            if specs["hardware"] == "GPU required" and not gpu_available:
                continue
            
            # Check latency constraint
            if specs["speed_ms"] > max_latency_ms:
                continue
            
            # Check accuracy constraint
            if min_accuracy and specs["accuracy_score"] < min_accuracy:
                continue
            
            compatible_models.append((name, specs))
        
        if not compatible_models:
            return "all-MiniLM-L6-v2"  # Fallback to lightest model
        
        # Sort based on priority
        if priority == "speed":
            selected = sorted(compatible_models, key=lambda x: x[1]["speed_ms"])[0][0]
        elif priority == "accuracy":
            selected = sorted(compatible_models, key=lambda x: x[1]["accuracy_score"], reverse=True)[0][0]
        else:  # balanced
            # Calculate efficiency score
            selected = sorted(compatible_models, 
                            key=lambda x: x[1]["accuracy_score"] / (x[1]["speed_ms"] * x[1]["size_mb"]), 
                            reverse=True)[0][0]
        
        return selected
    
    def create_comparison_matrix(self) -> pd.DataFrame:
        """Create model comparison DataFrame"""
        data = []
        for name, specs in self.model_db.items():
            data.append({
                "Model": name,
                "Size (MB)": specs["size_mb"],
                "Speed (ms)": specs["speed_ms"],
                "Context": specs["context_length"],
                "Accuracy": specs["accuracy_score"],
                "Best For": specs["best_for"],
                "Hardware": specs["hardware"]
            })
        
        return pd.DataFrame(data)
    
    def create_resource_speed_plot(self) -> go.Figure:
        """Create interactive resource-speed-accuracy visualization"""
        models = []
        sizes = []
        speeds = []
        accuracies = []
        
        for name, specs in self.model_db.items():
            models.append(name)
            sizes.append(specs["size_mb"])
            speeds.append(specs["speed_ms"])
            accuracies.append(specs["accuracy_score"])
        
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=speeds,
            y=accuracies,
            mode='markers+text',
            text=models,
            textposition="top center",
            marker=dict(
                size=[s/30 for s in sizes],  # Scale bubble size
                color=accuracies,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Accuracy Score")
            ),
            hovertemplate="<br>".join([
                "Model: %{text}",
                "Speed: %{x}ms",
                "Accuracy: %{y}",
                "Size: %{marker.size:.0f}MB"
            ])
        ))
        
        fig.update_layout(
            title="Model Performance Comparison (Bubble size = Model size)",
            xaxis_title="Inference Speed (ms)",
            yaxis_title="Accuracy Score (MTEB)",
            showlegend=False,
            height=600
        )
        
        return fig

def render_model_selection_dashboard(st_container):
    """Render the model selection dashboard in Streamlit"""
    selector = ModelSelector()
    
    with st_container:
        st.markdown("### 🔧 Dynamic Model Selection")
        
        # Resource monitoring widget
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Available RAM", f"{selector.system_resources['ram_mb']:.0f} MB")
        with col2:
            st.metric("CPU Cores", selector.system_resources['cpu_count'])
        with col3:
            if selector.system_resources['gpu_available']:
                st.metric("GPU Memory", f"{selector.system_resources['gpu_memory_mb']:.0f} MB")
                st.caption(selector.system_resources['gpu_name'])
            else:
                st.metric("GPU", "Not Available")
        
        st.markdown("---")
        
        # Model selection controls
        st.markdown("### Model Selection Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            priority = st.select_slider(
                "Optimization Priority",
                options=["speed", "balanced", "accuracy"],
                value="balanced",
                help="Choose between speed, accuracy, or balanced performance"
            )
            
            max_latency = st.slider(
                "Maximum Latency (ms)",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Maximum acceptable inference time"
            )
        
        with col2:
            min_accuracy = st.slider(
                "Minimum Accuracy Score",
                min_value=58.0,
                max_value=66.0,
                value=60.0,
                step=0.5,
                help="Minimum MTEB accuracy score"
            )
            
            selected_model = selector.select_model(
                max_latency_ms=max_latency,
                priority=priority,
                min_accuracy=min_accuracy
            )
        
        # Display selected model
        st.markdown("### 🎯 Recommended Model")
        
        selected_specs = MODEL_DATABASE[selected_model]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", selected_model)
        with col2:
            st.metric("Speed", f"{selected_specs['speed_ms']}ms")
        with col3:
            st.metric("Accuracy", f"{selected_specs['accuracy_score']}")
        with col4:
            st.metric("Size", f"{selected_specs['size_mb']}MB")
        
        st.info(f"**Best for:** {selected_specs['best_for']}")
        
        # Model comparison visualization
        with st.expander("📊 Model Comparison Dashboard", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Comparison Table", "Performance Plot", "Tradeoff Analysis"])
            
            with tab1:
                df = selector.create_comparison_matrix()
                st.dataframe(
                    df.style.background_gradient(subset=['Accuracy'], cmap='Greens')
                            .background_gradient(subset=['Speed (ms)'], cmap='Reds_r'),
                    use_container_width=True
                )
            
            with tab2:
                fig = selector.create_resource_speed_plot()
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Create tradeoff analysis
                fig_tradeoff = go.Figure()
                
                for name, specs in MODEL_DATABASE.items():
                    efficiency = specs["accuracy_score"] / (specs["speed_ms"] * specs["size_mb"] / 1000)
                    fig_tradeoff.add_trace(go.Bar(
                        name=name,
                        x=[name],
                        y=[efficiency],
                        text=f"{efficiency:.2f}",
                        textposition='auto',
                    ))
                
                fig_tradeoff.update_layout(
                    title="Model Efficiency Score (Accuracy / (Speed × Size))",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig_tradeoff, use_container_width=True)
        
        return selected_model