import os
import time
import asyncio
import json
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union

import chainlit as cl
from chainlit.types import AskFileResponse
import numpy as np
import chromadb

# Import modules
from modules.system_setup import (
    ensure_dependencies, setup_ollama, refresh_available_models,
    download_model, DEFAULT_MODEL_NAME, TOURISM_RECOMMENDED_MODELS
)
from modules.vector_store import initialize_vector_db, reset_vector_db, get_chroma_collection, hybrid_retrieval
from modules.nlp_models import load_nltk_resources, load_spacy_model, load_embedding_model, extract_tourism_entities
from modules.pdf_processor import process_uploaded_pdf
from modules.vector_store import add_chunks_to_collection
from modules.llm_interface import query_llm, SlidingWindowMemory
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
    
    "General Tourism Assistant": "You are a helpful tourism information assistant. Provide clear, accurate, and balanced information about travel topics based on the provided document context. Offer insights on destinations, travel planning, industry trends, and tourism services. Present information in an accessible manner suitable for both industry professionals and travelers. Focus on presenting factual information rather than personal opinions."
}

# --- Global state ---
class AppState:
    def __init__(self):
        self.current_role = "Travel Trends Analyst"
        self.conversation_memory = SlidingWindowMemory(max_tokens=2048)
        self.processed_files = set()
        self.extracted_entities = {}
        self.tourism_metrics = {}
        self.initialization_complete = False
        self.collection = None
        self.embedding_model = None
        self.nlp_model = None
        self.settings = {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "overlap": DEFAULT_OVERLAP,
            "top_n": DEFAULT_TOP_N,
            "model": DEFAULT_MODEL_NAME,
            "use_hybrid_retrieval": True,
            "use_reranker": True,
            "use_query_reformulation": True,
            "hybrid_alpha": DEFAULT_HYBRID_ALPHA
        }
        
    def get_system_prompt(self):
        """Get the current system prompt based on selected role."""
        return AGENT_ROLES.get(self.current_role, AGENT_ROLES["General Tourism Assistant"])

# Initialize app state
app_state = AppState()

# --- Setup and initialization ---
@cl.on_chat_start
async def on_chat_start():
    """Initialize the tourism RAG chatbot when a new chat starts."""
    # Send a welcome message
    await cl.Message(
        content="""# 🌍 Welcome to Tourism Insights Explorer!
        
Analyze your tourism documents to extract valuable insights about:
- 📊 Travel market trends
- 💳 Payment methods across segments
- 👥 Customer segmentation strategies
- 🌱 Sustainability initiatives

**Upload your tourism PDF documents to get started!**
        """,
        metadata={"role": "system"}
    ).send()
    
    # Initialize tourism system components
    if not app_state.initialization_complete:
        with cl.Step("System Initialization", show_feedback=True) as step:
            step.input = "Initializing Tourism Analysis System"
            
            try:
                # Initialize NLP resources
                step.status = cl.StepStatus.RUNNING
                await step.update()
                
                load_nltk_resources()
                app_state.nlp_model = load_spacy_model()
                app_state.embedding_model = load_embedding_model()
                
                # Initialize vector database
                initialize_vector_db()
                app_state.collection = get_chroma_collection()
                
                if app_state.nlp_model and app_state.embedding_model and app_state.collection:
                    app_state.initialization_complete = True
                    step.output = "✅ Tourism Analysis System initialized successfully!"
                    step.status = cl.StepStatus.COMPLETED
                else:
                    step.output = "❌ Failed to initialize Tourism Analysis System. Please check logs."
                    step.status = cl.StepStatus.FAILED
            except Exception as e:
                logger.error(f"Initialization error: {str(e)}")
                step.output = f"❌ Error during initialization: {str(e)}"
                step.status = cl.StepStatus.FAILED
            
            await step.update()
    
    # Create action buttons for tourism expertise roles
    actions = [
        cl.Action(name="trends", value="Travel Trends Analyst", label="📊 Travel Trends"),
        cl.Action(name="payment", value="Payment Specialist", label="💳 Payment Analysis"),
        cl.Action(name="segments", value="Market Segmentation Expert", label="👥 Market Segments"),
        cl.Action(name="sustainability", value="Sustainability Tourism Advisor", label="🌱 Sustainability"),
        cl.Action(name="genz", value="Gen Z Travel Specialist", label="👧 Gen Z Travel"),
        cl.Action(name="luxury", value="Luxury Tourism Consultant", label="💎 Luxury Tourism"),
        cl.Action(name="analytics", value="Tourism Analytics Expert", label="📈 Tourism Analytics"),
        cl.Action(name="general", value="General Tourism Assistant", label="🧭 General Tourism")
    ]
    
    # Add expertise selector
    await cl.Message(
        content="Select a tourism expertise focus for your analysis:",
        actions=actions,
        metadata={"role": "system"}
    ).send()
    
    # Set up file upload element
    await cl.Message(
        content="📤 Upload your tourism documents (PDFs) to begin analysis.",
        author="Tourism Assistant"
    ).send()
    
    # Create settings element
    settings = cl.ChatSettings(
        [
            cl.ChatSettingsOption(
                name="chunk_size",
                label="Document Chunk Size",
                initial_value=app_state.settings["chunk_size"],
                description="Size of document chunks in words",
                component=cl.ChatSettingsSlider(min=100, max=1000, step=50)
            ),
            cl.ChatSettingsOption(
                name="top_n",
                label="Search Results",
                initial_value=app_state.settings["top_n"],
                description="Number of document chunks to retrieve per query",
                component=cl.ChatSettingsSlider(min=1, max=20, step=1)
            ),
            cl.ChatSettingsOption(
                name="use_hybrid_retrieval",
                label="Hybrid Search",
                initial_value=app_state.settings["use_hybrid_retrieval"],
                description="Combine vector and keyword search for better results",
                component=cl.ChatSettingsSwitch()
            ),
            cl.ChatSettingsOption(
                name="use_reranker",
                label="Neural Reranking",
                initial_value=app_state.settings["use_reranker"],
                description="Use AI to improve search result quality",
                component=cl.ChatSettingsSwitch()
            ),
            cl.ChatSettingsOption(
                name="use_query_reformulation",
                label="Query Enhancement",
                initial_value=app_state.settings["use_query_reformulation"],
                description="Automatically improve queries with context",
                component=cl.ChatSettingsSwitch()
            )
        ]
    )
    await settings.send()

@cl.on_settings_update
async def on_settings_update(settings: Dict[str, Any]):
    """Update app settings when user changes them."""
    # Update app state with new settings
    for key, value in settings.items():
        if key in app_state.settings:
            app_state.settings[key] = value
    
    # Notify user of updated settings
    await cl.Message(
        content=f"✅ Tourism analysis settings updated successfully!",
        author="System",
        disable_feedback=True
    ).send()

@cl.on_action
async def on_action(action: cl.Action):
    """Handle action button clicks for tourism expertise selection."""
    if action.value in AGENT_ROLES:
        app_state.current_role = action.value
        
        # Send confirmation of new expertise
        await cl.Message(
            content=f"📚 Tourism expertise changed to: **{action.value}**\n\nYour questions will now be analyzed from this perspective.",
            author="Tourism Assistant"
        ).send()
        
        # Update sidebar
        elements = []
        elements.append(cl.Image(name="role", path="./assets/roles/" + action.value.lower().replace(" ", "_") + ".png"))
        elements.append(cl.Text(name="description", content=f"**Current Expertise:** {action.value}\n\n{AGENT_ROLES[action.value]}"))
        await cl.ChatSettings(elements).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Process user messages and generate responses."""
    if not app_state.initialization_complete:
        await cl.Message(
            content="⚠️ Tourism Analysis System is not fully initialized. Please refresh and try again.",
            author="System"
        ).send()
        return
    
    if not app_state.processed_files:
        await cl.Message(
            content="⚠️ No tourism documents have been processed. Please upload PDF documents to begin analysis.",
            author="Tourism Assistant"
        ).send()
        return
    
    # Check if we have required components
    if not app_state.embedding_model or not app_state.collection:
        await cl.Message(
            content="⚠️ Tourism knowledge base is not available. Please refresh and try again.",
            author="System"
        ).send()
        return
    
    # Get user query
    user_query = message.content
    if not user_query or not user_query.strip():
        await cl.Message(
            content="Please enter a valid question about your tourism documents.",
            author="Tourism Assistant"
        ).send()
        return
    
    # Process user query with streaming response
    try:
        # Add loading message
        msg = cl.Message(author=app_state.current_role)
        await msg.send()
        
        # Get conversation memory
        conversation_memory = app_state.conversation_memory.get_formatted_history()
        
        # Create a TokenStream for analyzing and returning results
        token_stream = cl.TokenStream(message_id=msg.id)
        await token_stream.init()
        
        # Start a background task to process the query
        with cl.Step(name="Searching Tourism Knowledge Base", show_feedback=True) as step:
            step.input = user_query
            
            results = await asyncio.to_thread(
                hybrid_retrieval,
                query=user_query,
                embedding_model=app_state.embedding_model,
                collection=app_state.collection,
                top_n=app_state.settings["top_n"],
                alpha=app_state.settings["hybrid_alpha"],
                use_reranker=app_state.settings["use_reranker"]
            )
            
            # Create source documents for citation
            source_documents = []
            for i, result in enumerate(results):
                if 'metadata' in result and 'filename' in result['metadata']:
                    source_documents.append(
                        cl.SourceDocument(
                            page_content=result['text'],
                            metadata={
                                "source": result['metadata'].get('filename', f"Document {i+1}"),
                                "score": f"{result['score']:.2f}"
                            }
                        )
                    )
                else:
                    source_documents.append(
                        cl.SourceDocument(
                            page_content=result['text'],
                            metadata={"source": f"Document {i+1}", "score": f"{result['score']:.2f}"}
                        )
                    )
            
            if not results:
                step.output = "No relevant information found in the tourism documents."
                await step.update()
                await token_stream.append("I couldn't find relevant information in your tourism documents. Please try a different question or upload more documents related to this topic.")
                await token_stream.end()
                return
                
            # Log the search results
            step.output = f"Found {len(results)} relevant document segments"
            await step.update()
        
        # Process query with LLM
        with cl.Step(name="Generating Tourism Insights", show_feedback=True) as step:
            step.input = user_query
            
            # Simulate token streaming for faster response
            async def stream_tokens():
                answer = await asyncio.to_thread(
                    query_llm,
                    user_query=user_query,
                    top_n=app_state.settings["top_n"],
                    local_llm_model=app_state.settings["model"],
                    embedding_model=app_state.embedding_model,
                    collection=app_state.collection,
                    conversation_memory=conversation_memory,
                    system_prompt=app_state.get_system_prompt(),
                    use_hybrid_retrieval=app_state.settings["use_hybrid_retrieval"],
                    use_query_reformulation=app_state.settings["use_query_reformulation"],
                    hybrid_alpha=app_state.settings["hybrid_alpha"],
                    use_reranker=app_state.settings["use_reranker"]
                )
                
                # Stream tokens
                tokens = answer.split()
                for token in tokens:
                    await token_stream.append(token + " ")
                    # Random small delay for natural reading feel
                    await asyncio.sleep(0.02)
                
                # Update memory
                app_state.conversation_memory.add("User", user_query)
                app_state.conversation_memory.add("Assistant", answer)
                
                # Update step
                step.output = "Generated tourism insights based on document analysis."
                await step.update()
                
                return answer
            
            # Stream tokens and wait for completion
            final_answer = await stream_tokens()
            
            # End the token stream
            await token_stream.end()
            
            # Add source documents to the message
            await msg.update(source_documents=source_documents)
    
    except Exception as e:
        logger.error(f"Error processing tourism query: {str(e)}")
        await cl.Message(
            content=f"❌ An error occurred while processing your tourism query: {str(e)}",
            author="System"
        ).send()

@cl.on_file_upload
async def on_file_upload(file: AskFileResponse):
    """Process uploaded tourism document files."""
    if file.type != "application/pdf":
        await cl.Message(
            content="⚠️ Only PDF documents are supported for tourism analysis.",
            author="System"
        ).send()
        return
    
    try:
        # Check if we have the required components
        if not app_state.embedding_model or app_state.collection is None:
            app_state.embedding_model = load_embedding_model()
            app_state.collection = get_chroma_collection()
            
            if not app_state.embedding_model or app_state.collection is None:
                await cl.Message(
                    content="⚠️ Tourism analysis system is not properly initialized. Please refresh and try again.",
                    author="System"
                ).send()
                return
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.content)
            temp_file_path = temp_file.name
        
        # Create processing message
        process_msg = cl.Message(content=f"🔍 Analyzing tourism document: {file.name}...", author="System")
        await process_msg.send()
        
        # Process the document with advanced analysis
        with cl.Step(f"Processing {file.name}", show_feedback=True) as step:
            step.input = file.name
            
            try:
                # Convert file to proper format for processing
                uploaded_file = type('obj', (object,), {
                    'name': file.name,
                    'read': lambda: file.content
                })
                
                # Process with tourism-focused extraction
                chunks = await asyncio.to_thread(
                    process_uploaded_pdf,
                    uploaded_file=uploaded_file,
                    chunk_size=app_state.settings["chunk_size"],
                    overlap=app_state.settings["overlap"],
                    extract_images=True
                )
                
                if not chunks:
                    step.output = f"No content could be extracted from {file.name}."
                    step.status = cl.StepStatus.FAILED
                    await step.update()
                    
                    await process_msg.update(content=f"⚠️ No content could be extracted from {file.name}. Please check if the PDF contains text content.")
                    return
                
                # Process chunks and extract tourism entities
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
                
                for chunk in chunks:
                    # Extract entities
                    if "tourism_entities" not in chunk.get("metadata", {}):
                        entities = extract_tourism_entities(chunk["text"])
                        chunk["metadata"]["tourism_entities"] = entities
                    else:
                        entities = chunk["metadata"]["tourism_entities"]
                    
                    # Aggregate entities
                    for entity_type, items in entities.items():
                        if entity_type in all_tourism_entities:
                            all_tourism_entities[entity_type].update(items)
                    
                    # Process segment information
                    if "segment_matches" in chunk["metadata"]:
                        for segment, has_match in chunk["metadata"]["segment_matches"].items():
                            if has_match:
                                tourism_metrics["segments"][segment] = tourism_metrics["segments"].get(segment, 0) + 1
                    
                    # Process payment information
                    if chunk["metadata"].get("has_payment_info", False):
                        payment_keywords = ["credit card", "debit card", "cash", "digital wallet", 
                                        "mobile payment", "cryptocurrency"]
                        
                        for payment in payment_keywords:
                            if payment in chunk["text"].lower():
                                tourism_metrics["payment_methods"][payment] = tourism_metrics["payment_methods"].get(payment, 0) + 1
                
                # Add to collection
                add_success = await asyncio.to_thread(
                    add_chunks_to_collection,
                    chunks=[c["text"] for c in chunks],
                    embedding_model=app_state.embedding_model,
                    collection=app_state.collection
                )
                
                if add_success:
                    app_state.processed_files.add(file.name)
                    
                    # Convert sets to lists for storage
                    for entity_type in all_tourism_entities:
                        all_tourism_entities[entity_type] = list(all_tourism_entities[entity_type])
                    
                    # Merge with existing entities
                    for entity_type, entities in all_tourism_entities.items():
                        if entity_type not in app_state.extracted_entities:
                            app_state.extracted_entities[entity_type] = []
                        app_state.extracted_entities[entity_type].extend(entities)
                        # Remove duplicates
                        app_state.extracted_entities[entity_type] = list(set(app_state.extracted_entities[entity_type]))
                    
                    # Process tourism metrics
                    if tourism_metrics["segments"]:
                        for segment, count in tourism_metrics["segments"].items():
                            app_state.tourism_metrics.setdefault("segments", {})
                            app_state.tourism_metrics["segments"][segment] = app_state.tourism_metrics["segments"].get(segment, 0) + count
                    
                    if tourism_metrics["payment_methods"]:
                        for payment, count in tourism_metrics["payment_methods"].items():
                            app_state.tourism_metrics.setdefault("payment_methods", {})
                            app_state.tourism_metrics["payment_methods"][payment] = app_state.tourism_metrics["payment_methods"].get(payment, 0) + count
                    
                    # Update step
                    step.output = f"Successfully processed {file.name} with {len(chunks)} content segments."
                    step.status = cl.StepStatus.COMPLETED
                    
                    # Create entity summary
                    entity_summary = ""
                    for entity_type, entities in all_tourism_entities.items():
                        if entities:
                            entity_summary += f"**{entity_type}**: {', '.join(list(entities)[:5])}"
                            if len(entities) > 5:
                                entity_summary += f" and {len(entities) - 5} more"
                            entity_summary += "\n\n"
                    
                    # Update processing message
                    await process_msg.update(
                        content=f"""✅ Successfully analyzed tourism document: **{file.name}**

**Document Statistics:**
- 📄 {len(chunks)} content segments extracted
- 🔍 {sum(len(entities) for entities in all_tourism_entities.values())} tourism entities identified
- 📊 Document added to knowledge base

**Key Entities Identified:**
{entity_summary}

You can now ask questions about this document!"""
                    )
                else:
                    step.output = f"Failed to add {file.name} to tourism knowledge base."
                    step.status = cl.StepStatus.FAILED
                    await step.update()
                    
                    await process_msg.update(content=f"❌ Failed to add {file.name} to tourism knowledge base. Please try again.")
            except Exception as e:
                logger.error(f"Error processing tourism document {file.name}: {str(e)}")
                step.output = f"Error processing document: {str(e)}"
                step.status = cl.StepStatus.FAILED
                await step.update()
                
                await process_msg.update(content=f"❌ Error processing tourism document {file.name}: {str(e)}")
            
            # Display tourism insights if available
            if app_state.extracted_entities:
                # Create entity display
                entities_data = {
                    "DESTINATION": ", ".join(app_state.extracted_entities.get("DESTINATION", [])[:10]),
                    "ACCOMMODATION": ", ".join(app_state.extracted_entities.get("ACCOMMODATION", [])[:10]),
                    "TRANSPORTATION": ", ".join(app_state.extracted_entities.get("TRANSPORTATION", [])[:10]),
                    "ACTIVITY": ", ".join(app_state.extracted_entities.get("ACTIVITY", [])[:10]),
                    "ATTRACTION": ", ".join(app_state.extracted_entities.get("ATTRACTION", [])[:10])
                }
                
                # Create segments display if available
                segments_data = None
                if "segments" in app_state.tourism_metrics and app_state.tourism_metrics["segments"]:
                    total_segments = sum(app_state.tourism_metrics["segments"].values())
                    segments_data = {
                        segment: f"{round((count / total_segments) * 100, 1)}%"
                        for segment, count in app_state.tourism_metrics["segments"].items()
                    }
                
                # Create payment methods display if available
                payment_data = None
                if "payment_methods" in app_state.tourism_metrics and app_state.tourism_metrics["payment_methods"]:
                    total_payments = sum(app_state.tourism_metrics["payment_methods"].values())
                    payment_data = {
                        payment: f"{round((count / total_payments) * 100, 1)}%"
                        for payment, count in app_state.tourism_metrics["payment_methods"].items()
                    }
                
                # Create elements for display
                elements = []
                
                # Create tourism entities element
                entities_element = cl.Dataframe(
                    data=[[entities_data["DESTINATION"], entities_data["ACCOMMODATION"], entities_data["TRANSPORTATION"], entities_data["ACTIVITY"], entities_data["ATTRACTION"]]],
                    columns=["Destinations", "Accommodations", "Transportation", "Activities", "Attractions"],
                )
                
                # Create segments element if available
                if segments_data:
                    segments_element = cl.Dataframe(
                        data=[[segment, percentage] for segment, percentage in segments_data.items()],
                        columns=["Market Segment", "Percentage"]
                    )
                    elements.append(segments_element)
                
                # Create payment methods element if available
                if payment_data:
                    payment_element = cl.Dataframe(
                        data=[[payment, percentage] for payment, percentage in payment_data.items()],
                        columns=["Payment Method", "Percentage"]
                    )
                    elements.append(payment_element)
                
                # Send insights message with elements
                await cl.Message(
                    content=f"## 📊 Tourism Document Insights\n\nHere's a summary of key tourism entities and insights extracted from your documents:",
                    elements=elements,
                    author="Tourism Insights"
                ).send()
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass
    
    except Exception as e:
        logger.error(f"Error handling tourism file upload: {str(e)}")
        await cl.Message(
            content=f"❌ An error occurred while processing your tourism document: {str(e)}",
            author="System"
        ).send()

# Custom CSS to enhance the UI for tourism industry professionals
custom_css = """
/* Tourism-themed UI */
:root {
    --primary-color: #1E88E5;
    --secondary-color: #26A69A;
    --accent-color: #FFC107;
    --success-color: #4CAF50;
    --warning-color: #FF9800;
    --error-color: #F44336;
}

/* Custom font for tourism professionalism */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

body {
    font-family: 'Poppins', sans-serif;
}

/* Header styling */
.cl-main-header {
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    color: white !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.cl-main-header-appname {
    font-weight: 600 !important;
}

/* Message styling for tourism chat */
.cl-message-container[data-author="Tourism Assistant"] .cl-message-content,
.cl-message-container[data-author="Travel Trends Analyst"] .cl-message-content,
.cl-message-container[data-author="Payment Specialist"] .cl-message-content,
.cl-message-container[data-author="Market Segmentation Expert"] .cl-message-content,
.cl-message-container[data-author="Sustainability Tourism Advisor"] .cl-message-content,
.cl-message-container[data-author="Gen Z Travel Specialist"] .cl-message-content,
.cl-message-container[data-author="Luxury Tourism Consultant"] .cl-message-content,
.cl-message-container[data-author="Tourism Analytics Expert"] .cl-message-content,
.cl-message-container[data-author="General Tourism Assistant"] .cl-message-content,
.cl-message-container[data-author="Tourism Insights"] .cl-message-content {
    background-color: #E8F5E9;
    border-radius: 12px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

/* User message styling */
.cl-message-container[data-author="user"] .cl-message-content {
    background-color: #E3F2FD;
    border-radius: 12px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

/* System message styling */
.cl-message-container[data-author="system"] .cl-message-content {
    background-color: #FFF8E1;
    border-radius: 8px;
    border-left: 4px solid var(--accent-color);
}

/* File upload styling */
.cl-file-dropzone {
    border: 2px dashed var(--secondary-color);
    background-color: rgba(38, 166, 154, 0.05);
    transition: all 0.3s ease;
}

.cl-file-dropzone:hover {
    background-color: rgba(38, 166, 154, 0.1);
    border-color: var(--primary-color);
}

/* Action button styling */
.cl-action-button {
    border-radius: 20px;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.cl-action-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

/* Tables for tourism data */
table {
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

th {
    background-color: var(--primary-color);
    color: white;
    font-weight: 500;
    text-align: left;
    padding: 12px 15px;
}

td {
    padding: 10px 15px;
    border-bottom: 1px solid #f0f0f0;
}

tr:last-child td {
    border-bottom: none;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

/* Input field styling */
.cl-chat-input-container {
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
}

.cl-chat-input-container:focus-within {
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.12);
}

/* Loading indicators */
.cl-step-loading {
    color: var(--primary-color);
}

/* Chat settings */
.cl-chat-settings-container {
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.cl-chat-settings-option-label {
    font-weight: 500;
}

/* Source references styling */
.cl-reference-list-header {
    color: var(--primary-color);
    font-weight: 600;
}

.cl-reference-item {
    border-left: 3px solid var(--secondary-color);
    padding-left: 10px;
    margin: 8px 0;
    background-color: rgba(38, 166, 154, 0.05);
    border-radius: 0 6px 6px 0;
}

/* Mobile optimization */
@media (max-width: 768px) {
    .cl-main-container {
        padding: 10px;
    }
    
    .cl-message-content {
        padding: 12px;
    }
    
    .cl-action-panel {
        flex-wrap: wrap;
    }
}
"""

# Configure Chainlit with tourism theme
cl.configure(
    title="Tourism Insights Explorer",
    description="Analyze tourism documents for trends, payment methods, market segments, and sustainability",
    logo_url="https://cdn-icons-png.flaticon.com/512/2161/2161470.png",
    theme=cl.Theme(
        primary="#1E88E5",
        secondary="#26A69A",
        accent="#FFC107",
        success="#4CAF50",
        warning="#FF9800",
        error="#F44336",
    ),
    custom_css=custom_css,
    code=True,  # For code highlighting
    markdown=True,  # For better formatting
    host="0.0.0.0",  # Accept connections from all interfaces
    port=8000,  # Default port
    debug=False,  # Disable debug mode for production
    watch=False  # Disable hot reloading for better performance
)