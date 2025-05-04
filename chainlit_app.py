import os
import time
import asyncio
import json
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union

import chainlit as cl
# Update import for newer Chainlit versions - AskFileResponse was renamed
try:
    from chainlit.types import AskFileResponse  # For older versions
except ImportError:
    # For newer versions, we'll handle file objects directly
    pass

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

# Check Chainlit version and set compatibility flags
try:
    cl_version = cl.__version__
    log_error(f"Detected Chainlit version: {cl_version}")

    # Parse version for compatibility checks
    version_parts = cl_version.split('.')
    MAJOR_VERSION = int(version_parts[0]) if version_parts else 0
    MINOR_VERSION = int(version_parts[1]) if len(version_parts) > 1 else 0

    # Set flags based on version
    USE_NEW_FILE_API = MAJOR_VERSION >= 1
    USE_ASYNC_ACTIONS = MAJOR_VERSION >= 1

except (AttributeError, ValueError, IndexError):
    # Default to assuming newer version if we can't detect
    log_error("Could not determine Chainlit version, assuming newer API")
    USE_NEW_FILE_API = True
    USE_ASYNC_ACTIONS = True

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

# --- Global state --- # This class definition remains the same as provided
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

# --- Rest of the code follows with appropriate adaptations ---
# ... (The rest of your code like on_chat_start, etc.)
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

    # Check and fix Chainlit config if needed
    try:
        config_dir = os.path.join(os.path.expanduser("~"), ".chainlit")
        config_file = os.path.join(config_dir, "config.toml")

        if os.path.exists(config_file):
            # Check if config file is valid
            try:
                with open(config_file, 'r') as f:
                    config_content = f.read()

                # Simple validation - check if it contains expected sections
                if "[project]" not in config_content or "[UI]" not in config_content:
                    # Config file may be corrupt - backup and remove
                    backup_file = f"{config_file}.bak"
                    os.rename(config_file, backup_file)
                    log_error(f"Detected potentially corrupt Chainlit config. Backed up to {backup_file}")

                    await cl.Message(
                        content="⚠️ Chainlit configuration has been reset due to potential corruption. Please refresh the page.",
                        author="System"
                    ).send()

                    return
            except Exception as e:
                # Error reading config file - might be corrupt
                log_error(f"Error reading Chainlit config file: {str(e)}")
                try:
                    # Try to backup and remove
                    backup_file = f"{config_file}.error"
                    os.rename(config_file, backup_file)
                    log_error(f"Backed up problematic config to {backup_file}")

                    await cl.Message(
                        content="⚠️ Chainlit configuration has been reset due to errors. Please refresh the page.",
                        author="System"
                    ).send()

                    return
                except Exception as rename_err:
                    log_error(f"Failed to backup/remove corrupt config: {str(rename_err)}")
    except Exception as config_check_err:
        log_error(f"Error checking Chainlit config: {str(config_check_err)}")

    # Initialize tourism system components
    if not app_state.initialization_complete:
        with cl.Step("System Initialization", show_feedback=True) as step:
            step.input = "Initializing Tourism Analysis System"

            try:
                # Initialize NLP resources
                step.status = cl.StepStatus.RUNNING
                await step.update()

                # More robust NLTK resources loading
                nltk_success = await asyncio.to_thread(load_nltk_resources)
                if not nltk_success:
                    step.output = "❌ Failed to initialize NLTK resources. Document processing may fail."
                    step.status = cl.StepStatus.WARNING
                    await step.update()

                    # Send a warning message
                    await cl.Message(
                        content="⚠️ NLTK resources could not be loaded. Document processing may be limited.",
                        author="System"
                    ).send()

                    # Don't return - attempt to initialize other components

                # Load NLP and embedding models
                app_state.nlp_model = await asyncio.to_thread(load_spacy_model)
                app_state.embedding_model = await asyncio.to_thread(load_embedding_model)

                # Check for critical model failures
                if not app_state.nlp_model:
                    step.output = "❌ Failed to load NLP model. Document processing will be limited."
                    step.status = cl.StepStatus.WARNING
                    await step.update()

                    await cl.Message(
                        content="⚠️ NLP model could not be loaded. Entity extraction will be limited.",
                        author="System"
                    ).send()

                    # Continue with limited functionality

                if not app_state.embedding_model:
                    step.output = "❌ Failed to load embedding model. Search functionality will not work."
                    step.status = cl.StepStatus.FAILED
                    await step.update()

                    await cl.Message(
                        content="❌ Embedding model could not be loaded. Please refresh and try again.",
                        author="System"
                    ).send()

                    # This is a critical failure - return early
                    return

                # Initialize vector database
                db_success = await asyncio.to_thread(initialize_vector_db)
                if not db_success:
                    step.output = "❌ Failed to initialize vector database. Search will not work."
                    step.status = cl.StepStatus.FAILED
                    await step.update()

                    await cl.Message(
                        content="❌ Vector database initialization failed. Please refresh and try again.",
                        author="System"
                    ).send()

                    # This is a critical failure - return early
                    return

                app_state.collection = await asyncio.to_thread(get_chroma_collection)

                if not app_state.collection:
                    step.output = "❌ Failed to get vector collection. Search will not work."
                    step.status = cl.StepStatus.FAILED
                    await step.update()

                    await cl.Message(
                        content="❌ Vector collection could not be accessed. Please refresh and try again.",
                        author="System"
                    ).send()

                    # This is a critical failure - return early
                    return

                # If we made it here with all critical components, mark as complete
                if app_state.embedding_model and app_state.collection:
                    app_state.initialization_complete = True
                    step.output = "✅ Tourism Analysis System initialized successfully!"
                    step.status = cl.StepStatus.COMPLETED
                else:
                    step.output = "⚠️ Tourism Analysis System initialized with limitations."
                    step.status = cl.StepStatus.WARNING
            except Exception as e:
                logger.error(f"Initialization error: {str(e)}")
                step.output = f"❌ Error during initialization: {str(e)}"
                step.status = cl.StepStatus.FAILED

                await cl.Message(
                    content=f"❌ System initialization failed: {str(e)}\n\nPlease refresh and try again.",
                    author="System"
                ).send()

                return

            await step.update()

    # Add a message indicating readiness for file upload
    await cl.Message(
        content="System initialized. You can now upload your PDF documents for analysis.",
        author="System"
    ).send()

    # Create action buttons for tourism expertise roles
    # ... [rest of the function remains unchanged] ...


@cl.on_chat_start
async def on_chat_start_old(): # Renamed the old function temporarily
    """Initialize the tourism RAG chatbot when a new chat starts."""
    print("--- DEBUG: Entering on_chat_start ---")
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
    
    # Initialize tourism system components (OLD VERSION - TO BE REMOVED)
    if not app_state.initialization_complete:
        print("--- DEBUG: Starting System Initialization block ---")
        async with cl.Step(name="System Initialization", show_input=True) as step:
            step.input = "Initializing Tourism Analysis System"
            print("--- DEBUG: System Initialization Step created ---")
            
            try:
                # Initialize NLP resources
                step.status = cl.StepStatus.RUNNING
                await step.update()
                
                print("--- DEBUG: Calling load_nltk_resources() ---")
                load_nltk_resources()
                print("--- DEBUG: Calling load_spacy_model() ---")
                app_state.nlp_model = load_spacy_model()
                print("--- DEBUG: Calling load_embedding_model() ---")
                app_state.embedding_model = load_embedding_model()
                print("--- DEBUG: Calling initialize_vector_db() ---")
                # Initialize vector database
                initialize_vector_db()
                app_state.collection = get_chroma_collection()
                
                if app_state.nlp_model and app_state.embedding_model and app_state.collection:
                    app_state.initialization_complete = True
                    step.output = "✅ Tourism Analysis System initialized successfully!"
                    step.status = cl.StepStatus.COMPLETED
                    print("--- DEBUG: System Initialization successful ---")
                else:
                    step.output = "❌ Failed to initialize Tourism Analysis System. Please check logs."
                    step.status = cl.StepStatus.FAILED
                    print("--- DEBUG: System Initialization failed (component check) ---")
            except Exception as e:
                logger.error(f"Initialization error: {str(e)}")
                step.output = f"❌ Error during initialization: {str(e)}"
                step.status = cl.StepStatus.FAILED
                print(f"--- DEBUG: System Initialization failed with exception: {e} ---")
            
            await step.update()
        print("--- DEBUG: Exiting System Initialization block ---")
    
    
    # Wait for all file processing tasks to complete
    if file_processing_tasks:
        print(f"--- DEBUG: Waiting for {len(file_processing_tasks)} file processing tasks ---")
        await asyncio.gather(*file_processing_tasks)
        print("--- DEBUG: File processing tasks complete ---")
    
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
    print("--- DEBUG: Sent role selection message ---")
    
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
    print("--- DEBUG: Exiting on_chat_start ---")


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

# Register callback for each tourism expertise action
@cl.action_callback("trends")
@cl.action_callback("payment")
@cl.action_callback("segments")
@cl.action_callback("sustainability")
@cl.action_callback("genz")
@cl.action_callback("luxury")
@cl.action_callback("analytics")
@cl.action_callback("general")
async def on_action(action: cl.Action):
    """Handle action button clicks for tourism expertise selection."""
    print(f"--- DEBUG: Action callback triggered: {action.name} ({action.value}) ---")
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
        print(f"--- DEBUG: Updated role to {action.value} ---")

@cl.on_message
async def on_message(message: cl.Message):
    """Process user messages and generate responses."""
    print(f"--- DEBUG: Entering on_message for query: '{message.content[:50]}...' ---")
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
        async with cl.Step(name="Searching Tourism Knowledge Base", show_input=True) as step:
            print(f"--- DEBUG: Starting hybrid retrieval for query: '{user_query}' ---")
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
            print(f"--- DEBUG: Hybrid retrieval returned {len(results)} results ---")
            
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
        async with cl.Step(name="Generating Tourism Insights", show_input=True) as step:
            print(f"--- DEBUG: Starting LLM query for: '{user_query}' ---")
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
                print(f"--- DEBUG: LLM query returned answer (length: {len(answer)}) ---")
                
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
            print("--- DEBUG: Token streaming complete ---")
            
            # End the token stream
            await token_stream.end()
            
            # Add source documents to the message
            await msg.update(source_documents=source_documents)
    
    except Exception as e:
        error_message = f"Error processing tourism query: {str(e)}"
        logger.error(error_message)
        await cl.Message(
            content=f"❌ An error occurred: {error_message}",
            author="System"
        ).send()
        print(f"--- DEBUG: Error in on_message: {error_message} ---")
        return
    
    print("--- DEBUG: Exiting on_message ---")

# For most recent Chainlit versions:
@cl.on_chat_message_files
async def on_chat_message_files(files: List[cl.File]): # Use cl.File for type hint if available
    """Process uploaded tourism document files."""
    if not files: # Check if the list is empty
        return

    # First check if system is initialized
    if not app_state.initialization_complete:
        await cl.Message(
            content="❌ System is not fully initialized. Please wait for initialization to complete or refresh the page.",
            author="System"
        ).send()
        return

    # Check if we have the required components
    if not app_state.embedding_model or app_state.collection is None:
        # Try to load them if not available (useful after refresh)
        log_error("Missing components during file upload, attempting to reload...")
        app_state.embedding_model = await asyncio.to_thread(load_embedding_model)
        app_state.collection = await asyncio.to_thread(get_chroma_collection)

        if not app_state.embedding_model or app_state.collection is None:
            log_error("Failed to load required components for file processing")
            await cl.Message(
                content="⚠️ Tourism analysis system is not properly initialized. Please refresh and try again.",
                author="System"
            ).send()
            return

    for file in files:
        # Check file type using the file object's attributes
        if not hasattr(file, 'type') or not file.type.startswith("application/pdf"):
            await cl.Message(
                content=f"⚠️ Only PDF documents are supported for tourism analysis. Skipped: {file.name}",
                author="System"
            ).send()
            continue

        temp_file_path = None
        try:
            # Get file content and name from the cl.File object
            file_content = await file.get_bytes()
            file_name = file.name

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Create processing message
            process_msg = cl.Message(content=f"🔍 Analyzing tourism document: {file_name}...", author="System")
            await process_msg.send()

            # Process the document with advanced analysis
            async with cl.Step(name=f"Processing {file_name}", show_input=True, show_feedback=True) as step:
                step.input = file_name

                try:
                    # Convert file to proper format for processing
                    # The 'file' object from AskFileResponse already has 'name' and 'content'
                    # We need a 'read' method for process_uploaded_pdf
                    uploaded_file_obj = type('obj', (object,), {
                        'name': file.name,
                        'read': lambda: file_content # Use the fetched content
                    })

                    # Process with tourism-focused extraction
                    print(f"--- DEBUG: Calling process_uploaded_pdf for {file_name} ---")
                    chunks = await asyncio.to_thread(
                        process_uploaded_pdf,
                        uploaded_file=uploaded_file_obj,
                        chunk_size=app_state.settings["chunk_size"],
                        overlap=app_state.settings["overlap"],
                        extract_images=True # Keep image extraction if needed
                    )
                    print(f"--- DEBUG: process_uploaded_pdf returned {len(chunks)} chunks for {file_name} ---")

                    if not chunks:
                        step.output = f"No content could be extracted from {file.name}."
                        step.status = cl.StepStatus.FAILED
                        await step.update()

                        await process_msg.update(content=f"⚠️ No content could be extracted from {file.name}. Please check if the PDF contains text content.")
                        print(f"--- DEBUG: No content extracted from {file_name} ---")
                        continue # Process next file

                    # Process chunks and extract tourism entities/metrics
                    file_tourism_entities = {
                        "DESTINATION": set(), "ACCOMMODATION": set(), "TRANSPORTATION": set(),
                        "ACTIVITY": set(), "ATTRACTION": set()
                    }
                    file_tourism_metrics = {"segments": {}, "payment_methods": {}}

                    batch_size = 10 # Process in batches if needed
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        step.output = f"Analyzing content batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}..."
                        await step.update()

                        for chunk in batch:
                            try:
                                # Extract entities
                                if "tourism_entities" not in chunk.get("metadata", {}):
                                    entities = extract_tourism_entities(chunk["text"])
                                    chunk["metadata"]["tourism_entities"] = entities
                                else:
                                    entities = chunk["metadata"]["tourism_entities"]

                                # Aggregate entities for this file
                                for entity_type, items in entities.items():
                                    if entity_type in file_tourism_entities:
                                        file_tourism_entities[entity_type].update(items)

                                # Process segment information
                                if "segment_matches" in chunk["metadata"]:
                                    for segment, has_match in chunk["metadata"]["segment_matches"].items():
                                        if has_match:
                                            file_tourism_metrics["segments"][segment] = file_tourism_metrics["segments"].get(segment, 0) + 1

                                # Process payment information
                                if chunk["metadata"].get("has_payment_info", False):
                                    payment_keywords = ["credit card", "debit card", "cash", "digital wallet",
                                                    "mobile payment", "cryptocurrency"]
                                    for payment in payment_keywords:
                                        if payment in chunk["text"].lower():
                                            file_tourism_metrics["payment_methods"][payment] = file_tourism_metrics["payment_methods"].get(payment, 0) + 1

                            except Exception as chunk_err:
                                log_error(f"Error processing chunk from {file_name}: {str(chunk_err)}")
                                # Continue with next chunk

                    # Add chunks to collection (ensure embedding_model and collection are valid)
                    print(f"--- DEBUG: Calling add_chunks_to_collection for {len(chunks)} chunks from {file.name} ---")
                    add_success = await asyncio.to_thread(
                        add_chunks_to_collection,
                        chunks=[c["text"] for c in chunks], # Pass only text
                        metadatas=[c.get("metadata", {}) for c in chunks], # Pass metadata
                        embedding_model=app_state.embedding_model,
                        collection=app_state.collection
                    )
                    print(f"--- DEBUG: add_chunks_to_collection returned: {add_success} for {file_name} ---")

                    if add_success:
                        app_state.processed_files.add(file_name)

                        # Convert sets to lists for storage and merge with global state
                        for entity_type in file_tourism_entities:
                            entities_list = list(file_tourism_entities[entity_type])
                            if entity_type not in app_state.extracted_entities:
                                app_state.extracted_entities[entity_type] = []
                            app_state.extracted_entities[entity_type].extend(entities_list)
                            # Remove duplicates globally
                            app_state.extracted_entities[entity_type] = list(set(app_state.extracted_entities[entity_type]))

                        # Merge metrics with global state
                        if file_tourism_metrics["segments"]:
                            app_state.tourism_metrics.setdefault("segments", {})
                            for segment, count in file_tourism_metrics["segments"].items():
                                app_state.tourism_metrics["segments"][segment] = app_state.tourism_metrics["segments"].get(segment, 0) + count

                        if file_tourism_metrics["payment_methods"]:
                            app_state.tourism_metrics.setdefault("payment_methods", {})
                            for payment, count in file_tourism_metrics["payment_methods"].items():
                                app_state.tourism_metrics["payment_methods"][payment] = app_state.tourism_metrics["payment_methods"].get(payment, 0) + count

                        # Update step
                        step.output = f"Successfully processed {file_name} with {len(chunks)} content segments."
                        step.status = cl.StepStatus.COMPLETED

                        # Create entity summary for this file
                        entity_summary = ""
                        total_entities_found = 0
                        for entity_type, entities in file_tourism_entities.items():
                            if entities:
                                entity_list = list(entities)
                                total_entities_found += len(entity_list)
                                entity_summary += f"**{entity_type}**: {', '.join(entity_list[:5])}"
                                if len(entity_list) > 5:
                                    entity_summary += f" and {len(entity_list) - 5} more"
                                entity_summary += "\n\n"

                        # Update processing message
                        await process_msg.update(
                            content=f"""✅ Successfully analyzed tourism document: **{file_name}**

**Document Statistics:**
- 📄 {len(chunks)} content segments extracted
- 🔍 {total_entities_found} tourism entities identified in this document
- 📊 Document added to knowledge base

**Key Entities Identified in this Document:**
{entity_summary if entity_summary else "None"}

You can now ask questions about this document!"""
                        )

                    else:
                        step.output = f"Failed to add {file_name} to tourism knowledge base."
                        step.status = cl.StepStatus.FAILED
                        await step.update()

                        await process_msg.update(content=f"❌ Failed to add {file_name} to tourism knowledge base. Please try again.")
                        print(f"--- DEBUG: Failed to add {file_name} to collection ---")

                except Exception as e:
                    logger.error(f"Error processing tourism document {file_name}: {str(e)}")
                    step.output = f"Error processing document: {str(e)}"
                    step.status = cl.StepStatus.FAILED
                    await step.update()

                    await process_msg.update(content=f"❌ Error processing tourism document {file_name}: {str(e)}")
                    print(f"--- DEBUG: Exception during PDF processing for {file_name}: {e} ---")

                await step.update() # Final update for the step

            # Display updated global tourism insights if available
            if app_state.extracted_entities or app_state.tourism_metrics:
                await display_global_insights()

        except Exception as e:
            logger.error(f"Error handling tourism file upload for {file.name}: {str(e)}")
            await cl.Message(
                content=f"❌ An error occurred while processing {file.name}: {str(e)}",
                author="System"
            ).send() # Use file.name here as file_name might not be set if error is early
            print(f"--- DEBUG: Outer exception in on_file_upload for {file.name}: {e} ---")

        finally:
            # Clean up temporary file
            try:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as cleanup_err:
                log_error(f"Error cleaning up temporary file {temp_file_path}: {str(cleanup_err)}")

async def display_global_insights():
    """Helper function to display aggregated insights in the UI."""
    print("--- DEBUG: Updating global insights display ---")
    elements = []

    # Create tourism entities element (using global state)
    entities_data = {
        "DESTINATION": ", ".join(app_state.extracted_entities.get("DESTINATION", [])[:10]),
        "ACCOMMODATION": ", ".join(app_state.extracted_entities.get("ACCOMMODATION", [])[:10]),
        "TRANSPORTATION": ", ".join(app_state.extracted_entities.get("TRANSPORTATION", [])[:10]),
        "ACTIVITY": ", ".join(app_state.extracted_entities.get("ACTIVITY", [])[:10]),
        "ATTRACTION": ", ".join(app_state.extracted_entities.get("ATTRACTION", [])[:10])
    }
    entities_element = cl.Dataframe(
        data=[[entities_data["DESTINATION"], entities_data["ACCOMMODATION"], entities_data["TRANSPORTATION"], entities_data["ACTIVITY"], entities_data["ATTRACTION"]]],
        columns=["Destinations (Top 10)", "Accommodations (Top 10)", "Transportation (Top 10)", "Activities (Top 10)", "Attractions (Top 10)"],
        title="Aggregated Tourism Entities"
    )
    elements.append(entities_element)

    # Create segments element if available (using global state)
    if "segments" in app_state.tourism_metrics and app_state.tourism_metrics["segments"]:
        total_segments = sum(app_state.tourism_metrics["segments"].values())
        if total_segments > 0:
            segments_data = {
                segment: f"{round((count / total_segments) * 100, 1)}%"
                for segment, count in sorted(app_state.tourism_metrics["segments"].items(), key=lambda item: item[1], reverse=True)
            }
            segments_element = cl.Dataframe(
                data=[[segment, percentage] for segment, percentage in segments_data.items()],
                columns=["Market Segment", "Mention Frequency"],
                title="Aggregated Market Segment Mentions"
            )
            elements.append(segments_element)

    # Create payment methods element if available (using global state)
    if "payment_methods" in app_state.tourism_metrics and app_state.tourism_metrics["payment_methods"]:
        total_payments = sum(app_state.tourism_metrics["payment_methods"].values())
        if total_payments > 0:
            payment_data = {
                payment: f"{round((count / total_payments) * 100, 1)}%"
                for payment, count in sorted(app_state.tourism_metrics["payment_methods"].items(), key=lambda item: item[1], reverse=True)
            }
            payment_element = cl.Dataframe(
                data=[[payment, percentage] for payment, percentage in payment_data.items()],
                columns=["Payment Method", "Mention Frequency"],
                title="Aggregated Payment Method Mentions"
            )
            elements.append(payment_element)

    # Send insights message with elements
    if elements:
        await cl.Message(
            content=f"## 📊 Updated Global Tourism Insights\n\nHere's a summary of key tourism entities and insights extracted from **all processed documents**:",
            elements=elements,
            author="Tourism Insights"
        ).send()
        print(f"--- DEBUG: Sent updated global tourism insights message ---")
    else:
        print(f"--- DEBUG: No global insights to display yet ---")