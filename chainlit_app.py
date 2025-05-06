import os
import time
import asyncio
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union

import chainlit as cl
# Use cl.File for newer versions
if hasattr(cl, 'File'):
    FileType = cl.File
else:
    # Fallback for older versions if needed, though cl.File is standard now
    FileType = Any # Or define a placeholder if strict typing is needed

import numpy as np
import chromadb

# Import modules
from modules.system_setup import (
    ensure_dependencies, setup_ollama, refresh_available_models,
    install_package, download_model, DEFAULT_MODEL_NAME, TOURISM_RECOMMENDED_MODELS
)
from modules.vector_store import initialize_vector_db, reset_vector_db, get_chroma_collection, hybrid_retrieval
from modules.nlp_models import load_embedding_model, get_embedding_dimensions, extract_tourism_entities_streaming # Use streaming version
from modules.pdf_processor import process_uploaded_pdf
from modules.vector_store import add_chunks_to_collection
from modules.llm_interface import query_llm, SlidingWindowMemory
from modules.utils import log_error, TourismLogger, extract_tourism_metrics_from_text
from modules.memory_utils import memory_monitor # Import memory monitor
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
logger = TourismLogger(log_dir=".chainlit/logs") # Log within chainlit folder if possible

# --- Constants and AGENT_ROLES ---
DEFAULT_CHUNK_SIZE = 512 # Match app.py
DEFAULT_OVERLAP = 64   # Match app.py
DEFAULT_TOP_N = 5      # Match app.py
DEFAULT_CONVERSATION_MEMORY = 3
DEFAULT_HYBRID_ALPHA = 0.7  # Weight balance between vector and BM25 search
DEFAULT_PERFORMANCE_TARGET = "balanced"
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
        self.tourism_insights = {} # Renamed from tourism_metrics for consistency
        self.initialization_complete = False
        self.collection = None # Chroma collection
        self.embedding_model = None
        self.nlp_model = None
        self.settings = {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "overlap": DEFAULT_OVERLAP,
            "top_n": DEFAULT_TOP_N,
            "model": DEFAULT_MODEL_NAME,
            "performance_target": DEFAULT_PERFORMANCE_TARGET,
            "use_hybrid_retrieval": True,
            "use_reranker": True,
            "hybrid_alpha": DEFAULT_HYBRID_ALPHA
        }
        self.available_models = [DEFAULT_MODEL_NAME] # Store available LLMs
        
    def get_system_prompt(self):
        """Get the current system prompt based on selected role."""
        return AGENT_ROLES.get(self.current_role, AGENT_ROLES["General Tourism Assistant"])

# Initialize app state
app_state = AppState()

# --- Setup and initialization ---
@cl.on_chat_start
async def on_chat_start():
    """Initialize the tourism RAG chatbot when a new chat starts."""
    # Send welcome message
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
        async with cl.Step(name="System Initialization", show_input=False) as step:
            step.input = "Initializing Tourism Analysis System"
            await step.update()

            try:
                step.status = cl.StepStatus.RUNNING
                await step.update()

                # 1. Check dependencies
                step.output = "Checking dependencies..."
                await step.update()
                mismatched = await asyncio.to_thread(ensure_dependencies)
                if mismatched:
                    pkgs = ", ".join([f"{p[0]}=={p[1]}" for p in mismatched])
                    step.output = f"⚠️ Missing/mismatched dependencies: {pkgs}. Please install them."
                    step.status = cl.StepStatus.FAILED
                    await step.update()
                    await cl.Message(
                        content=f"⚠️ Missing dependencies: `{pkgs}`. Please install them (`pip install ...`) and restart.",
                        author="System"
                    ).send()
                    return # Stop initialization if dependencies are missing

                # 2. Setup Ollama
                step.output = "Setting up Ollama..."
                await step.update()
                ollama_ok = await asyncio.to_thread(setup_ollama)
                if not ollama_ok:
                    step.output = "❌ Ollama setup failed. Please ensure Ollama is installed and running."
                    step.status = cl.StepStatus.FAILED
                    await step.update()
                    await cl.Message(
                        content="❌ Ollama is not detected or failed to set up. Please install and run Ollama.",
                        author="System"
                    ).send()
                    return # Stop if Ollama isn't ready

                # 3. Check available LLM models
                step.output = "Checking available LLM models..."
                await step.update()
                app_state.available_models = await asyncio.to_thread(refresh_available_models)
                if not app_state.available_models:
                    step.output = f"⚠️ No Ollama models found. Attempting to download default: {DEFAULT_MODEL_NAME}"
                    await step.update()
                    success, msg = await asyncio.to_thread(download_model, DEFAULT_MODEL_NAME)
                    if success:
                        step.output = f"✅ Downloaded {DEFAULT_MODEL_NAME}. Refreshing models..."
                        await step.update()
                        app_state.available_models = await asyncio.to_thread(refresh_available_models)
                    else:
                        step.output = f"❌ Failed to download {DEFAULT_MODEL_NAME}: {msg}"
                        step.status = cl.StepStatus.FAILED
                        await step.update()
                        await cl.Message(content=f"❌ Failed to download default model: {msg}", author="System").send()
                        return # Stop if no model available

                # 4. Load Embedding Model
                step.output = f"Loading embedding model ({app_state.settings['performance_target']} target)..."
                await step.update()
                app_state.embedding_model = await asyncio.to_thread(
                    load_embedding_model,
                    performance_target=app_state.settings['performance_target']
                )
                if not app_state.embedding_model:
                    step.output = "❌ Failed to load embedding model. Search functionality will not work."
                    step.status = cl.StepStatus.FAILED
                    await step.update()
                    await cl.Message(
                        content="❌ Embedding model could not be loaded. Please refresh and try again.",
                        author="System"
                    ).send()
                    return

                # 5. Initialize Vector Database
                step.output = "Initializing vector database..."
                await step.update()
                db_success = await asyncio.to_thread(initialize_vector_db)
                if not db_success:
                    step.output = "❌ Failed to initialize vector database. Search will not work."
                    step.status = cl.StepStatus.FAILED
                    await step.update()

                    await cl.Message(
                        content="❌ Vector database initialization failed. Please refresh and try again.",
                        author="System"
                    ).send()
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
                    return

                # If we made it here with all critical components, mark as complete
                if app_state.embedding_model and app_state.collection is not None:
                    app_state.initialization_complete = True
                    step.output = "✅ Tourism Analysis System initialized successfully!"
                    step.status = cl.StepStatus.COMPLETED
                else:
                    step.output = "⚠️ Tourism Analysis System initialized with limitations."
                    step.status = cl.StepStatus.WARNING
            except Exception as e:
                logger.error(f"Initialization error: {str(e)}", exc_info=True)
                step.output = f"❌ Error during initialization: {str(e)}"
                step.status = cl.StepStatus.FAILED
                await step.update()
                await cl.Message(
                    content=f"❌ System initialization failed: {str(e)}\n\nPlease refresh and try again.",
                    author="System"
                ).send()

                return

    # Add a message indicating readiness for file upload
    if app_state.initialization_complete:
        await cl.Message(content="System initialized. You can now upload PDF documents.", author="System").send()

    # Create action buttons for tourism expertise roles
    actions = [
        cl.Action(name="trends", value="Travel Trends Analyst", label="📊 Travel Trends"),
        cl.Action(name="payment", value="Payment Specialist", label="💳 Payment Analysis"),
        cl.Action(name="segments", value="Market Segmentation Expert", label="👥 Market Segments"),
        cl.Action(name="sustainability", value="Sustainability Tourism Advisor", label="🌱 Sustainability"),
        cl.Action(name="genz", value="Gen Z Travel Specialist", label="👧 Gen Z Travel"),
        cl.Action(name="luxury", value="Luxury Tourism Consultant", label="💎 Luxury Tourism"),
        cl.Action(name="analytics", value="Tourism Analytics Expert", label="📈 Tourism Analytics"),
        cl.Action(name="general", value="General Tourism Assistant", label="🧭 General Tourism"),
        cl.Action(name="reset_db", value="reset", label="🗑️ Reset Database") # Add reset action
    ]
    
    # Add expertise selector
    await cl.Message(
        content="Select a tourism expertise focus for your analysis:",
        actions=actions,
        metadata={"role": "system"}
    ).send()
    
    # Create settings element
    settings_elements = [
        cl.ChatSettingsSlider(
            id="chunk_size", # Use id for mapping in on_settings_update
            label="Document Chunk Size",
            initial=app_state.settings["chunk_size"],
            min=100, max=1000, step=50,
            description="Size of document chunks in characters (approx)."
        ),
        cl.ChatSettingsSlider(
            id="overlap",
            label="Chunk Overlap",
            initial=app_state.settings["overlap"],
            min=0, max=200, step=10,
            description="Overlap between chunks."
        ),
        cl.ChatSettingsSlider(
            id="top_n",
            label="Search Results (Top N)",
            initial=app_state.settings["top_n"],
            min=1, max=20, step=1,
            description="Number of document chunks to retrieve per query."
        ),
        cl.ChatSettingsSelect(
            id="model",
            label="LLM Model",
            values=app_state.available_models,
            initial_value=app_state.settings["model"]
        ),
        cl.ChatSettingsSwitch(
            id="use_hybrid_retrieval",
            label="Hybrid Search (Vector + Keyword)",
            initial=app_state.settings["use_hybrid_retrieval"],
            description="Combine vector and keyword search for potentially better results."
        ),
        cl.ChatSettingsSlider(
            id="hybrid_alpha",
            label="Hybrid Search Balance (Alpha)",
            initial=app_state.settings["hybrid_alpha"],
            min=0.0, max=1.0, step=0.1,
            description="Weight for vector search (1.0 = pure vector, 0.0 = pure keyword)."
        ),
        cl.ChatSettingsSwitch(
            id="use_reranker",
            label="Neural Reranking",
            initial=app_state.settings["use_reranker"],
            description="Use AI to improve search result relevance (requires more computation)."
        ),
        cl.ChatSettingsSelect(
            id="performance_target",
            label="Embedding Performance Target",
            values=["low_latency", "balanced", "high_accuracy"],
            initial_value=app_state.settings["performance_target"],
            description="Optimize embedding model selection for speed, accuracy, or balance."
        )
    ]
    await cl.ChatSettings(elements=settings_elements).send()
    logger.info("Chat started and UI elements sent.")

@cl.on_settings_update
async def on_settings_update(settings: Dict[str, Any]):
    """Update app settings when user changes them."""
    logger.info(f"Settings update received: {settings}")
    needs_model_reload = False
    # Update app state with new settings
    for key, value in settings.items():
        if key in app_state.settings:
            if key == "performance_target" and app_state.settings[key] != value:
                needs_model_reload = True
            app_state.settings[key] = value
            logger.info(f"Updated setting '{key}' to '{value}'")
        else:
            logger.warning(f"Received unknown setting key: {key}")

    # Reload embedding model if performance target changed
    if needs_model_reload and app_state.initialization_complete:
        await cl.Message(content="🔄 Performance target changed. Reloading embedding model...", author="System").send()
        # Removed stray opening square bracket
        async with cl.Step(name="Reloading Embedding Model") as step:
            step.output = f"Loading model for '{app_state.settings['performance_target']}' target..."
            await step.update()
            app_state.embedding_model = await asyncio.to_thread(
                load_embedding_model,
                performance_target=app_state.settings['performance_target']
            )
            if app_state.embedding_model:
                step.output = f"✅ Embedding model reloaded successfully!"
                step.status = cl.StepStatus.COMPLETED
                await cl.Message(content="✅ Embedding model reloaded.", author="System").send()
            else:
                step.output = "❌ Failed to reload embedding model."
                step.status = cl.StepStatus.FAILED
                await cl.Message(content="❌ Failed to reload embedding model.", author="System").send()
            await step.update()

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
@cl.action_callback("general") # Role selection callbacks
@cl.action_callback("reset_db") # Reset DB callback
async def on_action(action: cl.Action):
    """Handle action button clicks for tourism expertise selection."""
    logger.info(f"Action callback triggered: {action.name} ({action.value})")
    if action.value in AGENT_ROLES:
        app_state.current_role = action.value
        
        # Send confirmation of new expertise
        await cl.Message(
            content=f"📚 Tourism expertise changed to: **{action.value}**\n\nYour questions will now be analyzed from this perspective.",
            author="Tourism Assistant"
        ).send()
        
        # Optionally update a sidebar element if you have one showing the role
        # Example: await cl.ChatSettings(elements=[cl.Text(name="current_role_display", content=f"Role: {action.value}")]).send()
        logger.info(f"Updated role to {action.value}")

    elif action.value == "reset":
        await cl.Message(content="🗑️ Resetting vector database and clearing processed files...", author="System").send()
        async with cl.Step(name="Resetting Database") as step:
            success, message = await asyncio.to_thread(reset_vector_db)
            if success:
                app_state.processed_files.clear()
                app_state.conversation_memory.clear()
                app_state.extracted_entities = {}
                app_state.tourism_insights = {}
                # Re-initialize collection (it might be None after reset)
                app_state.collection = await asyncio.to_thread(get_chroma_collection)
                step.output = "✅ Database reset successfully."
                step.status = cl.StepStatus.COMPLETED
                await cl.Message(content=f"✅ {message}", author="System").send()
            else:
                step.output = f"❌ {message}"
                step.status = cl.StepStatus.FAILED
                await cl.Message(content=f"❌ {message}", author="System").send()
            await step.update()

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
        await cl.Message( # Check if files have been processed
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
    logger.info(f"Received query: '{user_query[:100]}...'")
    if not user_query or not user_query.strip():
        await cl.Message(
            content="Please enter a valid question about your tourism documents.",
            author="Tourism Assistant"
        ).send()
        return
    
    # Check memory before processing
    memory_monitor.check()

    # Process user query with streaming response
    try:
        # Add loading message
        msg = cl.Message(author=app_state.current_role)
        await msg.send()
        
        # Get conversation memory
        conversation_memory = app_state.conversation_memory.get_formatted_history()
        
        # Start a background task to process the query
        async with cl.Step(name="Searching Tourism Knowledge Base", show_input=False) as search_step:
            step.input = user_query
            await search_step.update()
            
            results = await asyncio.to_thread(
                hybrid_retrieval, # Use hybrid retrieval from vector_store module
                query=user_query,
                embedding_model=app_state.embedding_model,
                collection=app_state.collection,
                top_n=app_state.settings["top_n"],
                alpha=app_state.settings["hybrid_alpha"],
                use_reranker=app_state.settings["use_reranker"]
            )
            logger.info(f"Hybrid retrieval returned {len(results)} results")
            
            # Create source documents for citation
            source_documents = []
            for i, result in enumerate(results):
                if 'metadata' in result and 'filename' in result['metadata']:
                    source_documents.append(
                        cl.SourceDocument(
                            page_content=result['text'],
                            metadata={
                                "source": result['metadata'].get('filename', f"Source {i+1}"),
                                "score": f"{result['score']:.2f}"
                            }
                        )
                    )
                else:
                    source_documents.append(
                        cl.SourceDocument(
                            page_content=result['text'],
                            metadata={"source": f"Source {i+1}", "score": f"{result['score']:.2f}"}
                        )
                    )
            
            if not results:
                search_step.output = "No relevant information found in the tourism documents."
                search_step.status = cl.StepStatus.WARNING
                await search_step.update()
                await msg.update(content="I couldn't find relevant information in your tourism documents for that query. Please try rephrasing or asking about topics covered in the uploaded files.")
                return
                
            # Log the search results
            search_step.output = f"Found {len(results)} relevant document segments."
            search_step.status = cl.StepStatus.COMPLETED
            await search_step.update()
        
        # Process query with LLM
        async with cl.Step(name="Generating Tourism Insights", show_input=False) as step:
            step.input = user_query
            await step.update()

            # Use TokenStream for LLM response
            stream = cl.Message(content="", author=app_state.current_role, parent_id=msg.parent_id).stream_token
            
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
                    hybrid_alpha=app_state.settings["hybrid_alpha"],
                    use_reranker=app_state.settings["use_reranker"]
                )
                logger.info(f"LLM query returned answer (length: {len(answer)})")
                
                # Stream tokens
                # Simple streaming by word - more sophisticated tokenization could be used
                for word in answer.split():
                    await stream(word + " ")
                    await asyncio.sleep(0.01) # Small delay for effect
                
                # Update memory
                app_state.conversation_memory.add("User", user_query)
                app_state.conversation_memory.add("Assistant", answer)
                
                # Update step
                step.output = "Generated tourism insights."
                step.status = cl.StepStatus.COMPLETED
                await step.update()
                
                return answer
            
            # Stream tokens and wait for completion
            final_answer = await stream_tokens()
            logger.info("Token streaming complete.")
            
            # End the token stream
            await stream.end()
            
            # Add source documents to the message
            await cl.Message(content="", author=app_state.current_role, source_documents=source_documents).send() # Send sources separately
    
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        logger.error(error_message, exc_info=True)
        await cl.Message(
            content=f"❌ An error occurred: {error_message}",
            author="System"
        ).send()
        return
    
    print("--- DEBUG: Exiting on_message ---")

# For most recent Chainlit versions:
@cl.on_chat_message_files
async def on_chat_message_files(files: List[cl.File]): # Use cl.File for type hint if available
    """Process uploaded tourism document files (PDFs)."""
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
        logger.warning("Missing components during file upload, attempting to reload...")
        app_state.embedding_model = await asyncio.to_thread(load_embedding_model)
        app_state.collection = await asyncio.to_thread(get_chroma_collection)

        if not app_state.embedding_model or app_state.collection is None: # Check again
            log_error("Failed to load required components for file processing")
            await cl.Message(
                content="⚠️ Tourism analysis system is not properly initialized. Please refresh and try again.",
                author="System"
            ).send()
            return

    for file in files:
        # Check file type using the file object's attributes
        if not file.mime or not file.mime == "application/pdf":
            await cl.Message(
                content=f"⚠️ Only PDF documents are supported for tourism analysis. Skipped: {file.name}",
                author="System"
            ).send()
            continue

        temp_file_path = None
        try:
            # Get file content and name from the cl.File object
            file_content = await file.get_bytes()
            file_name = file.name # Use file.name

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Create processing message
            process_msg = cl.Message(content=f"🔍 Analyzing tourism document: {file_name}...", author="System")
            await process_msg.send()

            # Check memory before processing
            memory_monitor.check()

            # Process the document with advanced analysis
            async with cl.Step(name=f"Processing {file_name}", show_input=False) as step:
                step.input = file_name
                await step.update()
                try:
                    # Convert file to proper format for processing
                    # The 'file' object from AskFileResponse already has 'name' and 'content'
                    # We need a 'read' method for process_uploaded_pdf
                    uploaded_file_obj = type('obj', (object,), {
                        'name': file.name,
                        'read': lambda: file_content # Use the fetched content
                    })

                    # Process with tourism-focused extraction
                    logger.info(f"Calling process_uploaded_pdf for {file_name}")
                    chunks = await asyncio.to_thread(
                        process_uploaded_pdf,
                        uploaded_file=uploaded_file_obj,
                        chunk_size=app_state.settings["chunk_size"],
                        overlap=app_state.settings["overlap"],
                        extract_images=True # Keep image extraction if needed
                    )
                    logger.info(f"process_uploaded_pdf returned {len(chunks)} chunks for {file_name}")

                    if not chunks:
                        step.output = f"No content could be extracted from {file.name}."
                        step.status = cl.StepStatus.FAILED
                        await step.update()
                        await process_msg.update(content=f"⚠️ No content could be extracted from {file.name}. Please check if the PDF contains text content.")
                        logger.warning(f"No content extracted from {file_name}")
                        continue # Process next file

                    # Process chunks and extract tourism entities/metrics
                    file_tourism_entities = {
                        "DESTINATION": set(), "ACCOMMODATION": set(), "TRANSPORTATION": set(),
                        "ACTIVITY": set(), "ATTRACTION": set()
                    }
                    file_tourism_insights = {"segments": {}, "payment_methods": {}} # Use insights naming

                    batch_size = 10 # Process in batches if needed
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i+batch_size]
                        step.output = f"Analyzing content batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}..."
                        await step.update()

                        for chunk in batch:
                            try:
                                # Extract entities
                                # Use streaming/optimized entity extraction if available
                                if "tourism_entities" not in chunk.get("metadata", {}): # Check if already processed
                                    # Assuming extract_tourism_entities_streaming exists and works on single chunks
                                    entities = extract_tourism_entities_streaming(chunk["text"]) # Adapt if needed
                                    chunk["metadata"]["tourism_entities"] = entities
                                else:
                                    entities = chunk["metadata"]["tourism_entities"] # Reuse if present

                                # Aggregate entities for this file
                                for entity_type, items in entities.items():
                                    if entity_type in file_tourism_entities:
                                        file_tourism_entities[entity_type].update(items)

                                # Process segment information
                                if "segment_matches" in chunk["metadata"]:
                                    for segment, has_match in chunk["metadata"]["segment_matches"].items():
                                        if has_match:
                                            file_tourism_insights["segments"][segment] = file_tourism_insights["segments"].get(segment, 0) + 1

                                # Process payment information
                                if chunk["metadata"].get("has_payment_info", False):
                                    payment_keywords = ["credit card", "debit card", "cash", "digital wallet",
                                                    "mobile payment", "cryptocurrency"]
                                    for payment in payment_keywords:
                                        if payment in chunk["text"].lower(): # Simple keyword check
                                            file_tourism_insights["payment_methods"][payment] = file_tourism_insights["payment_methods"].get(payment, 0) + 1

                            except Exception as chunk_err:
                                log_error(f"Error processing chunk from {file_name}: {str(chunk_err)}")
                                # Continue with next chunk

                    # Add chunks to collection (ensure embedding_model and collection are valid)
                    logger.info(f"Calling add_chunks_to_collection for {len(chunks)} chunks from {file.name}")
                    add_success = await asyncio.to_thread(
                        add_chunks_to_collection,
                        chunks=[c["text"] for c in chunks],
                        metadatas=[c.get("metadata", {}) for c in chunks],
                        embedding_model=app_state.embedding_model,
                        collection=app_state.collection
                    )
                    logger.info(f"add_chunks_to_collection returned: {add_success} for {file_name}")

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
                        if file_tourism_insights["segments"]:
                            app_state.tourism_insights.setdefault("segments", {})
                            for segment, count in file_tourism_insights["segments"].items():
                                app_state.tourism_insights["segments"][segment] = app_state.tourism_insights["segments"].get(segment, 0) + count

                        if file_tourism_insights["payment_methods"]:
                            app_state.tourism_insights.setdefault("payment_methods", {})
                            for payment, count in file_tourism_insights["payment_methods"].items():
                                app_state.tourism_insights["payment_methods"][payment] = app_state.tourism_insights["payment_methods"].get(payment, 0) + count
                        
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
                        logger.error(f"Failed to add {file_name} to collection")

                except Exception as e:
                    logger.error(f"Error processing tourism document {file_name}: {str(e)}", exc_info=True)
                    step.output = f"Error processing document: {str(e)}"
                    step.status = cl.StepStatus.FAILED
                    await step.update()

                    await process_msg.update(content=f"❌ Error processing tourism document {file_name}: {str(e)}")
                    logger.error(f"Exception during PDF processing for {file_name}: {e}")

                await step.update() # Final update for the step

            # Display updated global tourism insights if available
            if app_state.extracted_entities or app_state.tourism_insights:
                await display_global_insights()

        except Exception as e:
            logger.error(f"Error handling tourism file upload for {file.name}: {str(e)}", exc_info=True)
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
    logger.info("Updating global insights display")
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
    if "segments" in app_state.tourism_insights and app_state.tourism_insights["segments"]:
        total_segments = sum(app_state.tourism_insights["segments"].values())
        if total_segments > 0:
            segments_data = {
                segment: f"{round((count / total_segments) * 100, 1)}%"
                for segment, count in sorted(app_state.tourism_insights["segments"].items(), key=lambda item: item[1], reverse=True)
            } # Sort by count desc
            segments_element = cl.Dataframe(
                data=[[segment, percentage] for segment, percentage in segments_data.items()],
                columns=["Market Segment", "Mention Frequency"],
                title="Aggregated Market Segment Mentions"
            )
            elements.append(segments_element)

    # Create payment methods element if available (using global state)
    if "payment_methods" in app_state.tourism_insights and app_state.tourism_insights["payment_methods"]:
        total_payments = sum(app_state.tourism_insights["payment_methods"].values())
        if total_payments > 0:
            payment_data = {
                payment: f"{round((count / total_payments) * 100, 1)}%"
                for payment, count in sorted(app_state.tourism_insights["payment_methods"].items(), key=lambda item: item[1], reverse=True)
            } # Sort by count desc
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
        logger.info("Sent updated global tourism insights message")
    else:
        logger.info("No global insights to display yet")