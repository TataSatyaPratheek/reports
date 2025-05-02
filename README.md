# Enhanced PDF Analyzer: Converse with Your Documents using Local LLMs

A powerful, modular application that allows you to chat with PDF documents using local Large Language Models (LLMs) through Ollama. This application leverages advanced Retrieval-Augmented Generation (RAG) techniques including hybrid vector and keyword search, query reformulation, and neural reranking to provide a secure, private way to analyze and extract information from PDF documents without sending your data to external APIs.

## 🌟 Key Features

- **Fully Local Processing**: All data stays on your machine - no external API calls
- **Modular Architecture**: Well-organized codebase for easy maintenance and extension
- **Hybrid Retrieval**: Combines vector similarity and BM25 keyword search for optimal results
- **Neural Reranking**: Uses Jina Reranker to improve search result relevance
- **Smart Chunking**: Intelligent document splitting for better context preservation
- **Dynamic Query Reformulation**: Automatically improves queries based on conversation context
- **Context-Aware Memory**: Sliding window approach for maintaining conversation context
- **Customizable LLM Roles**: Configure the assistant's expertise based on document type
- **Resource Monitoring**: Adaptive performance optimization based on system resources
- **User-Friendly Interface**: Clean, intuitive UI with configurable advanced features
- **Permission System**: Transparent initialization with user-approved actions

## 📋 Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) for running local language models
- Required Python packages (installed automatically via PDM):
  - streamlit
  - openparse
  - sentence-transformers
  - chromadb
  - nltk
  - spacy
  - ollama (Python client)
  - psutil
  - jina-reranker
  - rank-bm25
  - langchain-core
  - cachetools

## 🚀 Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pdf-analyzer.git
   cd pdf-analyzer
   ```

2. Install PDM (if not already installed):
   ```bash
   pip install pdm
   ```

3. Install dependencies:
   ```bash
   pdm install
   ```

4. Run the application:
   ```bash
   pdm run start
   ```

5. Open your browser at `http://localhost:8501`

6. Follow the initialization steps in the interface:
   - Approve required permissions
   - Click "Initialize System"
   - Upload your PDF documents
   - Start chatting with your documents!

## 🧰 System Architecture

The application is built with a modular design for maintainability and extensibility:

```
pdf-analyzer/
├── app.py (main.py)      # Main application entry point
├── modules/              # Modular components
│   ├── __init__.py       # Package initialization
│   ├── system_setup.py   # Dependency and Ollama management
│   ├── nlp_models.py     # NLP model loading and management
│   ├── vector_store.py   # ChromaDB integration with hybrid search
│   ├── pdf_processor.py  # PDF parsing and chunking
│   ├── llm_interface.py  # Enhanced Ollama integration
│   ├── ui_components.py  # Reusable UI elements
│   └── utils.py          # Utility functions
├── chroma_vector_db/     # Vector database storage (created at runtime)
├── pyproject.toml        # Project configuration and dependencies
└── README.md             # Project documentation
```

### Enhanced Processing Pipeline

1. **Document Upload**: PDFs are uploaded through the Streamlit interface
2. **Text Extraction**: [OpenParse](https://github.com/hyperonym/openparse) extracts text while preserving structure
3. **Smart Chunking**: Documents are split into semantic chunks with overlap for context preservation
4. **Vector Embedding**: [Sentence-Transformers](https://www.sbert.net/) convert chunks into vector embeddings
5. **Database Storage**: [ChromaDB](https://docs.trychroma.com/) stores embeddings for efficient retrieval
6. **Neural Caching**: Caches embeddings to avoid redundant processing
7. **Query Reformulation**: Dynamically improves queries based on conversation context
8. **Hybrid Retrieval**: Combines vector similarity and BM25 keyword search
9. **Reranking**: [Jina Reranker](https://jina.ai/) improves search result relevance
10. **Context Compilation**: Top matching chunks are combined to create context
11. **LLM Response**: [Ollama](https://ollama.ai/) generates responses based on the context and query

## 📊 Features in Detail

### Advanced Retrieval Pipeline

The application uses state-of-the-art retrieval techniques:

- **Hybrid Search**: Combines the strengths of dense vector retrieval (semantic understanding) and sparse retrieval (keyword matching)
- **Configurable Balance**: Adjust the weight between vector similarity and BM25 keyword search
- **Neural Reranking**: Reorders search results to prioritize most relevant content
- **Dynamic Query Reformulation**: Automatically enhances queries based on conversation context

### Context-Aware Memory

The memory system maintains conversation context while managing token usage:

- **Sliding Window Memory**: Maintains recent conversation turns within token limits
- **Dynamic Sizing**: Automatically adjusts memory based on conversation complexity
- **Conversation Persistence**: Maintains context across multiple questions

### Neural Caching

The caching system improves performance for repeated operations:

- **Embedding Cache**: Avoids redundant text encoding operations
- **Reranker Cache**: Stores reranking results for similar queries
- **TTL-Based Expiry**: Automatically manages cache freshness and size

### User Interface Enhancements

The interface provides intuitive controls for advanced features:

- **Feature Toggles**: Enable/disable hybrid retrieval, reranking, and query reformulation
- **Parameter Controls**: Fine-tune retrieval balance and other parameters
- **Resource Monitoring**: View system resource usage in real-time
- **Detailed Status Updates**: Clear feedback during processing operations

## ⚙️ Configuration Options

### Advanced Retrieval Settings

Fine-tune the document retrieval behavior:

- **Hybrid Retrieval**: Toggle between standard vector search and hybrid vector+BM25
- **Vector/Keyword Balance**: Adjust the balance between semantic and keyword search
- **Reranker**: Enable/disable neural reranking of search results
- **Query Reformulation**: Enable/disable automatic query improvement

### Assistant Roles

Choose from predefined roles or create custom behavior:

- **Financial Analyst**: Specialized in financial reports and data
- **Academic Research Assistant**: Helps with scholarly content
- **Technical Documentation Expert**: Explains technical documents
- **Legal Document Analyzer**: Interprets legal documents and terminology
- **Medical Literature Assistant**: Assists with medical publications
- **General Assistant**: All-purpose document assistant
- **Custom**: Define your own system prompt for specialized tasks

### Processing Settings

Fine-tune the document processing behavior:

- **Chunk Size**: Adjust the size of document segments (in words)
- **Overlap**: Control context preservation between chunks
- **Top Results**: Number of chunks to retrieve for each query
- **Conversation Memory**: Number of previous Q&A pairs to include as context

### Model Settings

Configure the underlying language model:

- **Model Selection**: Choose from locally available Ollama models
- **Model Download**: Add new models to your local collection

## 🔧 Advanced Usage

### Custom System Prompts

Create specialized assistants by defining custom system prompts. For example:

```
You are an expert patent attorney specializing in technology patents. 
Help analyze patent documents by identifying claims, potential prior art issues, 
and explanations of technical concepts in accessible language.
```

### Vector Database Management

The application provides tools to manage the vector database:

- **Reset Database**: Clear all document embeddings to start fresh
- **View Processed Files**: See which documents have been processed and stored

### Error Handling and Logging

The application includes comprehensive error handling:

- **Error Logging**: Detailed logs for troubleshooting
- **Graceful Degradation**: Continues functioning even when components fail
- **User Feedback**: Clear error messages explain issues to users

## 🛡️ Privacy and Security

This application is designed with privacy in mind:

- **All Local Processing**: No data leaves your machine
- **No External APIs**: All computation happens locally
- **No Data Collection**: Your documents remain private
- **Permission System**: The application asks for explicit permission before installing components

## 🔍 How It Works: Technical Details

### Hierarchical NSW Indexing

The application uses Hierarchical Navigable Small World (HNSW) graph indexing in ChromaDB for extremely fast approximated nearest neighbor search. This approach significantly improves search performance for large document collections.

### Neural Reranking

After initial retrieval, the Jina Reranker applies a neural model to reorder results based on a more sophisticated relevance assessment, dramatically improving the quality of context provided to the LLM.

### Hybrid BM25 + Vector Search

The application combines two complementary search methods:

1. **Vector Search**: Captures semantic meaning and conceptual relationships
2. **BM25 Okapi**: Captures keyword matches and rare/unique terms
3. **Weighted Combination**: Blends results for optimal retrieval

### Query Reformulation

The application can intelligently reformulate user queries to incorporate conversation context, resolve references, and make implicit questions explicit, resulting in better document retrieval.

## 📚 Credits and References

This project builds upon several open-source libraries and tools:

- [Streamlit](https://streamlit.io/) - The web application framework
- [OpenParse](https://github.com/hyperonym/openparse) - PDF parsing library
- [Sentence-Transformers](https://www.sbert.net/) - Vector embedding models
- [ChromaDB](https://docs.trychroma.com/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [NLTK](https://www.nltk.org/) - Natural Language Toolkit for text processing
- [spaCy](https://spacy.io/) - NLP framework
- [Jina Reranker](https://jina.ai/) - Neural reranking library
- [Rank-BM25](https://github.com/dorianbrown/rank_bm25) - BM25 keyword search implementation
- [psutil](https://github.com/giampaolo/psutil) - System monitoring
- [PDM](https://pdm.fming.dev/) - Python dependency manager

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

Name - Satya Pratheek Tata

Email - satyapratheek.tata@edhec.com

Project Link: [https://github.com/TataSatyaPratheek/pdf-analyzer](https://github.com/TataSatyaPratheek/reports)