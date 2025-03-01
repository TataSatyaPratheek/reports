# PDF Analyzer: Converse with Your Documents using Local LLMs

A powerful, modular application that allows you to chat with PDF documents using local Large Language Models (LLMs) through Ollama. This application leverages vector embeddings, semantic search, and local language models to provide a secure, private way to analyze and extract information from PDF documents without sending your data to external APIs.

## 🌟 Key Features

- **Fully Local Processing**: All data stays on your machine - no external API calls
- **Modular Architecture**: Well-organized codebase for easy maintenance and extension
- **Vector Search**: Efficient retrieval of relevant document sections
- **Smart Chunking**: Intelligent document splitting for better context preservation
- **Customizable LLM Roles**: Configure the assistant's expertise based on document type
- **Resource Monitoring**: Adaptive performance optimization based on system resources
- **User-Friendly Interface**: Clean, intuitive UI with three-panel layout
- **Permission System**: Transparent initialization with user-approved actions

## 📋 Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) for running local language models
- Required Python packages (installed automatically):
  - streamlit
  - openparse
  - sentence-transformers
  - chromadb
  - nltk
  - spacy
  - ollama (Python client)
  - psutil

## 🚀 Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pdf-analyzer.git
   cd pdf-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

5. Follow the initialization steps in the interface:
   - Approve required permissions
   - Click "Initialize System"
   - Upload your PDF documents
   - Start chatting with your documents!

## 🧰 System Architecture

The application is built with a modular design for maintainability and extensibility:

```
pdf-analyzer/
├── app.py                  # Main application entry point
├── modules/                # Modular components
│   ├── __init__.py         # Package initialization
│   ├── system_setup.py     # Dependency and Ollama management
│   ├── nlp_models.py       # NLP model loading and management
│   ├── vector_store.py     # ChromaDB integration
│   ├── pdf_processor.py    # PDF parsing and chunking
│   ├── llm_interface.py    # Ollama integration
│   ├── ui_components.py    # Reusable UI elements
│   └── utils.py            # Utility functions
├── chroma_vector_db/       # Vector database storage (created at runtime)
├── requirements.txt        # Dependencies list
└── README.md               # Project documentation
```

### Processing Pipeline

1. **Document Upload**: PDFs are uploaded through the Streamlit interface
2. **Text Extraction**: [OpenParse](https://github.com/hyperonym/openparse) extracts text while preserving structure
3. **Smart Chunking**: Documents are split into semantic chunks with overlap for context preservation
4. **Vector Embedding**: [Sentence-Transformers](https://www.sbert.net/) convert chunks into vector embeddings
5. **Database Storage**: [ChromaDB](https://docs.trychroma.com/) stores embeddings for efficient retrieval
6. **Query Processing**: User questions are embedded and matched against document chunks
7. **Context Compilation**: Top matching chunks are combined to create context
8. **LLM Response**: [Ollama](https://ollama.ai/) generates responses based on the context and query

## 📊 Features in Detail

### Document Processing

The application uses a sophisticated document processing pipeline:

- **PDF Parsing**: Extracts text with structural awareness
- **Smart Chunking Algorithm**: Creates semantic chunks that preserve context
- **Overlap Strategy**: Ensures context flows between chunks
- **Batch Processing**: Handles large documents efficiently
- **Progress Tracking**: Real-time feedback on processing status

### Vector Search

The vector database enables semantic search capabilities:

- **Semantic Understanding**: Matches concepts, not just keywords
- **Relevance Ranking**: Retrieves the most relevant document sections
- **Efficient Storage**: Optimized for quick retrieval and minimum disk usage
- **Persistence**: Retains document knowledge between sessions

### Local LLM Integration

Integration with Ollama provides powerful local language model capabilities:

- **Model Selection**: Choose from available local models
- **Role Customization**: Configure the assistant's expertise and behavior
- **Conversation Memory**: Maintains context across multiple questions
- **System Prompts**: Custom instructions for specialized knowledge domains

### User Interface

The three-panel interface provides an intuitive user experience:

- **Left Sidebar**: Document management and uploads
- **Main Panel**: Chat interface with conversation history
- **Right Sidebar**: Configuration options and settings
- **Top Status Bar**: System resource monitoring

## ⚙️ Configuration Options

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

### Vector Embeddings

The application uses the Sentence-Transformers library with the "all-MiniLM-L6-v2" model to convert text chunks into vector embeddings. These vectors represent the semantic meaning of text in a high-dimensional space, allowing for similarity matching based on meaning rather than exact keyword matches.

### Retrieval-Augmented Generation (RAG)

The application implements the RAG pattern:

1. **Retrieval**: Finding the most relevant chunks from documents
2. **Augmentation**: Using these chunks to provide context to the LLM
3. **Generation**: Generating a response based on the query and context

This approach combines the knowledge from your documents with the language model's capabilities, resulting in responses that are both relevant and coherent.

### Adaptive Processing

The application monitors system resources and adjusts its behavior:

- **Batch Size Adaptation**: Adjusts processing batch sizes based on available memory
- **Worker Thread Optimization**: Scales thread usage based on CPU load
- **Memory Management**: Implements efficient cleanup to prevent memory leaks

## 📚 Credits and References

This project builds upon several open-source libraries and tools:

- [Streamlit](https://streamlit.io/) - The web application framework
- [OpenParse](https://github.com/hyperonym/openparse) - PDF parsing library
- [Sentence-Transformers](https://www.sbert.net/) - Vector embedding models
- [ChromaDB](https://docs.trychroma.com/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [NLTK](https://www.nltk.org/) - Natural Language Toolkit for text processing
- [spaCy](https://spacy.io/) - NLP framework
- [psutil](https://github.com/giampaolo/psutil) - System monitoring

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

Your Name - Satya Pratheek Tata

Email - satyapratheek.tata@edhec.com

Project Link: [https://github.com/TataSatyaPratheek/pdf-analyzer](https://github.com/TataSatyaPratheek/pdf-analyzer)