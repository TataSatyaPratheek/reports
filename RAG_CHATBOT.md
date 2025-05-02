# Tourism Insights RAG Chatbot

A powerful, tourism-focused RAG (Retrieval Augmented Generation) chatbot with dual interface options, designed for tourism industry professionals to analyze travel trends, payment methods, market segments, and sustainability initiatives.

## 🏝️ Features

- **Tourism-Focused Analysis**: Specialized extraction of tourism entities, segments, and metrics
- **Hybrid Retrieval**: Combines vector similarity and BM25 keyword search for optimal results
- **Neural Reranking**: Uses Jina Reranker Client to improve search result relevance
- **Flash Attention Integration**: Optimized performance for embeddings and attention mechanisms
- **Market Segmentation Analysis**: Automatically identifies and analyzes tourism market segments
- **Payment Method Analysis**: Detects and compares payment preferences across segments
- **Travel Trend Detection**: Identifies macro trends in the tourism industry
- **Sustainability Analysis**: Focuses on ecological and social sustainability initiatives
- **Dual Interface Options**: Choose between Streamlit or Chainlit based on your needs
- **Private & Secure**: All processing happens locally - your data never leaves your machine

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [PDM](https://pdm.fming.dev/) package manager (recommended)
- [Ollama](https://ollama.ai/) for running local language models

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/tourism-rag-chatbot.git
   cd tourism-rag-chatbot
   ```

2. Run the setup script to create directories and check dependencies:
   ```bash
   python setup.py --all
   ```

3. Download a recommended model for tourism analysis:
   ```bash
   ollama pull llama3.2:latest
   ```

## 💻 Running the Application

### Option 1: Streamlit Interface

The Streamlit interface provides a rich, visual experience with dashboard views and comprehensive analytics:

```bash
# Start the Streamlit app
pdm run start
```

Then open your browser at `http://localhost:8501`

### Option 2: Chainlit Interface (Recommended for RAG)

The Chainlit interface is optimized for conversational RAG with better performance and intuitive citation:

```bash
# Start the Chainlit app
pdm run start_chainlit
```

Then open your browser at `http://localhost:8000`

## 🧩 Interface Comparison

| Feature | Streamlit | Chainlit |
|---------|-----------|----------|
| Visual Dashboards | ✅ Rich | ✅ Basic |
| Source Citations | ❌ No | ✅ Yes |
| Response Speed | ⚡ Good | ⚡⚡⚡ Excellent |
| UI Responsiveness | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memory Usage | 🧠🧠🧠 | 🧠🧠 |
| Mobile Friendly | ✅ Good | ✅ Excellent |
| Tourism Analysis | ✅ Complete | ✅ Complete |
| Document Management | ✅ Basic | ✅ Advanced |

## 🗂️ Project Structure

```
tourism-rag-chatbot/
├── app.py                  # Streamlit interface
├── chainlit_app.py         # Chainlit interface
├── modules/                # Core functionality
│   ├── __init__.py
│   ├── llm_interface.py    # Enhanced LLM integration with Flash Attention
│   ├── nlp_models.py       # Tourism-optimized NLP models
│   ├── pdf_processor.py    # Tourism document processing
│   ├── system_setup.py     # System initialization
│   ├── ui_components.py    # UI elements
│   ├── utils.py            # Utility functions
│   └── vector_store.py     # Vector database with hybrid search and Jina reranking
├── assets/                 # UI assets
│   └── roles/              # Expert role images
├── tests/                  # Test suites
│   ├── stress/             # Load testing with Locust
│   └── e2e/                # E2E testing with Playwright
├── pyproject.toml          # PDM project configuration
├── setup.py                # Setup script
└── README.md               # Documentation
```

## 🧠 Tourism Expert Modes

The chatbot offers specialized expertise modes for different tourism analysis needs:

1. **Travel Trends Analyst**: Focus on macro industry trends and forecasts
2. **Payment Specialist**: Analyze payment methods across travel segments
3. **Market Segmentation Expert**: Identify and analyze distinct customer segments
4. **Sustainability Tourism Advisor**: Focus on ecological and social sustainability
5. **Gen Z Travel Specialist**: Understand unique preferences of younger travelers
6. **Luxury Tourism Consultant**: Analyze high-end travel market trends
7. **Tourism Analytics Expert**: Data-driven insights and metrics
8. **General Tourism Assistant**: Broad tourism information assistance

## 📊 Analytics Capabilities

The system automatically extracts and analyzes:

- **Tourism Entities**: Destinations, accommodations, transportation, activities, attractions
- **Market Segments**: Luxury, budget, family, solo, adventure, cultural
- **Payment Methods**: Credit card, cash, digital wallets, cryptocurrencies
- **Sustainability Initiatives**: Eco-friendly practices, certifications, green initiatives
- **Demographics**: Generation-specific trends and preferences
- **Tourism Metrics**: Visitor numbers, spending patterns, growth rates

## 🔍 2025 Enhanced Features

The system has been upgraded with several cutting-edge features for 2025:

### Flash Attention Integration

Uses the Flash Attention library to dramatically speed up transformer attention operations, resulting in faster embedding generation and LLM inference.

### Jina Reranker Client

Implements the updated `jina-reranker-client` package for enhanced result reranking, replacing the older Jina Reranker implementation.

### Async Processing

Fully asynchronous processing pathways for both embedding generation and LLM queries, providing better responsiveness especially in the Chainlit interface.

### PDM Build System

Uses the modern PDM package management system for more reliable dependency management and development workflow.

### Performance Benchmarking

Includes Locust-based load testing for measuring performance under various usage patterns and user loads.

## 💎 Tourism Industry Use Cases

- **Destination Management Organizations**: Analyze visitor trends and preferences
- **Hotel Chains**: Understand payment methods and booking patterns
- **Tour Operators**: Identify emerging market segments and preferences
- **Tourism Boards**: Track sustainability initiatives and their impact
- **Travel Technology Companies**: Analyze digital payment adoption
- **Luxury Travel Providers**: Understand high-end traveler expectations
- **Tourism Consultants**: Extract actionable insights from industry reports

## 🛠️ Configuration

Both interfaces allow configuring:

- **Chunk Size**: Adjust the granularity of document segments
- **Search Results**: Control how many document sections are retrieved
- **Hybrid Retrieval Balance**: Fine-tune the mix of vector and keyword search
- **Neural Reranking**: Enable/disable sophisticated result re-ordering
- **Query Enhancement**: Turn on/off automatic query improvement
- **LLM Model**: Select different Ollama models optimized for tourism

## 🧪 Testing

The system includes comprehensive testing capabilities:

```bash
# Run unit tests with parallelization
pdm run test

# Run performance benchmarking
pdm run benchmark

# Type checking
pdm run types

# Style and linting
pdm run lint
pdm run format
```

## 🌐 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install dev dependencies (`pdm install -G dev`)
4. Make your changes
5. Run tests (`pdm test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.