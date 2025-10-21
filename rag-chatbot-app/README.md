# 🤖 RAG Chatbot - Document Q&A Application

A powerful Retrieval-Augmented Generation (RAG) chatbot application that allows you to upload documents and ask questions about their content using local LLM models via Ollama.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🎯 Overview

The RAG Chatbot is a sophisticated document analysis application that combines:

- **Document Processing**: Supports PDF, TXT, DOCX, and DOC files
- **Vector Search**: Uses ChromaDB for efficient document retrieval
- **Local LLM Integration**: Works with Ollama models (llama3.2, gemma2, qwen2.5, etc.)
- **Multi-Document Analysis**: Can analyze multiple documents simultaneously
- **Performance Monitoring**: Real-time system resource monitoring
- **Intelligent Model Selection**: Automatically recommends optimal models based on system performance

## ✨ Features

### 🔍 Core Features
- **Multi-Format Support**: PDF, TXT, DOCX, DOC files
- **Intelligent Chunking**: Configurable text splitting with overlap
- **Vector Embeddings**: Uses HuggingFace sentence transformers
- **Local LLM Integration**: Works with Ollama models
- **Real-time Performance Monitoring**: Memory and CPU load tracking
- **Smart Model Selection**: Automatic model recommendations

### 📊 Analysis Modes
- **Standard RAG**: Single query analysis
- **Cross-Document Analysis**: Multi-document synthesis
- **Individual Document Focus**: Document-specific analysis

### 🛠️ Management Features
- **Document Statistics**: File counts, chunk analysis, type detection
- **Database Inspection**: ChromaDB status and size monitoring
- **Document Verification**: Content validation and type detection
- **Clear & Reset**: Complete database and session cleanup

### 🎨 User Interface
- **Streamlit Web Interface**: Modern, responsive UI
- **Real-time Chat**: Interactive Q&A interface
- **Debug Mode**: Detailed logging and context information
- **Performance Indicators**: System resource monitoring
- **Document Summaries**: Auto-generated document overviews

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Chatbot Application                  │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)                                           │
│  ├── Document Upload Interface                                  │
│  ├── Chat Interface                                             │
│  ├── Configuration Panel                                        │
│  └── Performance Monitor                                        │
├─────────────────────────────────────────────────────────────────┤
│  Core RAG Engine                                                │
│  ├── Document Loader (LangChain)                               │
│  ├── Text Splitter (RecursiveCharacterTextSplitter)            │
│  ├── Embeddings (HuggingFace)                                  │
│  ├── Vector Store (ChromaDB)                                   │
│  └── QA Chain (RetrievalQA)                                    │
├─────────────────────────────────────────────────────────────────┤
│  LLM Integration (Ollama)                                       │
│  ├── Model Management                                           │
│  ├── Performance Monitoring                                     │
│  └── Response Generation                                        │
├─────────────────────────────────────────────────────────────────┤
│  Data Storage                                                   │
│  ├── ChromaDB (Vector Database)                                │
│  ├── Session State (Streamlit)                                 │
│  └── Temporary Files                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Document Upload → Text Extraction → Chunking → Embeddings → Vector Store
                                                                    ↓
User Query → Vector Search → Context Retrieval → LLM Processing → Response
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Ollama installed and running
- At least 4GB RAM (8GB+ recommended)
- 2GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/vishnu-krishnan/ai-powered-apps.git
cd ai-powered-apps/rag-chatbot-app
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama Models

```bash
# Install recommended models
ollama pull llama3.2:1b    # Fast, efficient (1.3GB)
ollama pull gemma2:2b      # Balanced (1.6GB)
ollama pull mistral:7b     # High quality (4.4GB)
ollama pull qwen2.5:7b     # High quality (4.7GB)
```

### Step 5: Start Ollama Server

```bash
ollama serve
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

## 📖 Usage

### 1. Initial Setup

1. **Start Ollama**: Ensure Ollama server is running
2. **Launch App**: Run `streamlit run app.py`
3. **Check Connection**: Use "Check Ollama Connection" button
4. **Select Model**: Choose appropriate model based on system performance

### 2. Document Upload

1. **Upload Files**: Use the file uploader in the sidebar
2. **Configure Processing**: Adjust chunk size and overlap
3. **Process Documents**: Click "Process Documents" button
4. **Verify Upload**: Check document statistics

### 3. Ask Questions

1. **Type Question**: Use the chat input at the bottom
2. **Select Analysis Mode**: Choose analysis type
3. **Get Response**: View AI-generated answers
4. **Debug Mode**: Enable for detailed information

### 4. Document Management

- **View Statistics**: Check file counts and processing status
- **Generate Summaries**: Create document overviews
- **Verify Content**: Validate document processing
- **Clear Documents**: Reset the system

## ⚙️ Configuration

### Model Selection

The application automatically recommends models based on system performance:

| System Status | Recommended Model | Reason |
|---------------|-------------------|---------|
| High CPU Load (>4.0) | llama3.2:1b | Fastest, most efficient |
| Moderate Load (2.0-4.0) | gemma2:2b | Balanced performance |
| Good Performance (<2.0) | mistral:7b, qwen2.5:7b | High quality responses |

### Text Processing Settings

- **Chunk Size**: 1000-3000 characters (default: 2000)
- **Chunk Overlap**: 100-600 characters (default: 400)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2

### Performance Monitoring

The application monitors:
- **Memory Usage**: Available RAM in GB
- **CPU Load**: 1-minute load average
- **Model Performance**: Response times and success rates

## 🔧 Troubleshooting

### Common Issues

#### 1. Ollama Connection Errors
```
Error: Cannot connect to Ollama server
```
**Solution**: 
- Ensure Ollama is running: `ollama serve`
- Check if port 11434 is available
- Verify Ollama installation

#### 2. Model Timeout Errors
```
Error: Model timeout: Read timed out
```
**Solution**:
- Use smaller models (llama3.2:1b, gemma2:2b)
- Increase system memory
- Close other applications

#### 3. Memory Errors
```
Error: Memory error: Out of memory
```
**Solution**:
- Use smaller models
- Reduce chunk size
- Free up system memory

#### 4. ChromaDB Errors
```
Error: Database error: readonly database
```
**Solution**:
- Click "Clear All Documents"
- Check file permissions
- Restart the application

### Performance Optimization

#### For Low-Memory Systems (<4GB)
- Use `llama3.2:1b` model
- Set chunk size to 1000
- Close other applications
- Enable swap space

#### For High-Performance Systems (>8GB)
- Use `mistral:7b` or `qwen2.5:7b`
- Increase chunk size to 3000
- Enable debug mode for detailed logging

## 📚 API Reference

### RAGChatbot Class

#### Methods

##### `initialize_embeddings()`
Initialize the HuggingFace embedding model.

##### `load_documents(file_paths, original_names=None)`
Load documents from various file types with enhanced metadata.

**Parameters**:
- `file_paths`: List of file paths to load
- `original_names`: Dictionary mapping temp paths to original names

**Returns**: List of Document objects

##### `split_documents(documents, chunk_size=2000, chunk_overlap=400)`
Split documents into chunks for processing.

**Parameters**:
- `documents`: List of Document objects
- `chunk_size`: Size of each chunk in characters
- `chunk_overlap`: Overlap between chunks

**Returns**: List of Document chunks

##### `create_vectorstore(documents, collection_name="rag_documents")`
Create and populate the ChromaDB vector store.

**Parameters**:
- `documents`: List of Document chunks
- `collection_name`: Name of the ChromaDB collection

**Returns**: Boolean success status

##### `setup_qa_chain(model_name, ollama_base_url)`
Setup the Q&A chain with specified model.

**Parameters**:
- `model_name`: Name of the Ollama model
- `ollama_base_url`: URL of the Ollama server

**Returns**: Boolean success status

##### `query(question)`
Query the RAG system with a question.

**Parameters**:
- `question`: User question string

**Returns**: Dictionary with result and source documents

##### `get_cross_document_analysis(question)`
Perform cross-document analysis.

**Parameters**:
- `question`: User question string

**Returns**: Dictionary with comprehensive analysis

##### `get_document_statistics()`
Get statistics about processed documents.

**Returns**: Dictionary with file counts and processing status

##### `clear_documents()`
Clear all processed documents and reset the system.

**Returns**: Boolean success status

### Configuration Options

#### Environment Variables
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `CHUNK_SIZE`: Default chunk size (default: 2000)
- `CHUNK_OVERLAP`: Default chunk overlap (default: 400)

#### Model Configuration
- `temperature`: LLM temperature (default: 0.3)
- `timeout`: Request timeout in seconds (default: 120)
- `num_predict`: Maximum tokens to generate (default: 1024)

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make your changes
5. Add tests if applicable
6. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions
- Include error handling

### Testing

```bash
# Run basic tests
python -m pytest tests/

# Test with different models
python test_models.py

# Performance testing
python performance_test.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Ollama](https://ollama.ai/) for local LLM integration
- [Streamlit](https://streamlit.io/) for the web interface
- [HuggingFace](https://huggingface.co/) for embeddings

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting guide
- Review the documentation

---

**Made with ❤️ for document analysis and AI-powered Q&A**