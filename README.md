# 🚀 LLM Applications Suite

A comprehensive collection of AI-powered applications built with Large Language Models (LLMs) and modern web technologies.

## 📋 Applications Overview

This repository contains five distinct AI applications, each designed to solve specific real-world problems:

### 1. 📄 Document Summary App
**Location**: `doc-summary-app/`
- **Purpose**: Extract and summarize content from PDF documents
- **Technology**: Streamlit + Transformers (Pegasus model)
- **Features**: PDF text extraction, intelligent chunking, AI-powered summarization

### 2. 💬 Local LLM Chat App
**Location**: `local-llm-chat-app/`
- **Purpose**: Interactive chat interface with local LLM models
- **Technology**: FastAPI + Streamlit + Ollama
- **Features**: Real-time chat, session management, multiple model support

### 3. 🌍 Global News Topic Tracker
**Location**: `global-news-tracker/`
- **Purpose**: Scrape and summarize trending news topics
- **Technology**: Streamlit + BeautifulSoup + Transformers (BART)
- **Features**: Google News scraping, trending topic detection, AI summarization

### 4. 🤖 Multi-Modal Assistant
**Location**: `multimodal-assistant/`
- **Purpose**: AI assistant for both text and image queries
- **Technology**: Streamlit + Ollama + Vision-Language Models
- **Features**: Image analysis, text processing, multi-modal interactions

### 5. 🎙️ Meeting Notes & Action Item Extractor
**Location**: `meeting-notes-extractor/`
- **Purpose**: Convert meeting audio into structured notes and task lists
- **Technology**: Streamlit + OpenAI Whisper + GPT-3.5-turbo
- **Features**: Audio transcription, note structuring, action item extraction

### 6. 🤖 RAG Chatbot - Document Q&A
**Location**: `rag-chatbot-app/`
- **Purpose**: AI-powered document Q&A using Retrieval-Augmented Generation
- **Technology**: Streamlit + LangChain + ChromaDB + Ollama (Local LLMs)
- **Features**: Multi-format document upload, vector search, intelligent Q&A, source attribution

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Applications Suite                   │
├─────────────────────────────────────────────────────────────┤
│  📄 Document Summary    │  💬 Local LLM Chat               │
│  • PDF Processing       │  • Real-time Chat                │
│  • Pegasus Summarizer   │  • Ollama Integration            │
│  • Streamlit UI         │  • FastAPI Backend               │
├─────────────────────────────────────────────────────────────┤
│  🌍 News Tracker        │  🤖 Multi-Modal Assistant        │
│  • Google News Scraping │  • Text + Image Processing       │
│  • BART Summarizer      │  • LLaVA Vision Model            │
│  • Trending Topics      │  • Interactive Chat Interface    │
├─────────────────────────────────────────────────────────────┤
│  🎙️ Meeting Extractor  │  🤖 RAG Chatbot                  │
│  • Audio Transcription  │  • Document Q&A                  │
│  • Note Structuring     │  • Vector Search                 │
│  • Action Item Tracking │  • Source Attribution            │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishnu-krishnan/ai-powered-apps.git
   cd ai-powered-apps
   ```

2. **Choose an application to run**
   ```bash
   # For Document Summary App
   cd doc-summary-app
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   
   # For Local LLM Chat App
   cd local-llm-chat-app
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # Start backend: uvicorn backend.main:app --reload
   # Start frontend: streamlit run frontend/app.py
   
   # For Global News Tracker
   cd global-news-tracker
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   
   # For Multi-Modal Assistant
   cd multimodal-assistant
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # Install Ollama and models first
   streamlit run app.py
   
   # For Meeting Notes Extractor
   cd meeting-notes-extractor
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # Install FFmpeg and get OpenAI API key
   streamlit run app.py
   
   # For RAG Chatbot
   cd rag-chatbot-app
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # Get OpenAI API key
   streamlit run app.py
   ```

## 📊 Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Streamlit**: Web application framework
- **FastAPI**: High-performance API framework
- **Transformers**: Hugging Face model library
- **PyTorch**: Deep learning framework

### AI/ML Models
- **Pegasus**: Document summarization
- **BART**: News summarization
- **LLaVA**: Vision-language understanding
- **Llama3/Mistral**: Text generation
- **Whisper**: Speech-to-text transcription
- **GPT-3.5-turbo**: Meeting analysis and structuring
- **Sentence Transformers**: Text embeddings for RAG
- **ChromaDB**: Vector database for similarity search

### Additional Libraries
- **BeautifulSoup4**: Web scraping
- **Requests**: HTTP client
- **PIL**: Image processing
- **PyMuPDF**: PDF processing

## 🎯 Use Cases

### Document Summary App
- **Academic Research**: Summarize research papers and articles
- **Business Reports**: Extract key insights from lengthy documents
- **Legal Documents**: Quick overview of contracts and agreements
- **Educational Content**: Summarize textbooks and study materials

### Local LLM Chat App
- **Personal Assistant**: Daily task management and information retrieval
- **Code Assistance**: Programming help and debugging
- **Creative Writing**: Story generation and content creation
- **Learning Support**: Educational Q&A and explanations

### Global News Tracker
- **Media Monitoring**: Track news about specific topics or brands
- **Market Research**: Monitor industry trends and developments
- **Academic Research**: Stay updated on research and scientific news
- **Personal News**: Customized news summaries for busy professionals

### Multi-Modal Assistant
- **Educational Support**: Analyze diagrams, charts, and visual content
- **Accessibility**: Image description for visually impaired users
- **Professional Workflows**: Document analysis and visual feedback
- **Research Applications**: Scientific image analysis and interpretation

### Meeting Notes Extractor
- **Business Meetings**: Team standups, client meetings, board meetings
- **Educational Applications**: Lecture recordings and study group sessions
- **Legal and Compliance**: Deposition recordings and training sessions
- **Healthcare**: Patient consultations and medical team meetings

### RAG Chatbot
- **Research Papers**: Query academic papers and research documents
- **Legal Documents**: Ask questions about contracts, policies, and legal texts
- **Technical Documentation**: Interactive help with manuals and guides
- **Business Intelligence**: Analyze reports and business documents
- **Educational Content**: Create interactive learning experiences with textbooks

## 🔧 Configuration

### Environment Setup
Each application has its own virtual environment and requirements. Follow the individual README files for specific setup instructions.

### Model Requirements
- **Document Summary**: Requires internet connection for model download
- **Local LLM Chat**: Requires Ollama server and compatible models
- **News Tracker**: Requires internet connection for news scraping
- **Multi-Modal**: Requires Ollama server with vision models
- **Meeting Extractor**: Requires OpenAI API key and FFmpeg installation
- **RAG Chatbot**: Requires Ollama server with local LLM models

### Hardware Requirements
- **Minimum**: 8GB RAM, 4GB free disk space
- **Recommended**: 16GB RAM, 8GB free disk space
- **GPU**: Optional but recommended for faster processing

## 📈 Performance Considerations

### Optimization Tips
1. **Model Caching**: Models are cached after first load
2. **Batch Processing**: Process multiple items when possible
3. **Resource Management**: Monitor memory usage during processing
4. **Network Optimization**: Use local models when possible

### Scaling Considerations
- **Local Deployment**: Suitable for personal and small team use
- **Cloud Deployment**: Consider cloud services for production use
- **Load Balancing**: Implement for high-traffic scenarios
- **Caching**: Add Redis or similar for session management

## 🛠️ Development

### Project Structure
```
ai-powered-apps/
├── doc-summary-app/          # Document summarization application
├── local-llm-chat-app/       # Local LLM chat application
├── global-news-tracker/      # News tracking application
├── multimodal-assistant/      # Multi-modal AI assistant
├── meeting-notes-extractor/  # Meeting audio processing application
├── rag-chatbot-app/          # RAG document Q&A application
├── README.md                 # This file
└── .gitignore                # Git ignore rules
```

### Contributing Guidelines
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Standards
- Follow PEP 8 Python style guidelines
- Add docstrings to functions and classes
- Include error handling and logging
- Write clear and concise comments

## 🔒 Security Considerations

### Data Privacy
- **Local Processing**: Most applications process data locally
- **No Data Storage**: Applications don't store user data permanently
- **Secure APIs**: Use HTTPS for external API calls
- **Input Validation**: Sanitize all user inputs

### Best Practices
- Keep dependencies updated
- Use virtual environments
- Implement proper error handling
- Follow security guidelines for web applications

## 📚 Documentation

Each application includes comprehensive documentation:
- **Problem Statement**: Clear definition of the problem being solved
- **Solution Architecture**: Technical architecture and design decisions
- **Installation Guide**: Step-by-step setup instructions
- **Usage Examples**: How to use each feature
- **Troubleshooting**: Common issues and solutions

## 🆘 Support

### Getting Help
1. Check the individual application README files
2. Review the troubleshooting sections
3. Create an issue in the repository
4. Check existing issues for similar problems

### Common Issues
- **Model Loading**: Ensure sufficient system resources
- **Network Connectivity**: Check internet connection for external services
- **Dependencies**: Verify all requirements are installed
- **Permissions**: Check file and directory permissions

## 🚀 Future Roadmap

### Planned Features
- **API Integration**: REST APIs for all applications
- **Mobile Apps**: Native mobile applications
- **Cloud Deployment**: Docker containers and cloud deployment guides
- **Advanced Analytics**: Usage analytics and performance monitoring
- **Multi-language Support**: Support for multiple languages
- **Custom Model Training**: Tools for training custom models

### Community Contributions
- **New Applications**: Additional AI-powered applications
- **Model Integrations**: Support for new AI models
- **UI/UX Improvements**: Enhanced user interfaces
- **Performance Optimizations**: Faster processing and better resource usage

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For providing the Transformers library and pre-trained models
- **Streamlit**: For the excellent web application framework
- **Ollama**: For local LLM serving capabilities
- **Open Source Community**: For the various libraries and tools used

---

**Built with ❤️ using Python, Streamlit, and AI/ML technologies**

*For questions, suggestions, or contributions, please open an issue or submit a pull request.*
