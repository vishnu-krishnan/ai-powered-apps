# 📄 Document Summary App

A powerful AI-powered document summarization tool that extracts text from PDF documents and generates intelligent summaries using state-of-the-art language models.

## 🚀 Features

- **PDF Text Extraction**: Automatically extracts text from PDF documents
- **AI-Powered Summarization**: Uses BART and Pegasus models for intelligent summarization
- **Simple Summary Fallback**: Extractive summarization when AI models fail
- **Multiple Summary Options**: Choose between AI and simple summarization methods
- **User-Friendly Interface**: Clean Streamlit-based web interface
- **Error Handling**: Robust error handling with helpful user feedback

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - Primary: Facebook BART-large-cnn
  - Fallback: Google Pegasus-xsum
- **PDF Processing**: PyMuPDF (fitz)
- **Text Processing**: NLTK, Regular Expressions
- **Language**: Python 3.8+

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for model downloads)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishnu-krishnan/ai-powered-apps.git
   cd ai-powered-apps/doc-summary-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the app**
   Open your browser to `http://localhost:8501`

## 🎯 Usage

### Basic Usage

1. **Upload PDF**: Click "Browse files" and select a PDF document
2. **View Extracted Text**: Expand "Show extracted text" to verify content
3. **Generate Summary**: Choose between two options:
   - **"Generate AI Summary"**: Uses BART model for advanced summarization
   - **"Generate Simple Summary"**: Uses extractive method for reliable results

### Supported File Types

- **PDF**: Primary format (MP3, WAV, M4A, FLAC, OGG)
- **File Size**: Up to 25MB (Streamlit default)
- **Text Quality**: Works best with text-heavy PDFs

### Best Practices

- **Document Quality**: Use clear, well-formatted PDFs
- **Text Content**: Works best with text-heavy documents (not image-heavy)
- **Document Length**: Handles both short and long documents
- **Language**: Optimized for English text

## 🔧 Troubleshooting

### Common Issues

| Issue | Symptom | Quick Fix |
|-------|---------|-----------|
| **Wrong Summary** | Generates unrelated content | Try "Generate Simple Summary" |
| **IndentationError** | Python syntax errors | Check code indentation |
| **NLTK Missing** | `punkt_tab not found` | Run NLTK download command |
| **Model Failed** | No summarization model | Restart app, check internet |
| **Upload Failed** | Can't upload PDF | Check file size/format |

### Quick Solutions

```bash
# Fix NLTK issues
python -c "import nltk; nltk.download('punkt_tab')"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check system status
python -c "import fitz, nltk, transformers; print('All OK')"
```

### Detailed Troubleshooting

- 📋 [Complete Troubleshooting Guide](TROUBLESHOOTING.md)
- ⚡ [Quick Fixes Reference](QUICK_FIXES.md)

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Extraction│───▶│  Text Cleaning  │
│   (PyMuPDF)     │    │  (fitz library) │    │  (Regex/NLTK)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◀───│  Summary Engine │◀───│  Model Selection│
│  (User Interface)│    │  (BART/Pegasus) │    │  (AI/Simple)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **PDF Processing**: Upload → Text Extraction → Cleaning
2. **Model Selection**: AI Summary (BART) or Simple Summary (Extractive)
3. **Text Processing**: Chunking → Summarization → Final Output
4. **User Interface**: Display → Feedback → Error Handling

## 📊 Performance

### Model Performance

| Model | Speed | Quality | Reliability |
|-------|-------|---------|-------------|
| **BART-large-cnn** | Medium | High | High |
| **Pegasus-xsum** | Fast | Medium | Medium |
| **Simple Extraction** | Fast | Medium | Very High |

### System Requirements

- **Minimum**: 4GB RAM, 2GB free disk space
- **Recommended**: 8GB RAM, 5GB free disk space
- **Network**: Internet connection for model downloads

## 🔒 Security & Privacy

- **Local Processing**: All processing happens locally
- **No Data Storage**: Documents are not permanently stored
- **Temporary Files**: Automatically cleaned after processing
- **Model Downloads**: Only downloads publicly available models

## 🚀 Future Enhancements

### Planned Features

- **Multi-language Support**: Support for multiple languages
- **Custom Models**: User-uploaded custom models
- **Batch Processing**: Multiple document processing
- **API Integration**: REST API for programmatic access
- **Advanced Analytics**: Document analysis and insights

### Performance Improvements

- **Model Optimization**: Quantized models for faster processing
- **Caching**: Intelligent caching for repeated documents
- **Parallel Processing**: Multi-threaded document processing
- **GPU Support**: CUDA acceleration for faster processing

## 🤝 Contributing

### Development Setup

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

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Getting Help

1. Check the troubleshooting guides
2. Review the quick fixes reference
3. Create an issue in the repository
4. Check existing issues for similar problems

### Resources

- **Documentation**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Quick Fixes**: [QUICK_FIXES.md](QUICK_FIXES.md)
- **Requirements**: [requirements.txt](requirements.txt)

## 🙏 Acknowledgments

- **Hugging Face**: For providing the Transformers library and pre-trained models
- **Streamlit**: For the excellent web application framework
- **PyMuPDF**: For robust PDF processing capabilities
- **NLTK**: For natural language processing tools

---

**Built with ❤️ using Python, Streamlit, and AI/ML technologies**

*For questions, suggestions, or contributions, please open an issue or submit a pull request.*