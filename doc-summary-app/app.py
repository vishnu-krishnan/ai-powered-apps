import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import re
import nltk
from collections import Counter
from docx import Document
import io
import os

# Available models configuration
AVAILABLE_MODELS = {
    "BART (Recommended)": {
        "model_name": "facebook/bart-large-cnn",
        "description": "Best for general summarization, high quality",
        "max_length": 150,
        "min_length": 50
    },
    "Pegasus": {
        "model_name": "google/pegasus-xsum", 
        "description": "Fast and efficient, good for news articles",
        "max_length": 120,
        "min_length": 40
    },
    "T5": {
        "model_name": "t5-small",
        "description": "Lightweight, good for quick summaries",
        "max_length": 100,
        "min_length": 30
    }
}

# Load selected summarization model
@st.cache_resource
def load_summarizer(model_name):
    try:
        return pipeline("summarization", model=model_name)
    except Exception as e:
        st.error(f"Failed to load model {model_name}: {e}")
        return None

# Clean up LaTeX-style or noisy text
def clean_text(text):
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)   # Remove LaTeX commands with braces
    text = re.sub(r'\\[a-zA-Z]+', '', text)            # Remove LaTeX commands without braces
    text = re.sub(r'\s{2,}', ' ', text)                # Replace multiple spaces with one
    return text.strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    pdf_file = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf_file:
        text += page.get_text()
    return clean_text(text)

# Extract text from Word document
def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return clean_text(text)
    except Exception as e:
        st.error(f"Error reading Word document: {str(e)}")
        return ""

# Extract text from plain text file
def extract_text_from_txt(file):
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                file.seek(0)  # Reset file pointer
                text = file.read().decode(encoding)
                return clean_text(text)
            except UnicodeDecodeError:
                continue
        # If all encodings fail, try with errors='ignore'
        file.seek(0)
        text = file.read().decode('utf-8', errors='ignore')
        return clean_text(text)
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return ""

# Universal text extraction function
def extract_text_from_file(file):
    """Extract text from various file types"""
    file_type = file.type.lower()
    file_name = file.name.lower()
    
    if file_type == "application/pdf" or file_name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.endswith('.docx'):
        return extract_text_from_docx(file)
    elif file_type == "text/plain" or file_name.endswith(('.txt', '.md', '.csv')):
        return extract_text_from_txt(file)
    else:
        # Try to read as text file for unknown types
        st.warning(f"⚠️ Unknown file type: {file_type}. Attempting to read as text file...")
        return extract_text_from_txt(file)

# Split text into manageable chunks
def split_into_chunks(text, max_tokens=400):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Simple extractive summarization as fallback
def simple_summarize(text, max_sentences=3):
    """Simple extractive summarization using sentence scoring"""
    try:
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        # Also try the older punkt tokenizer as fallback
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
        
        # Simple scoring based on word frequency
        words = nltk.word_tokenize(text.lower())
        word_freq = Counter([w for w in words if w.isalpha() and len(w) > 3])
        
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = nltk.word_tokenize(sentence.lower())
            score = sum([word_freq[word] for word in sentence_words if word in word_freq])
            sentence_scores[sentence] = score
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        summary_sentences = [sent[0] for sent in top_sentences]
        
        return " ".join(summary_sentences)
    except Exception as e:
        # Fallback to basic sentence extraction without NLTK
        return basic_sentence_extraction(text, max_sentences)

def basic_sentence_extraction(text, max_sentences=3):
    """Basic sentence extraction without NLTK dependency"""
    try:
        # Split by common sentence endings
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple scoring based on word count and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Score based on length and position (first sentences are often important)
            score = len(sentence.split()) + (10 - i)  # First sentences get higher score
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent[0] for sent in scored_sentences[:max_sentences]]
        
        return ". ".join(top_sentences) + "."
    except Exception as e:
        return f"Basic extraction failed: {str(e)}"

# Clean and Professional UI
st.set_page_config(
    page_title="Document Summarization Tool",
    page_icon="📄",
    layout="wide"
)

# Theme-aware CSS styling
st.markdown("""
<style>
    /* Main header with theme-aware colors */
    .main-header {
        background: var(--primary-color, #2c3e50);
        color: var(--text-color, white);
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 600;
        color: white;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
        color: white;
    }
    
    /* File info with theme-aware background */
    .file-info {
        background: var(--secondary-background-color, #ecf0f1);
        color: var(--text-color, #2c3e50);
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color, #3498db);
    }
    
    /* Summary container with theme-aware styling */
    .summary-container {
        background: var(--background-color, #ffffff);
        color: var(--text-color, #2c3e50);
        border: 1px solid var(--border-color, #bdc3c7);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Button styling with theme awareness */
    .stButton > button {
        background: var(--primary-color, #3498db);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: var(--primary-hover-color, #2980b9);
        transform: none;
    }
    
    /* Dark mode specific adjustments */
    @media (prefers-color-scheme: dark) {
        .main-header {
            background: #1a1a1a;
            color: white;
        }
        
        .file-info {
            background: #2d2d2d;
            color: #ffffff;
            border-left-color: #4a9eff;
        }
        
        .summary-container {
            background: #2d2d2d;
            color: #ffffff;
            border-color: #404040;
        }
        
        .stButton > button {
            background: #4a9eff;
        }
        
        .stButton > button:hover {
            background: #3a8bef;
        }
    }
    
    /* Light mode specific adjustments */
    @media (prefers-color-scheme: light) {
        .main-header {
            background: #2c3e50;
            color: white;
        }
        
        .file-info {
            background: #f8f9fa;
            color: #2c3e50;
            border-left-color: #3498db;
        }
        
        .summary-container {
            background: #ffffff;
            color: #2c3e50;
            border-color: #dee2e6;
        }
    }
    
    /* Streamlit dark theme overrides */
    .stApp[data-theme="dark"] .main-header {
        background: #1a1a1a;
        color: white;
    }
    
    .stApp[data-theme="dark"] .file-info {
        background: #2d2d2d;
        color: #ffffff;
        border-left-color: #4a9eff;
    }
    
    .stApp[data-theme="dark"] .summary-container {
        background: #2d2d2d;
        color: #ffffff;
        border-color: #404040;
    }
    
    .stApp[data-theme="dark"] .stButton > button {
        background: #4a9eff;
    }
    
    .stApp[data-theme="dark"] .stButton > button:hover {
        background: #3a8bef;
    }
    
    /* Additional dark mode overrides for better visibility */
    .stApp[data-theme="dark"] .file-info {
        background: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    .stApp[data-theme="dark"] .summary-container {
        background: #2d2d2d !important;
        color: #ffffff !important;
        border-color: #404040 !important;
    }
    
    /* Ensure text is visible in dark mode */
    .stApp[data-theme="dark"] .file-info strong {
        color: #ffffff !important;
    }
    
    .stApp[data-theme="dark"] .summary-container p {
        color: #ffffff !important;
    }
    
    /* Override any white text on white background issues */
    .stApp[data-theme="dark"] * {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# Clean header
st.markdown("""
<div class="main-header">
    <h1>📄 Document Summarization Tool</h1>
    <p>AI-powered document analysis and summarization</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with model selection
with st.sidebar:
    st.markdown("### 🤖 AI Model Selection")
    
    # Model selection
    selected_model = st.selectbox(
        "Choose AI Model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Select the AI model for summarization"
    )
    
    # Show model info
    model_info = AVAILABLE_MODELS[selected_model]
    st.markdown(f"**Model:** {model_info['model_name']}")
    st.markdown(f"**Description:** {model_info['description']}")
    
    # Load the selected model
    with st.spinner("Loading AI model..."):
        summarizer = load_summarizer(model_info['model_name'])
    
    if summarizer is not None:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Failed to load model")
    
    st.markdown("### 📋 Supported Formats")
    st.markdown("- PDF Documents")
    st.markdown("- Text Files (.txt)")
    st.markdown("- Word Documents (.docx)")
    st.markdown("- Markdown Files (.md)")
    st.markdown("- CSV Files (.csv)")
    
    st.markdown("### 📊 Model Comparison")
    st.markdown("**BART**: Best quality, slower")
    st.markdown("**Pegasus**: Fast, good for news")
    st.markdown("**T5**: Lightweight, quick")
    
    st.markdown("### ⚡ Features")
    st.markdown("- AI-powered summarization")
    st.markdown("- Multiple model options")
    st.markdown("- Model comparison")
    st.markdown("- Fast processing")
    st.markdown("- Local processing (private)")
    st.markdown("- Multiple summary options")
    

# Main content
st.markdown("### 📁 Upload Document")
uploaded_file = st.file_uploader(
    "Choose a file to upload",
    type=["pdf", "txt", "docx", "md", "csv"],
    help="Supported formats: PDF, TXT, DOCX, MD, CSV"
)

if uploaded_file:
    # Clean file information
    st.markdown(f"""
    <div class="file-info">
        <strong>📁 File:</strong> {uploaded_file.name} | 
        <strong>Type:</strong> {uploaded_file.type} | 
        <strong>Size:</strong> {len(uploaded_file.getvalue()):,} bytes
    </div>
    """, unsafe_allow_html=True)
    
    # Process document
    with st.spinner("Extracting text from document..."):
        text = extract_text_from_file(uploaded_file)

    # Show document stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Characters", f"{len(text):,}")
    with col2:
        st.metric("Words", f"{len(text.split()):,}")
    with col3:
        st.metric("Estimated Reading Time", f"{len(text.split()) // 200} min")

    # Text preview
    with st.expander("📄 View extracted text"):
        st.text_area("", text[:2000] + ("..." if len(text) > 2000 else ""), height=150, disabled=True)

    # Summary options
    st.markdown("### 🧠 Generate Summary")
    col1, col2 = st.columns(2)
    
    # Initialize session state for summaries
    if 'ai_summary' not in st.session_state:
        st.session_state.ai_summary = None
    if 'simple_summary' not in st.session_state:
        st.session_state.simple_summary = None
    
    with col1:
        if st.button("🤖 AI Summary", use_container_width=True):
            if summarizer is None:
                st.error("❌ No AI summarization model available. Please restart the app.")
            else:
                with st.spinner("🧠 Generating AI summary..."):
                    try:
                        # For short documents, summarize directly
                        if len(text.split()) < 1000:
                            # Clean and prepare text
                            clean_text_input = re.sub(r'\s+', ' ', text).strip()
                            
                            # Ensure text is not too long
                            if len(clean_text_input) > 2000:
                                clean_text_input = clean_text_input[:2000] + "..."
                            
                            # Generate summary with model-specific parameters
                            summary = summarizer(
                                clean_text_input, 
                                max_length=model_info['max_length'], 
                                min_length=model_info['min_length'], 
                                do_sample=False,
                                temperature=0.1
                            )
                            
                            final_summary = summary[0]['summary_text']
                            
                        else:
                            # For longer documents, use chunking
                            chunks = split_into_chunks(text)
                            partial_summaries = []

                            for i, chunk in enumerate(chunks):
                                try:
                                    # Clean chunk
                                    clean_chunk = re.sub(r'\s+', ' ', chunk).strip()
                                    if len(clean_chunk) > 1000:
                                        clean_chunk = clean_chunk[:1000] + "..."
                                    
                                    summary = summarizer(
                                        clean_chunk, 
                                        max_length=model_info['max_length']//2, 
                                        min_length=model_info['min_length']//2, 
                                        do_sample=False,
                                        temperature=0.1
                                    )
                                    partial_summaries.append(summary[0]['summary_text'])
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ Skipped chunk {i+1} due to error: {e}")

                            if partial_summaries:
                                combined_summary = " ".join(partial_summaries)
                                
                                # Final re-summarization
                                if len(combined_summary) > 1000:
                                    combined_summary = combined_summary[:1000] + "..."
                                    
                                final_summary = summarizer(
                                    combined_summary,
                                    max_length=model_info['max_length'],
                                    min_length=model_info['min_length'],
                                    do_sample=False,
                                    temperature=0.1
                                )[0]['summary_text']
                            else:
                                final_summary = "Unable to generate summary due to processing errors."

                        # Store AI summary in session state
                        st.session_state.ai_summary = final_summary
                        st.success("✅ AI Summary generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
                        st.info("💡 Try with a shorter document or restart the app.")
    
    with col2:
        if st.button("📋 Simple Summary", use_container_width=True):
            with st.spinner("Generating simple summary..."):
                simple_summary = simple_summarize(text)
                
                # Store simple summary in session state
                st.session_state.simple_summary = simple_summary
                st.success("✅ Simple Summary generated successfully!")

    # Unified results display section
    if st.session_state.ai_summary or st.session_state.simple_summary:
        st.markdown("---")
        st.markdown("### 📊 Summary Results")
        
        # Create tabs for different summaries
        tab_names = []
        if st.session_state.ai_summary:
            tab_names.append("🤖 AI Summary")
        if st.session_state.simple_summary:
            tab_names.append("📋 Simple Summary")
        
        if len(tab_names) > 1:
            tabs = st.tabs(tab_names)
        else:
            tabs = [st.container()]
        
        tab_index = 0
        
        # AI Summary Tab
        if st.session_state.ai_summary:
            with tabs[tab_index]:
                st.markdown(f"#### 🤖 AI-Powered Summary ({selected_model})")
                st.markdown(f"""
                <div class="summary-container">
                    {st.session_state.ai_summary}
                </div>
                """, unsafe_allow_html=True)
                
                # AI Summary stats
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Words", f"{len(st.session_state.ai_summary.split())}")
                with col_stat2:
                    st.metric("Compression", f"{len(st.session_state.ai_summary.split()) / len(text.split()) * 100:.1f}%")
                with col_stat3:
                    st.metric("Model", selected_model.split(' ')[0])
                with col_stat4:
                    st.metric("Type", "AI-Generated")
            tab_index += 1
        
        # Simple Summary Tab
        if st.session_state.simple_summary:
            with tabs[tab_index]:
                st.markdown("#### 📋 Extractive Summary")
                st.markdown(f"""
                <div class="summary-container">
                    {st.session_state.simple_summary}
                </div>
                """, unsafe_allow_html=True)
                
                # Simple Summary stats
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Words", f"{len(st.session_state.simple_summary.split())}")
                with col_stat2:
                    st.metric("Compression", f"{len(st.session_state.simple_summary.split()) / len(text.split()) * 100:.1f}%")
                with col_stat3:
                    st.metric("Type", "Extractive")
        
        # Clear summaries button
        if st.button("🗑️ Clear All Summaries", use_container_width=True):
            st.session_state.ai_summary = None
            st.session_state.simple_summary = None
            st.rerun()


