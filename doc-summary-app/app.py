import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import re
import nltk
from collections import Counter
from docx import Document
import io
import os

# Load summarization model with better configuration
@st.cache_resource
def load_summarizer():
    try:
        # Try BART model first (more reliable)
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.warning(f"BART model failed, trying Pegasus: {e}")
        try:
            return pipeline("summarization", model="google/pegasus-xsum")
        except Exception as e2:
            st.error(f"All models failed: {e2}")
            return None

summarizer = load_summarizer()

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

# Streamlit App UI
st.title("📄 Document Summarization Tool")

uploaded_file = st.file_uploader(
    "Upload a document", 
    type=["pdf", "txt", "docx", "md", "csv"],
    help="Supported formats: PDF, TXT, DOCX, MD, CSV"
)

if uploaded_file:
    # Show file information
    st.info(f"📁 **File:** {uploaded_file.name} ({uploaded_file.type})")
    
    with st.spinner("📄 Extracting text from document..."):
        text = extract_text_from_file(uploaded_file)

    st.subheader("📃 Extracted Document Text")
    with st.expander("Show extracted text"):
        st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Generate AI Summary"):
            if summarizer is None:
                st.error("❌ No AI summarization model available. Please restart the app.")
            else:
                with st.spinner("🧠 Generating summary..."):
                    try:
                        # For short documents, summarize directly
                        if len(text.split()) < 1000:
                            # Clean and prepare text
                            clean_text_input = re.sub(r'\s+', ' ', text).strip()
                            
                            # Ensure text is not too long
                            if len(clean_text_input) > 2000:
                                clean_text_input = clean_text_input[:2000] + "..."
                            
                            # Generate summary with better parameters
                            summary = summarizer(
                                clean_text_input, 
                                max_length=150, 
                                min_length=50, 
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
                                        max_length=100, 
                                        min_length=30, 
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
                                    max_length=150,
                                    min_length=50,
                                    do_sample=False,
                                    temperature=0.1
                                )[0]['summary_text']
                            else:
                                final_summary = "Unable to generate summary due to processing errors."

                        st.subheader("✅ Final Summary")
                        st.write(final_summary)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
                        st.info("💡 Try with a shorter document or restart the app.")
    
    with col2:
        if st.button("📋 Generate Simple Summary"):
            with st.spinner("📝 Generating simple summary..."):
                simple_summary = simple_summarize(text)
                st.subheader("✅ Simple Summary")
                st.write(simple_summary)

