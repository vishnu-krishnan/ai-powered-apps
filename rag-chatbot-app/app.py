import streamlit as st
import os
import tempfile
import requests
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.processed_files = {}  # Track processed files with metadata
        self.document_summaries = {}  # Store document summaries
        
    def initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            st.error(f"Error initializing embeddings: {e}")
    
    def load_documents(self, file_paths: List[str], original_names: dict = None) -> List[Document]:
        """Load documents from various file types with enhanced metadata"""
        documents = []
        
        for file_path in file_paths:
            try:
                # Use original name if available, otherwise use file path
                display_name = original_names.get(file_path, file_path) if original_names else file_path
                
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_path.endswith(('.docx', '.doc')):
                    loader = Docx2txtLoader(file_path)
                else:
                    st.warning(f"Unsupported file type: {display_name}")
                    continue
                
                docs = loader.load()
                
                # Add enhanced metadata for each document
                for doc in docs:
                    # Determine document type based on filename (general approach)
                    doc_type = "general"
                    content_lower = doc.page_content.lower()
                    
                    # Simple document type detection based on filename
                    filename_lower = display_name.lower()
                    if any(keyword in filename_lower for keyword in ["report", "analysis", "study", "research"]):
                        doc_type = "report"
                    elif any(keyword in filename_lower for keyword in ["manual", "guide", "instruction", "tutorial"]):
                        doc_type = "manual"
                    elif any(keyword in filename_lower for keyword in ["policy", "procedure", "rule", "regulation"]):
                        doc_type = "policy"
                    elif any(keyword in filename_lower for keyword in ["meeting", "minutes", "notes", "agenda"]):
                        doc_type = "meeting"
                    elif any(keyword in filename_lower for keyword in ["contract", "agreement", "legal"]):
                        doc_type = "legal"
                    elif any(keyword in filename_lower for keyword in ["financial", "budget", "invoice", "receipt"]):
                        doc_type = "financial"
                    elif any(keyword in filename_lower for keyword in ["technical", "specification", "design"]):
                        doc_type = "technical"
                    else:
                        # Try to detect from content if filename doesn't give clues
                        if any(keyword in content_lower[:500] for keyword in ["executive summary", "introduction", "conclusion", "findings"]):
                            doc_type = "report"
                        elif any(keyword in content_lower[:500] for keyword in ["step", "procedure", "how to", "instructions"]):
                            doc_type = "manual"
                        elif any(keyword in content_lower[:500] for keyword in ["policy", "rule", "regulation", "guideline"]):
                            doc_type = "policy"
                        else:
                            doc_type = "general"
                    
                    doc.metadata.update({
                        'source_file': display_name,  # Use original name for display
                        'file_type': display_name.split('.')[-1],
                        'document_type': doc_type,  # Add document type for filtering
                        'processed_at': str(datetime.now()),
                        'document_id': f"{display_name}_{len(documents)}"
                    })
                    
                    # Debug logging
                    logger.info(f"Document type detected: {doc_type} for {display_name}")
                
                documents.extend(docs)
                
                # Track processed files using original name
                self.processed_files[display_name] = {
                    'status': 'loaded',
                    'document_count': len(docs),
                    'total_chars': sum(len(doc.page_content) for doc in docs),
                    'file_type': display_name.split('.')[-1]
                }
                
                logger.info(f"Loaded {len(docs)} documents from {display_name}")
                
            except Exception as e:
                logger.error(f"Error loading {display_name}: {e}")
                st.error(f"Error loading {display_name}: {e}")
                self.processed_files[display_name] = {'status': 'error', 'error': str(e)}
        
        return documents
    
    def split_documents(self, documents: List[Document], chunk_size: int = 2000, chunk_overlap: int = 400):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(splits)} chunks")
        return splits
    
    def create_vectorstore(self, documents: List[Document], collection_name: str = "rag_documents"):
        """Create and populate the vector store"""
        try:
            # Remove any existing database to ensure clean start
            import shutil
            import os
            chroma_db_path = "./chroma_db"
            if os.path.exists(chroma_db_path):
                shutil.rmtree(chroma_db_path)
                logger.info("Removed existing ChromaDB to ensure clean start")
            
            # Create directory with proper permissions
            os.makedirs(chroma_db_path, mode=0o755, exist_ok=True)
            
            # Initialize ChromaDB with error handling
            try:
                client = chromadb.PersistentClient(path=chroma_db_path)
                logger.info("ChromaDB client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing ChromaDB client: {e}")
                # Try alternative approach
                client = chromadb.Client()
                logger.info("Using in-memory ChromaDB client as fallback")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=client,
                collection_name=collection_name
            )
            
            logger.info(f"Vector store created with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            st.error(f"Error creating vector store: {e}")
            return False
    
    def setup_qa_chain(self, model_name: str = "llama3", ollama_base_url: str = "http://localhost:11434"):
        """Setup the Q&A chain"""
        try:
            # Check if Ollama is running
            try:
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    raise Exception("Ollama server not responding")
            except Exception as e:
                raise Exception(f"Cannot connect to Ollama server at {ollama_base_url}. Please ensure Ollama is running.")
            
            # Initialize LLM with better settings for natural responses
            llm = OllamaLLM(
                model=model_name,
                base_url=ollama_base_url,
                temperature=0.3,  # Slightly higher for more natural responses
                timeout=120,  # Increased timeout for better responses
                num_predict=1024  # Allow longer, more detailed responses
            )
            
            # Create natural and intelligent prompt template for RAG
            prompt_template = """You are an intelligent document analysis assistant. Analyze the provided documents and answer the user's question in a natural, helpful way.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Read the document content carefully
- Answer the question based ONLY on the information in the documents
- Be accurate and precise when counting or listing items
- Provide a natural, conversational response
- If asked for specific numbers or lists, be thorough and accurate
- If the information isn't available in the documents, say so clearly
- Cite which document(s) contain the information when relevant

Your response:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create retrieval QA chain with better settings
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}  # Get more documents for better context
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            logger.info("Q&A chain setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Q&A chain: {e}")
            st.error(f"Error setting up Q&A chain: {e}")
            return False
    
    def query(self, question: str) -> dict:
        """Query the RAG system"""
        try:
            if self.qa_chain is None:
                return {"error": "Q&A chain not initialized"}
            
            # Use regular retriever for all questions (generalized approach)
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 12}  # Get more documents for better context
            )
            retrieved_docs = retriever.invoke(question)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for question: {question}")
            for i, doc in enumerate(retrieved_docs):
                doc_type = doc.metadata.get('document_type', 'unknown')
                logger.info(f"Document {i+1} ({doc_type}): {doc.page_content[:100]}...")
            
            # Use regular QA chain for all questions (generalized approach)
            result = self.qa_chain.invoke({"query": question})
            
            # Check if we got a valid response
            if "result" in result and result["result"].strip():
                # Add source document information to the response
                result["source_documents"] = retrieved_docs
                return result
            else:
                # Fallback: provide information about retrieved documents
                doc_summaries = []
                for i, doc in enumerate(retrieved_docs[:5]):  # Show first 5 docs
                    source = doc.metadata.get('source_file', 'Unknown source')
                    summary = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    doc_summaries.append(f"📄 **{source}**: {summary}")
                
                return {
                    "result": f"🔍 **Analysis Results:**\n\nI found {len(retrieved_docs)} relevant documents in your uploaded files. Here's what I discovered:\n\n" + "\n\n".join(doc_summaries),
                    "source_documents": retrieved_docs
                }
            
        except Exception as e:
            logger.error(f"Error querying: {e}")
            error_msg = str(e)
            
            # Provide specific guidance for common errors
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                return {"error": f"Model timeout: {error_msg}. Try using a smaller model like llama3.2:1b for faster responses."}
            elif "connection" in error_msg.lower():
                return {"error": f"Connection error: {error_msg}. Make sure Ollama is running with 'ollama serve'."}
            elif "memory" in error_msg.lower():
                return {"error": f"Memory error: {error_msg}. Try using a smaller model or free up system memory."}
            else:
                return {"error": f"Query failed: {error_msg}"}
    
    def get_document_summary(self, file_path: str) -> str:
        """Generate a summary for a specific document"""
        try:
            if file_path not in self.processed_files:
                return "Document not found"
            
            # Get all chunks from this specific file
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10, "filter": {"source_file": file_path}}
            )
            
            # Get a sample of content from the document
            sample_docs = retriever.get_relevant_documents("summary overview")
            
            if not sample_docs:
                return "No content found in document"
            
            # Create a summary prompt
            content_sample = "\n".join([doc.page_content[:500] for doc in sample_docs[:3]])
            
            summary_prompt = f"""Please provide a concise summary of the following document content:

{content_sample}

Summary:"""
            
            # Use the LLM to generate summary
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": summary_prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get('response', 'Unable to generate summary')
                self.document_summaries[file_path] = summary
                return summary
            else:
                return "Error generating summary"
                
        except Exception as e:
            logger.error(f"Error generating summary for {file_path}: {e}")
            return f"Error: {str(e)}"
    
    def get_cross_document_analysis(self, question: str) -> dict:
        """Analyze question across all documents and provide comprehensive answer"""
        try:
            # Get relevant documents from all files
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 12}  # Get more documents for cross-analysis
            )
            
            retrieved_docs = retriever.invoke(question)
            
            # Group documents by source file
            docs_by_file = {}
            for doc in retrieved_docs:
                source_file = doc.metadata.get('source_file', 'unknown')
                if source_file not in docs_by_file:
                    docs_by_file[source_file] = []
                docs_by_file[source_file].append(doc)
            
            # Create comprehensive context
            context_parts = []
            for file_path, docs in docs_by_file.items():
                file_content = "\n".join([doc.page_content for doc in docs])
                context_parts.append(f"From {os.path.basename(file_path)}:\n{file_content}")
            
            context = "\n\n".join(context_parts)
            
            # Enhanced prompt for cross-document analysis
            prompt = f"""You are analyzing multiple documents to answer a question. Use information from all relevant documents to provide a comprehensive answer.

Documents analyzed: {len(docs_by_file)} files
Total relevant sections: {len(retrieved_docs)}

Context from multiple documents:
{context}

Question: {question}

Instructions:
- Synthesize information from all relevant documents
- If documents contradict each other, mention the differences
- Provide a comprehensive answer that draws from all sources
- Cite which document(s) contain specific information when possible

Comprehensive Answer:"""
            
            # Get response from LLM with increased timeout
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120  # Increased timeout to match main LLM
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "result": result.get('response', 'No response'),
                    "source_documents": retrieved_docs,
                    "files_analyzed": list(docs_by_file.keys()),
                    "analysis_type": "cross_document"
                }
            else:
                return {"error": f"Model request failed: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            logger.error("Cross-document analysis timed out")
            return {"error": "Analysis timed out. The model is taking too long to respond. Try using a smaller model or asking a simpler question."}
        except Exception as e:
            logger.error(f"Error in cross-document analysis: {e}")
            return {"error": str(e)}
    
    def verify_document_content(self, question: str) -> dict:
        """Verify document content before answering questions"""
        try:
            if not self.vectorstore:
                return {"error": "No documents loaded"}
            
            # Get all documents to verify content
            all_docs = self.vectorstore.similarity_search(question, k=20)
            
            # Analyze document types
            doc_types = {}
            for doc in all_docs:
                doc_type = doc.metadata.get('document_type', 'unknown')
                if doc_type not in doc_types:
                    doc_types[doc_type] = []
                doc_types[doc_type].append({
                    'source': doc.metadata.get('source_file', 'Unknown'),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            return {
                "total_documents": len(all_docs),
                "document_types": doc_types,
                "verification_status": "completed"
            }
        except Exception as e:
            logger.error(f"Error verifying document content: {e}")
            return {"error": str(e)}
    
    def get_document_statistics(self) -> dict:
        """Get statistics about processed documents"""
        try:
            stats = {
                "total_files": len(self.processed_files),
                "successful_files": len([f for f in self.processed_files.values() if f.get('status') == 'loaded']),
                "failed_files": len([f for f in self.processed_files.values() if f.get('status') == 'error']),
                "total_documents": sum(f.get('document_count', 0) for f in self.processed_files.values() if f.get('status') == 'loaded'),
                "total_chunks": len(self.documents) if hasattr(self, 'documents') and self.documents else 0,
                "total_characters": sum(f.get('total_chars', 0) for f in self.processed_files.values() if f.get('status') == 'loaded'),
                "file_types": {}
            }
            
            # Count file types
            for file_info in self.processed_files.values():
                if file_info.get('status') == 'loaded':
                    file_type = file_info.get('file_type', 'unknown')
                    stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            
            return stats
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {"error": str(e)}
    
    def clear_documents(self):
        """Clear all processed documents and reset the chatbot"""
        try:
            import shutil
            import os
            
            # Clear session data
            self.documents = []
            self.processed_files = {}
            self.document_summaries = {}
            self.vectorstore = None
            self.qa_chain = None
            
            # Actually delete the ChromaDB database
            chroma_db_path = "./chroma_db"
            if os.path.exists(chroma_db_path):
                # Force remove with proper permissions
                os.system(f"chmod -R 755 {chroma_db_path}")
                shutil.rmtree(chroma_db_path)
                logger.info("ChromaDB database deleted successfully")
            
            # Also clear any temporary files
            temp_files = ["./chroma_db", "./.chroma", "./chroma"]
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    try:
                        os.system(f"chmod -R 755 {temp_path}")
                        shutil.rmtree(temp_path)
                        logger.info(f"Cleared {temp_path}")
                    except Exception as e:
                        logger.warning(f"Could not clear {temp_path}: {e}")
            
            logger.info("All documents and database cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False

def main():
    st.set_page_config(
        page_title="RAG Chatbot - Document Q&A",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot - Document Q&A")
    st.markdown("Upload documents and ask questions using Retrieval-Augmented Generation (RAG)")
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = "standard"  # standard, cross_document, individual
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ollama Configuration
        st.subheader("🤖 Local LLM Settings")
        
        # Ollama Base URL
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value="http://localhost:11434",
            help="URL of your Ollama server"
        )
        
        # Get available models
        available_models = []
        try:
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json().get('models', [])
                available_models = [model['name'] for model in models_data]
                # Sort models by size (smaller first) for better memory management
                model_sizes = {
                    'llama3.2:1b': 1.3,
                    'gemma2:2b': 1.6,
                    'llama3.2:3b': 2.0,
                    'llama3:8b': 4.7,
                    'qwen2.5:7b': 4.7,
                    'llava:latest': 4.7,
                    'llama3:70b': 40.0
                }
                available_models.sort(key=lambda x: model_sizes.get(x, 10.0))
            else:
                available_models = ["llama3.2:1b", "llama3.2:3b", "llama3:8b", "qwen2.5:7b", "llava:latest"]
        except:
            available_models = ["llama3.2:1b", "llama3.2:3b", "llama3:8b", "qwen2.5:7b", "llava:latest"]
        
        # Performance monitoring
        try:
            import psutil
            memory = psutil.virtual_memory()
            load_avg = psutil.getloadavg()
            available_gb = memory.available / (1024**3)
            load_1min = load_avg[0]
            
            # Performance status
            if available_gb < 2.0:
                st.error("🚨 CRITICAL: Less than 2GB available memory")
            elif available_gb < 4.0:
                st.warning("⚠️ WARNING: Less than 4GB available memory")
            else:
                st.success("✅ Good: Sufficient memory available")
            
            if load_1min > 4.0:
                st.error("🚨 HIGH CPU LOAD: System is overloaded")
            elif load_1min > 2.0:
                st.warning("⚠️ MODERATE CPU LOAD: System is busy")
            else:
                st.success("✅ Good: CPU load is manageable")
            
            # Show current resources
            st.info(f"📊 Available Memory: {available_gb:.1f}GB | CPU Load: {load_1min:.1f}")
            
        except ImportError:
            st.warning("⚠️ psutil not available for performance monitoring")
        except Exception as e:
            st.warning(f"⚠️ Could not check system resources: {e}")
            available_gb = 8.0  # Default assumption
            load_1min = 1.0     # Default assumption

        # Model selection with recommendations
        if available_models:
            # Recommend the best model based on system performance
            if available_gb < 2.0 or load_1min > 4.0:
                # Critical performance - use smallest model
                if "llama3.2:1b" in available_models:
                    recommended_model = "llama3.2:1b"
                elif "gemma2:2b" in available_models:
                    recommended_model = "gemma2:2b"
                else:
                    recommended_model = available_models[0]
            elif available_gb < 4.0 or load_1min > 2.0:
                # Moderate performance - use small models
                if "llama3.2:1b" in available_models:
                    recommended_model = "llama3.2:1b"
                elif "gemma2:2b" in available_models:
                    recommended_model = "gemma2:2b"
                elif "mistral:7b" in available_models:
                    recommended_model = "mistral:7b"
                else:
                    recommended_model = available_models[0]
            else:
                # Good performance - can use larger models
                if "llama3.2:1b" in available_models:
                    recommended_model = "llama3.2:1b"
                elif "gemma2:2b" in available_models:
                    recommended_model = "gemma2:2b"
                elif "mistral:7b" in available_models:
                    recommended_model = "mistral:7b"
                elif "qwen2.5:7b" in available_models:
                    recommended_model = "qwen2.5:7b"
                else:
                    recommended_model = available_models[0]
            
            model_name = st.selectbox(
                "Model",
                available_models,
                index=available_models.index(recommended_model) if recommended_model in available_models else 0,
                help="Select the local model to use. Smaller models are faster and use less memory."
            )
            
            # Show model recommendations based on performance
            if model_name == "qwen2.5:7b" or model_name == "llava:latest" or model_name == "mistral:7b":
                if available_gb < 4.0 or load_1min > 2.0:
                    st.error("❌ Large model selected on resource-constrained system. May cause timeouts!")
                else:
                    st.warning("⚠️ Large model selected. May be slow and use significant memory.")
            elif model_name == "llama3.2:1b":
                st.success("✅ Efficient model selected. Best for resource-constrained systems!")
            elif model_name == "gemma2:2b":
                st.success("✅ Excellent choice! Gemma2:2b offers great balance of speed and quality!")
            elif model_name == "llama3.2:3b":
                st.info("ℹ️ Good model selected. Balanced performance and quality.")
            else:
                st.info("ℹ️ Model selected. Performance may vary.")
        else:
            model_name = st.selectbox(
                "Model",
                ["llama3.2:1b", "llama3.2:3b", "llama3:8b", "qwen2.5:7b", "llava:latest"],
                help="Select the local model to use"
            )
        
        # Check Ollama connection
        if st.button("🔍 Check Ollama Connection"):
            try:
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    available_models = [model['name'] for model in models]
                    st.success(f"✅ Connected! Available models: {', '.join(available_models)}")
                else:
                    st.error("❌ Ollama server not responding")
            except Exception as e:
                st.error(f"❌ Cannot connect to Ollama: {str(e)}")
                st.info("💡 Make sure Ollama is running: `ollama serve`")
        
        # Test model directly
        if st.button("🧪 Test Model Directly"):
            try:
                test_prompt = "Hello, can you tell me about artificial intelligence?"
                response = requests.post(
                    f"{ollama_base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": test_prompt,
                        "stream": False
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Model test successful!")
                    st.text_area("Model Response:", result.get('response', 'No response'), height=100)
                else:
                    st.error(f"❌ Model test failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Model test error: {str(e)}")
        
        # Document upload
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files"
        )
        
        # Chunk size configuration
        st.header("🔧 Text Processing")
        chunk_size = st.slider("Chunk Size", 1000, 3000, 2000, help="Size of text chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", 100, 600, 400, help="Overlap between chunks")
        
        # Debug mode
        debug_mode = st.checkbox("🐛 Debug Mode", help="Show detailed logging and context information")
        
        # Analysis mode selection
        st.header("📊 Analysis Mode")
        analysis_mode = st.selectbox(
            "Analysis Type",
            ["standard", "cross_document", "individual"],
            format_func=lambda x: {
                "standard": "🔍 Standard RAG (Single Query)",
                "cross_document": "📚 Cross-Document Analysis", 
                "individual": "📄 Individual Document Focus"
            }[x],
            help="Choose how to analyze your documents"
        )
        st.session_state.analysis_mode = analysis_mode
        
        # Process documents button
        if st.button("🔄 Process Documents", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one document")
            else:
                with st.spinner("Processing documents..."):
                    # Initialize embeddings
                    st.session_state.chatbot.initialize_embeddings()
                    
                    # Save uploaded files temporarily and track original names
                    temp_files = []
                    original_names = {}
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_files.append(tmp_file.name)
                            original_names[tmp_file.name] = uploaded_file.name
                    
                    # Load documents with original names
                    documents = st.session_state.chatbot.load_documents(temp_files, original_names)
                    
                    if documents:
                        # Split documents
                        splits = st.session_state.chatbot.split_documents(
                            documents, chunk_size, chunk_overlap
                        )
                        
                        # Create vector store
                        if st.session_state.chatbot.create_vectorstore(splits):
                            # Setup Q&A chain
                            if st.session_state.chatbot.setup_qa_chain(model_name, ollama_base_url):
                                st.session_state.documents_loaded = True
                                st.success(f"✅ Successfully processed {len(documents)} documents!")
                                st.session_state.messages = []  # Clear chat history
                            else:
                                st.error("Failed to setup Q&A chain")
                        else:
                            st.error("Failed to create vector store")
                    
                    # Clean up temporary files
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
        
        # Document Management Section
        if st.session_state.documents_loaded:
            st.header("📁 Document Management")
            
            # Document management buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                if st.button("🗑️ Clear All Documents", type="secondary"):
                    if st.session_state.chatbot.clear_documents():
                        st.session_state.documents_loaded = False
                        st.session_state.messages = []
                        st.success("✅ All documents and database cleared!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear documents")
            
            with col2:
                if st.button("🔄 Refresh Statistics"):
                    st.rerun()
            
            with col3:
                if st.button("🔍 Inspect Database"):
                    try:
                        import os
                        chroma_db_path = "./chroma_db"
                        if os.path.exists(chroma_db_path):
                            st.info("📊 **Database Status:** Found ChromaDB database")
                            st.write(f"**Database Path:** `{chroma_db_path}`")
                            
                            # Show database size
                            import subprocess
                            result = subprocess.run(['du', '-sh', chroma_db_path], capture_output=True, text=True)
                            if result.returncode == 0:
                                st.write(f"**Database Size:** {result.stdout.strip()}")
                            
                            # List database files
                            files = os.listdir(chroma_db_path)
                            st.write(f"**Database Files:** {files}")
                            
                            st.warning("⚠️ **Issue Found:** Old documents are persisting in the database!")
                            st.info("💡 **Solution:** Click 'Clear All Documents' to delete the database completely.")
                        else:
                            st.success("✅ **Database Status:** No persistent database found")
                    except Exception as e:
                        st.error(f"❌ Error inspecting database: {e}")
            
            with col4:
                if st.button("✅ Verify Documents"):
                    try:
                        verification = st.session_state.chatbot.verify_document_content("test question")
                        if "error" not in verification:
                            st.success("✅ **Document Verification Complete**")
                            st.write(f"**Total Documents:** {verification['total_documents']}")
                            
                            # Show document types
                            for doc_type, docs in verification['document_types'].items():
                                st.write(f"**{doc_type.title()}:** {len(docs)} documents")
                                for doc in docs[:2]:  # Show first 2 examples
                                    st.write(f"  - {doc['source']}: {doc['content_preview'][:50]}...")
                        else:
                            st.error(f"❌ Verification failed: {verification['error']}")
                    except Exception as e:
                        st.error(f"❌ Error verifying documents: {e}")
            
            # Show document statistics
            stats = st.session_state.chatbot.get_document_statistics()
            if "error" not in stats:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📁 Files", stats["total_files"])
                with col2:
                    st.metric("✅ Successful", stats["successful_files"])
                with col3:
                    st.metric("📄 Documents", stats["total_documents"])
                with col4:
                    st.metric("🧩 Chunks", stats["total_chunks"])
                
                # Explanation of the difference
                st.info("💡 **Files vs Documents vs Chunks:**\n"
                       "- **Files**: Original uploaded files\n"
                       "- **Documents**: Pages/sections extracted from files\n"
                       "- **Chunks**: Text segments for AI processing")
                
                # Show file types
                if stats["file_types"]:
                    st.write("**📊 File Types:**")
                    for file_type, count in stats["file_types"].items():
                        st.write(f"- {file_type}: {count}")
            
            # Document summaries
            if st.button("📋 Generate Document Summaries"):
                with st.spinner("Generating summaries..."):
                    for file_path in st.session_state.chatbot.processed_files.keys():
                        if st.session_state.chatbot.processed_files[file_path].get('status') == 'loaded':
                            summary = st.session_state.chatbot.get_document_summary(file_path)
                            st.session_state.chatbot.document_summaries[file_path] = summary
            
            # Show document summaries
            if st.session_state.chatbot.document_summaries:
                st.subheader("📄 Document Summaries")
                for file_path, summary in st.session_state.chatbot.document_summaries.items():
                    with st.expander(f"📄 {os.path.basename(file_path)}"):
                        st.write(summary)
    
    # Main chat interface - Full width layout
    st.header("💬 Chat Interface")
    
    # Display chat messages first (in correct order: question above answer)
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show additional info for assistant messages
                if message["role"] == "assistant":
                    if st.session_state.analysis_mode == "cross_document":
                        st.info(f"📚 Cross-Document Analysis")
                    elif st.session_state.analysis_mode == "individual":
                        st.info("📄 Individual Document Analysis")
                    
                    # Debug information
                    if debug_mode:
                        st.info("🐛 Debug Information:")
                        debug_info = {
                            "analysis_mode": st.session_state.analysis_mode,
                            "result_length": len(message["content"])
                        }
                        st.json(debug_info)
    
    # Chat input at the bottom (after messages)
    if prompt := st.chat_input("Ask a question about your documents...", key="main_chat_input"):
        if not st.session_state.documents_loaded:
            st.warning("Please upload and process documents first!")
        else:
            # Add user message to chat history and display immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display the user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response from RAG system based on analysis mode
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Choose analysis method based on mode
                    if st.session_state.analysis_mode == "cross_document":
                        response = st.session_state.chatbot.get_cross_document_analysis(prompt)
                    elif st.session_state.analysis_mode == "individual":
                        # For individual mode, we'll use standard query but show which documents are most relevant
                        response = st.session_state.chatbot.query(prompt)
                    else:  # standard mode
                        response = st.session_state.chatbot.query(prompt)
                    
                    if "error" in response:
                        st.error(f"Error: {response['error']}")
                    else:
                        # Display the response
                        st.markdown(response["result"])
                        
                        # Show analysis mode info
                        if st.session_state.analysis_mode == "cross_document":
                            st.info(f"📚 Cross-Document Analysis")
                        elif st.session_state.analysis_mode == "individual":
                            st.info("📄 Individual Document Analysis")
                        
                        # Debug information
                        if debug_mode:
                            st.info("🐛 Debug Information:")
                            debug_info = {
                                "query": prompt,
                                "result_length": len(response.get("result", "")),
                                "analysis_mode": st.session_state.analysis_mode
                            }
                            st.json(debug_info)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response["result"]})
    
    # Clear chat button (moved to bottom for better UX)
    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", type="secondary"):
                st.rerun()
    
    # Status indicator
    if st.session_state.documents_loaded:
        st.success("✅ Documents loaded - Ready for questions!")
    else:
        st.warning("⚠️ No documents loaded - Upload documents in the sidebar to get started")

if __name__ == "__main__":
    main()
