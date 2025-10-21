import streamlit as st
import openai
import whisper
import tempfile
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any
import uuid

# Free LLM imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    FREE_LLM_AVAILABLE = True
except ImportError:
    FREE_LLM_AVAILABLE = False

# Additional free NLP imports
try:
    import spacy
    import nltk
    from textblob import TextBlob
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    import re
    FREE_NLP_AVAILABLE = True
except ImportError:
    FREE_NLP_AVAILABLE = False

# Ollama integration
try:
    import ollama
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="🎙️ Meeting Notes & Action Item Extractor",
    page_icon="🎙️",
    layout="wide"
)

# Session state setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "structured_notes" not in st.session_state:
    st.session_state.structured_notes = {}
if "action_items" not in st.session_state:
    st.session_state.action_items = []

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    """Load Whisper model for speech-to-text conversion"""
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

# Load Free LLM model
@st.cache_resource
def load_free_llm_model():
    """Load a free local LLM model for note structuring"""
    if not FREE_LLM_AVAILABLE:
        return None
    
    try:
        # Use a smaller, faster model for local processing
        model_name = "microsoft/DialoGPT-medium"  # Lightweight model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create text generation pipeline
        generator = pipeline("text-generation", 
                           model=model, 
                           tokenizer=tokenizer,
                           max_length=512,
                           do_sample=True,
                           temperature=0.7)
        return generator
    except Exception as e:
        st.warning(f"Free LLM model could not be loaded: {str(e)}")
        return None

# Ollama model functions
def check_ollama_connection():
    """Check if Ollama is running and available"""
    if not OLLAMA_AVAILABLE:
        return False
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_ollama_models():
    """Get list of available Ollama models"""
    if not check_ollama_connection():
        return []
    
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
    except:
        pass
    return []

def analyze_with_ollama(transcription: str, model_name: str = "llama3.2"):
    """Analyze meeting transcription using Ollama"""
    if not OLLAMA_AVAILABLE or not check_ollama_connection():
        return None
    
    try:
        prompt = f"""
        Analyze the following meeting transcription and extract structured information:

        Transcription:
        {transcription}

        Please provide a JSON response with the following structure:
        {{
            "meeting_title": "Brief title of the meeting",
            "date": "Meeting date if mentioned",
            "participants": ["List of participants mentioned"],
            "key_topics": ["Main topics discussed"],
            "decisions": ["Key decisions made"],
            "summary": "Brief summary of the meeting"
        }}

        Only return valid JSON, no additional text.
        """
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a meeting analysis expert. Extract structured information from meeting transcriptions and return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response['message']['content'].strip()
        # Clean up the response to ensure it's valid JSON
        result = re.sub(r'^```json\s*', '', result)
        result = re.sub(r'\s*```$', '', result)
        
        return json.loads(result)
    except Exception as e:
        st.warning(f"Ollama analysis failed: {str(e)}")
        return None

def extract_action_items_ollama(transcription: str, model_name: str = "llama3.2"):
    """Extract action items using Ollama"""
    if not OLLAMA_AVAILABLE or not check_ollama_connection():
        return []
    
    try:
        prompt = f"""
        Analyze the following meeting transcription and extract action items:

        Transcription:
        {transcription}

        For each action item, provide:
        - task: The specific task to be done
        - assignee: Who is responsible (if mentioned)
        - due_date: When it's due (if mentioned)
        - priority: High/Medium/Low (based on context)
        - status: Open

        Return a JSON array of action items. If no action items are found, return an empty array.
        Only return valid JSON, no additional text.
        """
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert at extracting action items from meeting transcriptions. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response['message']['content'].strip()
        # Clean up the response
        result = re.sub(r'^```json\s*', '', result)
        result = re.sub(r'\s*```$', '', result)
        
        action_items = json.loads(result)
        # Ensure each action item has required fields
        for item in action_items:
            if "task" not in item:
                item["task"] = "Unknown task"
            if "assignee" not in item:
                item["assignee"] = "Not specified"
            if "due_date" not in item:
                item["due_date"] = "Not specified"
            if "priority" not in item:
                item["priority"] = "Medium"
            if "status" not in item:
                item["status"] = "Open"
        
        return action_items
    except Exception as e:
        st.warning(f"Ollama action item extraction failed: {str(e)}")
        return []

# Initialize models
whisper_model = load_whisper_model()
free_llm_model = load_free_llm_model()

# OpenAI API configuration
def setup_openai():
    """Setup OpenAI API key"""
    api_key = st.sidebar.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        help="Enter your OpenAI API key for premium GPT features, or use free local models"
    )
    if api_key:
        openai.api_key = api_key
        return True
    return False

# Audio processing functions
def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe audio file using Whisper"""
    if whisper_model is None:
        return "Error: Whisper model not loaded"
    
    try:
        result = whisper_model.transcribe(audio_file_path)
        return result["text"]
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def structure_meeting_notes_free(transcription: str) -> Dict[str, Any]:
    """Structure meeting notes using free NLP libraries"""
    try:
        # Enhanced free analysis using multiple NLP libraries
        lines = transcription.split('\n')
        
        # Extract participants using NLTK and TextBlob
        participants = []
        key_topics = []
        decisions = []
        
        if FREE_NLP_AVAILABLE:
            try:
                # Use NLTK for better name extraction
                sentences = sent_tokenize(transcription)
                for sentence in sentences:
                    words = word_tokenize(sentence)
                    pos_tags = pos_tag(words)
                    
                    # Extract proper nouns (potential names)
                    for word, pos in pos_tags:
                        if pos == 'NNP' and len(word) > 2 and word[0].isupper():
                            participants.append(word)
                    
                    # Extract topics using TextBlob sentiment and keywords
                    blob = TextBlob(sentence)
                    if any(keyword in sentence.lower() for keyword in ['discuss', 'topic', 'issue', 'problem', 'solution', 'plan', 'project']):
                        if len(sentence) > 10 and len(sentence) < 100:
                            key_topics.append(sentence.strip())
                    
                    # Extract decisions
                    if any(keyword in sentence.lower() for keyword in ['decide', 'decision', 'agree', 'approved', 'concluded', 'resolved']):
                        if len(sentence) > 10 and len(sentence) < 100:
                            decisions.append(sentence.strip())
                            
            except Exception as e:
                st.warning(f"Advanced NLP features not available: {str(e)}")
                # Fallback to basic pattern matching
                pass
        
        # Fallback to basic pattern matching if advanced NLP fails
        if not participants or not key_topics:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Extract participants (names mentioned)
                if any(word in line.lower() for word in ['said', 'mentioned', 'suggested', 'proposed']):
                    words = line.split()
                    for i, word in enumerate(words):
                        if word.lower() in ['said', 'mentioned', 'suggested', 'proposed'] and i > 0:
                            potential_name = words[i-1]
                            if potential_name[0].isupper() and len(potential_name) > 2:
                                participants.append(potential_name)
                
                # Extract key topics
                topic_keywords = ['discuss', 'topic', 'issue', 'problem', 'solution', 'plan', 'project']
                if any(keyword in line.lower() for keyword in topic_keywords):
                    if len(line) > 10 and len(line) < 100:
                        key_topics.append(line.strip())
                
                # Extract decisions
                decision_keywords = ['decide', 'decision', 'agree', 'approved', 'concluded', 'resolved']
                if any(keyword in line.lower() for keyword in decision_keywords):
                    if len(line) > 10 and len(line) < 100:
                        decisions.append(line.strip())
        
        # Create enhanced summary using TextBlob if available
        if FREE_NLP_AVAILABLE:
            try:
                blob = TextBlob(transcription)
                # Get the most important sentences (simple extractive summary)
                sentences = blob.sentences
                summary_sentences = sentences[:3]  # First 3 sentences
                summary = ' '.join([str(s) for s in summary_sentences])
            except:
                # Fallback summary
                sentences = transcription.split('.')
                summary = '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else transcription[:200] + '...'
        else:
            # Basic summary
            sentences = transcription.split('.')
            summary = '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else transcription[:200] + '...'
        
        return {
            "meeting_title": "Meeting Notes",
            "date": "Not specified",
            "participants": list(set(participants))[:5],  # Remove duplicates, limit to 5
            "key_topics": key_topics[:5],  # Limit to 5 topics
            "decisions": decisions[:5],  # Limit to 5 decisions
            "summary": summary
        }
    except Exception as e:
        return {"error": f"Error structuring notes with free model: {str(e)}"}

def structure_meeting_notes(transcription: str) -> Dict[str, Any]:
    """Structure meeting notes using OpenAI GPT, Ollama, or free alternative"""
    if openai.api_key:
        # Use OpenAI if available
        prompt = f"""
        Please analyze the following meeting transcription and extract structured information:

        Transcription:
        {transcription}

        Please provide a JSON response with the following structure:
        {{
            "meeting_title": "Brief title of the meeting",
            "date": "Meeting date if mentioned",
            "participants": ["List of participants mentioned"],
            "key_topics": ["Main topics discussed"],
            "decisions": ["Key decisions made"],
            "summary": "Brief summary of the meeting"
        }}

        Only return valid JSON, no additional text.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a meeting analysis expert. Extract structured information from meeting transcriptions and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            # Clean up the response to ensure it's valid JSON
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'\s*```$', '', result)
            
            return json.loads(result)
        except Exception as e:
            return {"error": f"Error structuring notes with OpenAI: {str(e)}"}
    elif check_ollama_connection():
        # Use Ollama if available
        available_models = get_available_ollama_models()
        if available_models:
            # Try different models in order of preference
            for model in ["qwen2.5:7b", "llava:latest", "llama3.2", "llama3.1", "mistral", "gemma2"]:
                if any(model in available_model for available_model in available_models):
                    result = analyze_with_ollama(transcription, model)
                    if result and not result.get("error"):
                        return result
            # Fallback to first available model
            result = analyze_with_ollama(transcription, available_models[0])
            if result and not result.get("error"):
                return result
        return structure_meeting_notes_free(transcription)
    else:
        # Use free alternative
        return structure_meeting_notes_free(transcription)

def extract_action_items_free(transcription: str) -> List[Dict[str, str]]:
    """Extract action items using free pattern matching"""
    action_items = []
    
    # Keywords that indicate action items
    action_keywords = [
        'action', 'task', 'todo', 'need to', 'should', 'must', 'will do', 
        'follow up', 'next steps', 'assign', 'responsible', 'deadline', 'due'
    ]
    
    # Priority indicators
    high_priority_words = ['urgent', 'asap', 'immediately', 'critical', 'important']
    medium_priority_words = ['soon', 'this week', 'next week']
    low_priority_words = ['eventually', 'when possible', 'low priority']
    
    lines = transcription.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_lower = line.lower()
        
        # Check if line contains action keywords
        if any(keyword in line_lower for keyword in action_keywords):
            # Extract potential assignee (look for names before action words)
            assignee = "Not specified"
            words = line.split()
            for i, word in enumerate(words):
                if word.lower() in ['will', 'should', 'must', 'need'] and i > 0:
                    potential_assignee = words[i-1]
                    if potential_assignee[0].isupper() and len(potential_assignee) > 2:
                        assignee = potential_assignee
                        break
            
            # Determine priority
            priority = "Medium"
            if any(word in line_lower for word in high_priority_words):
                priority = "High"
            elif any(word in line_lower for word in low_priority_words):
                priority = "Low"
            
            # Extract due date (look for date patterns)
            due_date = "Not specified"
            date_patterns = ['today', 'tomorrow', 'next week', 'this week', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
            for pattern in date_patterns:
                if pattern in line_lower:
                    due_date = pattern.title()
                    break
            
            # Clean up the task description
            task = line
            if len(task) > 100:
                task = task[:100] + "..."
            
            action_items.append({
                "task": task,
                "assignee": assignee,
                "due_date": due_date,
                "priority": priority,
                "status": "Open"
            })
    
    return action_items[:10]  # Limit to 10 action items

def extract_action_items(transcription: str) -> List[Dict[str, str]]:
    """Extract action items using OpenAI, Ollama, or free alternative"""
    if openai.api_key:
        # Use OpenAI if available
        prompt = f"""
        Please analyze the following meeting transcription and extract action items:

        Transcription:
        {transcription}

        For each action item, provide:
        - task: The specific task to be done
        - assignee: Who is responsible (if mentioned)
        - due_date: When it's due (if mentioned)
        - priority: High/Medium/Low (based on context)
        - status: Open

        Return a JSON array of action items. If no action items are found, return an empty array.
        Only return valid JSON, no additional text.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting action items from meeting transcriptions. Return only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            # Clean up the response
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'\s*```$', '', result)
            
            action_items = json.loads(result)
            # Ensure each action item has required fields
            for item in action_items:
                if "task" not in item:
                    item["task"] = "Unknown task"
                if "assignee" not in item:
                    item["assignee"] = "Not specified"
                if "due_date" not in item:
                    item["due_date"] = "Not specified"
                if "priority" not in item:
                    item["priority"] = "Medium"
                if "status" not in item:
                    item["status"] = "Open"
            
            return action_items
        except Exception as e:
            return [{"error": f"Error extracting action items with OpenAI: {str(e)}"}]
    elif check_ollama_connection():
        # Use Ollama if available
        available_models = get_available_ollama_models()
        if available_models:
            # Try different models in order of preference
            for model in ["qwen2.5:7b", "llava:latest", "llama3.2", "llama3.1", "mistral", "gemma2"]:
                if any(model in available_model for available_model in available_models):
                    result = extract_action_items_ollama(transcription, model)
                    if result and not any(item.get("error") for item in result):
                        return result
            # Fallback to first available model
            result = extract_action_items_ollama(transcription, available_models[0])
            if result and not any(item.get("error") for item in result):
                return result
        return extract_action_items_free(transcription)
    else:
        # Use free alternative
        return extract_action_items_free(transcription)

def save_meeting_data(transcription: str, structured_notes: Dict, action_items: List[Dict]):
    """Save meeting data to a JSON file"""
    meeting_data = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().isoformat(),
        "transcription": transcription,
        "structured_notes": structured_notes,
        "action_items": action_items
    }
    
    # Create downloadable JSON
    json_str = json.dumps(meeting_data, indent=2)
    return json_str

# Main App UI
st.title("🎙️ Meeting Notes & Action Item Extractor")
st.markdown("**Convert meeting audio into structured notes and actionable tasks**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI API key setup
    openai_available = setup_openai()
    
    if openai_available:
        st.success("✅ OpenAI API configured (Premium features enabled)")
    else:
        st.info("🆓 Using free alternatives (No API key required)")
    
    # Whisper model status
    if whisper_model:
        st.success("✅ Whisper model loaded")
    else:
        st.error("❌ Whisper model failed to load")
    
    # Free LLM status
    if FREE_LLM_AVAILABLE:
        st.success("✅ Free LLM models available")
    else:
        st.warning("⚠️ Free LLM models not available")
    
    # Free NLP status
    if FREE_NLP_AVAILABLE:
        st.success("✅ Advanced NLP libraries available")
    else:
        st.info("ℹ️ Basic pattern matching available")
    
    # Ollama status
    if check_ollama_connection():
        available_models = get_available_ollama_models()
        if available_models:
            st.success(f"✅ Ollama connected ({len(available_models)} models)")
            st.markdown("**Available Models:**")
            for model in available_models[:3]:  # Show first 3 models
                st.markdown(f"• {model}")
        else:
            st.warning("⚠️ Ollama connected but no models found")
    else:
        st.info("ℹ️ Ollama not running (optional for better analysis)")
    
    st.markdown("---")
    st.markdown("**📋 Features:**")
    st.markdown("• 🎙️ Audio transcription with Whisper (Free)")
    if openai_available:
        st.markdown("• 🧠 AI-powered note structuring (Premium)")
        st.markdown("• 📋 Advanced action item extraction (Premium)")
    elif check_ollama_connection():
        st.markdown("• 🦙 Ollama LLM note structuring (Free)")
        st.markdown("• 📋 Ollama action item extraction (Free)")
    else:
        st.markdown("• 📝 Enhanced NLP note structuring (Free)")
        st.markdown("• 📋 Smart action item extraction (Free)")
        st.markdown("• 🔍 NLTK + TextBlob analysis (Free)")
    st.markdown("• 💾 Export to JSON format (Free)")
    
    st.markdown("---")
    st.markdown("**🆓 Free Alternatives Used:**")
    st.markdown("• OpenAI Whisper (Local)")
    st.markdown("• NLTK (Natural Language Toolkit)")
    st.markdown("• TextBlob (Text Processing)")
    st.markdown("• Pattern Matching Algorithms")
    if FREE_LLM_AVAILABLE:
        st.markdown("• Hugging Face Transformers")
    if check_ollama_connection():
        st.markdown("• 🦙 Ollama (Local LLM)")
    
    st.markdown("---")
    st.markdown("**🦙 Ollama Setup (Optional):**")
    st.markdown("1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`")
    st.markdown("2. Pull a model: `ollama pull llama3.2`")
    st.markdown("3. Start Ollama: `ollama serve`")
    st.markdown("4. Refresh this page")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎤 Audio Upload")
    
    # Audio file upload
    uploaded_file = st.file_uploader(
        "Upload meeting audio file",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
        help="Supported formats: MP3, WAV, M4A, FLAC, OGG"
    )
    
    if uploaded_file:
        # Display file info
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Process audio button
        if st.button("🎯 Process Audio", disabled=not whisper_model):
            with st.spinner("🎙️ Transcribing audio..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Transcribe audio
                    transcription = transcribe_audio(tmp_file_path)
                    st.session_state.transcription = transcription
                    
                    if transcription and not transcription.startswith("Error"):
                        st.success("✅ Transcription completed!")
                        
                        # Structure notes (works with or without OpenAI)
                        with st.spinner("🧠 Structuring notes..."):
                            structured_notes = structure_meeting_notes(transcription)
                            st.session_state.structured_notes = structured_notes
                        
                        with st.spinner("📋 Extracting action items..."):
                            action_items = extract_action_items(transcription)
                            st.session_state.action_items = action_items
                        
                        if openai_available:
                            st.success("✅ Analysis completed with premium OpenAI!")
                        elif check_ollama_connection():
                            st.success("✅ Analysis completed with Ollama LLM!")
                        else:
                            st.success("✅ Analysis completed with free alternatives!")
                    else:
                        st.error(f"❌ Transcription failed: {transcription}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

with col2:
    st.subheader("📝 Results")
    
    # Display transcription
    if st.session_state.transcription:
        st.markdown("**🎙️ Transcription:**")
        with st.expander("View full transcription"):
            st.text_area("Transcription", st.session_state.transcription, height=200, disabled=True)
    
    # Display structured notes
    if st.session_state.structured_notes and not st.session_state.structured_notes.get("error"):
        st.markdown("**📊 Structured Notes:**")
        
        notes = st.session_state.structured_notes
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.markdown(f"**Title:** {notes.get('meeting_title', 'N/A')}")
            st.markdown(f"**Date:** {notes.get('date', 'N/A')}")
            st.markdown(f"**Participants:** {', '.join(notes.get('participants', []))}")
        
        with col2b:
            st.markdown("**Key Topics:**")
            for topic in notes.get('key_topics', []):
                st.markdown(f"• {topic}")
        
        st.markdown("**Decisions:**")
        for decision in notes.get('decisions', []):
            st.markdown(f"• {decision}")
        
        st.markdown(f"**Summary:** {notes.get('summary', 'N/A')}")
    
    # Display action items
    if st.session_state.action_items and not st.session_state.action_items[0].get("error"):
        st.markdown("**📋 Action Items:**")
        
        for i, item in enumerate(st.session_state.action_items, 1):
            with st.expander(f"Action Item {i}: {item.get('task', 'Unknown')[:50]}..."):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Task:** {item.get('task', 'N/A')}")
                    st.markdown(f"**Assignee:** {item.get('assignee', 'N/A')}")
                with col_b:
                    st.markdown(f"**Due Date:** {item.get('due_date', 'N/A')}")
                    st.markdown(f"**Priority:** {item.get('priority', 'N/A')}")
                    st.markdown(f"**Status:** {item.get('status', 'N/A')}")

# Export section
if st.session_state.transcription:
    st.subheader("💾 Export Data")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("📄 Export to JSON"):
            json_data = save_meeting_data(
                st.session_state.transcription,
                st.session_state.structured_notes,
                st.session_state.action_items
            )
            
            st.download_button(
                label="⬇️ Download JSON",
                data=json_data,
                file_name=f"meeting_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col4:
        if st.button("🗑️ Clear Session"):
            st.session_state.transcription = ""
            st.session_state.structured_notes = {}
            st.session_state.action_items = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🎙️ Meeting Notes & Action Item Extractor | Powered by Whisper & OpenAI
    </div>
    """,
    unsafe_allow_html=True
)


