import streamlit as st
import requests
import base64
import json
from PIL import Image
import io
from typing import Optional, Dict, Any
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Modal Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS to hide file uploader details and improve layout
st.markdown("""
<style>
/* Hide file uploader details */
div[data-testid="stFileUploader"] > div > div > div > div {
    display: none !important;
}
/* Hide the drag and drop text */
div[data-testid="stFileUploader"] > div > div > div > div > div {
    display: none !important;
}
/* Make uploader more compact and smaller */
div[data-testid="stFileUploader"] {
    border: 1px dashed #ccc;
    border-radius: 5px;
    padding: 5px;
    text-align: center;
    font-size: 12px;
    min-height: 40px;
}
/* Remove the large rectangle box around text input */
div[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
/* Style the textarea directly - Dark mode compatible */
div[data-testid="stChatInput"] textarea {
    border: 1px solid #555 !important;
    box-shadow: none !important;
    font-size: 16px;
    padding: 10px;
    border-radius: 8px;
    outline: none !important;
    background: #2b2b2b !important;
    color: #ffffff !important;
}
/* Remove focus outline and any red borders - Dark mode compatible */
div[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    border: 1px solid #007bff !important;
    box-shadow: none !important;
    background: #2b2b2b !important;
    color: #ffffff !important;
}
/* Dark mode placeholder text */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #888 !important;
}
/* Remove all container styling */
div[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
/* Ensure chat messages stay above input */
div[data-testid="stChatMessage"] {
    margin-bottom: 10px;
}
/* Fix chat container positioning */
div[data-testid="stChatMessage"] > div {
    position: relative;
    z-index: 1;
}
/* Dark mode improvements */
.main .block-container {
    background: transparent;
    padding-bottom: 150px;
}
/* Ensure chat messages are visible in dark mode */
div[data-testid="stChatMessage"] {
    color: inherit;
}
/* Dark mode file uploader styling */
div[data-testid="stFileUploader"] {
    background: #2b2b2b !important;
    border: 1px dashed #555 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# Session state setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "has_image" not in st.session_state:
    st.session_state.has_image = False
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "has_file" not in st.session_state:
    st.session_state.has_file = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
if "image_already_shown" not in st.session_state:
    st.session_state.image_already_shown = False
if "last_uploaded_file_name" not in st.session_state:
    st.session_state.last_uploaded_file_name = None
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# Helper functions
def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def process_image_with_vision_model(image: Image.Image, prompt: str, model: str) -> str:
    """Process image with vision model via Ollama"""
    try:
        # Convert image to base64
        img_base64 = encode_image_to_base64(image)
        
        # Prepare request for Ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        }
        
        # Make request to Ollama with longer timeout for vision models
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=300  # Increased to 5 minutes for vision models
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response received")
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "⏱️ Image processing is taking longer than expected. The vision model is working on your image - please wait a moment and try again, or try with a smaller image."
    except requests.exceptions.ConnectionError:
        return "🔌 Connection error with Ollama server. Please ensure Ollama is running and try again."
    except Exception as e:
        return f"❌ Error processing image: {str(e)}"

def process_text_with_llm(message: str, model: str, conversation_history: list = None) -> str:
    """Process text-only queries with LLM via Ollama with conversation history"""
    try:
        # Build messages array for chat API
        messages = []
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    # Include user messages with or without images
                    if "image" in msg:
                        # For user messages with images, mention the image context
                        messages.append({"role": "user", "content": f"[Image uploaded] {msg['content']}"})
                    else:
                        messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    # Include all assistant responses for context
                    messages.append({"role": "assistant", "content": msg["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "No response received")
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error processing text: {str(e)}"

def get_available_models() -> Dict[str, list]:
    """Get available models from Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("models", [])
            
            # Separate vision and text models
            vision_models = []
            text_models = []
            
            for model in models:
                model_name = model.get("name", "")
                # Check if it's a vision model (contains 'llava', 'vision', 'moondream', etc.)
                if any(keyword in model_name.lower() for keyword in ['llava', 'vision', 'multimodal', 'moondream']):
                    vision_models.append(model_name)
                else:
                    text_models.append(model_name)
            
            return {
                "vision": vision_models,
                "text": text_models
            }
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
    
    # Fallback models - prioritize lighter models for memory efficiency
    return {
        "vision": ["moondream:latest", "llava:7b", "llava:latest"],
        "text": ["llama3.2:1b", "gemma2:2b", "llama3.2:3b"]
    }

# Main App UI
st.title("🤖 Multi-Modal Assistant")
st.markdown("**Chat with AI using both text and images**")

# Sidebar for model selection
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Get available models
    models = get_available_models()
    
    # Vision model selection
    vision_model = st.selectbox(
        "🖼️ Vision Model",
        options=models["vision"],
        index=0 if models["vision"] else 0,
        help="Model for image analysis"
    )
    
    # Text model selection
    text_model = st.selectbox(
        "📝 Text Model", 
        options=models["text"],
        index=0 if models["text"] else 0,
        help="Model for text-only queries"
    )
    
    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("• Upload an image and ask questions about it")
    st.markdown("• Chat normally for text-only conversations")
    st.markdown("• Images are processed with vision models")
    st.markdown("• Try: 'What do you see in this image?'")
    st.markdown("• **Image processing may take 1-2 minutes**")
    st.markdown("• Smaller images process faster")
    st.markdown("• **💾 Smaller models use less memory**")
    st.markdown("• **⚡ Larger models offer better quality**")
    st.markdown("• Close other apps if processing is slow")
    st.markdown("• **🔄 Low memory? Try text-only mode**")
    st.markdown("• **💡 Tip: Stop unused models with `ollama stop`**")
    
    # Add a toggle for text-only mode
    text_only_mode = st.checkbox("📝 Text-Only Mode (Saves Memory)", value=False, help="Disable image processing to save memory")
    if text_only_mode:
        st.session_state.text_only_mode = True
    else:
        st.session_state.text_only_mode = False

# Main chat interface
st.subheader("💬 Chat Interface")

# Create a container for chat messages
chat_container = st.container()

# Display chat history in the container
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                if "image" in message:
                    # Show small thumbnail in chat history
                    st.image(message["image"], width=150, caption="📎 Image")
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])
    
    # Show processing status right after the last user message if awaiting response
    if st.session_state.awaiting_response and len(st.session_state.messages) > 0:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            with st.chat_message("assistant"):
                if "image" in last_message and not st.session_state.get("text_only_mode", False):
                    st.info(f"🔍 Analyzing image with {vision_model}... Please wait...")
                else:
                    st.info("💭 Thinking...")

# Clear chat button - only show if there are messages
if len(st.session_state.messages) > 0:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Fixed input area at bottom
st.markdown("---")
st.markdown("**💬 Chat Input**")

# Create a container for the input area at bottom
input_container = st.container()

with input_container:
    # Layout: Chat input on left (larger), file upload on right (much smaller)
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Chat input area (left side - larger)
        if st.session_state.has_file:
            if st.session_state.file_type == "image":
                st.markdown("📎 Image attached - ask about it or chat normally")
            elif st.session_state.file_type == "document":
                st.markdown("📎 Document attached - ask questions about it")
            prompt = st.chat_input("Type a message...", key="main_chat_input")
        else:
            prompt = st.chat_input("Type a message...", key="main_chat_input")
    
    with col2:
        # File upload section (right side - much smaller)
        uploaded_file = st.file_uploader(
            "📎",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf', 'txt', 'docx'],
            help="Upload file (max 200MB)",
            key="chat_file_upload",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Check file size (200MB limit)
            file_size = len(uploaded_file.getvalue())
            max_size = 200 * 1024 * 1024  # 200MB in bytes
            
            if file_size > max_size:
                st.error("File size too large! Please upload a file smaller than 200MB.")
                st.session_state.has_file = False
            else:
                # Determine file type
                file_type = uploaded_file.type
                file_name = uploaded_file.name
                
                if file_type.startswith('image/'):
                    # Handle images
                    st.markdown("📎 **Image attached**")
                    image = Image.open(uploaded_file)
                    
                    # Check if this is a new image
                    if st.session_state.last_uploaded_file_name != file_name:
                        st.session_state.current_image = image
                        st.session_state.has_image = True
                        st.session_state.image_already_shown = False
                        st.session_state.last_uploaded_file_name = file_name
                    
                    st.session_state.has_file = True
                    st.session_state.file_type = "image"
                elif file_type in ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    # Handle documents
                    st.markdown(f"📎 **Document attached: {file_name}**")
                    st.session_state.current_file = uploaded_file
                    st.session_state.has_file = True
                    st.session_state.file_type = "document"
                    st.session_state.has_image = False
                else:
                    st.error(f"Unsupported file type: {file_type}")
                    st.session_state.has_file = False
        else:
            st.session_state.has_image = False
            st.session_state.has_file = False
            st.session_state.image_already_shown = False
            st.session_state.last_uploaded_file_name = None

# Process awaiting response AFTER chat display
if st.session_state.awaiting_response and len(st.session_state.messages) > 0:
    # Get the last user message
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user":
        prompt = last_message["content"]
        use_vision_model = "image" in last_message and not st.session_state.get("text_only_mode", False)
        
        # Generate assistant response
        if use_vision_model:
            # Use vision model for first query with image
            try:
                response = process_image_with_vision_model(
                    st.session_state.current_image, 
                    prompt, 
                    vision_model
                )
            except Exception as e:
                response = f"⚠️ Vision model error. Using text mode. Error: {str(e)}"
            
        elif st.session_state.has_image and st.session_state.get("text_only_mode", False):
            response = f"📝 **Text-Only Mode Active**: I can't process images in this mode to save memory. Please describe the image in text, and I'll help you with your question: '{prompt}'"
            
        else:
            # Use text model for follow-up questions or text-only queries
            response = process_text_with_llm(prompt, text_model, st.session_state.messages)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.awaiting_response = False
        st.rerun()

# Process the input
if prompt:
    # Add user message to chat history
    user_message = {"role": "user", "content": prompt}
    
    # Only attach image to chat if it's the first time showing it
    if st.session_state.has_image and not st.session_state.image_already_shown:
        user_message["image"] = st.session_state.current_image
        st.session_state.image_already_shown = True
    
    st.session_state.messages.append(user_message)
    st.session_state.awaiting_response = True
    
    # Rerun to show user message, then process on next render
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🤖 Multi-Modal Assistant | Powered by Ollama & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)



