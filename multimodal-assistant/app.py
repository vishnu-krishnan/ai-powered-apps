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
    page_title="🤖 Multi-Modal Assistant",
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
/* Make uploader more compact */
div[data-testid="stFileUploader"] {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}
/* Make input area more prominent at bottom */
div[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0;
    background: white;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# Session state setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llava"
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

# Helper functions
def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def process_image_with_llava(image: Image.Image, prompt: str, model: str = "llava") -> str:
    """Process image with LLaVA model via Ollama"""
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
        
        # Make request to Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response received")
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_text_with_llm(message: str, model: str = "llama3") -> str:
    """Process text-only queries with LLM via Ollama"""
    try:
        payload = {
            "model": model,
            "prompt": message,
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response received")
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
                # Check if it's a vision model (contains 'llava', 'vision', etc.)
                if any(keyword in model_name.lower() for keyword in ['llava', 'vision', 'multimodal']):
                    vision_models.append(model_name)
                else:
                    text_models.append(model_name)
            
            return {
                "vision": vision_models,
                "text": text_models
            }
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
    
    # Fallback models
    return {
        "vision": ["llava:latest"],
        "text": ["qwen2.5:7b"]
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

# Main chat interface
st.subheader("💬 Chat Interface")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            if "image" in message:
                # Show small thumbnail in chat history
                st.image(message["image"], width=150, caption="📎 Image")
            st.markdown(message["content"])
        else:
            st.markdown(message["content"])

# Add some space to push input to bottom
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Fixed input area at bottom
st.markdown("---")
st.markdown("**💬 Chat Input**")

# Create a container for the input area at bottom
input_container = st.container()

with input_container:
    # Image upload section (inline with chat)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📎",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf', 'txt', 'docx'],
            help="Upload an image or document (max 10MB)",
            key="chat_file_upload",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Check file size (10MB limit)
            file_size = len(uploaded_file.getvalue())
            max_size = 10 * 1024 * 1024  # 10MB in bytes
            
            if file_size > max_size:
                st.error("File size too large! Please upload a file smaller than 10MB.")
                st.session_state.has_file = False
            else:
                # Determine file type
                file_type = uploaded_file.type
                file_name = uploaded_file.name
                
                if file_type.startswith('image/'):
                    # Handle images
                    st.markdown("📎 **Image attached**")
                    image = Image.open(uploaded_file)
                    st.session_state.current_image = image
                    st.session_state.has_image = True
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
    
    with col2:
        # Chat input area
        if st.session_state.has_file:
            if st.session_state.file_type == "image":
                st.markdown("📎 Image attached - ask about it or chat normally")
            elif st.session_state.file_type == "document":
                st.markdown("📎 Document attached - ask questions about it")
            prompt = st.chat_input("Type a message...", key="main_chat_input")
        else:
            prompt = st.chat_input("Type a message...", key="main_chat_input")

# Process the input
if prompt:
    # Add user message to chat history
    user_message = {"role": "user", "content": prompt}
    if st.session_state.has_image:
        user_message["image"] = st.session_state.current_image
    
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        if st.session_state.has_image:
            # Show small thumbnail instead of full image
            st.image(st.session_state.current_image, width=150, caption="📎 Image")
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.has_image:
                # Use vision model for image + text
                try:
                    response = process_image_with_llava(
                        st.session_state.current_image, 
                        prompt, 
                        vision_model
                    )
                except:
                    # Fallback to text model if vision model fails
                    response = process_text_with_llm(f"User uploaded an image and asked: {prompt}", text_model)
            else:
                # Use text model for text-only
                response = process_text_with_llm(prompt, text_model)
            
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear the uploaded image after processing
    st.session_state.has_image = False
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



