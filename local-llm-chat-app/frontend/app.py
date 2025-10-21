import streamlit as st
import requests
import uuid

# 📄 Streamlit config
st.set_page_config(page_title="🧠 Chat with AI", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 Let's Chat</h1>", unsafe_allow_html=True)

# 🧠 Session state setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3"
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# 🔄 Fetch model list from backend
@st.cache_data
def fetch_available_models():
    try:
        res = requests.get("http://localhost:8000/models")
        return res.json().get("models", ["llama3"])
    except Exception:
        return ["llama3"]

model_options = fetch_available_models()
default_model = st.session_state.selected_model
default_index = model_options.index(default_model) if default_model in model_options else 0

# 🎛️ Sidebar
with st.sidebar:
    st.subheader("🧠 Choose Model")
    st.session_state.selected_model = st.selectbox(
        "Model",
        options=model_options,
        index=default_index
    )
    st.caption("⚡ Powered by Ollama + FastAPI + Streamlit")

# 💬 Render existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✏️ Chat input (only show if not processing)
if not st.session_state.is_processing:
    user_input = st.chat_input("Type your message...")
else:
    user_input = None

# 🚀 Handle new message
if user_input:
    # Set processing flag
    st.session_state.is_processing = True

    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    "http://localhost:8000/chat",
                    json={
                        "message": user_input,
                        "session_id": st.session_state.session_id,
                        "model": st.session_state.selected_model
                    },
                    timeout=90  # ⬅ Increase if needed
                )
                reply = res.json().get("response", "[Error]: No response field")
            except Exception as e:
                reply = f"[Connection error]: {str(e)}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    # Unset processing flag
    st.session_state.is_processing = False

