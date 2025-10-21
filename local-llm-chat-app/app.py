import streamlit as st
import requests
import uuid
import threading
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import logging
from uuid import uuid4
import traceback
from typing import Optional
import uvicorn

# 🚀 FastAPI app setup
fastapi_app = FastAPI()
logging.basicConfig(level=logging.INFO)

# 🌐 CORS setup (open for dev, restrict in prod)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 In-memory session chat history
session_memory = {}

# 📦 Request and response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = "llama3"  # default model

class ChatResponse(BaseModel):
    response: str
    session_id: str

# 🔍 Get list of available models
@fastapi_app.get("/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("http://localhost:11434/api/tags")
            models_data = response.json()
            model_names = [m["name"] for m in models_data.get("models", [])]
            return {"models": model_names}
    except Exception as e:
        logging.error(f"[Model Fetch Error] {str(e)}")
        return {"models": ["llama3"]}  # fallback default

# 💬 Chat endpoint
@fastapi_app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid4())
    logging.info(f"[REQ] Session: {session_id} | Model: {req.model} | Message: {req.message}")

    history = session_memory.get(session_id, [])
    history.append({"role": "user", "content": req.message})

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            ollama_resp = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": req.model,
                    "messages": history
                }
            )

        # NDJSON streaming support
        lines = ollama_resp.text.strip().splitlines()
        full_response = ""
        for line in lines:
            try:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    full_response += data["message"]["content"]
            except json.JSONDecodeError:
                continue

        response = full_response.strip() or "[Empty response]"
        history.append({"role": "assistant", "content": response})
        session_memory[session_id] = history

        logging.info(f"[RES] Session: {session_id} | Response: {response}")
        return ChatResponse(response=response, session_id=session_id)

    except Exception as e:
        logging.error("[ERROR] " + traceback.format_exc())
        return ChatResponse(
            response=f"[Error]: Unable to process message. Details: {str(e)}",
            session_id=session_id
        )

# 🚀 Start FastAPI server in a separate thread
def start_fastapi_server():
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="info")

# 📄 Streamlit config
st.set_page_config(
    page_title="Buddy AI", 
    page_icon="😊", 
    layout="centered"
)
st.markdown("<h1 style='text-align: center; color: #2E8B57; font-family: Arial, sans-serif; margin-bottom: 10px;'>Hi there! I'm Buddy AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 16px; margin-top: 0;'>Your Private Local AI Companion</p>", unsafe_allow_html=True)

# 🧠 Session state setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3"
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "fastapi_started" not in st.session_state:
    st.session_state.fastapi_started = False

# 🚀 Start FastAPI server if not already started
if not st.session_state.fastapi_started:
    st.session_state.fastapi_started = True
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=start_fastapi_server, daemon=True)
    fastapi_thread.start()
    # Give the server a moment to start
    time.sleep(2)

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
    st.subheader("Choose LLM Model")
    st.session_state.selected_model = st.selectbox(
        "Model",
        options=model_options,
        index=default_index
    )
    st.caption("⚡ Powered by Ollama + FastAPI + Streamlit")
    
    # Add a button to clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

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
