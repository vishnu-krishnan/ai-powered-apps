from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import logging
from uuid import uuid4
import traceback
from typing import Optional

# 🚀 FastAPI app setup
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# 🌐 CORS setup (open for dev, restrict in prod)
app.add_middleware(
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
@app.get("/models")
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
@app.post("/chat", response_model=ChatResponse)
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

