# 🧠 Local LLM Chat App

A real-time, privacy-preserving chat application built with:

* **Streamlit** for the user interface
* **FastAPI** for the backend API
* **Ollama** to run local Large Language Models (LLMs) like LLaMA3, Mistral, or Phi

> ✅ Fully local & offline | ✅ Private & secure | ✅ Fast | ✅ Free & Open Source

---

## 📌 Problem Statement

Most AI chat interfaces rely on cloud-based APIs, which:

* Compromise user data privacy
* Introduce latency and incur usage costs
* Require constant internet access

**Goal:** Build a lightweight, responsive, local LLM-powered chat app that works entirely on your **Ubuntu 24.04** machine using open-source tools and models.

---

## ⚙️ Tech Stack

| Layer      | Tool                   | Description                             |
| ---------- | ---------------------- | --------------------------------------- |
| UI         | Streamlit              | Fast, interactive Python-based UI       |
| API Server | FastAPI                | Async backend to handle chat requests   |
| LLM Engine | Ollama                 | Run local open-source models            |
| Model      | LLaMA3 / Mistral / Phi | Your choice of local LLM                |
| OS         | Ubuntu 24.04           | Target development and runtime platform |

---

## 📀 Architecture

```
User ────> Streamlit UI ────> FastAPI Backend ────> Ollama API ────> Local LLM
```

* Users interact via the Streamlit UI
* Messages are sent to the FastAPI backend
* Backend forwards them to Ollama
* Ollama responds with generated text
* UI displays the assistant's reply in real-time

---

## 🚀 Setup Instructions

### 1. ✅ Prerequisites

Make sure the following are installed on your system:

* Python 3.10+
* pip
* [Ollama](https://ollama.com) (to run local models)

---

### 2. 📦 Clone the Repository

```bash
git clone https://github.com/your-username/local-llm-chat-app.git
cd local-llm-chat-app
```

---

### 3. 🐍 Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 4. 🤖 Run Ollama with a Local Model

Download and run a model (e.g., LLaMA3):

```bash
ollama run llama3
```

> You can replace `llama3` with any model you prefer (like `mistral` or `phi`).

---

### 5. 🔌 Start the FastAPI Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

---

### 6. 🎨 Launch the Streamlit Frontend

Open a **new terminal** window/tab:

```bash
cd frontend
streamlit run app.py
```

Then visit: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
local-llm-chat-app/
├── backend/             # FastAPI backend
│   └── main.py
├── frontend/            # Streamlit frontend
│   └── app.py
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 💬 Example Chat

**User:**

> How does photosynthesis work?

**Assistant (LLaMA3):**

> Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen...

---

## 🙌 Contributing

Feel free to fork, enhance, or report issues. PRs are welcome!

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
