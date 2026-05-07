# Medical Research Agent

A production-grade AI medical research agent built with:
- **LangGraph** — agent state machine with tool orchestration
- **LangChain** — LLM chains and tool wrappers
- **FastAPI** — async REST API backend
- **Groq** — free LLM inference (Llama 3.3 70B, Mixtral, etc.)
- **Vanilla JS frontend** — single HTML file, no build step needed

---

## Features

- Search PubMed for real medical literature (NCBI API)
- Drug interaction lookup (RxNorm + ONCHigh database)
- FDA drug information (openFDA API)
- Web search via DuckDuckGo (no API key needed)
- Multi-session chat with history
- Streaming support
- Export conversations

---

## Project Structure

```
medical-agent/
├── backend/
│   ├── main.py          # FastAPI app + routes
│   ├── agent.py         # LangGraph agent + tools
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html       # Single-file React-less frontend
```

---

## Setup

### 1. Get a free Groq API key
Go to https://console.groq.com → sign up → create API key

### 2. Install backend dependencies

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at http://localhost:8000  
Swagger docs at http://localhost:8000/docs

### 4. Open the frontend

Just open `frontend/index.html` in your browser — no server needed.

Then:
1. Enter your Groq API key in the sidebar
2. Select a model (Llama 3.3 70B recommended)
3. Start asking medical research questions!

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Status check |
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/chat` | Send message, get response |
| POST | `/chat/stream` | Streaming response (SSE) |

### Example request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What are the side effects of metformin?"}],
    "groq_api_key": "gsk_your_key_here",
    "model": "llama-3.3-70b-versatile"
  }'
```

---

## Agent Tools

| Tool | Description | Data Source |
|------|-------------|-------------|
| `search_pubmed` | Search medical literature | NCBI PubMed API (free) |
| `search_web` | Search for guidelines/news | DuckDuckGo (free) |
| `lookup_drug_interactions` | Check drug-drug interactions | RxNorm + ONCHigh (free) |
| `get_drug_info` | Drug pharmacology details | openFDA (free) |

---

## Disclaimer

This tool is for **educational and research purposes only**. Always consult a qualified healthcare professional for medical decisions.
