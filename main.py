from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import asyncio
import os
from agent import MedicalResearchAgent

app = FastAPI(title="Medical Research Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    openai_api_key: str
    model: str = "gpt-4o-mini"
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join("index.html"))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/models")
async def get_models():
    return {
        "models": [
            {"id": "gpt-4o",      "name": "GPT-4o (best)"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini (recommended)"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo","name": "GPT-3.5 Turbo (cheapest)"},
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        agent = MedicalResearchAgent(
            openai_api_key=request.openai_api_key,
            model=request.model
        )
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        result = await agent.run(messages)
        return ChatResponse(
            response=result["response"],
            tools_used=result.get("tools_used", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        try:
            agent = MedicalResearchAgent(
                openai_api_key=request.openai_api_key,  # ← was groq_api_key
                model=request.model
            )
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            async for chunk in agent.stream(messages):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
