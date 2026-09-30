from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import graph

app = FastAPI(title="Agentic AI RAG Chatbot")


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke({
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0
    })

    return {
        "query": request.query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result["context"],
        "confidence_score": result["score"]
    }