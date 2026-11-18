from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import answer_question

app = FastAPI(
    title="RAD Human Resource Policy RAG API",
    description="Question-answering API based on the RAD Human Resource Policy 2021.",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "RAD Human Resource Policy RAG API"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        return answer_question(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(exc)}"
        )
