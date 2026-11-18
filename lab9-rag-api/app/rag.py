from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"

COLLECTION_NAME = "rad_hr_policy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
collection = client.get_collection(COLLECTION_NAME)


def retrieve_context(question: str, top_k: int = 5):
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    sources = []

    for document, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        sources.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"]
        })

    return sources


def generate_answer(question: str, sources: list):
    if not sources:
        return "I could not find relevant information in the RAD Human Resource Policy."

    context = "\n\n".join(
        f"Source: {item['source']}, Page: {item['page']}\n{item['text']}"
        for item in sources
    )

    prompt = f"""
You are a question-answering assistant for the Relief Adversity
and Development Organization (RAD).

Answer the user's question using ONLY the policy information
provided in the context below.

Rules:
- Do not invent policy information.
- If the answer is not supported by the context, clearly say:
  "The available policy excerpts do not provide this information."
- Keep the answer clear and direct.
- Refer to the relevant policy page numbers.

POLICY CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def answer_question(question: str):
    sources = retrieve_context(question)
    answer = generate_answer(question, sources)

    unique_sources = []
    seen = set()

    for source in sources:
        key = (source["source"], source["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append({
                "source": source["source"],
                "page": source["page"]
            })

    return {
        "question": question,
        "answer": answer,
        "sources": unique_sources
    }
