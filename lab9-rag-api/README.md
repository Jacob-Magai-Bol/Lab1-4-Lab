# Lab 9 — Retrieval-Augmented Generation (RAG) API

## 1. Project Overview
This project implements a Retrieval-Augmented Generation (RAG) question-answering API using the RAD Human Resource Policy 2021.

The system retrieves relevant information from the policy and uses a locally hosted Large Language Model (LLM) to generate answers grounded in the document.

The available institutional source document contains 16 pages.

## 2. Technologies Used
- Python 3.13
- FastAPI — REST API
- Ollama — local LLM execution
- Llama 3.2 3B — language model
- ChromaDB — local vector database
- Sentence Transformers — document embeddings
- all-MiniLM-L6-v2 — embedding model
- PyPDF — PDF text extraction
- Uvicorn — API server

## 3. Project Structure
- documents/: RAD HR policy PDF
- app/: document ingestion, vector database, RAG and API code
- tests/: API tests
- evaluation/: ten test questions and evaluation results
- vectorstore/: persistent ChromaDB database
- outputs/: generated outputs
- README.md: project documentation

## 4. Document Processing
Source document: RAD Human Resource Policy 2021
Document length: 16 pages
Chunk size: 1,000 characters
Chunk overlap: 150 characters
Total generated chunks: 56

The extracted chunks were converted into vector embeddings using all-MiniLM-L6-v2 and stored locally in ChromaDB.

## 5. RAG Architecture
1. The user submits a question through FastAPI.
2. The question is converted into an embedding.
3. ChromaDB retrieves the five most relevant document chunks.
4. Retrieved content is supplied to Llama 3.2 3B through Ollama.
5. The model generates an answer using the retrieved information.
6. The API returns the answer and source-page references.

## 6. Installation and Setup
Create and activate the virtual environment:

    python -m venv .venv-lab9
    .\.venv-lab9\Scripts\Activate.ps1

Install dependencies:

    pip install -r .\lab9-rag-api\requirements.txt

Download the model:

    ollama pull llama3.2:3b

Ensure Ollama is running before starting the API.

## 7. Running the Application
From the main project directory:

    python -m uvicorn main:app --app-dir .\lab9-rag-api\app --host 127.0.0.1 --port 8000

Interactive API documentation:

    http://127.0.0.1:8000/docs

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Check API health |
| POST | /ask | Submit a question |

## 8. Evaluation Results
Ten questions were prepared from different sections of the RAD Human Resource Policy.

- Questions submitted: 10
- Successful HTTP responses: 10
- Answers with source references: 10
- Fully correct answers after manual review: 7
- Manually assessed accuracy: 70%

The questions covered probation, vision, mission, core values, nepotism, performance assessments, working hours and overtime, leave, confidentiality, and workplace harassment.

## 9. Limitations
1. The available institutional document contains 16 pages rather than the 20 pages specified in the assignment.
2. Some generated answers omitted relevant policy details.
3. Retrieved source references sometimes included additional pages that were not directly relevant.
4. Answer quality depends on document extraction, chunking, retrieval and language-model generation.
5. The system requires Ollama to be running locally.

## 10. Recommendations
- Improve document chunking and retrieval relevance.
- Adjust the number of retrieved chunks where necessary.
- Improve prompts to encourage complete answers.
- Verify generated answers against the original policy.
- Add more institutional documents to expand the knowledge base.
- Add automated API tests and improve error handling.

## 11. Conclusion
The project successfully implements a local RAG question-answering API using FastAPI, ChromaDB, Sentence Transformers and Ollama.

All ten evaluation questions received successful API responses with source references. Manual review found seven fully correct answers, giving a manually assessed accuracy of 70%.

The project demonstrates how Retrieval-Augmented Generation can retrieve and answer questions from institutional documents while providing references to the original source.
