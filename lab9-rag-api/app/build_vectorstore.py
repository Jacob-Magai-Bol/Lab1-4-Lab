from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_DIR / "outputs" / "document_chunks.json"
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"

COLLECTION_NAME = "rad_hr_policy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    records = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    print(f"Loading {len(records)} document chunks...")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [record["text"] for record in records]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "RAD Human Resource Policy 2021"}
    )

    collection.add(
        ids=[record["id"] for record in records],
        documents=texts,
        metadatas=[record["metadata"] for record in records],
        embeddings=embeddings
    )

    print("\nVector store created successfully.")
    print("Collection:", COLLECTION_NAME)
    print("Stored chunks:", collection.count())
    print("Vector store location:", VECTORSTORE_DIR)


if __name__ == "__main__":
    main()
