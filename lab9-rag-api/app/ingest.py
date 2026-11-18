import json
from pathlib import Path
from pypdf import PdfReader

PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_DIR / "documents" / "RAD_Human_Resource_Policy_2021.pdf"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "document_chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        for chunk_number, chunk in enumerate(split_text(text), start=1):
            records.append({
                "id": f"page-{page_number}-chunk-{chunk_number}",
                "text": chunk,
                "metadata": {
                    "source": PDF_PATH.name,
                    "page": page_number,
                    "chunk": chunk_number
                }
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"PDF pages processed: {len(reader.pages)}")
    print(f"Text chunks created: {len(records)}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
