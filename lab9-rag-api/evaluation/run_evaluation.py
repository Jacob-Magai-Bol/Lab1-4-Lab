import json
from pathlib import Path

import requests
from test_questions import QUESTIONS

API_URL = "http://127.0.0.1:8000/ask"
OUTPUT_FILE = Path(__file__).resolve().parent / "evaluation_results.json"


def main():
    results = []

    for number, question in enumerate(QUESTIONS, start=1):
        print(f"\nQuestion {number}/10: {question}")

        try:
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=240
            )
            response.raise_for_status()
            result = response.json()

            results.append({
                "number": number,
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "http_status": response.status_code
            })

            print("Answer:", result["answer"])
            print("Sources:", result["sources"])

        except requests.RequestException as exc:
            print("Request failed:", exc)
            results.append({
                "number": number,
                "question": question,
                "error": str(exc)
            })

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nEvaluation completed.")
    print("Questions submitted:", len(QUESTIONS))
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
