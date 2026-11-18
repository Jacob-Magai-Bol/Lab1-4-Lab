from pathlib import Path
import json
import random
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "imagenette2-160" / "val"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

FP32_MODEL = MODEL_DIR / "resnet18_fp32.onnx"
INT8_MODEL = MODEL_DIR / "resnet18_int8.onnx"

SYNSET_TO_LABEL = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_validation_images():
    random.seed(42)
    selected = []

    # Select exactly 50 images from each of the 10 classes
    for synset, label in SYNSET_TO_LABEL.items():
        class_dir = DATA_DIR / synset

        images = sorted([
            path for path in class_dir.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if len(images) < 50:
            raise RuntimeError(
                f"Class {synset} has fewer than 50 images."
            )

        random.shuffle(images)

        for path in images[:50]:
            with Image.open(path) as image:
                tensor = preprocess(
                    image.convert("RGB")
                ).numpy().astype(np.float32)

            selected.append((tensor, label))

    return selected


def evaluate_model(model_path, dataset):
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    latencies = []
    correct = 0

    # Warm-up: exclude these runs from latency measurements
    for tensor, _ in dataset[:10]:
        session.run(None, {input_name: tensor[np.newaxis, ...]})

    # Measure all 500 images individually
    for tensor, true_label in dataset:
        start = time.perf_counter()

        output = session.run(
            None,
            {input_name: tensor[np.newaxis, ...]}
        )[0]

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        predicted_label = int(np.argmax(output[0]))

        if predicted_label == true_label:
            correct += 1

    accuracy = (correct / len(dataset)) * 100
    p50_latency = float(np.percentile(latencies, 50))
    size_mb = model_path.stat().st_size / (1024 * 1024)

    return {
        "model": model_path.stem,
        "images_evaluated": len(dataset),
        "correct_predictions": correct,
        "top1_accuracy_percent": round(accuracy, 2),
        "p50_latency_ms": round(p50_latency, 3),
        "model_size_mb": round(size_mb, 2),
    }


print("Loading exactly 500 validation images...")
dataset = load_validation_images()

print("Validation images loaded:", len(dataset))
print("Evaluating FP32 model...")

fp32_results = evaluate_model(FP32_MODEL, dataset)

print("Evaluating INT8 model...")

int8_results = evaluate_model(INT8_MODEL, dataset)

results = {
    "dataset": "ImageNette validation",
    "total_images": 500,
    "images_per_class": 50,
    "execution_provider": "CPUExecutionProvider",
    "results": [fp32_results, int8_results],
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_path = OUTPUT_DIR / "evaluation_results.json"
results_path.write_text(
    json.dumps(results, indent=4),
    encoding="utf-8"
)

print("\nLAB 7 EVALUATION RESULTS")
print("-" * 65)

for result in results["results"]:
    print("\nModel:", result["model"])
    print("Images:", result["images_evaluated"])
    print("Correct predictions:", result["correct_predictions"])
    print("Top-1 accuracy:", result["top1_accuracy_percent"], "%")
    print("CPU p50 latency:", result["p50_latency_ms"], "ms")
    print("Model size:", result["model_size_mb"], "MB")

print("\nResults saved to:", results_path)
