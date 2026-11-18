import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "lab7-quantization" / "data" / "imagenette2-160" / "val"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ImageNet class indices for the 10 ImageNette classes
CLASS_MAP = {
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

def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32)

    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)

# Select exactly 50 images from each validation class
samples = []

for synset, class_index in CLASS_MAP.items():
    class_dir = DATASET / synset
    files = sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if len(files) < 50:
        raise RuntimeError(
            f"Only {len(files)} images found in {synset}; expected at least 50."
        )

    for path in files[:50]:
        samples.append((path, class_index))

print("Total evaluation images:", len(samples))

def evaluate_model(model_name, model_path, int8=False):
    print(f"\nEvaluating {model_name}...")

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1
    )
    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    correct = 0
    latencies = []

    for index, (path, true_class) in enumerate(samples):
        image = load_image(path)

        if int8:
            scale, zero_point = input_info["quantization"]
            image = np.round(image / scale + zero_point)
            image = np.clip(image, -128, 127).astype(np.int8)
        else:
            image = image.astype(np.float32)

        # Warm up the first 10 images; do not include them in latency.
        if index < 10:
            interpreter.set_tensor(input_info["index"], image)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_info["index"])
        else:
            interpreter.set_tensor(input_info["index"], image)

            start = time.perf_counter_ns()
            interpreter.invoke()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000

            latencies.append(elapsed)
            prediction = interpreter.get_tensor(output_info["index"])

        predicted_class = int(np.argmax(prediction))
        if predicted_class == true_class:
            correct += 1

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(samples)} images")

    accuracy = correct / len(samples) * 100
    p50 = float(np.percentile(latencies, 50))

    size_bytes = model_path.stat().st_size

    result = {
        "model": model_name,
        "correct_predictions": correct,
        "total_images": len(samples),
        "accuracy_percent": round(accuracy, 2),
        "cpu_latency_p50_ms": round(p50, 3),
        "model_size_bytes": size_bytes,
        "model_size_mb": round(size_bytes / (1024 * 1024), 2),
    }

    print("Results:", result)
    return result

fp32 = evaluate_model(
    "MobileNetV2 TFLite FP32",
    MODELS / "mobilenetv2_fp32.tflite"
)

int8 = evaluate_model(
    "MobileNetV2 TFLite INT8",
    MODELS / "mobilenetv2_int8.tflite",
    int8=True
)

comparison = {
    "dataset": "ImageNette validation",
    "evaluation_images": len(samples),
    "images_per_class": 50,
    "results": [fp32, int8],
    "accuracy_difference_percentage_points": round(
        int8["accuracy_percent"] - fp32["accuracy_percent"], 2
    ),
    "size_reduction_percent": round(
        (1 - int8["model_size_bytes"] / fp32["model_size_bytes"]) * 100,
        2
    ),
}

output_file = OUTPUTS / "evaluation_results.json"
output_file.write_text(json.dumps(comparison, indent=4))

print("\n========== FINAL COMPARISON ==========")
print(json.dumps(comparison, indent=4))
print("\nResults saved to:", output_file)
