import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "mobilenetv2_fp32.keras"
DATASET_PATH = ROOT.parent / "lab7-quantization" / "data" / "imagenette2-160" / "train"
OUTPUT_PATH = ROOT / "models" / "mobilenetv2_int8.tflite"

print("Loading MobileNetV2 model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Preparing calibration images...")

image_paths = []
for class_dir in sorted(DATASET_PATH.iterdir()):
    if class_dir.is_dir():
        files = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        image_paths.extend(files[:10])

if len(image_paths) < 100:
    raise RuntimeError(
        f"Expected 100 calibration images, but found {len(image_paths)}."
    )

image_paths = image_paths[:100]

def representative_dataset():
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((224, 224))
            array = np.asarray(image, dtype=np.float32)

        array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
        yield [np.expand_dims(array, axis=0)]

print("Calibration images:", len(image_paths))
print("Converting to full INT8...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
OUTPUT_PATH.write_bytes(tflite_model)

print("INT8 quantisation completed successfully.")
print("Output:", OUTPUT_PATH)
print("Model size:", len(tflite_model), "bytes")
