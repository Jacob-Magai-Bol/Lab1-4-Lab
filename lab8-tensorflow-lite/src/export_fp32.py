import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mobilenetv2_fp32.keras"
OUTPUT_PATH = ROOT / "models" / "mobilenetv2_fp32.tflite"

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Converting model to TensorFlow Lite FP32...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

OUTPUT_PATH.write_bytes(tflite_model)

print("FP32 TensorFlow Lite model exported successfully.")
print("Output:", OUTPUT_PATH)
print("Model size:", len(tflite_model), "bytes")
