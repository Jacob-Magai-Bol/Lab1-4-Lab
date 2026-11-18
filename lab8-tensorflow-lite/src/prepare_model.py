import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("Loading pretrained MobileNetV2...")

model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=True,
    input_shape=(224, 224, 3)
)

model.save(MODEL_DIR / "mobilenetv2_fp32.keras")

print("Model saved successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Model parameters:", model.count_params())
