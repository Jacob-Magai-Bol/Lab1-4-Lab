from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "imagenette2-160"
MODEL_DIR = ROOT / "models"

FP32_MODEL = MODEL_DIR / "resnet18_fp32.onnx"
INT8_MODEL = MODEL_DIR / "resnet18_int8.onnx"

# ResNet18 ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = preprocess(image).unsqueeze(0)

        return {"input": tensor.numpy().astype(np.float32)}

    def rewind(self):
        self.index = 0


# Collect training images only for calibration
train_dir = DATA_DIR / "train"

image_paths = sorted([
    path
    for path in train_dir.rglob("*")
    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

random.seed(42)
random.shuffle(image_paths)

calibration_images = image_paths[:100]

if len(calibration_images) < 100:
    raise RuntimeError("Fewer than 100 calibration images were found.")

print("Calibration images:", len(calibration_images))
print("Loading FP32 ONNX model...")

reader = ImageCalibrationReader(calibration_images)

print("Starting static INT8 quantisation...")

quantize_static(
    model_input=str(FP32_MODEL),
    model_output=str(INT8_MODEL),
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)

print("INT8 quantisation completed successfully.")
print("INT8 model saved to:", INT8_MODEL)
print("FP32 size (MB):", round(
    FP32_MODEL.stat().st_size / (1024 * 1024), 2
))
print("INT8 size (MB):", round(
    INT8_MODEL.stat().st_size / (1024 * 1024), 2
))
