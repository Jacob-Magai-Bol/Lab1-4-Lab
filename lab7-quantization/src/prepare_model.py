from pathlib import Path
import json

import torch
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

print("Loading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

categories = weights.meta["categories"]

model_path = MODEL_DIR / "resnet18_imagenet.pth"
torch.save(model.state_dict(), model_path)

metadata = {
    "model": "ResNet18",
    "weights": "DEFAULT",
    "dataset": "ImageNette",
    "number_of_classes": 10,
    "image_size": [224, 224],
    "synset_to_imagenet_label": SYNSET_TO_LABEL,
}

metadata_path = OUTPUT_DIR / "model_metadata.json"
metadata_path.write_text(
    json.dumps(metadata, indent=4),
    encoding="utf-8"
)

print("ResNet18 model prepared successfully.")
print("Model saved to:", model_path)
print("Metadata saved to:", metadata_path)
print("Number of mapped classes:", len(SYNSET_TO_LABEL))

for synset, label in SYNSET_TO_LABEL.items():
    print(f"{synset}: ImageNet label {label} ({categories[label]})")
