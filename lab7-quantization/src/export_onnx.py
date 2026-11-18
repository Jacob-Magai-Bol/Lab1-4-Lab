from pathlib import Path

import torch
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=None)
model.load_state_dict(
    torch.load(
        MODEL_DIR / "resnet18_imagenet.pth",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = MODEL_DIR / "resnet18_fp32.onnx"

print("Exporting ResNet18 to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamo=False,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("ONNX export completed successfully.")
print("Model saved to:", onnx_path)
print("Model size (MB):", round(onnx_path.stat().st_size / (1024 * 1024), 2))
