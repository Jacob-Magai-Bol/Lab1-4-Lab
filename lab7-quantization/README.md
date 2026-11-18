# Lab 7 — ONNX Export and INT8 Quantisation

## 1. Objective

Export an image classifier to ONNX, quantise it to 8-bit integers, and compare the original FP32 model with the quantised INT8 model.

## 2. Model and Dataset

- Model: ResNet18 pretrained on ImageNet
- Dataset: ImageNette validation dataset
- Evaluation images: 500 (50 images from each of 10 classes)
- Calibration images: 100 training images
- Input size: 224 x 224
- Execution provider: ONNX Runtime CPUExecutionProvider
- Evaluation metric: Top-1 accuracy
- Performance metric: CPU p50 inference latency

The validation images were kept separate from the calibration images.

## 3. Quantisation Method

The FP32 ResNet18 model was exported to ONNX using opset 17.

Static INT8 quantisation was then applied using ONNX Runtime quantisation tools with QDQ format. The quantised model uses INT8 weights and activations.

## 4. Results

| Measurement | FP32 Model | INT8 Model |
|---|---:|---:|
| Model size | 44.58 MB | 11.20 MB |
| CPU p50 latency | 9.123 ms | 22.259 ms |
| Correct predictions | 336 / 500 | 345 / 500 |
| Top-1 accuracy | 67.2% | 69.0% |

### Analysis

- INT8 reduced the model size by approximately 74.9%.
- INT8 achieved 9 more correct predictions than FP32 on the 500-image evaluation.
- INT8 accuracy was 1.8 percentage points higher in this evaluation.
- FP32 had lower p50 latency: 9.123 ms compared with 22.259 ms for INT8.
- INT8 inference was approximately 2.44 times slower than FP32 on the tested CPU.

Quantisation reduced storage requirements but did not improve CPU inference speed in this experiment.

## 5. Recommendation

For applications where CPU inference speed is the main priority, the FP32 model is recommended because it achieved substantially lower p50 latency.

For applications where model storage is more important, the INT8 model is a useful alternative because it reduced the model size by approximately 74.9%.

The INT8 model also achieved slightly higher accuracy in this 500-image test. This observed difference should not be interpreted as proof that quantisation generally improves accuracy.

## 6. Conclusion

The experiment successfully exported ResNet18 to ONNX and produced an INT8 quantised model. Both models were evaluated on the same 500 validation images using ONNX Runtime on the CPU.

The results demonstrate the trade-off between model size and inference latency. The FP32 model was faster, while the INT8 model was considerably smaller.

## 7. Output Files

- `models/resnet18_imagenet.pth` — pretrained PyTorch model
- `models/resnet18_fp32.onnx` — original ONNX model
- `models/resnet18_int8.onnx` — quantised ONNX model
- `outputs/model_metadata.json` — model metadata
- `outputs/evaluation_results.json` — measured evaluation results
- `src/prepare_model.py` — model preparation
- `src/export_onnx.py` — ONNX export
- `src/quantize_model.py` — INT8 quantisation
- `src/evaluate_models.py` — evaluation and benchmarking
