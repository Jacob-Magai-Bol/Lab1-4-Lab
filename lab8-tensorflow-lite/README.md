# LAB 8 — TensorFlow Lite INT8 Quantisation and Android Deployment

## 1. Objective

Convert a pretrained image classification model into TensorFlow Lite format, apply 8-bit integer quantisation, deploy the quantised model on an Android phone, and compare the original and quantised models in terms of model size, inference latency, and classification accuracy.

## 2. Development Environment

| Component | Configuration |
|---|---|
| Model | MobileNetV2 |
| Framework | TensorFlow 2.21.0 |
| Deployment format | TensorFlow Lite |
| Quantisation | Full INT8 |
| Dataset | ImageNette |
| Evaluation images | 500 |
| Images per class | 50 |
| Android device | OPPO A96 |
| Programming language | Kotlin |
| Android UI | Jetpack Compose |
| Minimum Android version | Android 7.0 (API 24) |
| Input dimensions | 224 x 224 x 3 |
| Output classes | 1,000 |

## 3. Model Preparation

A pretrained MobileNetV2 model was downloaded using TensorFlow Keras with ImageNet weights.

Model specifications:

- Parameters: 3,538,984
- Input shape: (None, 224, 224, 3)
- Output shape: (None, 1000)

The original Keras model was saved as:

`models/mobilenetv2_fp32.keras`

## 4. TensorFlow Lite Conversion

Two TensorFlow Lite models were generated:

1. FP32 model: `models/mobilenetv2_fp32.tflite`
2. INT8 model: `models/mobilenetv2_int8.tflite`

The INT8 conversion used 100 representative calibration images from the ImageNette training dataset.

The quantised model was verified to use signed INT8 for both input and output tensors.

Input:
- Shape: [1, 224, 224, 3]
- Data type: INT8

Output:
- Shape: [1, 1000]
- Data type: INT8

## 5. Model Size Comparison

| Model | Size (bytes) | Size (MB) |
|---|---:|---:|
| TensorFlow Lite FP32 | 14,001,044 | 13.35 |
| TensorFlow Lite INT8 | 4,029,824 | 3.84 |

The INT8 model achieved a 71.22% reduction in model size compared with FP32.

## 6. Accuracy and CPU Latency Evaluation

Both models were evaluated using the same 500 ImageNette validation images, with 50 images per class.

| Measurement | FP32 | INT8 |
|---|---:|---:|
| Correct predictions | 348/500 | 343/500 |
| Accuracy | 69.6% | 68.6% |
| CPU p50 latency | 6.429 ms | 4.004 ms |

### Evaluation observations

- INT8 accuracy was 1.0 percentage point lower than FP32.
- INT8 reduced model size by 71.22%.
- INT8 CPU p50 latency was approximately 37.72% lower than FP32.
- The quantised model provided a useful balance between model size, speed, and classification accuracy.

## 7. Android Deployment

The INT8 model was integrated into an Android application developed using Kotlin and Jetpack Compose.

The application was successfully installed and tested on an OPPO A96.

Application features:

1. Select an image from the phone.
2. Resize the image to 224 x 224 pixels.
3. Apply MobileNetV2 preprocessing.
4. Quantise image input using model quantisation parameters.
5. Run TensorFlow Lite inference.
6. Display the predicted ImageNet class name.
7. Display prediction confidence.
8. Measure phone CPU p50 inference latency.

The application performs:
- 10 warm-up inference runs.
- 30 measured inference runs.

The latency measurement covers model inference, not the complete image-loading and preprocessing process.

### ImageNet labels

The application was updated to load all 1,000 ImageNet class names from:

`app/src/main/assets/imagenet_class_index.json`

The updated application successfully built and installed on the OPPO A96. Readable prediction names were verified.

## 8. Android On-Device Test Results

Two earlier phone tests produced the following measurements:

| Test | Prediction output | Confidence | Phone CPU p50 |
|---|---|---:|---:|
| Test 1 | ImageNet class #652 | 8.59% | 39.502 ms |
| Test 2 (dog image) | ImageNet class #235 | 88.67% | 64.919 ms |

After integrating the complete ImageNet label file, the application successfully displayed readable prediction names.

The latest post-update confidence and latency measurements remain to be recorded.

Note: Android measurements are separate from the Windows CPU evaluation. They should not be directly compared as equivalent hardware benchmarks.

## 9. Recommendation

The TensorFlow Lite INT8 model is recommended for Android deployment.

Reasons:

- It is approximately 71.22% smaller than the FP32 model.
- It achieved faster CPU inference in the evaluation.
- Its accuracy was only 1.0 percentage point below FP32 on the 500-image evaluation.
- It was successfully deployed and tested on a physical Android phone.

The FP32 model remains an alternative when preserving the original evaluation accuracy is more important than reducing model size.

## 10. Conclusion

Lab 8 successfully demonstrated the conversion of a pretrained MobileNetV2 image classifier into TensorFlow Lite format, full INT8 quantisation, model evaluation, and Android deployment.

The quantised model achieved a substantial reduction in storage requirements while maintaining competitive classification accuracy and CPU inference speed.

The Android application successfully classified selected images and displayed readable ImageNet prediction names, confidence values, and phone inference latency.

Final outcome: TensorFlow Lite INT8 model successfully deployed and tested on an OPPO A96 Android phone.
