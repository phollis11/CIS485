# Real-Time Multi-Model Optimization Guide

This document explains how to combine **Player Detection** and **Field Keypoint Detection** into a single, high-performance pipeline capable of real-time (30+ FPS) inference.

## 1. Model Architecture & Selection
Running two large models will bottleneck any system. For real-time applications, you must use the most lightweight variants.

| Model Type | Recommended Version | Reason |
| :--- | :--- | :--- |
| **Player Detection** | `yolo11n.pt` (Nano) | Extremely fast; handles 22+ players with ease. |
| **Field Keypoints** | `yolo11n-pose.pt` (Nano) | Optimized for lower latency; 32 points is small enough for Nano. |

---

## 2. Hardware Acceleration (TensorRT)
PyTorch (`.pt`) files are for training. For real-time inference, use **NVIDIA TensorRT**. 

### **Step-by-Step Optimization:**
1.  **Export to ONNX**:
    ```python
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")
    model.export(format="onnx", imgsz=640, dynamic=False)
    ```
2.  **Compile to TensorRT (`.engine`)**:
    Run this on your target machine:
    ```bash
    trtexec --onnx=yolo11n-pose.onnx --saveEngine=yolo11n-pose.engine --fp16
    ```
    *Using **FP16** (Half-Precision) can double your performance with almost zero loss in accuracy.*

---

## 3. Asynchronous Pipeline (Threading)
To prevent the camera from "stuttering" while the AI thinks, use a **Multi-Threaded Architecture**.

### **The "Consumer-Producer" Model:**
-   **Thread A (Producer)**: Constantly reads the latest frame from the video stream into a buffer.
-   **Thread B (Consumer)**: Grabs the *latest available* frame from the buffer, runs both AI models, and calculates homography.
-   **Thread C (Displayer)**: Overlays results (bounding boxes + 2D map) onto the UI.

This ensures that even if inference takes 40ms (25 FPS), the display remains smooth.

---

## 4. Efficient Combined Inference
Instead of running one model after another, use **Model Batching** or **Sequential Inference with Shared Pre-processing**.

### **Shared Pre-processing Logic:**
1.  Resize and Normalize the image **once** on the GPU.
2.  Pass the same pre-processed tensor to both the Detection Engine and the Pose Engine.
3.  Combine the results in the post-processing step.

```python
# Pseudo-code for combined inference
players = player_model.predict(frame, conf=0.4, device='cuda')
field = field_model.predict(frame, conf=0.7, device='cuda')

# Pass both to your Homography Engine
H = calculate_homography(field.keypoints)
top_down_dots = map_players_to_2d(players.boxes, H)
```

---

## 5. Performance Checklist
- [ ] **Export models to `.engine` (TensorRT)**.
- [ ] **Run in FP16 precision**.
- [ ] **Move resizing and normalization to the GPU**.
- [ ] **Use a sub-sampling strategy** (e.g., detect field keypoints every 3 frames, players every 1 frame, as the field doesn't move as fast as the players).

---
**Status**: Research & Findings Complete. Proceed to implementation when ready.
