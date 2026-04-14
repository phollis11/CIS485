# `models.py` — Dual Inference Engine

## Overview

`models.py` defines the `DualInferenceEngine` class, which acts as a centralized wrapper around the two YOLO models used in the pipeline. Its job is to load both models once at startup and expose a single clean method to run both on any given video frame.

This design keeps all inference logic in one place, so the rest of the pipeline (`main_realtime.py`) never has to interact with Ultralytics directly.

---

## Class: `DualInferenceEngine`

### `__init__(use_tensorrt=False)`

Called once when the pipeline starts. Responsible for:

1. Reading both model file paths from `config.py`
2. Optionally swapping `.pt` paths for compiled `.engine` files if `use_tensorrt=True` was requested (via `--trt` CLI flag)
3. Loading both models into memory using `YOLO()` from Ultralytics

```python
engine = DualInferenceEngine(use_tensorrt=False)
```

The models remain in memory for the entire session — this is intentional. Loading a neural network from disk is expensive (~1–2 seconds each), so we do it once upfront rather than on every frame.

| Attribute | Model | Purpose |
|---|---|---|
| `self.player_model` | `yolo26n_soccernet_best.pt` | Detects players and the ball |
| `self.keypoint_model` | `yolo8n-pose_final_train.pt` | Detects pitch landmark keypoints |

---

### `run_inference(frame, imgsz=640)`

The core method, called on every processed video frame. Runs both YOLO models **sequentially** and returns their results as a tuple.

```python
player_res, kpt_res = engine.run_inference(frame, imgsz=640)
```

**Step 1 — Player Detection:**
Passes the frame through `self.player_model`. Returns bounding boxes with class IDs:
- `class 0` → Ball
- `class 1` → Player

**Step 2 — Keypoint Detection:**
Passes the same frame through `self.keypoint_model`. Returns the (x, y) pixel coordinates and confidence scores for each detected pitch landmark (corners, line intersections, etc.).

The `imgsz` parameter controls the resolution the models process at. Lower values run faster but may reduce accuracy for small or distant objects.

| `imgsz` | Speed | Accuracy |
|---|---|---|
| 640 (default) | Moderate | Good |
| 480 | Faster | Slightly reduced |
| 320 | Fastest | Noticeably reduced |

Both models run **on the CPU** unless TensorRT engines have been compiled (GPU only).

---

### `_try_get_engine_path(pt_path)`

A private helper used internally when `use_tensorrt=True`. It checks whether a compiled `.engine` file exists alongside the `.pt` weight file.

- If found → returns the `.engine` path (GPU-optimized)
- If not found → prints a warning and falls back to the `.pt` file

---

### `export_to_tensorrt()`

Triggered by running the pipeline with the `--export` flag:

```powershell
py main_realtime.py --export
```

Uses Ultralytics to compile both `.pt` files into TensorRT `.engine` files. Once compiled, running with `--trt` routes inference through the GPU engine, providing a significant FPS increase. **Requires a CUDA-enabled GPU.**

---

## Execution Flow

```
Pipeline Start
      │
      ▼
DualInferenceEngine.__init__()
  ├── Load player_model  (yolo26n_soccernet_best.pt)
  └── Load keypoint_model (yolo8n-pose_final_train.pt)
      │
      ▼  (repeats each processed frame)
DualInferenceEngine.run_inference(frame)
  ├── player_model.predict(frame)   →  player/ball bounding boxes
  └── keypoint_model.predict(frame) →  pitch keypoint coordinates
      │
      ▼
Results returned to main_realtime.py for tracking,
team classification, homography, and minimap rendering
```

---

## Key Design Decisions

- **Single responsibility:** `models.py` only handles loading and inference. All downstream logic (tracking, team assignment, rendering) lives in `main_realtime.py`.
- **Sequential execution:** Both models run one after the other on the same thread. A future optimization could be multi-threaded parallel inference if both models could share GPU memory.
- **Config-driven paths:** Model file paths live in `config.py` and are never hardcoded here, making it easy to swap models without touching inference logic.
