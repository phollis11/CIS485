# Pipeline Execution (`pipeline/`)

This directory contains the core backend logic for the 2D Virtual Mapping System. It orchestrates the entire computer vision pipeline, from raw video ingestion to the generation of 2D tactical maps.

## Structure Overview

- **`main_realtime.py`**: The primary execution script. It handles video loading, frame processing, and the orchestration of all models.
- **`models.py`**: A centralized inference engine that loads and runs both the Player/Ball YOLO model and the Pitch Keypoint YOLO model.
- **`app_utils.py`**: Contains helper functions for video processing, team classification (using SigLIP), and homography calculations.
- **`config.py`**: A configuration hub for setting model paths, inference parameters, and file paths.
- **`models_explained.md`**: Detailed documentation explaining the architecture and function of the models used in this pipeline.

## How to Run

From the root project directory (where `requirements.txt` is located), you can run inference on a video using:

```bash
python pipeline/main_realtime.py --source <path_to_video>
```

To use the faster TensorRT engines (if compiled), add the `--trt` flag:

```bash
python pipeline/main_realtime.py --source <path_to_video> --trt
```

For a full list of arguments, run:

```bash
python pipeline/main_realtime.py --help
```
