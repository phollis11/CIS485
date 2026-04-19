# Dataset Organization: Football Analytics Project

This document provides a detailed breakdown of the two primary datasets used by the CIS 485 Football AI Pipeline for model training and tactical analysis.

---

## 1. Roboflow Football Pitch Keypoint Dataset

This dataset is used to train the **Pitch Registration** model, allowing the system to understand the camera's perspective and perform 2D tactical mapping.

- **Source**: [Roboflow Universe (Version 15)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi/dataset/15)
- **Task Type**: Keypoint Detection (Pose Estimation)
- **Classes**: 1 (`pitch`)
- **Keypoints**: 32 predefined landmarks.

### Directory Structure
```text
football-field-detection-15/
├── train/ images/ labels/ (annotated .txt files)
├── valid/ images/ labels/
└── test/  images/ labels/
```

### 32-Point Coordinate Mapping
The dataset adheres to a fixed sequence of points representing geometric anchors on the field.

| Index | Landmark | Index | Landmark |
| :--- | :--- | :--- | :--- |
| **0-1** | Left Boundary Corners | **14-16** | Midline Points |
| **2-5** | Left Penalty Box Corners | **17-23** | Center Circle Arcs |
| **6-9** | Left Goal Area Corners | **24-25** | Right Boundary Corners |
| **10** | Left Penalty Spot | **26-29** | Right Penalty Box Corners |
| **11-13** | Left Penalty Arc | **30-31** | Right Goal Area Corners |

> [!TIP]
> **Symmetry Mapping**: The dataset uses a "flip index" to handle horizontal flipping during training:
> `[24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30]`

---

## 2. SoccerNet Player Tracking Dataset (SoccerNet-v3)

The SoccerNet dataset provides the foundational data for the **Player Tracking** model, specifically optimized for multi-object tracking (MOT).

- **Source**: [SoccerNet Benchmarks](https://www.soccer-net.org/tasks/tracking)
- **Task Type**: Multi-Object Tracking (MOT) & Identification.
- **Project Model**: `yolo26n_soccernet_best.pt`

### Organization & Hierarchy
SoccerNet data is organized hierarchically to represent high-level match contexts:

1.  **League**: (e.g., England Premiere League, La Liga).
2.  **Season**: (e.g., 2014-2015, 2015-2016).
3.  **Match**: Individual game directory.
4.  **Clip**: Video chunks containing specific events (kickoff, goals, etc.).

### Annotation Format (`Labels-Tracking.json`)
annotations are provided in a centralized JSON file within each match folder.

```json
{
  "sequence": "match_name",
  "tracklets": [
    {
      "track_id": 1,
      "label": "player",
      "frames": [
        {"frame_id": 0, "bbox": [x, y, w, h], "conf": 1.0},
        ...
      ]
    }
  ]
}
```

### Local Project Class ID Mapping
While SoccerNet has generic classes, this project maps them in `config.py` for consistent detection behavior:

- **Class 0**: `ball`
- **Class 1**: `player`
- **Class 98**: `referee` (Diagnostic ID)
- **Class 99**: `goalkeeper` (Diagnostic ID)

> [!IMPORTANT]
> **SoccerNet Advantage**: Unlike generic person detectors, the SoccerNet dataset is trained on broadcast angles, making it highly robust against rapid camera pans and player occlusions typical in professional match footage.
