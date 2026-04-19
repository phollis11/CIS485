# Implementation Plan: 2D Field Mapping via Homography

This document outlines the professional-grade implementation strategy for translating broadcast football footage into a 2D tactical view, incorporating best practices from Roboflow and SoccerNet benchmarks.

## 1. Full Real-World Coordinate Reference (105m x 68m)
Origin `(0,0)` is the **Top-Left Corner**.
X-axis: Long sideline (105m).
Y-axis: Short goal line (68m).

| Index | Landmark Name | X (meters) | Y (meters) | Region |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Top-Left Corner | 0.0 | 0.0 | Boundary |
| **1** | Bottom-Left Corner | 0.0 | 68.0 | Boundary |
| **2** | Left Penalty Top-Sideline | 0.0 | 13.85 | Left |
| **3** | Left Penalty Bottom-Sideline | 0.0 | 54.15 | Left |
| **4** | Left Penalty Top-Inner | 16.5 | 13.85 | Left |
| **5** | Left Penalty Bottom-Inner | 16.5 | 54.15 | Left |
| **6** | Left Goal Area Top-Sideline | 0.0 | 24.85 | Left |
| **7** | Left Goal Area Bottom-Sideline | 0.0 | 43.15 | Left |
| **8** | Left Goal Area Top-Inner | 5.5 | 24.85 | Left |
| **9** | Left Goal Area Bottom-Inner | 5.5 | 43.15 | Left |
| **10** | Left Penalty Spot | 11.0 | 34.0 | Left |
| **11-13** | Left Penalty Arc (Points) | 16.5-20.1 | 25.0-43.0 | Left |
| **14-16** | Midline (Top, Center, Bottom)| 52.5 | 0.0-68.0 | Center |
| **17-23** | Center Circle (Arc Points) | 43.3-61.6 | 24.8-43.1 | Center |
| **24** | Top-Right Corner | 105.0 | 0.0 | Boundary |
| **25** | Bottom-Right Corner | 105.0 | 68.0 | Boundary |
| **26-31** | Right Penalty & Goal Features | Symmetrical | Symmetrical| Right |

---

## 2. Advanced Implementation Best Practices

### A. Confidence-Based Filtering
To prevent "ghost" keypoints (predicted but not truly visible) from ruining the map:
1.  **Thresholding**: Only use keypoints with `conf > 0.5`.
2.  **Minimum Points**: If a frame detection yields fewer than **4 high-confidence points**, do not update the Homography Matrix.

### B. Robust Homography Calculation
Instead of a standard transform, use **RANSAC** (Random Sample Consensus) to calculate the matrix `H`.
*   **Why?**: RANSAC identifies "outlier" points (points that the model misidentified) and excludes them from the calculation.
*   **OpenCV Command**: `H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)`

### C. Bottom-Center (Feet) Projection
When mapping players or referees to the 2D view:
- **Do not** use the center of the bounding box.
- **Do** use the **Bottom-Center** point of the detected box (e.g., `x_mid, y_max`). This represents the point where the object is actually touching the surface of the pitch.

### D. Temporal Stability (Smoothing)
*   **Matrix Persistence**: If the model fails to detect enough points in a single frame (due to motion blur or zoom), "carry over" the last valid Homography Matrix `H` for up to 5-10 frames.
*   **Interpolation**: Smooth the coordinates of the 2D dots over time to prevent flickering on the tactical map.

---

## 3. Implementation Workflow (Python)

```python
import cv2
import numpy as np

# 1. Prepare Points
src_pts = np.array([pt_pixels for pt in frame_kpts if pt.conf > 0.5])
dst_pts = np.array([pt_meters for pt in relevant_meter_coords])

# 2. Compute Matrix
if len(src_pts) >= 4:
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    last_valid_H = H
else:
    H = last_valid_H # Fallback to previous frame's perspective

# 3. Project a Player (from Image to Meters)
player_pixel = np.array([x, y, 1]) # Bottom-center of bounding box
real_coord = np.dot(H, player_pixel)
real_x, real_y = real_coord[0]/real_coord[2], real_coord[1]/real_coord[2]
```

---

## 4. Visualization: The Mini-Map
To display the results, the script should:
1.  Create a blank **green image** (aspect ratio 105:68).
2.  Draw the pitch lines once using the meter-coordinates.
3.  Each frame, draw circles at the calculated `(real_x, real_y)` for each player.
4.  Overlay this mini-map onto the original video corner.

---
**Status**: Pending Detailed Code Implementation.
