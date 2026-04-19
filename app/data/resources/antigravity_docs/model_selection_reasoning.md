# Why YOLO Pose Models are Necessary for Keypoint Datasets

This document explains the technical reasoning behind switching from a standard detection model (e.g., `yolo11n.pt`) to a pose-specific model (`yolo11n-pose.pt`) for your football field detection project.

## 1. Data Structure Difference
YOLO models use different data loaders based on the task type defined in the model's architecture.

- **Detection Models**: Expect 5 columns per object in the label file:
  `class_id  x_center  y_center  width  height`
- **Pose Models**: Expect 5 columns for the box + 3 columns for *each* keypoint ($x, y, \text{visibility}$):
  `class_id  x_center  y_center  width  height  k1_x  k1_y  k1_v  k2_x  k2_y  k2_v ...`

Since your dataset has 32 keypoints, each label line has **101 columns** (5 + 32 * 3).

## 2. The "Coordinate Out of Bounds" Error
When you use a standard detection model on your 101-column pose labels:
1. The detection loader only looks for the first 5 columns.
2. However, because the file format doesn't match its expectation, it often tries to parse the remaining 96 columns as additional bounding boxes or segments.
3. In your labels, the **visibility flag** (the 3rd value of each keypoint) is often `2.0` (meaning "labeled but not visible").
4. Detection coordinates *must* be normalized between `0.0` and `1.0`. When the loader sees a `2.0`, it throws a `ValueError` or `RuntimeError`:
   > `ignoring corrupt image/label: non-normalized or out of bounds coordinates [2. 2.]`

## 3. Why `yolo11n-pose.pt` Fixes This
By using a model with the `-pose` suffix:
- **Correct Parsing**: The internal YAML configuration of the model tells the Ultralytics engine to use the `PoseDataset` loader.
- **Handling Visibility**: The pose loader specifically knows that every 3rd value in the keypoint list is a visibility flag and allows values of `0`, `1`, or `2` without flagging them as "out-of-bounds" coordinates.
- **Keypoint Regression**: Standard detection models lack the "Pose Head" (the part of the neural network) required to actually predict keypoint locations. Even if the data loaded, a detection model would only predict the bounding box and ignore the keypoints entirely.

## 4. Summary
For any project involving "Keypoints," "Landmarks," or "Pose Estimation" (like your football field detector), a **Pose** model is mandatory to ensure the data is read correctly and the network is capable of learning point coordinates.

---
*Created on 2026-03-26 to resolve YOLO training errors.*
