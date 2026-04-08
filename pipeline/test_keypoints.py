"""
test_keypoints.py
-----------------
Runs keypoint detection on a video using the configured pose model,
draws the detected pitch keypoints on each frame, and saves the result.

Usage:
    py test_keypoints.py
    py test_keypoints.py --source "path/to/video.mp4" --output "path/to/output.mp4"
    py test_keypoints.py --imgsz 480
"""

import argparse
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import config

# Colour used to draw each keypoint dot
KP_COLOUR    = (0, 255, 0)   # bright green
KP_RADIUS    = 6
KP_THICKNESS = -1             # filled circle

LABEL_COLOUR      = (255, 255, 0)  # yellow text
LABEL_FONT        = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE  = 0.45
LABEL_THICKNESS   = 1

def draw_keypoints(frame: np.ndarray, kpt_result) -> np.ndarray:
    """Draw all detected keypoints directly onto `frame` and return it."""
    annotated = frame.copy()

    if kpt_result.keypoints is None:
        return annotated

    # kpt_result.keypoints.data → shape (n_instances, n_kpts, 3)  [x, y, conf]
    kp_data = kpt_result.keypoints.data.cpu().numpy()

    for instance in kp_data:
        for kp_idx, (x, y, conf) in enumerate(instance):
            if conf < config.CONF_THRESHOLD_KEYPOINTS:
                continue

            cx, cy = int(x), int(y)
            cv2.circle(annotated, (cx, cy), KP_RADIUS, KP_COLOUR, KP_THICKNESS)
            cv2.putText(
                annotated,
                str(kp_idx),
                (cx + KP_RADIUS + 2, cy + KP_RADIUS),
                LABEL_FONT,
                LABEL_FONT_SCALE,
                LABEL_COLOUR,
                LABEL_THICKNESS,
                cv2.LINE_AA,
            )

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Keypoint Detection Test Script")
    parser.add_argument(
        "--source",
        type=str,
        default=r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\raw\prem_vid_short.mp4",
        help="Path to input video file (or 0 for webcam)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\output\keypoint_test_output.mp4",
        help="Where to save the annotated output video",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO input resolution (lower = faster, e.g. 480)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    print(f"📦 Loading Keypoint Model: {config.KEYPOINT_MODEL_PATH}")
    model = YOLO(config.KEYPOINT_MODEL_PATH, task="pose")

    # ------------------------------------------------------------------ #
    # Open video source
    # ------------------------------------------------------------------ #
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"❌ Could not open video source: {args.source}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ------------------------------------------------------------------ #
    # Set up video writer
    # ------------------------------------------------------------------ #
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"🎬 Input : {args.source}  ({width}×{height} @ {fps:.1f} fps, {total} frames)")
    print(f"💾 Output: {args.output}")
    print("🚀 Processing... (press Ctrl+C to abort early)")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model.predict(
                source=frame,
                conf=config.CONF_THRESHOLD_KEYPOINTS,
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            annotated = draw_keypoints(frame, result)

            # Burn frame number + keypoint count onto the frame
            n_instances = len(result.keypoints.data) if result.keypoints else 0
            n_visible   = int((result.keypoints.data[:, :, 2] > config.CONF_THRESHOLD_KEYPOINTS).sum()) \
                          if result.keypoints is not None and len(result.keypoints.data) > 0 else 0
            cv2.putText(
                annotated,
                f"Frame {frame_idx:04d} | Instances: {n_instances} | KPs (conf): {n_visible}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 25 == 0:
                pct = (frame_idx / total * 100) if total > 0 else 0
                print(f"  {frame_idx}/{total} frames ({pct:.1f}%)")

    except KeyboardInterrupt:
        print("\n⚠️  Aborted by user.")

    cap.release()
    writer.release()
    print(f"\n✅ Done! {frame_idx} frames written to:\n   {args.output}")


if __name__ == "__main__":
    main()
