import argparse
import os
import cv2
import numpy as np
import time
import supervision as sv
import collections
import torch

from video_stream import VideoStream
from ultralytics import YOLO
import config

def main():
    parser = argparse.ArgumentParser(description="Standalone Keypoint Detection Pipeline")
    parser.add_argument("--source", type=str, 
                        default=r"C:\Users\jessi\Videos\city_vs_liverpool.mp4", 
                        help="Path to video or 0 for camera")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input resolution")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD_KEYPOINTS, help="Detection confidence threshold")
    parser.add_argument("--output", type=str,
                        default=r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\output\keypoint_detection_only.mp4",
                        help="Path to save the output video")
    args = parser.parse_args()

    # Set HF token if available
    if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN:
        os.environ["HF_TOKEN"] = config.HF_TOKEN

    # Load only the keypoint model
    print(f"📦 Loading Keypoint Model: {config.KEYPOINT_MODEL_PATH}")
    # task='pose' is essential for keypoint models in Ultralytics
    model = YOLO(config.KEYPOINT_MODEL_PATH, task='pose')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Running inference on: {device}")

    src = int(args.source) if args.source.isdigit() else args.source
    try:
        stream = VideoStream(src)
    except Exception as e:
        print(f"❌ Error opening video stream: {e}")
        return

    # Initialize soccer configuration for line drawing
    from sports.configs.soccer import SoccerPitchConfiguration
    PITCH_CONFIG = SoccerPitchConfiguration()

    # Annotators
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        thickness=2,
        edges=PITCH_CONFIG.edges
    )
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#00FF00'), 
        radius=3
    )
    print("🚀 Starting keypoint detection... Press 'q' to quit.")
    
    fps_window = collections.deque(maxlen=30)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = None

    try:
        while stream.more():
            ret, frame = stream.read()
            if not ret or frame is None:
                break

            # Run Detection
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False
            )[0]

            # Process Keypoints
            key_points = sv.KeyPoints.from_ultralytics(results)
            
            annotated_frame = frame.copy()
            
            if len(key_points.xy) > 0:
                # 1. Annotate Edges (Lines)
                annotated_frame = edge_annotator.annotate(
                    scene=annotated_frame,
                    key_points=key_points
                )
                
                # 2. Annotate Points (Small dots)
                annotated_frame = vertex_annotator.annotate(
                    scene=annotated_frame, 
                    key_points=key_points
                )
                
                # 3. Annotate Labels (Under the points) using cv2 for better control
                for inst_idx in range(len(key_points.xy)):
                    points_xy = key_points.xy[inst_idx]
                    confidences = key_points.confidence[inst_idx] if key_points.confidence is not None else [1.0] * len(points_xy)
                    
                    mask = confidences > args.conf
                    valid_points = points_xy[mask]
                    valid_labels = [str(i) for i, valid in enumerate(mask) if valid]
                    
                    for i, pt in enumerate(valid_points):
                        # Center the text slightly below the point
                        text = valid_labels[i]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.4
                        thickness = 1
                        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                        
                        text_x = int(pt[0] - text_w / 2)
                        text_y = int(pt[1] + text_h + 8) # Offset down
                        
                        cv2.putText(annotated_frame, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Performance Overlay
            fps_window.append(time.perf_counter())
            fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0]) if len(fps_window) > 1 else 0.0
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Keypoints Found: {len(key_points.xy[0]) if len(key_points.xy) > 0 else 0}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display
            display_frame = cv2.resize(annotated_frame, (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2))
            cv2.imshow("Keypoint Detection Only", display_frame)

            # Video Recording
            if writer is None:
                h, w = display_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, max(stream.fps, 25), (w, h))
            writer.write(display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        stream.stop()
        if writer is not None:
            writer.release()
            print(f"💾 Video saved to: {args.output}")
        cv2.destroyAllWindows()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
