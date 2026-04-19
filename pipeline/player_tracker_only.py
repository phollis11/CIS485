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
    parser = argparse.ArgumentParser(description="Standalone Player Tracking Pipeline")
    parser.add_argument("--source", type=str, 
                        default=r"C:\Users\jessi\Videos\city_vs_liverpool.mp4", 
                        help="Path to video or 0 for camera")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input resolution")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD_PLAYERS, help="Detection confidence threshold")
    parser.add_argument("--output", type=str,
                        default=r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\output\player_tracking_only.mp4",
                        help="Path to save the output video")
    args = parser.parse_args()

    # Set HF token if available to prevent unauth warnings
    if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN:
        os.environ["HF_TOKEN"] = config.HF_TOKEN

    # Load only the player model
    print(f"📦 Loading Player Model: {config.PLAYER_MODEL_PATH}")
    model = YOLO(config.PLAYER_MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Running inference on: {device}")

    src = int(args.source) if args.source.isdigit() else args.source
    try:
        stream = VideoStream(src)
    except Exception as e:
        print(f"❌ Error opening video stream: {e}")
        return

    # Annotators
    palette = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700', '#FFFFFF'])
    ellipse_annotator = sv.EllipseAnnotator(color=palette, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=palette,
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )

    tracker = sv.ByteTrack()
    tracker.reset()

    print("🚀 Starting player tracking... Press 'q' to quit.")
    
    frame_count = 0
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

            detections = sv.Detections.from_ultralytics(results)
            
            # Filter out the ball (ID 0) and track everyone else (Players, Refs, Goalies)
            people_detections = detections[detections.class_id != config.BALL_ID]
            people_detections = people_detections.with_nms(threshold=0.5, class_agnostic=True)
            
            # Update Trackers
            people_detections = tracker.update_with_detections(detections=people_detections)

            # Prepare Labels
            labels = []
            if people_detections.tracker_id is not None:
                labels = [f"#{t_id}" for t_id in people_detections.tracker_id]
            else:
                labels = [""] * len(people_detections)

            # Annotate Frame
            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=people_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=people_detections, labels=labels)

            # Performance Overlay
            fps_window.append(time.perf_counter())
            fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0]) if len(fps_window) > 1 else 0.0
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f} | People Tracked: {len(people_detections)}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Resize for display (smaller window)
            display_frame = cv2.resize(annotated_frame, (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2))
            cv2.imshow("Player Tracking Only", display_frame)

            # Video Recording
            if writer is None:
                h, w = display_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, max(stream.fps, 25), (w, h))
            writer.write(display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1

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
