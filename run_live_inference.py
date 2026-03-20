import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path

def run_live_inference(model_path, camera_index=0, frame_skip=1):
    """
    Run live YOLO inference on a camera stream.
    
    Args:
        model_path (str): Path to the YOLO .pt file.
        camera_index (int): Index of the camera (0 for default, or virtual camera index).
        frame_skip (int): Process every Nth frame to save CPU/GPU resources.
    """
    # Load the model
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)

    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return

    print(f"Started live inference on camera {camera_index}.")
    print(f"Processing every {frame_skip} frame(s). Press 'q' to quit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        frame_count += 1
        
        # Only process every Nth frame
        if frame_count % frame_skip == 0:
            # Run inference
            results = model.predict(source=frame, verbose=False)[0]
            
            # Use the built-in plot() method to draw boxes
            annotated_frame = results.plot()
            
            # Display the result
            cv2.imshow("Live YOLO Detection", annotated_frame)
        else:
            # Just show the raw frame if skipping (optional, keeps latency low)
            cv2.imshow("Live YOLO Detection", frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live YOLO Camera Inference")
    parser.add_argument("--model", type=str, default="weights/yolo26_soccernet_best.pt", help="Path to weights file")
    parser.add_argument("--cam", type=int, default=2, help="Camera index (e.g., 0, 1, 2)") # 2 is the virtual camera
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (default 2)")
    
    args = parser.parse_args()
    
    run_live_inference(args.model, args.cam, args.skip)
