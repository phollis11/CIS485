import cv2
import os
from pathlib import Path

def transcode_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Skip: {input_path} not found.")
        return

    print(f"Transcoding {input_path}...")
    cap = cv2.VideoCapture(str(input_path))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    
    # We'll write to a temp file first to avoid overwriting issues
    temp_output = str(output_path) + ".tmp.mp4"
    out = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Progress: {frame_count}/{total_frames} frames", end="\r")
            
    cap.release()
    out.release()
    
    # Replace the old file with the new one
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_output, output_path)
    print(f"\nDone: {output_path}")

def main():
    # Define paths consistent with the app
    DEMO_DIR = Path(r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\app\data\resources\demo_videos")
    
    videos = [
        "long_demo_output.mp4",
        "player_tracking_only.mp4",
        "keypoint_detection_only.mp4"
    ]
    
    for vid_name in videos:
        path = DEMO_DIR / vid_name
        transcode_video(path, path)

if __name__ == "__main__":
    main()
