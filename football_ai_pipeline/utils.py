import cv2
import numpy as np
import imageio

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, (width, height), total_frames

def save_video(frames, output_path, fps, size):
    # Enforces web-friendly H.264 encoding via imageio
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=None)
    for frame in frames:
        # Streamlit/Browsers expect RGB, whereas OpenCV processes in BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)
    writer.close()
