from ultralytics import YOLO
import os
import cv2
import json
import numpy as np

# 1️⃣ Paths
MODEL_PATH = r'C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\weights\yolo26_pose_train.pt'
VIDEO_PATH = r'C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\raw\prem_vid_short.mp4'
SAVE_VIDEO_PATH = r'C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\annotated_output.mp4'
SAVE_JSON_PATH = r'C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\keypoints.json'

os.makedirs(os.path.dirname(SAVE_VIDEO_PATH), exist_ok=True)

# 2️⃣ Confidence threshold for keypoints
CONF_THRESHOLD = 0.95  # keypoints with confidence below this will be set to 0

# 3️⃣ Load YOLO pose model
print(f"📦 Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task='pose')

# 4️⃣ Prepare video capture
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 Video info: {frame_count} frames, {fps} FPS, {width}x{height}")

# Video writer for annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(SAVE_VIDEO_PATH, fourcc, fps, (width, height))

# 5️⃣ Run inference frame by frame
all_keypoints = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO pose inference
    results = model.predict(source=frame, stream=False, show=False)

    # Annotated frame
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    # Process keypoints
    frame_kpts = []
    for r in results:
        for kpts in r.keypoints:
            kpts_array = kpts.data.cpu().numpy()
            if kpts_array.size == 0:
                continue

            # Reshape and apply confidence filter
            num_kpts = kpts_array.size // 3
            kpts_array = kpts_array.reshape(num_kpts, 3)
            kpts_array[kpts_array[:, 2] < CONF_THRESHOLD, 2] = 0  # set low-confidence points to 0

            # Scale keypoints to original frame size if the model output is resized
            scale_x = width / results[0].orig_shape[1]
            scale_y = height / results[0].orig_shape[0]
            kpts_array[:, 0] *= scale_x
            kpts_array[:, 1] *= scale_y

            frame_kpts.append(kpts_array.tolist())

    all_keypoints.append({
        'frame': frame_idx,
        'keypoints': frame_kpts
    })

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx}/{frame_count} frames...")

cap.release()
out.release()

# 6️⃣ Save keypoints to JSON
with open(SAVE_JSON_PATH, 'w') as f:
    json.dump(all_keypoints, f, indent=4)

print("\n✅ Inference complete!")
print(f"Annotated video saved at: {SAVE_VIDEO_PATH}")
print(f"Keypoints JSON saved at: {SAVE_JSON_PATH}")