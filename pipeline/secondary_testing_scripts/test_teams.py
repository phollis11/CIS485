import cv2
import numpy as np
import torch
import supervision as sv
from sports.common.team import TeamClassifier
from models import DualInferenceEngine
import config
import os

def main():
    print("Testing TeamClassifier isolation...")
    engine = DualInferenceEngine(use_tensorrt=False)
    
    vid_path = r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\raw\prem_vid_short.mp4"
    cap = cv2.VideoCapture(vid_path)
    
    crops = []
    unique_classes = set()
    
    print("Scanning frames for players...")
    for _ in range(120): # scan up to 120 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        player_res, _ = engine.run_inference(frame)
        detections = sv.Detections.from_ultralytics(player_res)
        
        if len(detections.class_id) > 0:
            unique_classes.update(detections.class_id.tolist())
            
        players = detections[detections.class_id == config.PLAYER_ID]
        
        frame_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
        crops.extend(frame_crops)
        
        if len(crops) >= 15:
            break
            
    cap.release()
    print("Classes found in stream:", unique_classes)
    
    print(f"Found {len(crops)} player crops.")
    if len(crops) < 2:
        print("Still not enough players found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing TeamClassifier on {device}...")
    
    # Set HF token for authenticated model requests
    if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN:
        os.environ["HF_TOKEN"] = config.HF_TOKEN
        
    tc = TeamClassifier(device=device)
    
    print("Fitting Classifier on crops...")
    try:
        tc.fit(crops)
        print("✅ Fit complete.")
        
        preds = tc.predict(crops)
        print("✅ Predictions successfully retrieved:", preds)
    except Exception as e:
        print("❌ Error during fit/predict:", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
