"""
test_players_teams.py
---------------------
Test script that only runs player detection and team classification on a single image.
"""

import argparse
import os
import cv2
import numpy as np
import torch
import supervision as sv
from sports.common.team import TeamClassifier
from ultralytics import YOLO

import config

def main():
    parser = argparse.ArgumentParser(description="Player and Team Detection Test")
    parser.add_argument(
        "--source",
        default=r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\image_raw\SNMOT-060_000001.jpg",
        help="Path to the input image"
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\output\players_teams_test.jpg",
        help="Path to save the result"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Load player model
    print(f"📦 Loading Player Model: {config.PLAYER_MODEL_PATH}")
    model = YOLO(config.PLAYER_MODEL_PATH)

    # 2. Read image
    frame = cv2.imread(args.source)
    if frame is None:
        print(f"❌ Error: Could not read image {args.source}")
        return

    # 3. Detect objects
    print("🔍 Detecting players...")
    results = model.predict(source=frame, conf=config.CONF_THRESHOLD_PLAYERS, imgsz=640, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filter for players only
    players = detections[detections.class_id == config.PLAYER_ID]
    players = players.with_nms(threshold=0.5) # Clean up overlapping boxes

    print(f"✅ Found {len(players)} players.")

    # 4. Team Classification
    if len(players) >= 2:
        crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🕒 Classifying teams (using {device})...")
        
        # Set HF token for authenticated model requests
        if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN:
            os.environ["HF_TOKEN"] = config.HF_TOKEN
            
        tc = TeamClassifier(device=device)
        tc.fit(crops)
        players.class_id = tc.predict(crops)
        print("✅ Teams assigned.")
    else:
        print("⚠️  Not enough players to classify teams.")
        players.class_id = np.zeros(len(players), dtype=int)

    # 5. Annotation
    # Palette matches the previous cyan style for diagnostics
    # Team 0: Cyan, Team 1: Green, Others: Gold
    palette = sv.ColorPalette.from_hex(["#00FFFF", "#00FF00", "#FFD700"])
    
    ellipse_annotator = sv.EllipseAnnotator(color=palette, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=palette, 
        text_color=sv.Color.BLACK,
        text_position=sv.Position.BOTTOM_CENTER
    )

    annotated = frame.copy()
    annotated = ellipse_annotator.annotate(scene=annotated, detections=players)
    
    labels = [f"Team {'A' if cid == 0 else 'B'}" for cid in players.class_id]
    annotated = label_annotator.annotate(scene=annotated, detections=players, labels=labels)

    # Add info text in Cyan as requested
    info_color = (255, 255, 0) # BGR Cyan
    cv2.putText(
        annotated,
        f"Detected: {len(players)} Players | Teams: Classified",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        info_color,
        2,
        cv2.LINE_AA
    )

    # 6. Save
    cv2.imwrite(args.output, annotated)
    print(f"💾 Result saved to: {args.output}")

if __name__ == "__main__":
    main()
