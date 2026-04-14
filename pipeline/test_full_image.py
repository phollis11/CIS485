"""
test_full_image.py
------------------
Unified test script for processing a single image through the complete tactical pipeline:
  - Player & Ball Detection (YOLO)
  - Pitch Keypoint Detection (YOLO Pose)
  - Team Classification (HuggingFace + K-Means)
  - Goalkeeper Resolution (Geometric Heuristic)
  - Homography & 2D Mapping (ViewTransformer)
  - Side-by-side Minimap Rendering

Enhanced with high-visibility Cyan diagnostics as requested.
"""

import argparse
import os
import cv2
import numpy as np
import torch
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.common.team import TeamClassifier

from models import DualInferenceEngine
import config

# Color theme
CYAN = (255, 255, 0) # BGR
BLACK = (0, 0, 0)

def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    """Assign goalkeepers to the closest team based on spatial centroids."""
    if len(goalkeepers) == 0:
        return np.array([])
    if len(players) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
        
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    team_0_players = players_xy[players.class_id == 0]
    team_1_players = players_xy[players.class_id == 1]
    
    team_0_centroid = team_0_players.mean(axis=0) if len(team_0_players) > 0 else np.array([0, 0])
    team_1_centroid = team_1_players.mean(axis=0) if len(team_1_players) > 0 else np.array([0, 0])
    
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id, dtype=int)

def main():
    parser = argparse.ArgumentParser(description="Full Tactical Image Test")
    parser.add_argument(
        "--source", 
        type=str, 
        default=r"C:\Users\jessi\OneDrive\Desktop\CIS 485\football-pitch-detection\data\valid\images\4b770a_7_9_png.rf.453855a5d935934ac5622be5981a153a.jpg", 
        help="Path to input image"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\samples\output\full_tactical_test.jpg", 
        help="Path to save result"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    args = parser.parse_args()

    # 1. Initialization
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    engine = DualInferenceEngine(use_tensorrt=False)
    PITCH_CONFIG = SoccerPitchConfiguration()

    # Load image
    frame = cv2.imread(args.source)
    if frame is None:
        print(f"❌ Could not read image: {args.source}")
        return
    print(f"🖼  Processing: {os.path.basename(args.source)}")

    # 2. Inference
    player_res, kpt_res = engine.run_inference(frame, imgsz=args.imgsz)

    # 3. Keypoint Processing & Homography
    key_points = sv.KeyPoints.from_ultralytics(kpt_res)
    frame_ref_pts = []
    pitch_ref_pts = []

    if len(key_points.xy) > 0 and len(key_points.xy[0]) > 0:
        if key_points.confidence is not None:
            filter_conf = key_points.confidence[0] > config.CONF_THRESHOLD_KEYPOINTS
        else:
            raw_conf = kpt_res.keypoints.data[0].cpu().numpy()[:, 2]
            filter_conf = raw_conf > config.CONF_THRESHOLD_KEYPOINTS
        
        if len(filter_conf) == len(key_points.xy[0]):
            frame_ref_pts = key_points.xy[0][filter_conf]
            pitch_ref_pts = np.array(PITCH_CONFIG.vertices)[filter_conf]

    transformer = None
    if len(frame_ref_pts) >= 4:
        transformer = ViewTransformer(source=frame_ref_pts, target=pitch_ref_pts)
        print(f"✅ Homography: Found {len(frame_ref_pts)} valid keypoints.")
    else:
        print(f"⚠️  Not enough keypoints ({len(frame_ref_pts)}) for 2D mapping.")

    # 4. Player & Team Processing
    detections = sv.Detections.from_ultralytics(player_res)
    
    # Separate ball and pad it for visibility
    ball_detections = detections[detections.class_id == config.BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    # Separate people and handle NMS
    people = detections[detections.class_id != config.BALL_ID].with_nms(0.5)
    players = people[people.class_id == config.PLAYER_ID]
    goalkeepers = people[people.class_id == config.GOALKEEPER_ID]
    referees = people[people.class_id == config.REFEREE_ID]

    # Team Classification Logic
    if len(players) >= 2:
        crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🕒 Fitting TeamClassifier on {device}...")
        tc = TeamClassifier(device=device)
        tc.fit(crops)
        players.class_id = tc.predict(crops)
        
        # Resolve goalkeepers
        if len(goalkeepers) > 0:
            goalkeepers.class_id = resolve_goalkeepers_team_id(players, goalkeepers)
        print("✅ Team classification complete.")
    else:
        players.class_id = np.zeros(len(players), dtype=int)
        goalkeepers.class_id = np.zeros(len(goalkeepers), dtype=int)

    # Referees are fixed to ID 2 (Yellow)
    referees.class_id = np.full(len(referees), 2, dtype=int)
    
    merged_people = sv.Detections.merge([players, goalkeepers, referees])

    # 5. Annotation (Main Frame)
    # Palette: 0->Blue, 1->Pink, 2->Yellow
    palette = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])
    ellipse_annotator = sv.EllipseAnnotator(color=palette, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=palette, text_color=sv.Color.BLACK, text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=20, height=17)

    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=merged_people)
    
    temp_labels = ["Ref" if cid == 2 else f"Team {cid}" for cid in merged_people.class_id]
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=merged_people, labels=temp_labels)
    annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)

    # Draw Enhanced Keypoints (Cyan dots with black outline)
    for i, (px, py) in enumerate(frame_ref_pts.astype(int)):
        cv2.circle(annotated_frame, (px, py), 8, BLACK, -1)
        cv2.circle(annotated_frame, (px, py), 7, CYAN, -1)
        # Add index label with shadow
        l_pos = (px + 10, py + 5)
        cv2.putText(annotated_frame, str(i), l_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, str(i), l_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)

    # Add Diagnostic Summary in Cyan
    cv2.putText(
        annotated_frame,
        f"Tactical Image Test | Players: {len(players)} | KPs: {len(frame_ref_pts)}",
        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2, cv2.LINE_AA
    )

    # 6. Minimap Rendering
    minimap = draw_pitch(PITCH_CONFIG)
    if transformer is not None:
        def plot_on_pitch(det, color, r):
            nonlocal minimap
            if len(det) == 0: return
            xy = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            try:
                p_xy = transformer.transform_points(xy)
                minimap = draw_points_on_pitch(
                    config=PITCH_CONFIG, xy=p_xy, face_color=color,
                    edge_color=sv.Color.BLACK, radius=r, pitch=minimap
                )
            except: pass

        plot_on_pitch(ball_detections, sv.Color.WHITE, 10)
        plot_on_pitch(players[players.class_id == 0], sv.Color.from_hex('#00BFFF'), 16)
        plot_on_pitch(players[players.class_id == 1], sv.Color.from_hex('#FF1493'), 16)
        plot_on_pitch(goalkeepers[goalkeepers.class_id == 0], sv.Color.from_hex('#00BFFF'), 20)
        plot_on_pitch(goalkeepers[goalkeepers.class_id == 1], sv.Color.from_hex('#FF1493'), 20)
        plot_on_pitch(referees, sv.Color.from_hex('#FFD700'), 16)

    # 7. Layout and Save
    target_h = annotated_frame.shape[0]
    mini_w = int(minimap.shape[1] * target_h / minimap.shape[0])
    minimap_res = cv2.resize(minimap, (mini_w, target_h))
    
    combined = np.hstack((annotated_frame, minimap_res))
    cv2.imwrite(args.output, combined)
    print(f"✅ Full Tactical Test Output saved to: {args.output}")

if __name__ == "__main__":
    main()
