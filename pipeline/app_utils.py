import cv2
import numpy as np
import time
import collections
import os
import supervision as sv
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch
)
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.common.team import TeamClassifier

import config
from models import DualInferenceEngine

def resolve_goalkeepers_team_id(players, goalkeepers):
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

def run_optimized_pipeline(video_path, output_path, progress_callback=None):
    # Setup environment
    if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN:
        os.environ["HF_TOKEN"] = config.HF_TOKEN
        
    # Initialize Models
    engine = DualInferenceEngine(use_tensorrt=False)
    team_classifier = TeamClassifier(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    
    # Video Setup
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output Setup (Standard mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # State
    tracker = sv.ByteTrack()
    PITCH_CONFIG = SoccerPitchConfiguration()
    pitch_all_points = np.array(PITCH_CONFIG.vertices)
    PITCH_TEMPLATE = draw_pitch(PITCH_CONFIG)
    
    palette = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])
    ellipse_annotator = sv.EllipseAnnotator(color=palette, thickness=2)
    label_annotator = sv.LabelAnnotator(color=palette, text_color=sv.Color.BLACK, text_position=sv.Position.BOTTOM_CENTER)
    triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=20, height=17)
    edge_annotator = sv.EdgeAnnotator(color=sv.Color.from_hex('#00BFFF'), thickness=2, edges=PITCH_CONFIG.edges)
    
    calibration_crops = []
    is_calibrated = False
    player_team_registry = {}
    frame_count = 0
    
    last_player_res = None
    last_kpt_res = None
    transformer = None
    inverse_transformer = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if progress_callback:
            progress_callback(frame_count, total_frames)
            
        # Inference Logic (Frame Skipping integrated)
        if frame_count % 2 == 1 or last_player_res is None:
            player_res, kpt_res = engine.run_inference(frame)
            last_player_res, last_kpt_res = player_res, kpt_res
        else:
            player_res, kpt_res = last_player_res, last_kpt_res
            
        # Keypoints & Homography
        key_points = sv.KeyPoints.from_ultralytics(kpt_res)
        if len(key_points.xy) > 0 and len(key_points.xy[0]) > 0:
            raw_conf = kpt_res.keypoints.data[0].cpu().numpy()[:, 2]
            filter_conf = raw_conf > config.CONF_THRESHOLD_KEYPOINTS
            if len(filter_conf) == len(key_points.xy[0]):
                frame_pts = key_points.xy[0][filter_conf]
                pitch_pts = np.array(PITCH_CONFIG.vertices)[filter_conf]
                if len(frame_pts) >= 4:
                    try:
                        transformer = ViewTransformer(source=frame_pts, target=pitch_pts)
                        inverse_transformer = ViewTransformer(source=pitch_pts, target=frame_pts)
                    except: pass

        # Tracking & Detection
        detections = sv.Detections.from_ultralytics(player_res)
        ball_detections = detections[detections.class_id == config.BALL_ID]
        people_detections = detections[detections.class_id != config.BALL_ID]
        people_detections = people_detections.with_nms(threshold=0.5)
        people_detections = tracker.update_with_detections(people_detections)
        
        goalkeepers = people_detections[people_detections.class_id == config.GOALKEEPER_ID]
        players = people_detections[people_detections.class_id == config.PLAYER_ID]
        referees = people_detections[people_detections.class_id == config.REFEREE_ID]
        
        # Team Classification Caching
        if not is_calibrated:
            calibration_crops += [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
            if len(calibration_crops) >= 15:
                team_classifier.fit(calibration_crops)
                is_calibrated = True
        
        if is_calibrated:
            if players.tracker_id is not None:
                unknown_crops = []
                unknown_ids = []
                for idx, t_id in enumerate(players.tracker_id):
                    if t_id not in player_team_registry:
                        unknown_ids.append(t_id)
                        unknown_crops.append(sv.crop_image(frame, players.xyxy[idx]))
                if unknown_crops:
                    preds = team_classifier.predict(unknown_crops)
                    for i, t_id in enumerate(unknown_ids):
                        player_team_registry[t_id] = int(preds[i])
                players.class_id = np.array([player_team_registry.get(tid, 0) for tid in players.tracker_id])
            if len(goalkeepers) > 0:
                goalkeepers.class_id = resolve_goalkeepers_team_id(players, goalkeepers)
        
        referees.class_id = np.full(len(referees), 2, dtype=int)
        merged = sv.Detections.merge([players, goalkeepers, referees])
        
        # Annotate
        annotated_frame = ellipse_annotator.annotate(scene=frame.copy(), detections=merged)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=merged, 
                                                   labels=[f"#{tid}" if tid is not None else "" for tid in merged.tracker_id] if merged.tracker_id is not None else [""]*len(merged))
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)
        
        if inverse_transformer:
            try:
                frame_all_pts = inverse_transformer.transform_points(points=pitch_all_points)
                annotated_frame = edge_annotator.annotate(scene=annotated_frame, key_points=sv.KeyPoints(xy=frame_all_pts[np.newaxis, ...]))
            except: pass
            
        # Draw Minimap
        if transformer:
            current_minimap = PITCH_TEMPLATE.copy()
            def draw_m(det, color, r):
                nonlocal current_minimap
                if len(det) > 0:
                    xy = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    try:
                        p_xy = transformer.transform_points(points=xy)
                        current_minimap = draw_points_on_pitch(config=PITCH_CONFIG, xy=p_xy, face_color=color, edge_color=sv.Color.BLACK, radius=r, pitch=current_minimap)
                    except: pass
            draw_m(ball_detections, sv.Color.WHITE, 10)
            draw_m(players[players.class_id==0], sv.Color.from_hex('00BFFF'), 16)
            draw_m(players[players.class_id==1], sv.Color.from_hex('FF1493'), 16)
            draw_m(referees, sv.Color.from_hex('FFD700'), 16)
            
            # Overlay Minimap
            mw, mh = w // 4, h // 4
            resized_minimap = cv2.resize(current_minimap, (mw, mh))
            annotated_frame[0:mh, 0:mw] = resized_minimap
            
        out.write(annotated_frame)
        
    cap.release()
    out.release()
