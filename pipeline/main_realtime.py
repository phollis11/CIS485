import argparse
import os
import cv2
import numpy as np
import time

import supervision as sv
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch
)
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.common.team import TeamClassifier
import collections

from video_stream import VideoStream
from models import DualInferenceEngine
import config

def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    if len(goalkeepers) == 0:
        return np.array([])
    if len(players) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
        
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    # Calculate centroids
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
    parser = argparse.ArgumentParser(description="Real-Time Tactical Football Mapping")
    parser.add_argument("--source", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline", "test.mp4"), help="Path to video or 0 for camera")
    parser.add_argument("--trt", action="store_true", help="Try to use TensorRT engines for maximum speed")
    parser.add_argument("--export", action="store_true", help="Export PyTorch models to TensorRT and exit")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input resolution (lower = faster, e.g. 480)")
    parser.add_argument("--frame-skip", type=int, default=2, help="Run YOLO every N frames; interpolate in between (default: 2)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "realtime_output.mp4"),
                        help="Path to save the output video")
    args = parser.parse_args()

    engine = DualInferenceEngine(use_tensorrt=args.trt)
    if args.export:
        engine.export_to_tensorrt()
        return

    src = int(args.source) if args.source.isdigit() else args.source
    stream = VideoStream(src)

    # Initialize soccer configuration
    PITCH_CONFIG = SoccerPitchConfiguration()
    pitch_all_points = np.array(PITCH_CONFIG.vertices)

    # Annotators
    # Color Palette: 0 -> Team A (Blue), 1 -> Team B (Pink), 2 -> Referee (Yellow)
    palette = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])
    ellipse_annotator = sv.EllipseAnnotator(color=palette, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=palette,
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=20, height=17
    )
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        thickness=2, edges=PITCH_CONFIG.edges
    )

    tracker = sv.ByteTrack()
    tracker.reset()
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set HF token for authenticated model requests (prevents 504 and unauth warnings)
    if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN:
        os.environ["HF_TOKEN"] = config.HF_TOKEN
        
    team_classifier = TeamClassifier(device=device)
    calibration_crops = []
    is_calibrated = False
    player_team_registry = {}

    print("🚀 Starting pipeline... Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    last_player_res = None
    last_kpt_res = None
    FRAME_SKIP = args.frame_skip
    IMGSZ = args.imgsz

    # PRE-RENDER pitch once to save CPU
    PITCH_TEMPLATE = draw_pitch(PITCH_CONFIG)
    transformer = None
    inverse_transformer = None
    
    # Accurate FPS
    fps_window = collections.deque(maxlen=30)

    # --- Video writer (lazy init on first frame so we know the output size) ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = None
    print(f"💾 Recording to: {args.output}")

    try:
        while stream.more():
            ret, frame = stream.read()
            if not ret or frame is None:
                break

            player_res, kpt_res = None, None
            if frame_count % FRAME_SKIP == 0 or last_player_res is None:
                player_res, kpt_res = engine.run_inference(frame, imgsz=IMGSZ)
                last_player_res = player_res
                last_kpt_res = kpt_res
            else:
                player_res = last_player_res
                kpt_res = last_kpt_res

            # Process Keypoints
            key_points = sv.KeyPoints.from_ultralytics(kpt_res)
            
            frame_reference_points = []
            pitch_reference_points = []
            
            if len(key_points.xy) > 0 and len(key_points.xy[0]) > 0:
                if key_points.confidence is not None:
                    filter_conf = key_points.confidence[0] > config.CONF_THRESHOLD_KEYPOINTS
                else:
                    raw_conf = kpt_res.keypoints.data[0].cpu().numpy()[:, 2]
                    filter_conf = raw_conf > config.CONF_THRESHOLD_KEYPOINTS
                
                if len(filter_conf) == len(key_points.xy[0]):
                    frame_reference_points = key_points.xy[0][filter_conf]
                    pitch_reference_points = np.array(PITCH_CONFIG.vertices)[filter_conf]

            # Update transformer if we have enough points
            if len(frame_reference_points) >= 4:
                try:
                    transformer = ViewTransformer(
                        source=frame_reference_points,
                        target=pitch_reference_points
                    )
                    # For drawing pitch lines on the main frame, we need the reverse mapping
                    inverse_transformer = ViewTransformer(
                        source=pitch_reference_points,
                        target=frame_reference_points
                    )
                except Exception:
                    pass 

            current_minimap = PITCH_TEMPLATE.copy()
            main_annotated = frame.copy()

            # Process Players
            detections = sv.Detections.from_ultralytics(player_res)
            
            # Separate Ball
            ball_detections = detections[detections.class_id == config.BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            # People Tracking
            people_detections = detections[detections.class_id != config.BALL_ID]
            people_detections = people_detections.with_nms(threshold=0.5, class_agnostic=True)
            people_detections = tracker.update_with_detections(detections=people_detections)

            # Sort classes out
            goalkeepers = people_detections[people_detections.class_id == config.GOALKEEPER_ID]
            players = people_detections[people_detections.class_id == config.PLAYER_ID]
            referees = people_detections[people_detections.class_id == config.REFEREE_ID]
            
            # Rolling Calibration Logic for Teams
            if not is_calibrated:
                players_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
                calibration_crops += players_crops
                if len(calibration_crops) >= 10:
                    print("🕒 Calibrating Teams from first frames... Please wait.")
                    team_classifier.fit(calibration_crops)
                    is_calibrated = True
                    print("✅ Teams Calibrated!")
            
            if is_calibrated:
                unknown_indices = []
                unknown_crops = []
                
                if players.tracker_id is not None:
                    for idx, t_id in enumerate(players.tracker_id):
                        if t_id not in player_team_registry:
                            unknown_indices.append(idx)
                            unknown_crops.append(sv.crop_image(frame, players.xyxy[idx]))
                            
                    if len(unknown_crops) > 0:
                        new_preds = team_classifier.predict(unknown_crops)
                        for i, idx in enumerate(unknown_indices):
                            t_id = players.tracker_id[idx]
                            player_team_registry[t_id] = int(new_preds[i])
                            
                    players.class_id = np.array([player_team_registry.get(t_id, 0) for t_id in players.tracker_id], dtype=int)
                else:
                    players.class_id = np.zeros(len(players), dtype=int)
                    
                if len(goalkeepers) > 0:
                    goalkeepers.class_id = resolve_goalkeepers_team_id(players, goalkeepers)
            else:
                # Draw as Default Team 0 until calibrated
                players.class_id = np.zeros(len(players), dtype=int)
                goalkeepers.class_id = np.zeros(len(goalkeepers), dtype=int)

            # Referees get Color ID 2 (Yellow)
            referees.class_id = np.full(len(referees), 2, dtype=int)

            merged_detections = sv.Detections.merge([players, goalkeepers, referees])
            labels = []
            if merged_detections.tracker_id is not None:
                labels = [f"#{t_id}" for t_id in merged_detections.tracker_id]
            else:
                # Fill with empty strings to avoid crashing LabelAnnotator
                labels = [""] * len(merged_detections)

            # Annotate Main Frame
            main_annotated = ellipse_annotator.annotate(
                scene=main_annotated,
                detections=merged_detections)
            main_annotated = label_annotator.annotate(
                scene=main_annotated,
                detections=merged_detections,
                labels=labels)
            main_annotated = triangle_annotator.annotate(
                scene=main_annotated,
                detections=ball_detections)

            # If homography valid, render minimap and pitch edges on main view
            if transformer is not None:
                # 1. Pitch edges on main frame (Using Inverse Homography)
                if inverse_transformer is not None:
                    try:
                        frame_all_points = inverse_transformer.transform_points(points=pitch_all_points)
                        frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
                        main_annotated = edge_annotator.annotate(
                            scene=main_annotated,
                            key_points=frame_all_key_points)
                    except Exception:
                        pass
                    
                # 2. Draw Minimap Points
                def map_and_draw(det_obj, color, rad):
                    nonlocal current_minimap
                    if len(det_obj) > 0:
                        xy = det_obj.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        try:
                            pitch_xy = transformer.transform_points(points=xy)
                            current_minimap = draw_points_on_pitch(
                                config=PITCH_CONFIG,
                                xy=pitch_xy,
                                face_color=color,
                                edge_color=sv.Color.BLACK,
                                radius=rad,
                                pitch=current_minimap)
                        except Exception:
                            pass
                
                map_and_draw(ball_detections, sv.Color.WHITE, 10)
                
                # Split Teams
                team_0_players = players[players.class_id == 0]
                team_1_players = players[players.class_id == 1]
                map_and_draw(team_0_players, sv.Color.from_hex('00BFFF'), 16)
                map_and_draw(team_1_players, sv.Color.from_hex('FF1493'), 16)
                
                team_0_gk = goalkeepers[goalkeepers.class_id == 0]
                team_1_gk = goalkeepers[goalkeepers.class_id == 1]
                map_and_draw(team_0_gk, sv.Color.from_hex('00BFFF'), 20) # Slightly larger to see GK
                map_and_draw(team_1_gk, sv.Color.from_hex('FF1493'), 20)
                
                map_and_draw(referees, sv.Color.from_hex('FFD700'), 16)

            # Layout: Side by Side Main Frame and Minimap
            target_height = main_annotated.shape[0]
            # Match height
            if current_minimap.shape[0] > 0:
                scale = target_height / current_minimap.shape[0]
                target_width = int(current_minimap.shape[1] * scale)
                resized_minimap = cv2.resize(current_minimap, (target_width, target_height))
                combined_view = np.hstack((main_annotated, resized_minimap))
            else:
                combined_view = main_annotated

            # Draw FPS Performance and Status (Sliding Window)
            fps_window.append(time.perf_counter())
            fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0]) if len(fps_window) > 1 else 0.0
            
            status_text = "Calibrating Teams..." if not is_calibrated else "Tracking..."
            cv2.putText(combined_view, f"Status: {status_text} | H-Matrix: {'Valid' if transformer is not None else 'Lost'} | FPS: {fps:.1f}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            frame_count += 1
            
            # Scale to fit screen
            final_display = cv2.resize(combined_view, (combined_view.shape[1] // 3, combined_view.shape[0] // 3))

            # --- Write frame to output video ---
            if writer is None:
                h, w = final_display.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, max(stream.fps, 15), (w, h))
            writer.write(final_display)

            # Show Output
            cv2.imshow("Tactical View", final_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream.stop()
        if writer is not None:
            writer.release()
            print(f"💾 Video saved to: {args.output}")
        cv2.destroyAllWindows()
        print("✅ Inference complete. Shutting down.")

if __name__ == "__main__":
    main()
