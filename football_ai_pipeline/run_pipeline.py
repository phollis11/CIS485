import cv2
import sys
import os
# Ensure the local directory is in path so internal imports work when called from Streamlit
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import FootballDetector
from pitch_reg import PitchRegistration
from visualizer import PitchVisualizer
import utils
import supervision as sv

def main(video_path, output_path, progress_callback=None):
    # Initialize components
    detector = FootballDetector()
    pitch_reg = PitchRegistration() # Ideally with trained pose model
    visualizer = PitchVisualizer()
    
    # Get video info
    fps, size, total_frames = utils.get_video_info(video_path)
    
    # Process video
    generator = sv.get_video_frames_generator(video_path)
    processed_frames = []
    
    # For demo, let's simple team colors
    TEAM_1_COLOR = (255, 0, 0) # Red (BGR)
    TEAM_2_COLOR = (0, 0, 255) # Blue (BGR)
    
    print(f"Processing {total_frames} frames...")
    
    for i, frame in enumerate(generator):
        # Console output
        print(f"Frame {i+1}/{total_frames}", end="\r")
        
        # UI Callback if provided
        if progress_callback:
            progress_callback(i + 1, total_frames)
            
        # 1. Detect and track
        detections = detector.detect_and_track(frame)
        
        # 2. Annotate frame
        annotated_frame = detector.annotate_frame(frame, detections)
        
        # 3. Pitch registration
        keypoints = pitch_reg.detect_keypoints(frame)
        H = None
        if keypoints is not None:
            # Draw keypoints for debugging
            for i, kpt in enumerate(keypoints):
                x, y = kpt
                if x > 0 and y > 0:
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(annotated_frame, str(i), (int(x), int(y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            H = pitch_reg.get_homography(keypoints)
        
        # 4. Transform player positions and draw pitch control
        if H is not None:
            # Get bottom center of bounding boxes for feet position
            feet_positions = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_coords = pitch_reg.transform_points(H, feet_positions)
            
            # For demo, split detected players into two teams randomly
            half = len(pitch_coords) // 2
            team_1 = pitch_coords[:half]
            team_2 = pitch_coords[half:]
            
            # Get tactical view
            tactical_pitch = visualizer.calculate_pitch_control(
                team_1, team_2, TEAM_1_COLOR, TEAM_2_COLOR
            )
            
            # Resize tactical view to fit a corner of the main frame
            tw, th = size[0] // 4, size[1] // 4
            small_tactical = cv2.resize(tactical_pitch, (tw, th))
            annotated_frame[0:th, 0:tw] = small_tactical
            
        processed_frames.append(annotated_frame)

    # Save video
    utils.save_video(processed_frames, output_path, fps, size)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    # Example usage
    # main('input_video.mp4', 'output_video.mp4')
    main(r"C:\Users\pholl\OneDrive\Desktop\_pycache_\CIS 485\prem_vid_short.mp4", r"C:\Users\pholl\OneDrive\Desktop\_pycache_\CIS 485\prem_vid_annotated_short.mp4")
