import cv2
from ultralytics import YOLO
import supervision as sv
from pathlib import Path

class FootballDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            # Dynamically resolve to project_root/models/player_tracking.pt
            model_path = str(Path(__file__).parent.parent / "models" / "player_tracking.pt")
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect_and_track(self, frame, confidence=0.3):
        # Predict on frame
        results = self.model.predict(frame, conf=confidence, verbose=False)[0]
        
        # Convert to supervision detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter classes if needed (0: person, though YOLOv8x might have specific classes if trained)
        # In the notebook, they might be using a specific model for football
        # Let's assume standard COCO for now or the user provided model
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        return detections

    def annotate_frame(self, frame, detections):
        labels = [
            f"#{tracker_id} {self.model.model.names[class_id]}"
            for class_id, tracker_id 
            in zip(detections.class_id, detections.tracker_id)
        ]
        
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections,
            labels=labels
        )
        return annotated_frame
