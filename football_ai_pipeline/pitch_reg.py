import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class PitchRegistration:
    def __init__(self, model_path=None):
        if model_path is None:
            # Dynamically resolve to project_root/models/pitch_keypoints.pt
            model_path = str(Path(__file__).parent.parent / "models" / "pitch_keypoints.pt")
        # In reality, this should be the trained football-field-detection model
        self.model = YOLO(model_path)

        # Standard pitch dimensions in meters (FIFA standard)
        self.pitch_width = 105
        self.pitch_height = 68

        # Ideal 2D coordinates for the 32 keypoints (normalized or meters)
        # Mapping indices 0-31 to real-world (x, y) coordinates on the pitch.
        # (0, 0) is top-left, (105, 68) is bottom-right.
        self.ideal_keypoints = self._get_ideal_keypoints()

    def _get_ideal_keypoints(self):
        """
        Returns a dictionary mapping keypoint indices (0-31) to (x, y) coordinates in meters.
        Based on the standard 32-keypoint Roboflow football pitch model.
        """
        kp = {
            0: [0, 0],         # Top-left corner
            1: [0, 13.85],     # Top-left penalty area corner (outer)
            2: [0, 24.85],     # Top-left goal area corner
            3: [0, 43.15],     # Bottom-left goal area corner
            4: [0, 54.15],     # Bottom-left penalty area corner (outer)
            5: [0, 68],        # Bottom-left corner
            6: [16.5, 13.85],  # Top-left penalty area corner (inner)
            7: [16.5, 54.15],  # Bottom-left penalty area corner (inner)
            8: [5.5, 24.85],   # Top-left goal area corner (inner)
            9: [5.5, 43.15],   # Bottom-left goal area corner (inner)
            10: [11, 34],      # Left penalty spot
            11: [52.5, 0],     # Top-middle (halfway line)
            12: [52.5, 34],    # Center spot
            13: [52.5, 68],    # Bottom-middle (halfway line)
            14: [105, 0],      # Top-right corner
            15: [105, 13.85],  # Top-right penalty area corner (outer)
            16: [105, 24.85],  # Top-right goal area corner
            17: [105, 43.15],  # Bottom-right goal area corner
            18: [105, 54.15],  # Bottom-right penalty area corner (outer)
            19: [105, 68],     # Bottom-right corner
            20: [88.5, 13.85], # Top-right penalty area corner (inner)
            21: [88.5, 54.15], # Bottom-right penalty area corner (inner)
            22: [99.5, 24.85], # Top-right goal area corner (inner)
            23: [99.5, 43.15], # Bottom-right goal area corner (inner)
            24: [94, 34],      # Right penalty spot
            25: [52.5, 24.85], # Center circle top
            26: [52.5, 43.15], # Center circle bottom
            27: [32.5, 34],    # Center circle left (intersection)
            28: [72.5, 34],    # Center circle right (intersection)
            29: [16.5, 24.85], # Penalty arc left (top intersection)
            30: [16.5, 43.15], # Penalty arc left (bottom intersection)
            31: [88.5, 34]     # Penalty arc right (center point)
        }
        return kp

    def detect_keypoints(self, frame):
        results = self.model.predict(frame, verbose=False)[0]
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            # results.keypoints.xy is [N, 32, 2]
            return results.keypoints.xy.cpu().numpy()[0]
        return None

    def get_homography(self, keypoints):
        """
        Calculate homography matrix using detected keypoints.
        keypoints: numpy array of shape (32, 2)
        """
        src_pts = []
        dst_pts = []

        for i in range(len(keypoints)):
            x, y = keypoints[i]
            if x != 0 and y != 0 and i in self.ideal_keypoints:
                src_pts.append([x, y])
                dst_pts.append(self.ideal_keypoints[i])

        if len(src_pts) < 4:
            return None

        src_pts = np.array(src_pts).reshape(-1, 1, 2).astype(np.float32)
        dst_pts = np.array(dst_pts).reshape(-1, 1, 2).astype(np.float32)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def transform_points(self, H, points):
        """Transform image points to pitch coordinates."""
        if H is None or len(points) == 0:
            return points
        
        points = np.array(points).reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(points, H)
        return transformed.reshape(-1, 2)
