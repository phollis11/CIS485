import os
from pathlib import Path
from ultralytics import YOLO

import config

class DualInferenceEngine:
    def __init__(self, use_tensorrt=False):
        """
        Initializes the YOLO models for Player Detection and Keypoint Detection.
        Optionally uses TensorRT .engine files if available for maximum FPS.
        """
        self.use_tensorrt = use_tensorrt
        
        player_path = config.PLAYER_MODEL_PATH
        keypoint_path = config.KEYPOINT_MODEL_PATH

        if use_tensorrt:
            player_path = self._try_get_engine_path(player_path)
            keypoint_path = self._try_get_engine_path(keypoint_path)

        print(f"📦 Loading Player Model: {player_path}")
        self.player_model = YOLO(player_path)
        
        print(f"📦 Loading Pitch Keypoint Model: {keypoint_path}")
        self.keypoint_model = YOLO(keypoint_path, task='pose')

    def _try_get_engine_path(self, pt_path):
        """Checks if a compiled .engine file exists, otherwise falls back to .pt"""
        engine_path = pt_path.replace('.pt', '.engine')
        if os.path.exists(engine_path):
            return engine_path
        print(f"⚠️ Warning: TensorRT engine not found at {engine_path}. Falling back to .pt format (Expect lower FPS).")
        return pt_path

    def run_inference(self, frame, imgsz=640):
        """
        Runs both models sequentially on the given frame.
        imgsz controls the input resolution - lower = faster but less accurate.
        """
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Detect Players
        player_results = self.player_model.predict(
            source=frame, 
            conf=config.CONF_THRESHOLD_PLAYERS, 
            imgsz=imgsz,
            device=device,
            verbose=False
        )[0]
        
        # 2. Detect Field Keypoints
        keypoint_results = self.keypoint_model.predict(
            source=frame, 
            conf=config.CONF_THRESHOLD_KEYPOINTS, 
            imgsz=imgsz,
            device=device,
            verbose=False
        )[0]

        return player_results, keypoint_results

    def export_to_tensorrt(self):
        """Helper to export models if the user wants to optimize them locally."""
        print("Starting TensorRT Export...")
        self.player_model.export(format="engine", half=True, dynamic=False, imgsz=640)
        self.keypoint_model.export(format="engine", half=True, dynamic=False, imgsz=640)
        print("Export complete. You can now initialize the Engine with use_tensorrt=True")
