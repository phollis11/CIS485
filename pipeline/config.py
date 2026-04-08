import os

# --- MODEL PATHS ---
PLAYER_MODEL_PATH = r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\weights\yolo26n_soccernet_best.pt"
KEYPOINT_MODEL_PATH = r"C:\Users\jessi\OneDrive\Desktop\CIS 485\CIS485\weights\yolo8n-pose_final_train.pt"

# --- INFERENCE SETTINGS ---
CONF_THRESHOLD_KEYPOINTS = 0.50
CONF_THRESHOLD_PLAYERS = 0.40

# --- CLASS MAP SETTINGS (Mapped to YOLO classes) ---
BALL_ID = 0
PLAYER_ID = 1
GOALKEEPER_ID = 99
REFEREE_ID = 98
