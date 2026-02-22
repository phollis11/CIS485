# Football AI CV Pipeline

A professional Computer Vision web application built with Streamlit and Ultralytics YOLO to track football players and perform 2D pitch registration.

## Features
- **Player Detection & Tracking**: Utilizes YOLO models to detect players and track them across frames.
- **Pitch Registration**: Uses keypoint detection models to map real-world camera coordinates onto a 2D digital pitch.
- **Demonstrations**: Includes pre-processed videos demonstrating tracking and keypoint detection using state-of-the-art AI models.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/phollis11/CIS485.git
   cd CIS485
   ```

2. **Install dependencies:**
   Ensure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python -m streamlit run app/app_main.py
   ```
   *The app will automatically open in your default browser at http://localhost:8501.*

## Repository Structure
- `app/`: Contains the Streamlit web application pages and styling.
- `football_ai_pipeline/`: Contains the core computer vision logic (`detector.py`, `pitch_reg.py`, `visualizer.py`).
- `models/`: Stores the YOLO weights (`.pt` files) for easy portability.
- `demo_videos/`: Contains pre-annotated demonstration videos transcoded for web compatibility.
