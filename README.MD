# Football AI CV Pipeline

A Computer Vision system that tracks players, classifies their teams, and maps their real-world coordinates onto a 2D digital pitch. Utlizes fine-tuned Ultralytics YOLO models and Roboflow and SoccerNet datasets.

## Features

- **Player Detection & Tracking**: Utilizes YOLO models to detect players and track them across frames.
- **Pitch Registration**: Uses keypoint detection models to map real-world camera coordinates onto a 2D digital pitch.
- **Team Classification**: Uses SigLIP to classify the team of each detected player.
- **Demonstrations**: Includes pre-processed videos demonstrating tracking and keypoint detection using state-of-the-art AI models.

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/phollis11/CIS485.git
   cd CIS485
   ```

2. **Install dependencies:**
   Ensure you have Python 3.10+ installed. First, install the standard packages:

   ```bash
   pip install -r requirements.txt
   ```

   Then, install the dedicated Roboflow open-source sports module required for pitch annotations:

   ```bash
   pip install git+https://github.com/roboflow/sports.git
   ```

3. **Run the Application:**
   ```bash
   python -m streamlit run app/app_main.py
   ```
   _The app will automatically open in your default browser at http://localhost:8501._

## Repository Structure

- `app/`: Contains the Streamlit web application dashboard, subpages, and frontend styling.
  - `app_main.py`: The entry point of the frontend. It features an architectural diagram, an introduction to the project goals, and explains how ByteTrack, YOLO, and SigLIP combine forces in our pipeline.
  - `pages/`: Contains additional multi-page tools (e.g. Demonstration Videos, Pipeline Execution runner, and Document References).
  - `data/`: Designated for local application caching, resources, and demonstration assets securely detached from the backend root.
  - `style.css`: The core styling sheet ensuring our application features a dark, vibrant, glassmorphism-inspired aesthetic per modern design expectations.
- `pipeline/`: Contains the core backend computer vision logic (`main_realtime.py`, `models.py`, `app_utils.py`).
- `models/`: Empty cache directory for inference states (if used).
- `weights/`: Stores the fine-tuned YOLO weights (`.pt` files) for player and pitch detection.
