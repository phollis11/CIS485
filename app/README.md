# Streamlit Application (`app/`)

This directory serves as the primary gateway for the 2D Virtual Mapping System. Built on top of Streamlit, it functions as the interactive web interface, allowing researchers, coaches, or analysts to upload videos and immediately visualize tracking, keypoint generation, and our tactical map implementations in real-time.

## Structure Overview

- **`app_main.py`**: The entry point of the frontend. It features an architectural diagram, an introduction to the project goals, and explains how ByteTrack, YOLO, and SigLIP combine forces in our pipeline.
- **`pages/`**: Contains additional multi-page tools (e.g. Demonstration Videos, Pipeline Execution runner, and Document References).
- **`data/`**: Designated for local application caching, resources, and demonstration assets securely detached from the backend root.
- **`style.css`**: The core styling sheet ensuring our application features a dark, vibrant, glassmorphism-inspired aesthetic per modern design expectations.

## How to Run

Before running, ensure all dependencies and models are set up in the project root directory.

From the root project directory (where the `requirements.txt` is located), simply execute:

```bash
streamlit run app/app_main.py
```

_The server will automatically boot up, and the GUI will be populated on your default browser locally at `http://localhost:8501`._
