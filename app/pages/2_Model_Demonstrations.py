import streamlit as st
import os
from pathlib import Path

# Need to import local_css specifically for this page again
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent.parent / "style.css"
if css_path.exists():
    local_css(css_path)

st.title("🎬 Model Demonstrations")

st.markdown("""
<div class="glass-container">
    <p>This page features demonstrations of our advanced Computer Vision models running on football footage.</p>
</div>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Define the paths to the annotated videos
demo_vid_model_large = PROJECT_ROOT / "demo_videos" / "prem_vid_annotated_web.mp4"
demo_vid_model_small = PROJECT_ROOT / "demo_videos" / "prem_vid_annotated_short_web.mp4"

st.header("Large YOLO Model (YOLO11x) - Player Tracking Only")
if demo_vid_model_large.exists():
    # Render video 1 using path directly
    st.video(str(demo_vid_model_large))
    
    with open(demo_vid_model_large, 'rb') as video_file_1:
        video_bytes_1 = video_file_1.read()
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.download_button(
            label="⬇️ Download YOLO11x Demo Video",
            data=video_bytes_1,
            file_name="demo_yolo11x_tracking.mp4",
            mime="video/mp4"
        )
else:
    st.error(f"Demo video not found at {demo_vid_model_large}.")

st.markdown("---")

st.header("Small YOLO Model (YOLO8n) - Tracking & Key Point Detection")
st.markdown("<p style='margin-top:-10px; margin-bottom: 15px;'>Finetuned with a short training on player tracking. Additionally, this video ran with key point detection using a finetuned YOLO8n model.</p>", unsafe_allow_html=True)

if demo_vid_model_small.exists():
    # Render video 2 using path directly
    st.video(str(demo_vid_model_small))
    
    with open(demo_vid_model_small, 'rb') as video_file_2:
        video_bytes_2 = video_file_2.read()
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.download_button(
            label="⬇️ Download YOLO8n Demo Video",
            data=video_bytes_2,
            file_name="demo_yolo8n_tracking_keypoints.mp4",
            mime="video/mp4"
        )
else:
    st.error(f"Demo video not found at {demo_vid_model_small}.")
