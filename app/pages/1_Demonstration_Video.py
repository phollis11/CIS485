import streamlit as st
import os
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Model Demonstrations", layout="wide")

# --- INJECT CUSTOM CSS ---
def local_css(file_name):
    if Path(file_name).exists():
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent.parent / "style.css"
local_css(css_path)

def main():
    st.title("Model Demonstrations")
    
    st.markdown("""
    <div class="glass-container">
        <p>This page showcases the integration of multiple custom-trained computer vision models. 
        The videos below demonstrate the system's ability to extract high-level tactical information 
        from raw broadcast footage.</p>
    </div>
    """, unsafe_allow_html=True)
    

    
    # Define the paths to the annotated videos
    DEMO_DIR = Path(__file__).parent.parent / "data" / "resources" / "demo_videos"
    demo_vid_full_pipeline = DEMO_DIR / "long_demo_output.mp4"
    demo_vid_player_tracking = DEMO_DIR / "player_tracking_only.mp4"
    demo_vid_keypoint_detection = DEMO_DIR / "keypoint_detection_only.mp4"
    
    # Section 1
    st.header("2D Virtual Mapping Full Demonstration")
    st.markdown("<p>This is a full example of the entire project pipeline on a random 1 minute video clip from FULL MATCH | Manchester City v Liverpool | Quarter-Final | Emirates FA Cup 2025-26</p>", unsafe_allow_html=True)
    
    if demo_vid_full_pipeline.exists():
        st.video(open(demo_vid_full_pipeline, 'rb').read())
        
        with open(demo_vid_full_pipeline, 'rb') as vf:
            st.download_button(
                label="Download Full Pipeline Demonstration",
                data=vf,
                file_name="full_pipeline_demonstration.mp4",
                mime="video/mp4",
                key="btn_full_pipeline"
            )
    else:
        st.error(f"Demonstration asset not found at {demo_vid_full_pipeline.name}.")
    
    st.markdown("---")
    
    # Section 2
    st.header("Player Tracking Only Demonstration")
    st.markdown(""" 
    <p>This is a demonstration of the player tracking model. This uses a fine-tuned YOLO26n model on the SoccerNet dataset found in the resources tab.</p>
    """, unsafe_allow_html=True)
    
    if demo_vid_player_tracking.exists():
        st.video(open(demo_vid_player_tracking, 'rb').read())
        
        with open(demo_vid_player_tracking, 'rb') as vf:
            st.download_button(
                label="Download Player Tracking Demonstration",
                data=vf,
                file_name="player_tracking_demo.mp4",
                mime="video/mp4",
                key="btn_player_tracking"
            )
    else:
        st.error(f"Demonstration asset not found at {demo_vid_player_tracking.name}.")

    # Section 3
    st.header("Keypoint Detection Only Demonstration")
    st.markdown(""" 
    <p>This is a demonstration of the keypoint detection model. This uses a fine-tuned YOLOv8n-pose model on the Roboflow dataset found in the resources tab.</p>
    """, unsafe_allow_html=True)
    
    if demo_vid_keypoint_detection.exists():
        st.video(open(demo_vid_keypoint_detection, 'rb').read())
        
        with open(demo_vid_keypoint_detection, 'rb') as vf:
            st.download_button(
                label="Download Keypoint Detection Demonstration",
                data=vf,
                file_name="keypoint_detection_demo.mp4",
                mime="video/mp4",
                key="btn_keypoint_detection"
            )
    else:
        st.error(f"Demonstration asset not found at {demo_vid_keypoint_detection.name}.")

    st.markdown("""
    <div class="app-footer">
        2D Virtual Mapping System &nbsp;|&nbsp; Shepherd University CIS Capstone Project - Peyton Hollis, Curtis Canby, Uchenna Ibe &nbsp;|&nbsp; Built with Streamlit and Ultralytics YOLO, utlizes SoccerNet and Roboflow datasets. 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
