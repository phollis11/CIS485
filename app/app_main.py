import streamlit as st
import base64
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Football AI CV Pipeline",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- INJECT CUSTOM CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Try loading the css file
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    local_css(css_path)
else:
    st.warning("Custom CSS not found. Application styling might be degraded.")

# --- NAVIGATION ---
# Native Streamlit multipage app from 'pages/' directory handles routing automatically.
# We will use the main page for a welcome screen.

def main():
    st.title("⚽ Football AI Tracking & Analysis")
    
    st.markdown("""
    Welcome to the Football AI Pipeline interface. This tool utilizes advanced Computer Vision models to detect, track, and map football players onto a 2D digital pitch representation.
    
    ### Capabilities
    - **Player Detection**: YOLOv8-based model to locate and track players and referees dynamically.
    - **Pitch Registration**: Keypoint detection model to identify pitch lines and calculate homography for 2D mapping.
    - **Team Assignment**: Automatically separate players based on positional data (Demo).
    
    ### Get Started
    Please select a module from the sidebar navigation:
    * **1. Video Processing**: Upload and run the full AI pipeline on your custom footage.
    * **2. Model Demonstrations**: View pre-analyzed videos demonstrating the system's tracking and mapping capabilities.
    * **3. 2D Pitch Mapping**: View a sample 2D mapping of the player tracking and key point detection models.
    * **4. Reference Materials**: Access documents, links, and other relevant research references.    
    ---
    *Built with Streamlit & Ultralytics YOLO*
    """)
    
    # Adding a decorative component
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Navigate using the sidebar to begin analyzing football videos.")

    st.markdown("""
    <div class="app-footer">
        &copy; 2026 Football AI CV Pipeline. Built with Streamlit & Ultralytics YOLO.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
