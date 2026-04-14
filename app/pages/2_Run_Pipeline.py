import streamlit as st
import os
import sys
import time
from pathlib import Path

# --- PROJECT SETUP ---
# Robustly find project root and pipeline directory
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent # CIS485
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

try:
    from pipeline.app_utils import run_optimized_pipeline
except ImportError:
    from app_utils import run_optimized_pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Run Pipeline", layout="wide")

# --- INJECT CUSTOM CSS ---
def local_css(file_name):
    if Path(file_name).exists():
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent.parent / "style.css"
local_css(css_path)

def main():
    st.title("Run Pipeline")
    
    st.markdown("""
    <div class="glass-container">
        <p>This module allows you to upload custom football footage and process it using our optimized 
        computer vision pipeline. The system will perform real-time object tracking, pitch 
        registration, and team identification.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload video footage (.mp4)", type=['mp4'])
    
    if uploaded_file is not None:
        st.info("File uploaded and ready for analysis.")
        
        # Setup temporary directories
        TEMP_DIR = PROJECT_ROOT / "app" / "temp"
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        input_path = TEMP_DIR / "user_input.mp4"
        output_name = f"processed_{int(time.time())}.mp4"
        output_path = TEMP_DIR / output_name
        
        # Write upload to disk
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.subheader("Input Preview")
        st.video(input_path)
        
        if st.button("Start Analysis"):
            # UI components for feedback
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            def update_ui(current, total):
                percent = current / total
                progress_bar.progress(percent)
                status_container.markdown(f"<p>Processing frame {current} of {total} ({percent*100:.1f}%)</p>", unsafe_allow_html=True)
            
            try:
                with st.spinner("Initializing neural networks and tracking algorithms..."):
                    run_optimized_pipeline(
                        video_path=str(input_path),
                        output_path=str(output_path),
                        progress_callback=update_ui
                    )
                
                st.success("Analysis Complete")
                
                st.subheader("Annotated Result")
                st.video(str(output_path))
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name=f"Tactical_Analysis_{output_name}",
                        mime="video/mp4"
                    )
                    
            except Exception as e:
                st.error(f"An unexpected error occurred during processing: {str(e)}")
                st.code(str(e))

    st.markdown("""
    <div class="app-footer">
        2D Virtual Mapping System &nbsp;|&nbsp; Shepherd University CIS Capstone Project - Peyton Hollis, Curtis Canby, Uchenna Ibe &nbsp;|&nbsp; Built with Streamlit and Ultralytics YOLO, utlizes SoccerNet and Roboflow datasets. 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
