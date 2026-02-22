import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import from football_ai_pipeline
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from football_ai_pipeline.run_pipeline import main as run_football_pipeline

# Need to import local_css specifically for this page again
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent.parent / "style.css"
if css_path.exists():
    local_css(css_path)

st.title("⚽ Process Your Video")

st.markdown("""
<div class="glass-container">
    <p>Upload a short football video clip (.mp4) to run it through the Player Tracking and Pitch Registration pipeline. The system will detect players, assign teams, and generate a 2D tactical minimap.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Create temp directory for processing if it doesn't exist
    TEMP_DIR = PROJECT_ROOT / "app" / "temp"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file temporarily
    input_video_path = TEMP_DIR / "uploaded_video.mp4"
    output_video_path = TEMP_DIR / "processed_video.mp4"
    
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.markdown("### Preview")
    st.video(input_video_path)
    
    if st.button("🚀 Run AI Pipeline"):
        st.markdown("### Processing...")
        
        # Setup progress bar UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current_frame, total_frames):
            progress = current_frame / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {current_frame} of {total_frames} ({(progress*100):.1f}%)")
            
        try:
            with st.spinner("Initializing models..."):
                # Call the pipeline with the progress callback
                run_football_pipeline(
                    video_path=str(input_video_path),
                    output_path=str(output_video_path),
                    progress_callback=update_progress
                )
                
            st.success("✅ Processing Complete!")
            st.balloons()
            
            st.markdown("### Annotated Result")
            
            # Display result directly from path
            st.video(str(output_video_path))
            
            with open(output_video_path, 'rb') as vf:
                out_video_bytes = vf.read()
                
                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=out_video_bytes,
                    file_name="annotated_output.mp4",
                    mime="video/mp4"
                )
                
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            
        finally:
            # Clean up temp files if desired, though keeping them might be useful for debugging
            pass
