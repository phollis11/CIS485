import streamlit as st
import base64
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="2D Virtual Mapping System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- INJECT CUSTOM CSS ---
def local_css(file_name):
    if Path(file_name).exists():
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent / "style.css"
local_css(css_path)

def main():
    st.title("2D Virtual Mapping System")
    st.subheader("Computer Vision Capstone Project")
    
    st.markdown("""
    <div class="glass-container">
        <p>This platform serves as the primary interface for our Computer Vision Capstone project. 
        The system demonstrates a multi-stage pipeline designed to transform standard broadcast 
        football footage into a virtual 2D tactical map, utilizing computer vision models 
        for player, ball, and key point detection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Pipeline Architecture")
    
    # Professional Diagram using Graphviz
    st.graphviz_chart('''
    digraph G {
        bgcolor="transparent"
        rankdir=LR
        node [shape=box, style=filled, fillcolor="#1E293B", color="#334155", fontcolor="#F8FAFC", fontname="Inter"]
        edge [color="#2563EB", penwidth=2]
        
        Input [label="Video Source", shape=ellipse]
        YOLO [label="Player & Ball Detection\\n(YOLO26n)"]
        Tracker [label="ByteTrack\\n(Temporal Association)"]
        Keypoints [label="Pitch Keypoints\\n(YOLO8n-pose)"]
        Homography [label="Homography Matrix\\n(RANSAC)"]
        SigLIP [label="Team Classification\\n(SigLIP ViT)"]
        Output [label="2D Tactical Map", shape=ellipse]
        
        Input -> YOLO
        YOLO -> Tracker
        Tracker -> SigLIP
        Input -> Keypoints
        Keypoints -> Homography
        Homography -> Output
        SigLIP -> Output
    }
    ''')

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Core Components
        
        #### 1. Player Detection and Tracking
        Utilizing a fine-tuned YOLO26n architecture trained on the SoccerNet dataset, the system identifies players and the ball with high precision. **ByteTrack** is employed to maintain consistent identities across occlusions and camera movements.
        
        #### 2. Pitch Registration
        The registration module identifies static pitch features—such as center circles, penalty areas, and touchlines—using a fine-tuned YOLO8n-pose model on the Roboflow dataset. By calculating the **Homography Matrix** via RANSAC, we map pixel coordinates to real-world metric coordinates on a standard 105x68m pitch.
        """)
        
    with col2:
        st.markdown("""
        #### 3. Team Classification
        A high-performance Vision Transformer (**SigLIP**) extracts deep visual features from individual player crops. These features are processed through a classification head to group players into teams and identify the goalkeeper and officials, even under varying lighting and jersey styles.
        
        #### 4. Tactical Mapping
        The final output projected onto a 2D digital representation of the field provides an objective view of team formations, player positioning, and spatial control, independent of the broadcast camera's perspective.
        """)

    st.markdown("""
    <div class="app-footer">
        2D Virtual Mapping System &nbsp;|&nbsp; Shepherd University CIS Capstone Project - Peyton Hollis, Curtis Canby, Uchenna Ibe &nbsp;|&nbsp; Built with Streamlit and Ultralytics YOLO, utlizes SoccerNet and Roboflow datasets. 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
