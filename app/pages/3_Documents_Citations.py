import streamlit as st
import os
import json
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Documents & Citations", layout="wide")

# --- INJECT CUSTOM CSS ---
def local_css(file_name):
    if Path(file_name).exists():
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent.parent / "style.css"
local_css(css_path)

def main():
    st.title("Capstone Documents, Materials, and References")
    
    st.markdown("""
    <div class="glass-container">
        <p>This repository stores the primary research, technical documentation, and references 
        supporting the development of this 2D virtual mapping system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- DIRECTORY SETUP ---
    RESOURCES_DIR = Path(__file__).parent.parent / "data" / "resources"
    DOCS_DIR = RESOURCES_DIR / "documents"
    ANTI_DIR = RESOURCES_DIR / "antigravity_docs"
    
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ANTI_DIR.mkdir(parents=True, exist_ok=True)

    tab1, tab2, tab3 = st.tabs(["Capstone Documents", "Academic Citations", "Antigravity Reports"])
    
    with tab1:
        st.header("Document Archive")
        
        # Display existing docs
        docs = list(DOCS_DIR.glob("*"))
        if docs:
            for doc in docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"<p style='margin:0;'>{doc.name}</p>", unsafe_allow_html=True)
                with col2:
                    with open(doc, "rb") as f:
                        st.download_button(
                            label="Download", 
                            data=f, 
                            file_name=doc.name, 
                            mime="application/octet-stream", 
                            key=f"dl_{doc.name}"
                        )
        else:
            st.info("No documents have been archived in this repository.")

    with tab2:
        st.header("Citations and References")
        
        st.markdown("""
        The following references and datasets have been instrumental in the research and development of this automated tactical mapping system:
        
        #### Research & Academic Works
        - **Fujii, K. (2025).** *Computer vision for sports analytics.* SpringerLink. [DOI/Link](https://link.springer.com/chapter/10.1007/978-981-96-1445-5_2)
        - **Jiang, L., Yang, Z., & Gang, L. (2022).** *Transformer-based multi-player tracking and skill recognition framework for volleyball analytics.* IEEE Xplore. [Source](https://ieeexplore.ieee.org/abstract/document/10830493)
        - **Kumari, S., et al. (2025).** *Hybrid vision transformer and Convolutional neural network for sports video classification.* IEEE Xplore. [Source](https://ieeexplore.ieee.org/abstract/document/10837289)
        - **Liao, W. (2025).** *Beyond Vibe Coding: Building Production-Grade Software with AI Agents and Specification-Driven Development.*
        - **Manafifard, M., Ebadi, H., & Moghaddam, H. (2017).** *A survey on player tracking in soccer videos.* ScienceDirect. [Source](https://www.sciencedirect.com/science/article/abs/pii/S1077314217300309)
        - **Tian, Y., et al. (2026).** *Development and evolution of YOLO in object detection: A survey.* ScienceDirect. [Source](https://www.sciencedirect.com/science/article/abs/pii/S092523122503108X)

        #### Technical Guides & Official Documentation
        - **AlwaysAI. (2026).** *12 game-changing applications of computer vision.* [alwaysAI Blog](https://alwaysai.co/blog/computer-vision-applications)
        - **GeeksforGeeks. (2025).** *Swin transformer.* [Tutorial](https://www.geeksforgeeks.org/computer-vision/swin-transformer/)
        - **Khandelwal, R. (2022).** *Evaluation Metrics for Multiple Object Tracking.* [Medium](https://arshren.medium.com/evaluation-metrics-for-multiple-object-tracking-7b26ef23ef5f)
        - **MoOngy Labs. (2025).** *Real-time player tracking and 2D Field mapping using Homography for football analytics.* [Article](https://labs.moongy.group/articles/real-time-player-tracking-and-2d-field-mapping-using-homography-for-football-analytics)
        - **OpenCV Documentation.** *Basic concepts of homography.* [Reference](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)
        - **Ultralytics YOLO Docs.** *YOLOv8 vs YOLO11 (YOLO26): Evolution of Real-time detection.* [Comparison](https://docs.ultralytics.com/compare/yolov8-vs-yolo26/)
        - **IBM - Winland, V.** *What is self-attention?* [IBM Technology](https://www.ibm.com/think/topics/self-attention)

        #### Community & Open Source Projects
        - **Rijo-1.** *Football-Analysis-using-Computer-Vision-with-Yolov8-OpenCV.* [GitHub Repository](https://github.com/Rijo-1/Football-Analysis-using-Computer-Vision-with-Yolov8-OpenCV)
        - **Roboflow.** *Football AI Tutorial: From Basics to Advanced Stats with Python.* [YouTube Video](https://www.youtube.com/watch?v=aBVGKoNZQUw)
        - **SoccerNet.** *Scalable Dataset for Computer Vision in Soccer.* [Official Data Portal](https://www.soccer-net.org/data)
        - **Statsbomb - Viloria, I. R. (2024).** *Creating better data: How to map Homography.* [Statsbomb Blog](https://blogarchive.statsbomb.com/articles/football/creating-better-data-how-to-map-homography/)
        """)

    with tab3:
        st.header("AI Analysis & Technical Reports")
        st.markdown("""
        This section lists technical deep-dives, pipeline architecture diagrams, 
        and AI-generated research artifacts produced during development.
        """)
        
        st.markdown("---")
        
        # Display existing docs
        anti_docs = list(ANTI_DIR.glob("*"))
        if anti_docs:
            for adoc in anti_docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"<p style='margin:0;'>{adoc.name}</p>", unsafe_allow_html=True)
                with col2:
                    with open(adoc, "rb") as f:
                        st.download_button(
                            label="Download", 
                            data=f, 
                            file_name=adoc.name, 
                            mime="application/octet-stream", 
                            key=f"dl_{adoc.name}"
                        )
        else:
            st.info("No AI reports have been archived yet. Add .md or .pdf files to the `antigravity_docs` directory to see them here.")

    st.markdown("""
    <div class="app-footer">
        2D Virtual Mapping System &nbsp;|&nbsp; Shepherd University CIS Capstone Project - Peyton Hollis, Curtis Canby, Uchenna Ibe &nbsp;|&nbsp; Built with Streamlit and Ultralytics YOLO, utlizes SoccerNet and Roboflow datasets. 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
