import streamlit as st
import os
from pathlib import Path

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent.parent / "style.css"
if css_path.exists():
    local_css(css_path)

st.title("🗺️ 2D Pitch Mapping")

st.markdown("""
<div class="glass-container">
    <p>This page is designed to allow users to view a sample 2D mapping of the player tracking and key point detection models. It serves as another demonstration of the pipeline's capabilities to map real-world camera coordinates onto a standardized digital 2D pitch.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Since the user requested it to just be another demo of the 2D mapping
st.info("The sample 2D mapping capability is currently demonstrated within the YOLO8n demo (see the 'Model Demonstrations' page) where the minimap is rendered alongside player tracks. Further isolated 2D mapping visualizations will be placed here.")
