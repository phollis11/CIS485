# Streamlit Pages (`app/pages/`)

This directory is utilized natively by Streamlit to structure a **Multi-Page Application**. Any Python script placed in this directory will automatically populate the sidebar of our central dashboard `app_main.py`, providing straightforward routing and navigation without the need for complex web backend tools.

## Included Pages

* **`1_Demonstration_Video.py`**: A display suite for static, pre-recorded, and pre-annotated footage. Allows reviewers to immediately preview perfect-run examples of YOLO tracking overlaid on Premier League broadcast footage.
* **`2_Run_Pipeline.py`**: The dynamic interactive tab. Upload your own video feeds and trigger the CV backend processing scripts asynchronously. Once completed, the tracking map results are streamed back directly inside the page.
* **`3_Documents_Citations.py`**: Central hub referencing white papers, IEEE articles, and models used to build this architecture, such as YOLOv8 pose logic and SigLIP ViT configurations.
