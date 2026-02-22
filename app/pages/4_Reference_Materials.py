import streamlit as st
import os
import json
from pathlib import Path

st.set_page_config(page_title="Resources", page_icon="📚", layout="wide")

st.title("📚 Resources & References")
st.markdown("Add and manage your links, references, and documents here.")

# --- DIRECTORY SETUP ---
# Create a data directory in the app root to store resources persistently
RESOURCES_DIR = Path(__file__).parent.parent / "data" / "resources"
DOCS_DIR = RESOURCES_DIR / "documents"
LINKS_FILE = RESOURCES_DIR / "links.json"

# Ensure directories exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize links file if it doesn't exist
if not LINKS_FILE.exists():
    with open(LINKS_FILE, "w") as f:
        json.dump([], f)

def load_links():
    try:
        with open(LINKS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


# --- TABS ---
tab1, tab2 = st.tabs(["📄 Documents", "🔗 Links & References"])

with tab1:
    st.header("Saved Documents")
    
    docs = list(DOCS_DIR.glob("*"))
    if docs:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 **{doc.name}**")
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
        st.info("No documents uploaded yet.")

with tab2:
    st.header("Saved Links & References")
    
    links = load_links()
    if links:
        for i, link in enumerate(reversed(links)):
            st.markdown(f"**[{link['title']}]({link['url']})**")
            if link['description']:
                st.write(link['description'])
            st.write("---")
    else:
        st.info("No links added yet.")
