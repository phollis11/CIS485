# App Resources (`app/data/`)

This directory functions as the central storage repository for localized frontend resources strictly required by the Streamlit user interfaces. It enforces a structural boundary between heavy backend pipeline dependencies (managed in `../pipeline/` or `../weights/`) versus GUI-specific data loading.

## Contents

- **Local Dictionaries & Cached Sets**: Caches specific metadata utilized when the app boots up quickly to prevent heavy re-loads of static dictionaries.
- **Resource Media/Static Images**: Demonstration videos, static diagram caches, and placeholder profile avatars or interface icons used in `app_main.py` and `pages/`.
- **Knowledge References (`resources/antigravity_docs/`)**: Local reference `.md` files that define structural layouts and dataset configurations for quick context fetching.
