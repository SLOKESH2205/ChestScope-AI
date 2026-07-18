"""Streamlit entrypoint for the chest X-ray classification dashboard."""

import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from chest_xray_classifier.app.streamlit_app import main


if __name__ == "__main__":
    main()
