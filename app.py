import sys
from pathlib import Path

# Add project root to sys.path to allow consistent package imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from chest_xray_classifier.app.streamlit_app import main

if __name__ == "__main__":
    main()
