"""
Modular Streamlit Dashboard for Chest X-Ray Disease Classification.
Main entry point. Orchestrates tab layout, sidebar metadata, and calls separate UI component modules.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from chest_xray_classifier.app.ui.components import (
    render_home_tab,
    render_diagnosis_tab,
    render_comparison_tab,
    render_performance_tab,
    render_xai_tab,
    render_error_tab,
    render_dataset_tab,
    render_about_model_tab
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("streamlit_app")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "chest_xray_classifier" / "models"

# Available model files check
AVAILABLE_MODELS = []
if (MODELS_DIR / "custom_cnn.h5").exists():
    AVAILABLE_MODELS.append("Custom CNN")
if (MODELS_DIR / "efficientnetb0.h5").exists():
    AVAILABLE_MODELS.append("EfficientNetB0")
if (MODELS_DIR / "mobilenetv2.h5").exists():
    AVAILABLE_MODELS.append("MobileNetV2")

def apply_custom_styles() -> None:
    """Inject modern Glassmorphism styling and fonts into Streamlit."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
        
        :root {
            --primary: #38bdf8;
            --success: #22c55e;
            --warning: #ef9f27;
            --danger: #ef4444;
            --background: #020617;
            --card-bg: rgba(15, 23, 42, 0.65);
            --border: rgba(148, 163, 184, 0.15);
        }
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background: 
                radial-gradient(circle at 5% 10%, rgba(56, 189, 248, 0.12), transparent 400px),
                radial-gradient(circle at 95% 85%, rgba(34, 197, 94, 0.08), transparent 400px),
                linear-gradient(180deg, #020617 0%, #0b1329 60%, #0f172a 100%);
        }
        
        /* Hero Banner */
        .hero-section {
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2.2rem;
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(15, 23, 42, 0.8) 100%);
            backdrop-filter: blur(12px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            margin-bottom: 2rem;
        }
        .hero-section h1 {
            margin: 0;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(90deg, #38bdf8, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-section p {
            color: #94a3b8;
            margin-top: 0.8rem;
            font-size: 1.15rem;
            max-width: 900px;
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            border: 1px solid var(--border);
            background: var(--card-bg);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            margin-bottom: 1rem;
        }
        
        div[data-testid="stMetric"] {
            border: 1px solid var(--border);
            background: rgba(30, 41, 59, 0.5);
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_sidebar_info() -> None:
    """Render the deployment and library specifications in the sidebar."""
    import cv2
    st.sidebar.markdown("### 🛠️ Workspace Deployment")
    
    st.sidebar.markdown(
        f"""
        *   **Python:** `{sys.version.split(' ')[0]}`
        *   **TensorFlow:** `{tf.__version__}`
        *   **Streamlit:** `{st.__version__}`
        *   **OpenCV:** `{cv2.__version__}`
        *   **NumPy:** `{np.__version__}`
        
        ---
        *   **Model version:** `v1.2.0-stable`
        *   **Last updated:** `2026-07-18`
        *   **Git Hash:** `8f4ab2d`
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size:0.8rem; color:#94a3b8; line-height:1.2;">
            <b>Clinical disclaimer:</b> This application is built for research & portfolio purposes. It does NOT provide medically valid diagnostics.
        </div>
        """,
        unsafe_allow_html=True
    )

def main() -> None:
    st.set_page_config(
        page_title="Chest X-Ray Diagnostic Workspace",
        page_icon="🫁",
        layout="wide"
    )
    apply_custom_styles()
    render_sidebar_info()
    
    # Initialize prediction history session state
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    # Hero section
    st.markdown(
        """
        <div class="hero-section">
            <h1>🫁 Chest X-Ray AI Diagnostic Workspace</h1>
            <p>Clinical Decision-Support Dashboard with Monte Carlo Uncertainty and Explainable AI (Grad-CAM attributions).</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load files
    dataset_report = None
    if (OUTPUTS_DIR / "dataset_report.json").exists():
        with open(OUTPUTS_DIR / "dataset_report.json") as f:
            dataset_report = json.load(f)
            
    final_metrics = None
    if (OUTPUTS_DIR / "final_metrics.json").exists():
        with open(OUTPUTS_DIR / "final_metrics.json") as f:
            final_metrics = json.load(f)
            
    # Setup Tabs
    tabs = st.tabs([
        "🏠 Home",
        "🩺 Diagnosis",
        "📊 Model Comparison",
        "📈 Performance Analytics",
        "🔥 Explainability",
        "⚠ Error Analysis",
        "📁 Cohort Dataset",
        "📁 Session History",
        "ℹ About the Model"
    ])
    
    with tabs[0]:
        render_home_tab(dataset_report, final_metrics)
        
    with tabs[1]:
        render_diagnosis_tab(AVAILABLE_MODELS)
        
    with tabs[2]:
        render_comparison_tab(final_metrics)
        
    with tabs[3]:
        render_performance_tab(final_metrics)
        
    with tabs[4]:
        render_xai_tab(AVAILABLE_MODELS)
        
    with tabs[5]:
        render_error_tab(AVAILABLE_MODELS)
        
    with tabs[6]:
        render_dataset_tab(dataset_report)
        
    with tabs[7]:
        st.subheader("Logged Diagnostic Session Predictions")
        if not st.session_state.prediction_history:
            st.info("No predictions have been made in the current session yet.")
        else:
            df_hist = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            
            # Export CSV
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Session History CSV",
                data=csv,
                file_name="diagnostic_session_history.csv",
                mime="text/csv"
            )
            
    with tabs[8]:
        render_about_model_tab()

if __name__ == "__main__":
    main()
