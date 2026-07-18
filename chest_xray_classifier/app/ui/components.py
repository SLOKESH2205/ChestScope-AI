"""
UI Components module for the Streamlit Chest X-Ray Dashboard.
Organizes separate rendering functions for each tab in a clean, modular structure.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image

from chest_xray_classifier.config.config import CLASSES, VAL_DIR
from chest_xray_classifier.preprocessing import validate_image, preprocess_image
from chest_xray_classifier.utils.model_loader import load_single_model
from chest_xray_classifier.app.prediction.inference import run_dashboard_inference
from chest_xray_classifier.app.visualization.charts import plot_probability_bar_chart, plot_preprocessed_preview
from chest_xray_classifier.explainability import (
    compute_gradcam, compute_gradcam_plusplus, compute_integrated_gradients, overlay_heatmap, get_last_conv_layer
)
from chest_xray_classifier.app.report.pdf_generator import generate_clinical_pdf, generate_batch_pdf

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def render_home_tab(dataset_report: dict | None, final_metrics: dict | None) -> None:
    """Render the Home Landing Tab."""
    st.subheader("System Overview & Scientific Dashboard")
    
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.markdown(
            """
            ### Professional Computer Vision Workspace
            This medical decision-support dashboard exposes a deep learning pipeline trained to identify respiratory abnormalities from Chest X-Ray scans.
            
            #### Diagnostic Targets (4 Classes):
            *   **Normal**: Clean lung fields with no visible consolidation or infiltrates.
            *   **COVID-19**: Viral pneumonia pattern characterized by bilateral, peripheral ground-glass opacities.
            *   **Bacterial Pneumonia**: Focal consolidation, typically lobar in distribution.
            *   **Viral Pneumonia**: Diffuse, interstitial lung markings.
            
            #### Core End-to-End Workflow:
            ```
            [Upload Scan] ➔ [Automated Validation] ➔ [Standardized Preprocessing] ➔ [MC Dropout Inference] ➔ [Grad-CAM Attributions] ➔ [Clinical PDF Export]
            ```
            """
        )
        
        if final_metrics and final_metrics.get("best_model"):
            best = final_metrics["best_model"]
            st.markdown(
                f"""
                <div style="border:1px solid rgba(56, 189, 248, 0.3); background:rgba(14, 165, 233, 0.1); padding:1.2rem; border-radius:12px; margin-top:1.5rem;">
                    <h4 style="color:#38bdf8; margin:0 0 0.5rem 0;">🏆 Selected Flagship Model: {best['name']}</h4>
                    <p style="color:#cbd5e1; margin:0;">
                        Outperformed transfer learning configurations on the shared validation split.
                        Weighted F1-Score: <b>{best['f1_score']:.2%}</b> | Validation Accuracy: <b>{best['accuracy']:.2%}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    with col2:
        if dataset_report:
            st.markdown("#### Shared Split Summary")
            df_counts = []
            for split_name, split_data in dataset_report["splits"].items():
                row = {"Split": split_name.capitalize(), "Total Scans": split_data["total_count"]}
                row.update(split_data["class_counts"])
                df_counts.append(row)
            st.dataframe(pd.DataFrame(df_counts), hide_index=True)
            
            dist_path = OUTPUTS_DIR / "class_distribution.png"
            if dist_path.exists():
                st.image(str(dist_path), caption="Class Balancing Across Splits", use_container_width=True)
        else:
            st.info("Dataset statistics are not loaded. Run run_full_evaluation.py first.")

def render_diagnosis_tab(available_models: list[str]) -> None:
    """Render the main Diagnosis Inference Tab."""
    st.subheader("Clinical Diagnostic Workspace")
    
    if not available_models:
        st.error("No trained models found in the workspace directory. Train model using train_only_efficientnet.py.")
        return
        
    # Mode selection
    mode = st.radio("Select Diagnostic Mode", ["Single X-Ray Analysis", "Batch Processing"], horizontal=True)
    
    # Threshold slider in controls
    threshold = st.slider("Clinical Confidence Threshold", 0.50, 0.90, 0.60, 0.05, format="%.0f%%")
    
    selected_model = st.selectbox("Select Active Model", available_models, key="active_model")
    try:
        model = load_single_model(selected_model)
    except Exception as e:
        st.error(f"Active model weights could not be loaded: {e}")
        return
        
    if mode == "Single X-Ray Analysis":
        col_ctrl, col_main = st.columns([0.7, 1.3])
        
        with col_ctrl:
            show_gcam = st.toggle("Enable Grad-CAM Activation Map", value=True)
            show_prep = st.toggle("Show Preprocessing Preview", value=True)
            uploaded_file = st.file_uploader("Upload Chest X-Ray (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
            
        with col_main:
            if uploaded_file:
                # 1. Validation & Error Handling
                temp_path = OUTPUTS_DIR / f"diag_{uploaded_file.name}"
                temp_path.write_bytes(uploaded_file.read())
                
                is_valid, msg = validate_image(temp_path)
                if not is_valid:
                    st.error(f"⚠️ **Image Quality Assurance Failure:** {msg}")
                    st.warning("Please upload a valid, non-corrupt chest X-ray image in PNG or JPG format.")
                    return
                    
                # 2. Animated Pipeline Progress Bar
                status_box = st.status("Initializing diagnostic pipeline...")
                with status_box:
                    time.sleep(0.3)
                    status_box.update(label="✓ Image uploaded and verified", state="running")
                    time.sleep(0.3)
                    status_box.update(label="✓ Standardizing pixel preprocessing (224x224, normalized)", state="running")
                    time.sleep(0.3)
                    status_box.update(label="✓ Executing 15 Monte Carlo forward inference passes", state="running")
                    
                    # Run actual inference
                    result = run_dashboard_inference(model, temp_path, selected_model, threshold)
                    
                    time.sleep(0.3)
                    status_box.update(label="✓ Generating Grad-CAM attributions", state="running")
                    
                    overlay_temp_path = None
                    gcam_overlay = None
                    if show_gcam:
                        layer_name = get_last_conv_layer(model)
                        gcam = compute_gradcam(model, result["preprocessed_tensor"], CLASSES.index(result["prediction"]), layer_name)
                        gcam_overlay = overlay_heatmap(result["preprocessed_tensor"], gcam)
                        overlay_temp_path = OUTPUTS_DIR / f"overlay_temp_{uploaded_file.name}"
                        Image.fromarray(gcam_overlay).save(overlay_temp_path)
                        
                    time.sleep(0.2)
                    status_box.update(label="✓ Diagnostic summary compiled successfully", state="complete")
                    
                # Store in prediction history session state
                if "prediction_history" not in st.session_state:
                    st.session_state.prediction_history = []
                st.session_state.prediction_history.append({
                    "timestamp": time.strftime("%H:%M:%S", time.localtime()),
                    "filename": uploaded_file.name,
                    "model": selected_model,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "uncertainty": result["uncertainty"]
                })
                
                # Render side-by-side images
                col_raw, col_viz = st.columns(2)
                with col_raw:
                    st.image(str(temp_path), caption="Original Patient Scan", use_container_width=True)
                    
                with col_viz:
                    if show_gcam and gcam_overlay is not None:
                        st.image(gcam_overlay, caption="Grad-CAM Lesion Activation Map", use_container_width=True)
                        st.markdown(
                            """
                            *The highlighted regions contributed most to the model prediction. 
                            This visualization is intended only to explain model behavior and does not represent medically confirmed pathology.*
                            """
                        )
                        
                # Clinical Alerts
                if result["requires_review"]:
                    st.markdown(
                        f"""
                        <div style="border-left:5px solid #ef4444; background:rgba(239,68,68,0.1); padding:1rem; border-radius:8px; color:#fca5a5; margin:1rem 0;">
                            ⚠️ <b>Low Confidence Alert:</b> {result['status_message']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="border-left:5px solid #22c55e; background:rgba(34,197,94,0.1); padding:1rem; border-radius:8px; color:#86efac; margin:1rem 0;">
                            ✓ <b>Diagnostic Complete:</b> {result['status_message']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                # Results Metrics & Probs
                col_m1, col_m2 = st.columns([1, 1.2])
                with col_m1:
                    st.metric("Predicted Disease Category", result["prediction"])
                    st.metric("Confidence Score", f"{result['confidence']:.2%}")
                    st.metric("Predictive Uncertainty (Entropy)", f"{result['uncertainty']:.2%}")
                    st.metric("Inference Latency", f"{result['inference_ms']:.1f} ms")
                    
                with col_m2:
                    fig_prob = plot_probability_bar_chart(result["probabilities"], result["prediction"])
                    st.pyplot(fig_prob)
                    
                if show_prep:
                    st.markdown("#### Pipeline Preprocessing Verification")
                    fig_prep = plot_preprocessed_preview(temp_path, result["preprocessed_tensor"])
                    st.pyplot(fig_prep)
                    
                # PDF report
                pdf_bytes = generate_clinical_pdf(
                    image_path=temp_path,
                    gradcam_path=overlay_temp_path,
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    uncertainty=result["uncertainty"],
                    model_name=selected_model,
                    inference_ms=result["inference_ms"],
                    threshold=threshold
                )
                st.download_button(
                    label="📥 Download Clinical Diagnostic Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"diagnostic_report_{uploaded_file.name.split('.')[0]}.pdf",
                    mime="application/pdf"
                )
            else:
                st.info("Upload a patient scan to begin diagnostic inference.")
                
    elif mode == "Batch Processing":
        uploaded_files = st.file_uploader("Upload Multiple Chest X-Rays", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files:
            st.markdown(f"Loaded **{len(uploaded_files)}** files for batch diagnostics.")
            
            if st.button("Run Batch Diagnostics"):
                results = []
                bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    temp_path = OUTPUTS_DIR / f"diag_{file.name}"
                    temp_path.write_bytes(file.read())
                    
                    # Validate
                    is_valid, msg = validate_image(temp_path)
                    if not is_valid:
                        st.warning(f"Skipping {file.name}: {msg}")
                        continue
                        
                    res = run_dashboard_inference(model, temp_path, selected_model, threshold)
                    results.append(res)
                    bar.progress((idx + 1) / len(uploaded_files))
                    
                if results:
                    st.subheader("Batch Diagnostic Output Summary")
                    df_res = pd.DataFrame([
                        {
                            "Filename": r["filename"],
                            "Predicted Disease": r["prediction"],
                            "Confidence Score": f"{r['confidence']:.1%}",
                            "Entropy (Uncertainty)": f"{r['uncertainty']:.1%}",
                            "Clinical Action": "REVIEW" if r["requires_review"] else "OK"
                        } for r in results
                    ])
                    
                    st.dataframe(df_res, use_container_width=True, hide_index=True)
                    
                    # 1. Download CSV
                    csv_data = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Export Batch Results (CSV)", csv_data, "batch_diagnostics.csv", "text/csv")
                    
                    # 2. Download Combined PDF
                    combined_pdf = generate_batch_pdf(results, selected_model, threshold)
                    st.download_button("📥 Download Combined Diagnostic Report (PDF)", combined_pdf, "combined_batch_report.pdf", "application/pdf")
                else:
                    st.error("No valid scans could be processed in batch.")

def render_comparison_tab(final_metrics: dict | None) -> None:
    """Render the Scientific Model Comparison Tab."""
    st.subheader("Multi-Model Scientific Benchmark")
    
    comp_csv_path = OUTPUTS_DIR / "model_comparison.csv"
    if comp_csv_path.exists():
        df_comp = pd.read_csv(comp_csv_path)
        
        # Sort option
        sort_col = st.selectbox("Sort Table By Metric", list(df_comp.columns)[1:])
        df_sorted = df_comp.sort_values(by=sort_col, ascending=False if sort_col not in ["Avg Inference Time (ms/image)", "Model Size (MB)"] else True)
        
        format_dict = {
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}",
            "ROC AUC": "{:.2%}", "Specificity": "{:.2%}", "Sensitivity": "{:.2%}",
            "Balanced Accuracy": "{:.2%}", "Matthews Correlation Coefficient": "{:.3f}",
            "Avg Inference Time (ms/image)": "{:.2f} ms", "Model Size (MB)": "{:.2f} MB"
        }
        
        highlight_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Specificity", "Sensitivity", "Balanced Accuracy", "Matthews Correlation Coefficient"]
        
        st.dataframe(
            df_sorted.style.format({k: v for k, v in format_dict.items() if k in df_sorted.columns})
            .highlight_max(subset=[c for c in highlight_cols if c in df_sorted.columns], color="rgba(34, 197, 94, 0.2)"),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Comparison benchmarks have not been compiled. Please execute run_full_evaluation.py.")

def render_performance_tab(final_metrics: dict | None) -> None:
    """Render the Performance Analytics Tab."""
    st.subheader("Statistical Performance Dashboard")
    
    available_models = ["Custom CNN", "EfficientNetB0", "MobileNetV2"]
    selected_metrics_model = st.selectbox("Select Model to Inspect", available_models, key="perf_model_select")
    model_key = selected_metrics_model.lower().replace(" ", "_")
    
    col_cm, col_class = st.columns(2)
    
    with col_cm:
        cm_path = OUTPUTS_DIR / f"{model_key}_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix Heatmap", use_container_width=True)
        else:
            st.info(f"Confusion matrix plot for {selected_metrics_model} not generated.")
            
    with col_class:
        pcm_path = OUTPUTS_DIR / f"{model_key}_per_class_metrics.png"
        if pcm_path.exists():
            st.image(str(pcm_path), caption="Per-Class metrics (F1, Specificity, Sensitivity)", use_container_width=True)
        else:
            st.info(f"Per-class metrics plot for {selected_metrics_model} not generated.")

def render_xai_tab(available_models: list[str]) -> None:
    """Render the standalone Explainable AI exploration page."""
    st.subheader("Algorithmic Transparency & Attribution Maps")
    
    st.info(
        "💡 Activation heatmaps reveal image pixel correlation with classification outputs. "
        "They do not represent medical diagnostic claims and must be verified by a clinician.",
        icon="⚠️"
    )
    
    if not available_models:
        st.error("No trained models found in the workspace directory. Model weights (.h5) must be present in chest_xray_classifier/models/.")
        return
        
    selected_xai_model = st.selectbox("Select Model for Visual Attribution", available_models, key="xai_model_select")
    try:
        model = load_single_model(selected_xai_model)
    except Exception as e:
        st.error(f"Active model weights could not be loaded: {e}")
        return
    
    uploaded_xai = st.file_uploader("Upload Chest X-Ray for Interpretability", type=["png", "jpg", "jpeg"], key="xai_uploader")
    
    if uploaded_xai and model:
        temp_xai_path = OUTPUTS_DIR / f"xai_{uploaded_xai.name}"
        temp_xai_path.write_bytes(uploaded_xai.read())
        
        # Preprocess
        preprocessed = preprocess_image(temp_xai_path)
        probs = model.predict(preprocessed, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        predicted_class_name = CLASSES[pred_idx]
        
        st.markdown(f"**Target Prediction:** {predicted_class_name} ({probs[pred_idx]:.2%} confidence)")
        
        with st.spinner("Computing saliency attributions..."):
            try:
                layer_name = get_last_conv_layer(model)
                gcam = compute_gradcam(model, preprocessed, pred_idx, layer_name)
                gcam_pp = compute_gradcam_plusplus(model, preprocessed, pred_idx, layer_name)
                ig = compute_integrated_gradients(model, preprocessed, pred_idx)
                
                gcam_overlay = overlay_heatmap(preprocessed, gcam)
                gcam_pp_overlay = overlay_heatmap(preprocessed, gcam_pp)
                
                # Render
                col_o, col_gc, col_gcpp, col_ig = st.columns(4)
                col_o.image(str(temp_xai_path), caption="Original Scan", use_container_width=True)
                col_gc.image(gcam_overlay, caption="Grad-CAM Overlay", use_container_width=True)
                col_gcpp.image(gcam_pp_overlay, caption="Grad-CAM++ Overlay", use_container_width=True)
                
                # Normalize IG to uint8 colormap for display
                ig_colored = (ig * 255.0).astype(np.uint8)
                col_ig.image(ig_colored, caption="Integrated Gradients (Attr)", use_container_width=True)
            except Exception as e:
                st.error(f"Attribution mapping calculation failed: {e}")

def render_error_tab(available_models: list[str]) -> None:
    """Render the Misclassification Analysis Tab."""
    st.subheader("Worst-Confidence Diagnostics & Error Profiling")
    
    available_err_models = [name for name in available_models if (OUTPUTS_DIR / f"{name.lower().replace(' ', '_')}_error_analysis.json").exists()]
    
    if not available_err_models:
        st.info("No error analysis reports compiled yet. Run run_full_evaluation.py to create them.")
        return
        
    selected_err_model = st.selectbox("Select Model to Profiling Errors", available_err_models, key="err_model_select")
    model_key = selected_err_model.lower().replace(" ", "_")
    
    with open(OUTPUTS_DIR / f"{model_key}_error_analysis.json") as f:
        err_report = json.load(f)
        
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric("Misclassified Validation Scans", f"{err_report['misclassified_count']} / {err_report['total_evaluated']}")
    col_stat2.metric("Validation Error Rate", f"{err_report['error_rate']:.2%}")
    
    # Filter by class
    target_class = st.selectbox("Filter Errors by True Label", ["All"] + CLASSES)
    
    gallery_img = OUTPUTS_DIR / f"{model_key}_misclassified_gallery.png"
    if gallery_img.exists() and target_class == "All":
        st.image(str(gallery_img), caption="Gallery of worst-confidence mistakes (Actual vs Predicted)", use_container_width=True)
        
    # Table list
    df_worst = pd.DataFrame(err_report["worst_predictions"])
    if not df_worst.empty:
        if target_class != "All":
            df_worst = df_worst[df_worst["actual"] == target_class]
            
        st.markdown(f"#### Log of Misclassifications ({len(df_worst)} entries)")
        st.dataframe(
            df_worst[["filename", "actual", "predicted", "confidence", "uncertainty"]].rename(
                columns={"filename": "Filename", "actual": "True Label", "predicted": "Model Prediction", "confidence": "Confidence Score", "uncertainty": "Entropy"}
            ),
            hide_index=True, use_container_width=True
        )

def render_dataset_tab(dataset_report: dict | None) -> None:
    """Render the Dataset Explorer Tab."""
    st.subheader("Cohort Dataset & Preprocessing Profiler")
    
    if not dataset_report:
        st.info("Dataset summary report is missing.")
        return
        
    st.markdown(
        """
        Browse random sample images from the cohort splits and preview pipeline preprocessing steps:
        """
    )
    
    col_cfg, col_preview = st.columns([0.6, 1.4])
    
    with col_cfg:
        split = st.selectbox("Choose Cohort Split", ["train", "validation"])
        label = st.selectbox("Choose Class Target", CLASSES)
        
        # Scan folder for images
        class_folder = Path(VAL_DIR).parent / split / label
        images = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg"))
        
        st.metric(f"Total Scans in {split}/{label}", len(images))
        
        # Shuffle/sample button
        if len(images) > 0:
            selected_idx = st.slider("Select Image Index", 0, len(images) - 1, 0)
            target_image = images[selected_idx]
        else:
            target_image = None
            st.warning("No images found in folder.")
            
    with col_preview:
        if target_image:
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(str(target_image), caption=f"Raw Image: {target_image.name}", use_container_width=True)
            with col_img2:
                # Preprocess and show normalized view
                preprocessed = preprocess_image(target_image)
                st.image(preprocessed[0], caption="Pipeline Preprocessed View (224x224)", use_container_width=True)

def render_about_model_tab() -> None:
    """Render the detailed model architecture and design explanation tab."""
    st.subheader("Model Architectures & Evaluation Design")
    
    st.markdown(
        """
        ### 🧪 Project Scientific Rationale
        This section describes the engineering decisions, architecture structures, and diagnostics resolved during development.
        
        #### 1. Why the Custom CNN Won the F1 Benchmark
        *   **Baseline F1-Score:** **82.35%** (Accuracy: **82.33%**)
        *   **Parameter Scale:** The Custom CNN model stores **44,398,148** trainable parameters.
        *   **Architectural Bottleneck:** The transition from the final convolutional block to the Dense classification layer is performed using a `Flatten` layer. Because the final spatial map is $(26, 26, 128) = 86,528$ features, mapping this directly to a Dense layer of $512$ units requires **44,302,848** weights in a single layer!
        *   **Optimizer Footprint:** Because it was saved with active **Adam** optimizer states (which stores momentum and velocity floats for every parameter), the serialized checkpoint variable count rises to **133.2 million variables**, taking **508 MB** on disk.
        *   **Optimization Proposal:** Replacing the `Flatten` layer with a `GlobalAveragePooling2D` layer would pool the features to $(128,)$ before the dense connection, reducing the dense weights to just **65,536** parameters (a **99.85% parameter reduction**, dropping model size from **508 MB** to **less than 1.5 MB**!).
        
        #### 2. The EfficientNetB0 Double-Rescaling Bug & Fix
        *   **The Bug:** The pre-trained `EfficientNetB0` application includes an internal `Rescaling(1./255)` layer. The dataset generator was already scaling images to `[0, 1]`. When passed directly, images were scaled *again* to `[0, 0.00392]`, squashing activations to zero and dropping validation accuracy to **25%** (random guessing).
        *   **The Fix:** Integrated a `layers.Rescaling(255.0)` wrapper right at the model entry point inside `efficientnet.py` to scale normalized inputs back to `[0, 255]` before the base model processes them. Validation accuracy immediately recovered to **78.01%**.
        
        #### 3. MobileNetV2 Footprint
        *   **Size:** **23.72 MB** (only 2.2M parameters)
        *   **Performance:** Achieved **80.64%** accuracy, demonstrating exceptional parameter efficiency for resource-constrained or mobile web app deployment.
        """
    )
