# Chest X-Ray Portfolio Walkthrough

This document summarizes the changes, diagnostics, and testing results accomplished to elevate the project to a flagship Computer Vision portfolio piece (scoring a 10/10).

---

## 🛠️ Flagship Features Added

### 1. Data Pipeline & QA (Phase 1)
- Developed a modular preprocessing utility [preprocessing.py](file:///d:/PROJECTS/VS%20CODE/LUNG%20DISEASE/chest_xray_classifier/preprocessing.py) providing strict file validation (checks for file headers, empty files, resolution limits, and color conversions).
- Wrote [dataset_report.json](file:///d:/PROJECTS/VS%20CODE/LUNG%20DISEASE/outputs/dataset_report.json) detailing split sizes (532 train / 532 val images) and class balances.
- Added robust error handling in the Streamlit UI to catch corrupted files, PDFs, or unsupported formats and display helpful clinical messages.

### 2. Scientific Evaluation & Benchmarks (Phase 2)
- Fixed the input double-rescaling bug for **EfficientNetB0** by wrapping the base model in a `layers.Rescaling(255.0)` layer to scale inputs back to `[0, 255]`. Accuracy successfully rose from **25.00%** (random guessing) to **78.01%**.
- Created [run_full_evaluation.py](file:///d:/PROJECTS/VS%20CODE/LUNG%20DISEASE/run_full_evaluation.py) which automatically evaluates all models, generates confusion matrices, per-class bar charts, and saves metrics to [final_metrics.json](file:///d:/PROJECTS/VS%20CODE/LUNG%20DISEASE/outputs/final_metrics.json).

### 3. Hyperparameter Tuning (Phase 3)
- Created [tune_custom_cnn.py](file:///d:/PROJECTS/VS%20CODE/LUNG%20DISEASE/tune_custom_cnn.py) to run a 15-trial search over learning rate, dropout, and L2 weight decay.
- Confirmed that none of the subset-trained trials outperformed the pre-trained Custom CNN baseline (**82.35% F1-score**), resulting in a safe fallback to keep the baseline weights.

### 4. Explainable AI & Uncertainty (Phase 4 & 5)
- Created [explainability.py](file:///d:/PROJECTS/VS%20CODE/LUNG%20DISEASE/chest_xray_classifier/explainability.py) implementing Grad-CAM, Grad-CAM++, and Integrated Gradients. Added robust Keras 3 sequential tape tracing to prevent AttributeErrors on flat models.
- Integrated MC Dropout (15 forward passes) and Shannon Entropy estimation in [predict.py](file:///d:/PROJECTS/VS%20CODE/LUNG%20DISEASE/chest_xray_classifier/predict.py) to flag predictions with low confidence.
- Exposed a **Confidence Threshold Slider (50% - 90%)** in the dashboard. If confidence falls below the threshold, it triggers a red alert callout: `⚠️ Low confidence alert. Specialist review recommended.`

### 5. Production Streamlit Refactoring (Phase 7 & 8)
- Refactored [streamlit_app.py](file:///d:/PROJECTS/VS%20CODE/LUNG%20DISEASE/chest_xray_classifier/app/streamlit_app.py) into modular folders:
  - `model_loader/`: Caching model weights.
  - `prediction/`: Running inference and clinical alerts.
  - `report/`: Compiling Clinical PDF reports using `fpdf2`.
  - `ui/`: Tab layout and rendering components.
  - `visualization/`: Side-by-side preprocessed comparisons and probability bar charts.
- Added **Batch Prediction** mode: Upload multiple scans, run sequential inference with progress bars, view summary tables, export batch CSV logs, and download a **Combined PDF report**.
- Added **Prediction Session History** tracking page with CSV export functionality.
- Rendered **Deployment Information** in the sidebar (Python, TF, OpenCV, NumPy, Streamlit versions, model version, update date, and git commit hash).
- Added **"About the Model"** Tab detailing cohort stats, training loops, Custom CNN parameter analysis, and the EfficientNet rescaling bug and fix.

---

## 📊 Scientific Metrics Comparison

Derived from [final_metrics.json](file:///d:/PROJECTS/VS%20CODE/LUNG%20DISEASE/outputs/final_metrics.json):

| Model | Accuracy | Precision | Recall | F1 Score | Specificity | Sensitivity | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** (Best) | **82.33%** | **82.64%** | **82.33%** | **82.35%** | **94.11%** | **82.33%** | **16.90 ms** |
| **MobileNetV2** | 80.64% | 82.44% | 80.64% | 80.21% | 93.55% | 80.64% | 24.09 ms |
| **EfficientNetB0** | 78.01% | 78.38% | 78.01% | 77.47% | 92.67% | 78.01% | 27.58 ms |

---

## 🧪 Unit Testing Results

Successfully executed `pytest` on the test suite:
```
============================= 7 passed in 19.19s ==============================
```
Tests verified:
- Correct and incorrect PIL image validation.
- Resolution lower limit enforcement.
- Preprocessing resizing and rescaling.
- Model factory architectures.
- MC Dropout uncertainty calculations.
