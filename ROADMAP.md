# ChestScope-AI Roadmap

Future updates and milestones planned for ChestScope-AI:

## 🟩 Q3 2026: Global Average Pooling (GAP) Optimization
- Rebuild Custom CNN's head using `GlobalAveragePooling2D` to shrink parameters from 44.4 million to 65,536.
- Re-train the GAP optimized Custom CNN on the full dataset and check validation metrics.

## 🟩 Q4 2026: Multi-Image Saliency Maps
- Extend Grad-CAM and Grad-CAM++ to batch outputs.
- Display a combined PDF report containing heatmaps for every batch scan.

## 🟩 Q1 2027: WSL2 GPU Training Guides
- Document local execution setup for WSL2 to enable Windows GPU training support.
- Incorporate MLflow tracking server details.
