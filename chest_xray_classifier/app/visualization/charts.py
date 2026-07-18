"""
Visualization and charts component.
Generates plotting overlays, probability bar charts, and training progression plots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

def plot_probability_bar_chart(probabilities: dict, predicted_class: str) -> plt.Figure:
    """Generate a horizontal bar chart of class probabilities with the predicted class highlighted."""
    df = pd.DataFrame([
        {"Class": k, "Probability": v * 100.0} for k, v in probabilities.items()
    ]).sort_values(by="Probability", ascending=True)
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    # Highlight predicted class
    colors = ['#38bdf8' if c == predicted_class else '#1e293b' for c in df["Class"]]
    edge_colors = ['#0ea5e9' if c == predicted_class else '#475569' for c in df["Class"]]
    
    bars = ax.barh(df["Class"], df["Probability"], color=colors, edgecolor=edge_colors, height=0.6)
    
    # Add values text
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 2, bar.get_y() + bar.get_height()/2,
            f"{width:.1f}%",
            va='center', ha='left', fontsize=9, color='#94a3b8', fontweight='bold'
        )
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#475569')
    ax.spines['bottom'].set_color('#475569')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.set_xlabel("Probability (%)", color='#94a3b8', fontsize=9)
    ax.set_xlim(0, 110)
    fig.patch.set_facecolor('#070d19')
    ax.set_facecolor('#070d19')
    plt.tight_layout()
    return fig

def plot_preprocessed_preview(original_image_path, preprocessed_tensor) -> plt.Figure:
    """Plot side-by-side original image and preprocessed image (rescaled)."""
    from PIL import Image
    orig = Image.open(original_image_path)
    
    prep = preprocessed_tensor[0] # Shape: (224, 224, 3)
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title("Original (Raw)", color='#94a3b8', fontsize=10)
    axes[0].axis('off')
    
    axes[1].imshow(prep)
    axes[1].set_title("Preprocessed (224x224, Norm)", color='#94a3b8', fontsize=10)
    axes[1].axis('off')
    
    fig.patch.set_facecolor('#070d19')
    plt.tight_layout()
    return fig
