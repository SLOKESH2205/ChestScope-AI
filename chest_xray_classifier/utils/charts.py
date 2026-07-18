"""Plotly chart helpers for model comparison."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def metric_bar_chart(df: pd.DataFrame, metric: str):
    """Create a model comparison bar chart for one metric."""
    plot_df = df[["Model", metric]].dropna()
    fig = px.bar(
        plot_df,
        x="Model",
        y=metric,
        color="Model",
        text=metric,
        template="plotly_dark",
        color_discrete_sequence=["#38bdf8", "#22c55e", "#f59e0b"],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=35, b=10), height=330)
    return fig


def inference_line_chart(df: pd.DataFrame):
    """Create inference-speed line chart."""
    plot_df = df[["Model", "Avg Inference Time (ms/image)"]].dropna()
    fig = px.line(
        plot_df,
        x="Model",
        y="Avg Inference Time (ms/image)",
        markers=True,
        template="plotly_dark",
        title="Average Inference Time per Image",
    )
    fig.update_traces(line=dict(width=4, color="#38bdf8"), marker=dict(size=11))
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=330)
    return fig


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return metric-by-model table for Accuracy, Precision, Recall, and F1."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Specificity", "Avg Inference Time (ms/image)"]
    available = [metric for metric in metrics if metric in df.columns]
    table = df.set_index("Model")[available].T.reset_index().rename(columns={"index": "Metric"})
    return table


def radar_chart(df: pd.DataFrame):
    """Create a radar chart summarizing model strengths."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "ROC AUC"]
    available = [metric for metric in metrics if metric in df.columns]
    fig = go.Figure()
    colors = ["#38bdf8", "#22c55e", "#f59e0b"]

    for color, (_, row) in zip(colors, df.iterrows()):
        fig.add_trace(
            go.Scatterpolar(
                r=[float(row[metric]) for metric in available],
                theta=available,
                fill="toself",
                name=row["Model"],
                line=dict(color=color, width=3),
                opacity=0.6,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
        title="Overall Model Strengths",
    )
    return fig
