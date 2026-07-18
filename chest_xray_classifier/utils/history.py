"""CSV prediction history helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from chest_xray_classifier.config.config import RESULTS_DIR


HISTORY_PATH = Path(RESULTS_DIR) / "prediction_log.csv"
HISTORY_COLUMNS = ["timestamp", "model", "filename", "prediction", "confidence", "inference_ms"]


def save_prediction_log(model: str, filename: str, prediction: str, confidence: float, inference_ms: float) -> None:
    """Append one prediction to results/prediction_log.csv."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "filename": filename,
                "prediction": prediction,
                "confidence": confidence,
                "inference_ms": inference_ms,
            }
        ]
    )
    if HISTORY_PATH.exists():
        existing = pd.read_csv(HISTORY_PATH)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(HISTORY_PATH, index=False)


def load_history() -> pd.DataFrame:
    """Load prediction history and normalize older CSV shapes."""
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(HISTORY_PATH)
    if "filename" not in df.columns and "image_name" in df.columns:
        df["filename"] = df["image_name"]
    if "model" not in df.columns:
        df["model"] = "Custom CNN"
    if "inference_ms" not in df.columns and "inference_time_ms" in df.columns:
        df["inference_ms"] = df["inference_time_ms"]
    for column in HISTORY_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[HISTORY_COLUMNS]

