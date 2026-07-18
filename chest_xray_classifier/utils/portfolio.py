"""
Portfolio utilities module for Chest X-Ray Disease Classification.

Provides execution time logging, SQLite and CSV dual-logger for predictions,
folders preparation, and model metadata utilities.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd

from chest_xray_classifier.config.config import RESULTS_DIR, DB_PATH

logger = logging.getLogger(__name__)

# CSV log path
CSV_HISTORY_PATH = Path(RESULTS_DIR) / "prediction_log.csv"
HISTORY_COLUMNS = ["timestamp", "model", "filename", "prediction", "confidence", "uncertainty", "inference_ms"]

def setup_project_directories() -> None:
    """Ensure all required folders exist inside the project structure."""
    dirs = [
        Path(RESULTS_DIR),
        Path(RESULTS_DIR) / "charts",
        Path(RESULTS_DIR) / "confusion_matrices",
        Path(RESULTS_DIR) / "model_metrics",
        Path(RESULTS_DIR) / "heatmap_examples",
        Path(__file__).resolve().parents[2] / "outputs",
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Project directories checked and ready.")

def get_file_size_mb(file_path: str | Path) -> float:
    """Return file size in Megabytes."""
    path = Path(file_path)
    if not path.exists():
        return 0.0
    return float(path.stat().st_size / (1024.0 * 1024.0))

def init_sqlite_db() -> None:
    """Initialize SQLite database for tracking predictions if not exists."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                uncertainty REAL NOT NULL,
                inference_ms REAL NOT NULL
            )
            """
        )
        
        # Schema migration checks for existing tables
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "image_path" in columns and "filename" not in columns:
            cursor.execute("ALTER TABLE predictions RENAME COLUMN image_path TO filename")
        if "predicted_class" in columns and "prediction" not in columns:
            cursor.execute("ALTER TABLE predictions RENAME COLUMN predicted_class TO prediction")
        if "inference_time_ms" in columns and "inference_ms" not in columns:
            cursor.execute("ALTER TABLE predictions RENAME COLUMN inference_time_ms TO inference_ms")
        if "model" not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN model TEXT DEFAULT 'Custom CNN'")
        if "uncertainty" not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN uncertainty REAL DEFAULT 0.0")
            
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"Failed to initialize SQLite database: {exc}")

def save_prediction_log(
    model_name: str,
    filename: str,
    prediction: str,
    confidence: float,
    uncertainty: float,
    inference_ms: float
) -> None:
    """
    Save one prediction entry to both SQLite database and CSV history log.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    
    # 1. Save to SQLite
    try:
        init_sqlite_db()
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (timestamp, model, filename, prediction, confidence, uncertainty, inference_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, model_name, filename, prediction, confidence, uncertainty, inference_ms)
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"Failed to log prediction to SQLite: {exc}")
        
    # 2. Save to CSV
    try:
        CSV_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_row = pd.DataFrame([{
            "timestamp": timestamp,
            "model": model_name,
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "inference_ms": inference_ms
        }])
        
        if CSV_HISTORY_PATH.exists():
            try:
                existing = pd.read_csv(CSV_HISTORY_PATH)
                df = pd.concat([existing, new_row], ignore_index=True)
            except Exception:
                df = new_row
        else:
            df = new_row
            
        df.to_csv(CSV_HISTORY_PATH, index=False)
    except Exception as exc:
        logger.warning(f"Failed to log prediction to CSV: {exc}")

def load_prediction_history() -> pd.DataFrame:
    """
    Load prediction log from the SQLite database.
    Falls back to CSV if SQLite is empty or fails.
    """
    # Try SQLite first
    try:
        if DB_PATH.exists():
            conn = sqlite3.connect(str(DB_PATH))
            df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
            conn.close()
            if not df.empty:
                return df
    except Exception as exc:
        logger.warning(f"Failed to load prediction history from SQLite: {exc}")
        
    # Fallback to CSV
    if CSV_HISTORY_PATH.exists():
        try:
            df = pd.read_csv(CSV_HISTORY_PATH)
            # Add missing columns
            for col in HISTORY_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0 if col in ('confidence', 'uncertainty', 'inference_ms') else ""
            return df.sort_index(ascending=False)
        except Exception:
            pass
            
    return pd.DataFrame(columns=HISTORY_COLUMNS)
