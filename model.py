import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")  # headless backend for server
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# Paths
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
MODEL_PATH = "saved_model.h5"

# Load artifacts
scaler = None
label_encoder = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        scaler = None

if os.path.exists(ENCODER_PATH):
    try:
        label_encoder = joblib.load(ENCODER_PATH)
    except Exception:
        label_encoder = None

# Load model
model = load_model(MODEL_PATH)

# Label column name in dataset
LABEL_COL = "Label"

# Optional: known class names order (can be used if label_encoder is None)
CLASS_NAMES = None

def _preprocess(df: pd.DataFrame):
    df = df.copy()

    # Drop duplicate/unwanted columns like in Colab
    drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label.1"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Clean missing or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Separate features and label
    if LABEL_COL in df.columns:
        y_true_raw = df[LABEL_COL]
        X = df.drop(columns=[LABEL_COL])
    else:
        y_true_raw = None
        X = df

    # Keep only numeric columns (drop text like IPs, Timestamp)
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # Scale features
    if scaler is not None:
        X_vals = scaler.transform(X.values)
    else:
        X_vals = X.values

    # Encode labels
    y_true = None
    classes_ = None
    if y_true_raw is not None:
        if label_encoder is not None:
            y_true = label_encoder.transform(y_true_raw)
            classes_ = list(label_encoder.classes_)
        else:
            from sklearn.preprocessing import LabelEncoder
            le_temp = LabelEncoder()
            y_true = le_temp.fit_transform(y_true_raw)
            classes_ = list(le_temp.classes_)

    return X_vals, y_true, classes_




def _save_plot(fig, name, run_id):
    os.makedirs("static/plots", exist_ok=True)
    fname = f"{name}_{run_id}.png"
    path = os.path.join("static", "plots", fname)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return fname   # ✅ return just the filename


def _plot_cm(y_true, y_pred, classes_, run_id):
    if classes_ is None and CLASS_NAMES is not None:
        classes_ = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(set(y_true))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_)
    disp.plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title("Confusion Matrix")
    return _save_plot(fig, "confusion_matrix", run_id)

def _plot_attack_distribution(y_pred, classes_, run_id):
    unique, counts = np.unique(y_pred, return_counts=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    labels = [classes_[i] if classes_ and i < len(classes_) else str(i) for i in unique]
    ax.bar(labels, counts)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Count")
    ax.set_title("Attack Type Distribution (Predicted)")
    ax.tick_params(axis='x', rotation=20)
    return _save_plot(fig, "attack_distribution", run_id)

def _plot_timeline(y_pred, classes_, run_id):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(y_pred, linewidth=1.2)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Predicted Class Id")
    ax.set_title("Timeline of Predicted Attacks")
    return _save_plot(fig, "timeline", run_id)

def run_ids(df: pd.DataFrame, run_id: str):
    """
    Run IDS predictions and generate outputs for Flask templates.
    """
    notes = []
    X_vals, y_true, classes_ = _preprocess(df)

    # Attempt reshape for LSTM input (3D)
    try:
        X_vals_reshaped = X_vals.reshape((X_vals.shape[0], 1, X_vals.shape[1]))
    except Exception as e:
        notes.append(f"Reshape failed: {e}")
        X_vals_reshaped = X_vals  # fallback to 2D input

    # Predict probabilities
    probs = model.predict(X_vals_reshaped, verbose=0)

    # Handle binary or multi-class output
    if probs.ndim == 1 or probs.shape[1] == 1:
        y_pred = (probs.ravel() > 0.5).astype(int)
        if not classes_:
            classes_ = ["Class 0", "Class 1"]
    else:
        y_pred = probs.argmax(axis=1)
        if not classes_ and CLASS_NAMES:
            classes_ = CLASS_NAMES

    # Generate confusion matrix and classification report if true labels exist
    if y_true is not None:
        cm_path = _plot_cm(y_true, y_pred, classes_, run_id)
        report_dict = classification_report(
            y_true, y_pred,
            target_names=classes_ if classes_ else None,
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).T
        report_html = report_df.to_html(classes="table table-striped", float_format="%.3f")
    else:
        cm_path = None
        report_html = (
            "<p><em>No ground-truth labels found in the CSV, so only predictions and charts are shown.</em></p>"
        )
        notes.append("Add a label column to compute confusion matrix and full report.")

    dist_path = _plot_attack_distribution(y_pred, classes_ or None, run_id)
    timeline_path = _plot_timeline(y_pred, classes_ or None, run_id)

    preview_html = df.head(10).to_html(classes="table table-bordered")

    return {
        "preview_html": preview_html,
        "report_html": report_html,
        "cm_path": cm_path,
        "dist_path": dist_path,
        "timeline_path": timeline_path,
        "notes": notes,
    }
