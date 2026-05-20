"""
============================================================
train_model.py
AI-Based Real-Time Human Posture Detection and Correction System
============================================================
Machine Learning Training Pipeline

Trains FOUR ML classifiers and picks the best:
  1. Random Forest Classifier
  2. Logistic Regression
  3. Support Vector Machine (SVC)
  4. K-Nearest Neighbors (KNN)

Saves:
  - models/posture_model.pkl  (best trained model)
  - models/scaler.pkl         (fitted StandardScaler)
  - models/label_encoder.pkl  (label encoder)
  - graphs/model_comparison.png
  - graphs/confusion_matrix.png
  - reports/training_report.txt

Run AFTER preprocessing.py
============================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocessing import run_preprocessing
from feature_extraction import normalize_features

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
GRAPHS_DIR   = os.path.join(BASE_DIR, "graphs")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
MERGED_DIR   = os.path.join(BASE_DIR, "dataset", "merged")

for d in [MODELS_DIR, GRAPHS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD TRAINING / TEST DATA
# ─────────────────────────────────────────────────────────────

def load_train_test():
    """
    Load train.csv and test.csv from dataset/merged/.
    If not found, trigger preprocessing first.

    Returns:
        train_df (pd.DataFrame): Training data
        test_df  (pd.DataFrame): Testing data
    """
    train_path = os.path.join(MERGED_DIR, "train.csv")
    test_path  = os.path.join(MERGED_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[TrainModel] Preprocessed data not found. Running preprocessing...")
        run_preprocessing()

    train_df = pd.read_csv(train_path, engine='python')
    test_df  = pd.read_csv(test_path, engine='python')
    print(f"[TrainModel] Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# SECTION 2: PREPARE FEATURES
# ─────────────────────────────────────────────────────────────

def prepare_features(train_df, test_df):
    """
    Load precomputed angle features from the preprocessed CSVs.
    The preprocessing pipeline already computed all features.

    Returns:
        X_train_sc, X_test_sc (np.ndarray): Scaled feature matrices
        y_train, y_test       (np.ndarray): Label vectors
        scaler                (StandardScaler): Fitted scaler
        feature_names         (list[str])
    """
    from preprocessing import FEATURE_COLS

    feat_names = FEATURE_COLS

    # Features are already computed — just select them
    def to_XY(df):
        available = [c for c in feat_names if c in df.columns]
        missing   = [c for c in feat_names if c not in df.columns]
        if missing:
            print(f"  [TrainModel] Missing cols (filled 0): {missing}")
            for c in missing:
                df[c] = 0.0
        X = df[feat_names].values.astype(np.float32)
        y = df["label"].values.astype(int)
        return X, y

    print("[TrainModel] Loading features from training set...")
    X_train, y_train = to_XY(train_df)

    print("[TrainModel] Loading features from test set...")
    X_test,  y_test  = to_XY(test_df)

    print(f"  X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

    # Fit scaler on training data only
    X_train_sc, scaler = normalize_features(X_train, fit=True)
    X_test_sc,  _      = normalize_features(X_test,  scaler=scaler)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, feat_names


# ─────────────────────────────────────────────────────────────
# SECTION 3: MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────

def get_models():
    """
    Return a dictionary of ML classifiers to compare.

    Returns:
        dict: {model_name: sklearn_classifier}
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs"
        ),
        "SVM": SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="euclidean",
            n_jobs=-1
        ),
    }


# ─────────────────────────────────────────────────────────────
# SECTION 4: TRAIN AND EVALUATE ALL MODELS
# ─────────────────────────────────────────────────────────────

def train_and_evaluate(models, X_train, X_test, y_train, y_test, label_names):
    """
    Train each model and evaluate on the test set.

    Parameters:
        models      (dict): {name: classifier}
        X_train     (np.ndarray): Training features
        X_test      (np.ndarray): Testing features
        y_train     (np.ndarray): Training labels
        y_test      (np.ndarray): Testing labels
        label_names (list[str]) : Class names

    Returns:
        results (list[dict]): Metrics for each model (sorted by accuracy DESC)
        trained_models (dict): {name: fitted_classifier}
    """
    results       = []
    trained_models = {}

    print("\n" + "-" * 55)
    print("  TRAINING ML MODELS")
    print("-" * 55)

    for name, model in models.items():
        print(f"\n  > Training: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        cr   = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)

        print(f"    Accuracy : {acc*100:.2f}%")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-Score : {f1:.4f}")

        results.append({
            "name":      name,
            "model":     model,
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "cm":        cm,
            "report":    cr,
        })
        trained_models[name] = model

    # Sort by accuracy descending
    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results, trained_models


# ─────────────────────────────────────────────────────────────
# SECTION 5: VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(results):
    """
    Create a side-by-side bar chart comparing all model metrics.

    Saves to graphs/model_comparison.png
    """
    names     = [r["name"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    f1_scores  = [r["f1"] * 100 for r in results]
    precisions = [r["precision"] * 100 for r in results]
    recalls    = [r["recall"] * 100 for r in results]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    colors = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b"]
    bars = [
        ax.bar(x - 1.5*width, accuracies, width, label="Accuracy",  color=colors[0], alpha=0.9),
        ax.bar(x - 0.5*width, precisions, width, label="Precision", color=colors[1], alpha=0.9),
        ax.bar(x + 0.5*width, recalls,    width, label="Recall",    color=colors[2], alpha=0.9),
        ax.bar(x + 1.5*width, f1_scores,  width, label="F1-Score",  color=colors[3], alpha=0.9),
    ]

    for bar_group in bars:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        color="white", fontsize=7)

    ax.set_xlabel("Model", color="white", fontsize=12)
    ax.set_ylabel("Score (%)", color="white", fontsize=12)
    ax.set_title("ML Model Performance Comparison — Posture Detection",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_ylim(0, 110)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.spines[:].set_color("#444")

    plt.tight_layout()
    save_path = os.path.join(GRAPHS_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Graphs] Model comparison saved: {save_path}")


def plot_confusion_matrix(cm, label_names, model_name):
    """
    Plot and save the confusion matrix for the best model.

    Saves to graphs/confusion_matrix.png
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_names,
                yticklabels=label_names,
                ax=ax,
                linewidths=0.5,
                linecolor="#333")

    ax.set_title(f"Confusion Matrix — {model_name}", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", color="white", fontsize=11)
    ax.set_ylabel("True Label", color="white", fontsize=11)
    ax.tick_params(colors="white")

    plt.tight_layout()
    save_path = os.path.join(GRAPHS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Graphs] Confusion matrix saved: {save_path}")


def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models (Random Forest).

    Saves to graphs/feature_importance.png
    """
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:20]  # top 20

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(indices)))
    bars = ax.barh(range(len(indices)),
                   importances[indices],
                   align="center",
                   color=colors)

    ax.set_yticks(range(len(indices)))
    names = [feature_names[i] if i < len(feature_names) else f"f_{i}"
             for i in indices]
    ax.set_yticklabels(names, color="white", fontsize=9)
    ax.set_xlabel("Importance Score", color="white", fontsize=11)
    ax.set_title(f"Top Feature Importances — {model_name}",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.invert_yaxis()
    ax.spines[:].set_color("#444")

    plt.tight_layout()
    save_path = os.path.join(GRAPHS_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Graphs] Feature importance saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# SECTION 6: SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────

def save_model_artifacts(best_result, scaler, label_names, feature_names):
    """
    Save the best model, scaler, and label info to disk.

    Files saved:
        models/posture_model.pkl
        models/scaler.pkl
        models/label_encoder.pkl
        models/model_info.pkl
    """
    best_model = best_result["model"]

    # Save best model
    model_path = os.path.join(MODELS_DIR, "posture_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"[TrainModel] Best model saved: {model_path}")

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[TrainModel] Scaler saved: {scaler_path}")

    # Save metadata
    info = {
        "model_name":    best_result["name"],
        "accuracy":      best_result["accuracy"],
        "precision":     best_result["precision"],
        "recall":        best_result["recall"],
        "f1":            best_result["f1"],
        "label_names":   label_names,
        "feature_names": feature_names,
    }
    info_path = os.path.join(MODELS_DIR, "model_info.pkl")
    with open(info_path, "wb") as f:
        pickle.dump(info, f)
    print(f"[TrainModel] Model info saved: {info_path}")


def write_training_report(results, best_result, label_names):
    """Write a detailed training report to reports/training_report.txt."""
    report_path = os.path.join(REPORTS_DIR, "training_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  POSENOVA — ML TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Model    : {best_result['name']}\n")
        f.write(f"Accuracy      : {best_result['accuracy']*100:.2f}%\n")
        f.write(f"Precision     : {best_result['precision']:.4f}\n")
        f.write(f"Recall        : {best_result['recall']:.4f}\n")
        f.write(f"F1-Score      : {best_result['f1']:.4f}\n\n")
        f.write("─" * 55 + "\n")
        f.write("ALL MODELS COMPARISON:\n")
        f.write("─" * 55 + "\n")
        for r in results:
            f.write(f"\n{r['name']}:\n")
            f.write(f"  Accuracy  : {r['accuracy']*100:.2f}%\n")
            f.write(f"  Precision : {r['precision']:.4f}\n")
            f.write(f"  Recall    : {r['recall']:.4f}\n")
            f.write(f"  F1-Score  : {r['f1']:.4f}\n")
        f.write("\n" + "─" * 55 + "\n")
        f.write("BEST MODEL CLASSIFICATION REPORT:\n")
        f.write("─" * 55 + "\n\n")
        f.write(best_result["report"])
        f.write("\n" + "=" * 60 + "\n")

    print(f"[TrainModel] Report saved: {report_path}")


# ─────────────────────────────────────────────────────────────
# SECTION 7: MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def run_training():
    """
    Execute the complete ML training pipeline.

    Steps:
        1. Preprocess data (if not done already)
        2. Load train/test splits
        3. Extract and scale features
        4. Train 4 ML models
        5. Evaluate all models
        6. Select best model
        7. Save artifacts
        8. Generate graphs and report

    Returns:
        dict: Training results summary
    """
    print("\n" + "=" * 60)
    print("  POSENOVA - ML TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1-2: Data ───────────────────────────────────────
    train_df, test_df = load_train_test()

    # ── Step 3: Features ─────────────────────────────────────
    (X_train, X_test, y_train, y_test,
     scaler, feature_names) = prepare_features(train_df, test_df)

    # Determine label names
    all_labels = sorted(set(y_train) | set(y_test))
    label_names_map = {0: "bad", 1: "good"}
    label_names = [label_names_map.get(l, str(l)) for l in all_labels]

    # ── Step 4-5: Train and evaluate ─────────────────────────
    models  = get_models()
    results, trained_models = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, label_names
    )

    best_result = results[0]

    print(f"\n{'-'*55}")
    print(f"  [BEST MODEL] {best_result['name']}")
    print(f"    Accuracy : {best_result['accuracy']*100:.2f}%")
    print(f"    F1-Score : {best_result['f1']:.4f}")
    print(f"{'-'*55}")

    # ── Step 6: Visualizations ───────────────────────────────
    print("\n[TrainModel] Generating visualizations...")
    plot_model_comparison(results)
    plot_confusion_matrix(best_result["cm"], label_names, best_result["name"])
    plot_feature_importance(best_result["model"], feature_names, best_result["name"])

    # ── Step 7: Save artifacts ───────────────────────────────
    save_model_artifacts(best_result, scaler, label_names, feature_names)
    write_training_report(results, best_result, label_names)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE (OK)")
    print("=" * 60 + "\n")

    return {
        "best_model":    best_result["name"],
        "accuracy":      best_result["accuracy"],
        "all_results":   results,
    }


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = run_training()
    print(f"\nBest model: {summary['best_model']} | "
          f"Accuracy: {summary['accuracy']*100:.2f}%")
