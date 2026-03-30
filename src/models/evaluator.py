"""
evaluator.py — Model evaluation: confusion matrix, ROC curve, feature importance.
Saves publication-quality plots.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    PROCESSED_CSV, FEATURE_COLS, TARGET_BINARY,
    BEST_MODEL_PATH, METRICS_PATH,
    FEATURE_IMP_PLOT, CONFUSION_PLOT, ROC_PLOT,
    TRAIN_TEST_SPLIT, RANDOM_SEED
)

# ─── Plotting Style ────────────────────────────────────────────────────────────
plt.style.use("dark_background")
PALETTE = {"primary": "#4FC3F7", "accent": "#FF6B6B", "success": "#69F0AE"}


def load_data():
    df = pd.read_csv(PROCESSED_CSV)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df[TARGET_BINARY].values.astype(int)
    return X, y, available


def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["On Time", "Delayed"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    disp.im_.set_clim(0, cm.max())

    ax.set_title("Confusion Matrix — Flight Delay Prediction",
                 color="white", fontsize=13, pad=15)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT, dpi=150, bbox_inches="tight",
                facecolor="#0D1117")
    plt.close()
    print(f"[Evaluator] Confusion matrix → {CONFUSION_PLOT}")
    return cm


def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val  = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2,
            label=f"ROC Curve (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555", lw=1.5,
            label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["primary"])
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate",  color="white")
    ax.set_title("ROC Curve — Flight Delay Prediction",
                 color="white", fontsize=13, pad=15)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1A1A2E", edgecolor="#444", labelcolor="white")
    ax.spines["bottom"].set_color("#444"); ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(ROC_PLOT, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[Evaluator] ROC curve → {ROC_PLOT}")
    return roc_auc_val


def plot_feature_importance(model, feature_names: list):
    """Handles both tree-based feature_importances_ and linear coef_."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("[Evaluator] Model has no feature importance attribute.")
        return

    indices  = np.argsort(importances)[::-1][:20]   # top 20
    top_feat = [feature_names[i] for i in indices]
    top_imp  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    colors = [PALETTE["primary"] if i == 0 else "#2A6899" for i in range(len(top_feat))]
    bars = ax.barh(range(len(top_feat)), top_imp[::-1], color=colors[::-1],
                   edgecolor="#1A1A2E", linewidth=0.5)
    ax.set_yticks(range(len(top_feat)))
    ax.set_yticklabels(top_feat[::-1], color="white", fontsize=10)
    ax.set_xlabel("Feature Importance", color="white")
    ax.set_title("Top 20 Feature Importances", color="white", fontsize=13, pad=15)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444"); ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)

    # Value labels
    for bar, val in zip(bars[::-1], top_imp[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(FEATURE_IMP_PLOT, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[Evaluator] Feature importance → {FEATURE_IMP_PLOT}")


def full_evaluation() -> dict:
    """Loads saved model and runs full evaluation suite."""
    print("[Evaluator] Loading model and data...")
    model        = joblib.load(BEST_MODEL_PATH)
    X, y, feats  = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT,
        stratify=y, random_state=RANDOM_SEED
    )

    print(f"[Evaluator] Test set: {len(y_te):,} samples "
          f"(delayed: {y_te.mean():.2%})")

    cm      = plot_confusion_matrix(model, X_te, y_te)
    roc_val = plot_roc_curve(model, X_te, y_te)
    plot_feature_importance(model, feats)

    y_pred = model.predict(X_te)
    report = classification_report(y_te, y_pred,
                                   target_names=["On Time", "Delayed"],
                                   output_dict=True)
    print("\n" + classification_report(y_te, y_pred,
                                       target_names=["On Time", "Delayed"]))
    return {"classification_report": report, "roc_auc": roc_val}


if __name__ == "__main__":
    full_evaluation()
