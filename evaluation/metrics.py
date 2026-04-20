import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    classification_report, average_precision_score
)
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger

log = get_logger("metrics")

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"


def compute_all_metrics(
    y_true   : np.ndarray,
    y_pred   : np.ndarray,
    y_scores : np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Compute full suite of evaluation metrics.

    Args:
        y_true    : ground truth binary labels
        y_pred    : binary predictions
        y_scores  : continuous probability/anomaly scores
        model_name: label for logging

    Returns:
        dict with all metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr     = fp / (fp + tn + 1e-9)
    fnr     = fn / (fn + tp + 1e-9)
    tpr     = tp / (tp + fn + 1e-9)   # recall
    tnr     = tn / (tn + fp + 1e-9)   # specificity

    metrics = {
        "model"           : model_name,
        "accuracy"        : round(float((tp + tn) / (tp + tn + fp + fn)), 4),
        "precision"       : round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall"          : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1"              : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc"         : round(float(roc_auc_score(y_true, y_scores)), 4),
        "avg_precision"   : round(float(average_precision_score(y_true, y_scores)), 4),
        "fpr"             : round(float(fpr), 4),
        "fnr"             : round(float(fnr), 4),
        "tpr"             : round(float(tpr), 4),
        "tnr"             : round(float(tnr), 4),
        "tp"              : int(tp),
        "tn"              : int(tn),
        "fp"              : int(fp),
        "fn"              : int(fn),
    }

    log.info(f"\n--- {model_name} Metrics ---")
    log.info(f"F1={metrics['f1']:.4f} | "
             f"AUC={metrics['roc_auc']:.4f} | "
             f"FPR={metrics['fpr']:.4f} | "
             f"Recall={metrics['recall']:.4f}")
    log.info(f"\n{classification_report(y_true, y_pred, target_names=['normal','attack'])}")

    return metrics


def evaluate_all_models(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load all scored test files and compute metrics for every model.
    Produces a unified comparison table.
    """
    results = []

    # 1. LightGBM
    if (PROCESSED_DIR / "nslkdd_test_lgb_scored.csv").exists():
        df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_lgb_scored.csv")
        m   = compute_all_metrics(
            df["label_binary"].values,
            df["lgb_prediction"].values,
            df["lgb_proba"].values,
            "LightGBM"
        )
        results.append(m)

    # 2. Isolation Forest
    if (PROCESSED_DIR / "nslkdd_test_if_scored.csv").exists():
        df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_if_scored.csv")
        m   = compute_all_metrics(
            df["label_binary"].values,
            df["if_prediction"].values,
            df["if_anomaly_score"].values,
            "IsolationForest"
        )
        results.append(m)

    # 3. GenAI Detector
    if (PROCESSED_DIR / "nslkdd_test_genai_scored.csv").exists():
        df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_genai_scored.csv")
        m   = compute_all_metrics(
            df["label_binary"].values,
            df["genai_prediction"].values,
            df["genai_anomaly_score"].values,
            "GenAI_OneClassSVM"
        )
        results.append(m)

    # 4. Ensemble
    if (PROCESSED_DIR / "nslkdd_test_ensemble.csv").exists():
        df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_ensemble.csv")
        m   = compute_all_metrics(
            df["label_binary"].values,
            df["ensemble_pred"].values,
            df["ensemble_score"].values,
            "Ensemble"
        )
        results.append(m)

    comparison_df = pd.DataFrame(results)
    return comparison_df


def save_metrics_report(comparison_df: pd.DataFrame) -> None:
    """Save metrics comparison table to outputs."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    out_csv = METRICS_DIR / "model_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)
    log.info(f"Metrics saved -> {out_csv}")

    # Pretty text report
    out_txt = METRICS_DIR / "model_comparison.txt"
    with open(out_txt, "w") as f:
        f.write("BEHAVIORAL SECURITY LAYER — MODEL COMPARISON\n")
        f.write("=" * 70 + "\n\n")

        display_cols = [
            "model", "f1", "precision", "recall",
            "roc_auc", "fpr", "accuracy"
        ]
        f.write(comparison_df[display_cols].to_string(index=False))
        f.write("\n\n")

        f.write("Proposal Targets:\n")
        f.write("  Detection Accuracy  : > 97%\n")
        f.write("  False Positive Rate : < 8%\n")
        f.write("  ROC-AUC             : maximize\n")

    log.info(f"Text report saved -> {out_txt}")


if __name__ == "__main__":
    test_df     = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")
    comparison  = evaluate_all_models(test_df)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_report(comparison)

    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    display = ["model", "f1", "precision", "recall", "roc_auc", "fpr", "accuracy"]
    print(comparison[display].to_string(index=False))

    print("\nProposal Targets: F1>0.97, FPR<0.08, AUC>0.95")
    print(f"Best FPR  : {comparison['fpr'].min():.4f}")
    print(f"Best AUC  : {comparison['roc_auc'].max():.4f}")
    print(f"Best F1   : {comparison['f1'].max():.4f}")