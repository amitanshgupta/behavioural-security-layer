import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import torch
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    PROCESSED_DIR, METADATA_DIR, CLUSTERING_FEATURES
)
from utils.logger import get_logger
from models.bilstm_model import (
    BiLSTMClassifier, SequenceDataset, LSTM_FEATURES, SEQ_LEN
)
from torch.utils.data import DataLoader

log = get_logger("model_pipeline")

# Decision thresholds per model
THRESHOLDS = {
    "lgb"   : 0.5,
    "if"    : 0.5,
    "bilstm": 0.5,
    "genai" : 0.5,
}

# Ensemble weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "lgb"   : 0.50,   # primary — highest individual performance
    "if"    : 0.20,   # catches novel/zero-day attacks
    "bilstm": 0.15,   # temporal patterns
    "genai" : 0.15,   # GenAI-specific signals
}


class ModelPipeline:
    """
    Unified inference pipeline combining all four models.
    Loads saved artifacts and runs ensemble prediction.
    """

    def __init__(self):
        self.lgb_data    = None
        self.if_data     = None
        self.bilstm_data = None
        self.genai_data  = None
        self.device      = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_all_models(self) -> None:
        """Load all saved model artifacts."""
        log.info("Loading all model artifacts...")

        # LightGBM
        self.lgb_data  = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
        log.info("LightGBM loaded")

        # Isolation Forest
        self.if_data   = joblib.load(METADATA_DIR / "isolation_forest.pkl")
        log.info("Isolation Forest loaded")

        # BiLSTM
        bilstm_ckpt    = torch.load(
            METADATA_DIR / "bilstm_model.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.bilstm_model = BiLSTMClassifier(
            input_size=bilstm_ckpt["input_size"]
        )
        self.bilstm_model.load_state_dict(bilstm_ckpt["model_state"])
        self.bilstm_model.eval()
        self.bilstm_data = bilstm_ckpt
        log.info("BiLSTM loaded")

        # GenAI detector
        self.genai_data = joblib.load(METADATA_DIR / "genai_detector.pkl")
        log.info("GenAI detector loaded")

        log.info("All models loaded successfully")

    def predict_lgb(self, df: pd.DataFrame) -> np.ndarray:
        """LightGBM probability scores."""
        feats = self.lgb_data["features"]
        X     = df[feats].fillna(0)  # keep as DataFrame — LGB needs column names
        return self.lgb_data["model"].predict_proba(X)[:, 1]

    def predict_if(self, df: pd.DataFrame) -> np.ndarray:
        """Isolation Forest anomaly scores normalised to [0,1]."""
        feats  = self.if_data["features"]
        X      = pd.DataFrame(df[feats].fillna(0).values, columns=feats)
        raw    = self.if_data["model"].decision_function(X)
        scores = -raw
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    def predict_bilstm(self, df: pd.DataFrame) -> np.ndarray:
        """
        BiLSTM sequence scores.
        Pads first SEQ_LEN-1 rows with LightGBM score as fallback
        since BiLSTM needs a warmup window.
        """
        feats    = self.bilstm_data["features"]
        min_vals = self.bilstm_data["min_vals"]
        rng      = self.bilstm_data["rng"]

        X        = df[[f for f in feats if f in df.columns]].fillna(0).values
        X_norm   = (X - min_vals) / (rng + 1e-9)

        dataset  = SequenceDataset(
            np.array(X_norm, copy=True),
            np.zeros(len(X_norm)),   # dummy labels for inference
        )
        loader   = DataLoader(dataset, batch_size=512, shuffle=False)

        scores   = []
        self.bilstm_model.eval()
        with torch.no_grad():
            for X_batch, _ in loader:
                preds = self.bilstm_model(X_batch.to(self.device))
                scores.extend(preds.cpu().numpy())

        # Pad first SEQ_LEN-1 rows with mean score
        scores   = np.array(scores)
        pad_val  = scores.mean()
        padded   = np.full(len(df), pad_val)
        padded[SEQ_LEN - 1:SEQ_LEN - 1 + len(scores)] = scores
        return padded

    def predict_genai(self, df: pd.DataFrame) -> np.ndarray:
        """GenAI One-Class SVM anomaly scores normalised to [0,1]."""
        feats    = self.genai_data["features"]
        scaler   = self.genai_data["scaler"]
        model    = self.genai_data["model"]
        X        = pd.DataFrame(df[feats].fillna(0).values, columns=feats)
        X_scaled = scaler.transform(X)
        raw      = model.decision_function(X_scaled)
        scores   = -raw
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    def predict_ensemble(
        self,
        df        : pd.DataFrame,
        threshold : float = 0.5,
    ) -> pd.DataFrame:
        """
        Run all four models and combine via weighted average.

        Returns DataFrame with individual scores + ensemble result.
        """
        log.info(f"Running ensemble inference on {len(df):,} rows...")

        lgb_scores    = self.predict_lgb(df)
        if_scores     = self.predict_if(df)
        bilstm_scores = self.predict_bilstm(df)
        genai_scores  = self.predict_genai(df)

        ensemble = (
            lgb_scores    * ENSEMBLE_WEIGHTS["lgb"]    +
            if_scores     * ENSEMBLE_WEIGHTS["if"]     +
            bilstm_scores * ENSEMBLE_WEIGHTS["bilstm"] +
            genai_scores  * ENSEMBLE_WEIGHTS["genai"]
        )

        results = df.copy()
        results["lgb_score"]      = lgb_scores
        results["if_score"]       = if_scores
        results["bilstm_score"]   = bilstm_scores
        results["genai_score"]    = genai_scores
        results["ensemble_score"] = ensemble
        results["ensemble_pred"]  = (ensemble >= threshold).astype(int)

        flagged = results["ensemble_pred"].sum()
        log.info(
            f"Ensemble complete - {flagged:,} rows flagged as attack "
            f"({flagged/len(df)*100:.1f}%)"
        )
        return results

    def evaluate_ensemble(self, results: pd.DataFrame) -> dict:
        """Compute metrics if ground truth labels are available."""
        if "label_binary" not in results.columns:
            log.warning("No label_binary column - skipping evaluation")
            return {}

        from sklearn.metrics import (
            f1_score, roc_auc_score,
            classification_report, confusion_matrix
        )

        y      = results["label_binary"]
        y_pred = results["ensemble_pred"]
        scores = results["ensemble_score"]

        f1  = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, scores)
        fpr = (
            ((y_pred == 1) & (y == 0)).sum() /
            ((y == 0).sum() + 1e-9)
        )

        log.info(f"\n--- Ensemble Results ---")
        log.info(f"F1      : {f1:.4f}")
        log.info(f"ROC-AUC : {auc:.4f}")
        log.info(f"FPR     : {fpr:.4f} ({fpr*100:.1f}%)")
        log.info(f"\n{classification_report(y, y_pred, target_names=['normal','attack'])}")
        log.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

        return {"f1": f1, "roc_auc": auc, "fpr": fpr}


def run_pipeline() -> dict:
    """Master function — load models, run ensemble on test set."""
    pipeline = ModelPipeline()
    pipeline.load_all_models()

    test_df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")
    results  = pipeline.predict_ensemble(test_df)
    metrics  = pipeline.evaluate_ensemble(results)

    # Save ensemble scored output
    out_path = PROCESSED_DIR / "nslkdd_test_ensemble.csv"
    results.to_csv(out_path, index=False)
    log.info(f"Ensemble results saved -> {out_path}")

    return metrics


if __name__ == "__main__":
    metrics = run_pipeline()
    print(f"\nFinal Ensemble Test Results:")
    print(f"  F1      : {metrics.get('f1', 0):.4f}")
    print(f"  ROC-AUC : {metrics.get('roc_auc', 0):.4f}")
    print(f"  FPR     : {metrics.get('fpr', 0)*100:.1f}%")