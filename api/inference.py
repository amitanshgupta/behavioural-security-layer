import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import METADATA_DIR, CLUSTERING_FEATURES
from utils.logger import get_logger

log = get_logger("inference")

SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.85,
    "HIGH"    : 0.65,
    "MEDIUM"  : 0.40,
    "LOW"     : 0.20,
}

_models = {}


def load_models() -> None:
    """Load all model artifacts once at startup."""
    global _models
    log.info("Loading models for inference...")

    _models["lgb"]   = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
    _models["if"]    = joblib.load(METADATA_DIR / "isolation_forest.pkl")
    _models["genai"] = joblib.load(METADATA_DIR / "genai_detector.pkl")
    _models["scaler"]= joblib.load(METADATA_DIR / "cluster_scaler.pkl")
    _models["km"]    = joblib.load(METADATA_DIR / "kmeans_model.pkl")

    log.info("All models loaded")


def get_severity(score: float) -> str:
    if score >= SEVERITY_THRESHOLDS["CRITICAL"]: return "CRITICAL"
    if score >= SEVERITY_THRESHOLDS["HIGH"]    : return "HIGH"
    if score >= SEVERITY_THRESHOLDS["MEDIUM"]  : return "MEDIUM"
    if score >= SEVERITY_THRESHOLDS["LOW"]     : return "LOW"
    return "INFO"


def build_shap_explanation(
    row      : pd.DataFrame,
    top_n    : int = 3,
) -> dict:
    """Fast SHAP explanation using TreeExplainer."""
    try:
        import shap
        model    = _models["lgb"]["model"]
        features = _models["lgb"]["features"]
        X        = row[[f for f in features if f in row.columns]].fillna(0)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

        row_shap = shap_values[0]
        top_idx  = np.argsort(np.abs(row_shap))[::-1][:top_n]

        top_features = []
        for i in top_idx:
            feat     = features[i]
            shap_val = float(row_shap[i])
            raw_val  = float(X.iloc[0][feat]) if feat in X.columns else 0.0
            top_features.append({
                "feature"   : feat,
                "shap_value": round(shap_val, 4),
                "raw_value" : round(raw_val, 4),
                "direction" : "increases risk" if shap_val > 0
                              else "decreases risk",
            })

        parts = []
        for f in top_features:
            name = f["feature"].replace("_", " ")
            parts.append(
                f"{name} is {'abnormally high' if f['shap_value'] > 0 else 'abnormally low'} "
                f"(SHAP: {'+' if f['shap_value'] > 0 else ''}{f['shap_value']:.3f})"
            )
        narrative = "Alert triggered: " + " | ".join(parts) if parts else ""

        return {
            "top_features": top_features,
            "narrative"   : narrative,
            "base_value"  : round(float(base_value), 4),
        }
    except Exception as e:
        log.warning(f"SHAP failed: {e}")
        return {"top_features": [], "narrative": "", "base_value": 0.0}


def predict(event: dict) -> dict:
    """
    Main inference function.
    Takes a raw event dict, returns a scored alert dict.

    Args:
        event: dict with feature values (from network flow or endpoint)

    Returns:
        alert dict with score, severity, explanation
    """
    if not _models:
        load_models()

    # Convert to DataFrame
    df = pd.DataFrame([event])

    # LightGBM score
    lgb_feats = _models["lgb"]["features"]
    X_lgb     = pd.DataFrame(
        df[[f for f in lgb_feats if f in df.columns]].fillna(0).values,
        columns=[f for f in lgb_feats if f in df.columns]
    )
    # Pad missing features with 0
    for f in lgb_feats:
        if f not in X_lgb.columns:
            X_lgb[f] = 0.0
    X_lgb = X_lgb[lgb_feats]

    lgb_score = float(
        _models["lgb"]["model"].predict_proba(X_lgb)[0][1]
    )

    # Isolation Forest score
    if_feats  = _models["if"]["features"]
    X_if      = pd.DataFrame(
        df[[f for f in if_feats if f in df.columns]].fillna(0).values,
        columns=[f for f in if_feats if f in df.columns]
    )
    for f in if_feats:
        if f not in X_if.columns:
            X_if[f] = 0.0
    X_if = X_if[if_feats]
    if_raw    = _models["if"]["model"].decision_function(X_if)
    if_score  = float(np.clip(-if_raw[0] / 0.5, 0, 1))

    # GenAI score
    genai_feats  = _models["genai"]["features"]
    genai_scaler = _models["genai"]["scaler"]
    X_genai      = pd.DataFrame(
        df[[f for f in genai_feats if f in df.columns]].fillna(0).values,
        columns=[f for f in genai_feats if f in df.columns]
    )
    for f in genai_feats:
        if f not in X_genai.columns:
            X_genai[f] = 0.0
    X_genai  = X_genai[genai_feats]
    X_gs     = genai_scaler.transform(X_genai)
    genai_raw= _models["genai"]["model"].decision_function(X_gs)
    genai_score = float(np.clip(-genai_raw[0] / 0.5, 0, 1))

    # Cluster assignment
    clust_feats = [f for f in CLUSTERING_FEATURES if f in df.columns]
    X_clust     = df[clust_feats].fillna(0)
    X_scaled    = _models["scaler"].transform(X_clust)
    cluster_id  = int(_models["km"].predict(X_scaled)[0])

    # Ensemble score
    ensemble = (
        lgb_score   * 0.50 +
        if_score    * 0.20 +
        genai_score * 0.15 +
        lgb_score   * 0.15   # BiLSTM placeholder = lgb_score
    )
    ensemble = float(np.clip(ensemble, 0, 1))

    severity = get_severity(ensemble)

    # SHAP explanation (only for HIGH and above)
    shap_expl = {}
    if ensemble >= 0.40:
        shap_expl = build_shap_explanation(df)

    alert_id  = f"ALT-{datetime.utcnow().strftime('%H%M%S%f')[:10]}"

    alert = {
        "alert_id"        : alert_id,
        "timestamp"       : datetime.utcnow().isoformat(),
        "severity"        : severity,
        "ensemble_score"  : round(ensemble, 4),
        "lgb_score"       : round(lgb_score, 4),
        "if_score"        : round(if_score, 4),
        "genai_score"     : round(genai_score, 4),
        "cluster_id"      : cluster_id,
        "shap_narrative"  : shap_expl.get("narrative", ""),
        "top_shap_features": shap_expl.get("top_features", []),
        "raw_event"       : event,
        "analyst_action"  : None,
    }

    return alert