import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger
from explainability.shap_explainer import (
    load_lgb_model, compute_shap_values, explain_single_prediction
)
from explainability.lime_explainer import (
    build_lime_explainer, explain_instance,
    parse_lime_explanation, build_alert_text
)

log = get_logger("alert_generator")

ALERTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "alerts"

# Alert severity thresholds
SEVERITY = {
    "CRITICAL": 0.85,
    "HIGH"    : 0.65,
    "MEDIUM"  : 0.40,
    "LOW"     : 0.20,
}


def get_severity(score: float) -> str:
    """Map risk score to severity label."""
    if score >= SEVERITY["CRITICAL"]: return "CRITICAL"
    if score >= SEVERITY["HIGH"]    : return "HIGH"
    if score >= SEVERITY["MEDIUM"]  : return "MEDIUM"
    if score >= SEVERITY["LOW"]     : return "LOW"
    return "INFO"


def get_severity_emoji(severity: str) -> str:
    return {
        "CRITICAL": "[!!!]",
        "HIGH"    : "[!! ]",
        "MEDIUM"  : "[!  ]",
        "LOW"     : "[.  ]",
        "INFO"    : "[   ]",
    }.get(severity, "[   ]")


def build_shap_narrative(top_features: list) -> str:
    """
    Convert SHAP top features into a natural language sentence.
    This is the core NLG component of Novel Contribution 2.

    Example output:
    'Alert triggered: src_bytes is abnormally high (SHAP: +2.85),
     dst_host_suspicion_score elevated (SHAP: +2.18),
     dst_host_srv_count unusually low (SHAP: +1.83)'
    """
    if not top_features:
        return "No significant SHAP signals detected."

    parts = []
    for feat in top_features:
        name      = feat["feature"].replace("_", " ")
        shap_val  = feat["shap_value"]
        raw_val   = feat["raw_value"]
        direction = "abnormally high" if shap_val > 0 else "abnormally low"

        parts.append(
            f"{name} is {direction} "
            f"(value={raw_val:.3f}, SHAP: "
            f"{'+' if shap_val > 0 else ''}{shap_val:.3f})"
        )

    return "Alert triggered: " + " | ".join(parts)


def build_lime_narrative(parsed_lime: list) -> str:
    """
    Convert LIME conditions into a natural language sentence.
    LIME gives range conditions — we simplify for readability.
    """
    if not parsed_lime:
        return "No LIME signals available."

    parts = []
    for item in parsed_lime[:3]:
        condition = item["condition"]
        weight    = item["weight"]
        direction = "pushing toward attack" if weight > 0 else "reducing risk"
        parts.append(f"{condition} ({direction}, weight={weight:+.3f})")

    return "LIME context: " + " | ".join(parts)


def generate_alert(
    row_idx      : int,
    ensemble_score: float,
    shap_explanation: dict,
    lime_results : list,
    raw_features : dict,
    true_label   : int = None,
) -> dict:
    """
    Generate a unified alert object combining all signals.

    This is the data structure the React dashboard will render.
    Contains everything needed for an analyst to make a decision.

    Args:
        row_idx          : row index in test set
        ensemble_score   : combined model score [0,1]
        shap_explanation : output of explain_single_prediction()
        lime_results     : output of parse_lime_explanation()
        raw_features     : dict of feature name -> value for this row
        true_label       : ground truth (None in production)

    Returns:
        Structured alert dict
    """
    severity = get_severity(ensemble_score)
    emoji    = get_severity_emoji(severity)

    shap_narrative = build_shap_narrative(
        shap_explanation.get("top_features", [])
    )
    lime_narrative = build_lime_narrative(lime_results)

    # Combined analyst summary
    summary = (
        f"{emoji} [{severity}] Risk Score: {ensemble_score:.3f}\n"
        f"  {shap_narrative}\n"
        f"  {lime_narrative}"
    )

    alert = {
        "alert_id"       : f"ALT-{row_idx:06d}",
        "timestamp"      : datetime.utcnow().isoformat(),
        "severity"       : severity,
        "ensemble_score" : round(ensemble_score, 4),
        "shap_score"     : shap_explanation.get("prediction", 0.0),
        "base_value"     : shap_explanation.get("base_value", 0.0),
        "shap_narrative" : shap_narrative,
        "lime_narrative" : lime_narrative,
        "summary"        : summary,
        "top_shap_features": shap_explanation.get("top_features", []),
        "lime_conditions"  : lime_results[:3],
        "raw_features"     : raw_features,
        "true_label"       : true_label,
        "analyst_action"   : None,   # filled by dashboard
    }

    return alert


def run_alert_generator(n_alerts: int = 10) -> list:
    """
    Master function.
    Generates alerts for the top n_alerts highest-risk predictions
    from the ensemble output.

    Returns list of alert dicts, saved to outputs/alerts/
    """
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load ensemble scores
    ensemble_path = PROCESSED_DIR / "nslkdd_test_ensemble.csv"
    if not ensemble_path.exists():
        raise FileNotFoundError(
            "Run models/model_pipeline.py first to generate ensemble scores"
        )
    ensemble_df = pd.read_csv(ensemble_path)

    # Load LightGBM for SHAP + LIME
    model, features = load_lgb_model()
    shared = [f for f in features if f in ensemble_df.columns]
    X_test = ensemble_df[shared].fillna(0)

    # Compute SHAP values
    log.info("Computing SHAP values for alert generation...")
    from shap import TreeExplainer
    explainer_shap = TreeExplainer(model)
    shap_vals      = explainer_shap.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Build LIME explainer on training data
    log.info("Building LIME explainer...")
    train_df     = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")
    X_train      = train_df[shared].fillna(0)
    lime_explainer = build_lime_explainer(X_train, shared)

    # Select top n_alerts by ensemble score
    top_rows = (
        ensemble_df
        .nlargest(n_alerts, "ensemble_score")
        .reset_index()
    )

    log.info(f"Generating {n_alerts} alerts for highest-risk predictions...")

    alerts = []
    for _, row in top_rows.iterrows():
        orig_idx       = int(row["index"])
        ensemble_score = float(row["ensemble_score"])
        true_label     = int(row["label_binary"]) \
                         if "label_binary" in row else None

        # SHAP explanation
        shap_expl = explain_single_prediction(
            explainer_shap, shap_vals, X_test,
            row_idx=orig_idx, top_n=3
        )

        # LIME explanation
        lime_raw  = explain_instance(
            lime_explainer, model,
            X_test.iloc[orig_idx].values,
            num_samples=300,
        )
        lime_parsed = parse_lime_explanation(lime_raw)

        # Raw feature snapshot
        raw_feats = X_test.iloc[orig_idx].to_dict()

        alert = generate_alert(
            row_idx        = orig_idx,
            ensemble_score = ensemble_score,
            shap_explanation = shap_expl,
            lime_results   = lime_parsed,
            raw_features   = raw_feats,
            true_label     = true_label,
        )

        alerts.append(alert)
        log.info(f"\n{alert['summary']}")

    # Save alerts as JSON
    out_path = ALERTS_DIR / "generated_alerts.json"
    with open(out_path, "w") as f:
        json.dump(alerts, f, indent=2, default=str)
    log.info(f"\n{len(alerts)} alerts saved -> {out_path}")

    # Also save as CSV for easy inspection
    csv_rows = []
    for a in alerts:
        csv_rows.append({
            "alert_id"      : a["alert_id"],
            "severity"      : a["severity"],
            "ensemble_score": a["ensemble_score"],
            "shap_narrative": a["shap_narrative"],
            "lime_narrative": a["lime_narrative"],
            "true_label"    : a["true_label"],
        })
    pd.DataFrame(csv_rows).to_csv(
        ALERTS_DIR / "generated_alerts.csv", index=False
    )
    log.info("CSV version saved -> generated_alerts.csv")

    return alerts


if __name__ == "__main__":
    alerts = run_alert_generator(n_alerts=10)

    print(f"\n{'='*70}")
    print(f"GENERATED ALERTS SUMMARY")
    print(f"{'='*70}")
    for a in alerts:
        label = ("ATTACK" if a["true_label"] == 1
                 else "NORMAL" if a["true_label"] == 0
                 else "UNKNOWN")
        print(f"\n{a['alert_id']} | {a['severity']:<8} | "
              f"score={a['ensemble_score']:.3f} | "
              f"true={label}")
        print(f"  {a['shap_narrative'][:100]}...")