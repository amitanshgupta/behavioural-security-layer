import pandas as pd
import numpy as np
from pathlib import Path
import lime
import lime.lime_tabular
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger

log = get_logger("lime_explainer")

PLOTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "lime_plots"


def load_lgb_model() -> tuple:
    data     = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
    log.info(f"LightGBM loaded — {len(data['features'])} features")
    return data["model"], data["features"]


def build_lime_explainer(
    X_train  : pd.DataFrame,
    features : list,
) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Build LIME TabularExplainer fitted on training data distribution.
    LIME perturbs input samples and fits a local linear model
    to approximate the black-box model behaviour around each point.
    """
    log.info("Building LIME TabularExplainer...")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data  = X_train.values,
        feature_names  = features,
        class_names    = ["normal", "attack"],
        mode           = "classification",
        random_state   = 42,
        discretize_continuous = True,
    )
    log.info("LIME explainer ready")
    return explainer


def explain_instance(
    explainer  : lime.lime_tabular.LimeTabularExplainer,
    model,
    row        : np.ndarray,
    top_n      : int = 10,
    num_samples: int = 500,
) -> lime.explanation.Explanation:
    """
    Explain a single prediction using LIME.
    num_samples = perturbation samples (500 balances speed vs accuracy).
    """
    explanation = explainer.explain_instance(
        data_row       = row,
        predict_fn     = model.predict_proba,
        top_labels     = 1,
        num_features   = top_n,
        num_samples    = num_samples,
    )
    return explanation


def parse_lime_explanation(
    explanation: lime.explanation.Explanation,
) -> list:
    """
    Extract feature contributions from LIME explanation.
    Uses whichever label is available in the explanation.
    """
    # Use the first available label rather than hardcoding 1
    label   = list(explanation.local_exp.keys())[0]
    raw     = explanation.as_list(label=label)
    parsed  = []
    for condition, weight in raw:
        parsed.append({
            "condition": condition,
            "weight"   : round(weight, 4),
            "direction": "increases risk" if weight > 0 else "decreases risk",
        })
    return sorted(parsed, key=lambda x: -abs(x["weight"]))


def build_alert_text(parsed: list, prediction_proba: float) -> str:
    """
    Convert LIME explanation into analyst-readable alert text.
    Format matches the SHAP alert text for consistency in dashboard.
    """
    parts = [f"Risk score: {prediction_proba:.3f}"]
    for item in parsed[:3]:
        sign = "+" if item["weight"] > 0 else ""
        parts.append(
            f"{item['condition']} "
            f"(LIME: {sign}{item['weight']:.3f}, {item['direction']})"
        )
    return " | ".join(parts)


def plot_lime_explanation(
    explanation : lime.explanation.Explanation,
    row_idx     : int,
    save        : bool = True,
) -> None:
    """Save LIME explanation bar chart for a single prediction."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use whichever label is available
    label = list(explanation.local_exp.keys())[0]
    
    fig = explanation.as_pyplot_figure(label=label)
    fig.set_size_inches(12, 6)
    plt.title(f"LIME Explanation — Row {row_idx}", fontsize=13)
    plt.tight_layout()

    if save:
        out = PLOTS_DIR / f"lime_explanation_row{row_idx}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        log.info(f"LIME plot saved -> {out}")
    plt.close()


def run_lime_explainer(
    n_samples  : int = 5,
    num_perturb: int = 500,
) -> list:
    """
    Master function.
    Explains n_samples predictions from the test set using LIME.

    Args:
        n_samples   : number of rows to explain
        num_perturb : LIME perturbation samples per explanation

    Returns:
        list of explanation dicts
    """
    model, features = load_lgb_model()

    # Load data
    train_df = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")

    shared   = [f for f in features if f in train_df.columns
                and f in test_df.columns]
    X_train  = train_df[shared].fillna(0)
    X_test   = test_df[shared].fillna(0)

    log.info(f"Using {len(shared)} features | "
             f"explaining {n_samples} test samples")

    # Build explainer on training distribution
    explainer = build_lime_explainer(X_train, shared)

    # Pick diverse samples — mix of attack and normal
    y_test   = test_df["label_binary"]
    attack_idx = y_test[y_test == 1].index[:3].tolist()
    normal_idx = y_test[y_test == 0].index[:2].tolist()
    sample_idx = attack_idx + normal_idx

    results = []
    for i, idx in enumerate(sample_idx):
        row        = X_test.loc[idx].values
        true_label = int(y_test.loc[idx])

        log.info(f"\nExplaining row {idx} "
                 f"(true label: {'attack' if true_label else 'normal'})...")

        explanation = explain_instance(
            explainer, model, row, num_samples=num_perturb
        )

        # Get attack class probability
        proba = model.predict_proba(
            pd.DataFrame([row], columns=shared)
        )[0][1]

        parsed     = parse_lime_explanation(explanation)
        alert_text = build_alert_text(parsed, proba)

        log.info(f"  Prediction probability: {proba:.3f}")
        log.info(f"  Alert: {alert_text}")

        plot_lime_explanation(explanation, row_idx=idx)

        results.append({
            "row_idx"   : idx,
            "true_label": true_label,
            "proba"     : round(proba, 4),
            "features"  : parsed,
            "alert_text": alert_text,
        })

    log.info(f"\nLIME explanations complete — {len(results)} samples explained")
    return results


if __name__ == "__main__":
    results = run_lime_explainer(n_samples=5, num_perturb=500)

    print("\n--- LIME Alert Summaries ---")
    for r in results:
        label = "ATTACK" if r["true_label"] else "NORMAL"
        print(f"\nRow {r['row_idx']} [{label}] | score={r['proba']:.3f}")
        print(f"  {r['alert_text']}")