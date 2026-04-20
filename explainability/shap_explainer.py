import pandas as pd
import numpy as np
from pathlib import Path
import shap
import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger

log = get_logger("shap_explainer")

PLOTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "shap_plots"


def load_lgb_model() -> tuple:
    """Load saved LightGBM model and feature list."""
    data     = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
    model    = data["model"]
    features = data["features"]
    log.info(f"LightGBM loaded — {len(features)} features")
    return model, features


def compute_shap_values(
    model,
    X        : pd.DataFrame,
    sample_n : int = 2000,
) -> tuple:
    """
    Compute SHAP values using TreeExplainer.
    Samples up to sample_n rows for speed — full dataset not needed
    for global explanation.

    Returns:
        (explainer, shap_values, X_sample)
    """
    if len(X) > sample_n:
        X_sample = X.sample(sample_n, random_state=42)
        log.info(f"Sampled {sample_n:,} rows for SHAP computation")
    else:
        X_sample = X

    log.info("Initialising SHAP TreeExplainer...")
    explainer   = shap.TreeExplainer(model)

    log.info("Computing SHAP values (this may take 1-2 minutes)...")
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary classifier returns list [class0, class1]
    # We want class 1 (attack) values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    log.info(f"SHAP values computed — shape: {shap_values.shape}")
    return explainer, shap_values, X_sample


def get_top_features(
    shap_values : np.ndarray,
    features    : list,
    top_n       : int = 20,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP value per feature.
    Higher = more important globally.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df   = pd.DataFrame({
        "feature"         : features,
        "mean_abs_shap"   : mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    log.info(f"\nTop {top_n} features by mean |SHAP|:")
    for _, row in imp_df.head(top_n).iterrows():
        log.info(f"  {row['feature']:<40} {row['mean_abs_shap']:.4f}")

    return imp_df


def plot_summary(
    shap_values : np.ndarray,
    X_sample    : pd.DataFrame,
    top_n       : int = 20,
    save        : bool = True,
) -> None:
    """
    SHAP summary beeswarm plot — shows feature importance
    and direction of effect (red=high value pushes toward attack).
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display = top_n,
        show        = False,
    )
    plt.title("SHAP Feature Importance — Attack Detection", fontsize=14)
    plt.tight_layout()

    if save:
        out = PLOTS_DIR / "shap_summary.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        log.info(f"Summary plot saved -> {out}")
    plt.close()


def plot_waterfall(
    explainer   : shap.TreeExplainer,
    X_sample    : pd.DataFrame,
    shap_values : np.ndarray,
    row_idx     : int = 0,
    save        : bool = True,
) -> None:
    """
    Waterfall plot for a single prediction — shows exactly which
    features pushed the score up or down from the base rate.
    This is what the analyst dashboard displays per alert.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build SHAP Explanation object
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    explanation = shap.Explanation(
        values     = shap_values[row_idx],
        base_values= base_value,
        data       = X_sample.iloc[row_idx].values,
        feature_names = list(X_sample.columns),
    )

    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Row {row_idx}", fontsize=13)
    plt.tight_layout()

    if save:
        out = PLOTS_DIR / f"shap_waterfall_row{row_idx}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        log.info(f"Waterfall plot saved -> {out}")
    plt.close()


def plot_bar(
    shap_values : np.ndarray,
    X_sample    : pd.DataFrame,
    top_n       : int = 20,
    save        : bool = True,
) -> None:
    """
    Bar plot of mean absolute SHAP values — clean version
    suitable for the analyst dashboard sidebar.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.Series(mean_abs, index=X_sample.columns)
    feat_imp = feat_imp.nlargest(top_n).sort_values()

    plt.figure(figsize=(10, 8))
    feat_imp.plot(kind="barh", color="steelblue")
    plt.xlabel("Mean |SHAP Value|")
    plt.title(f"Top {top_n} Features by SHAP Importance", fontsize=13)
    plt.tight_layout()

    if save:
        out = PLOTS_DIR / "shap_bar.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        log.info(f"Bar plot saved -> {out}")
    plt.close()


def explain_single_prediction(
    explainer   : shap.TreeExplainer,
    shap_values : np.ndarray,
    X_sample    : pd.DataFrame,
    row_idx     : int,
    top_n       : int = 3,
) -> dict:
    """
    Generate structured explanation for a single prediction.
    This feeds directly into alert_generator.py.

    Returns dict with:
      - base_value    : model base rate
      - prediction    : final score
      - top_features  : list of {feature, shap_value, raw_value, direction}
      - alert_text    : pre-formatted alert string
    """
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    row_shap = shap_values[row_idx]
    row_data = X_sample.iloc[row_idx]

    # Final prediction score = base + sum(shap values)
    prediction = float(base_value + row_shap.sum())
    prediction = np.clip(prediction, 0.0, 1.0)

    # Top contributing features
    top_idx  = np.argsort(np.abs(row_shap))[::-1][:top_n]
    top_feats = []
    for i in top_idx:
        feat_name  = X_sample.columns[i]
        shap_val   = float(row_shap[i])
        raw_val    = float(row_data.iloc[i])
        direction  = "increases risk" if shap_val > 0 else "decreases risk"
        top_feats.append({
            "feature"   : feat_name,
            "shap_value": round(shap_val, 4),
            "raw_value" : round(raw_val, 4),
            "direction" : direction,
        })

    # Build alert text
    alert_parts = []
    for f in top_feats:
        sign = "+" if f["shap_value"] > 0 else ""
        alert_parts.append(
            f"{f['feature']}={f['raw_value']:.3f} "
            f"(SHAP: {sign}{f['shap_value']:.3f}, {f['direction']})"
        )
    alert_text = " | ".join(alert_parts)

    return {
        "base_value"  : round(float(base_value), 4),
        "prediction"  : round(prediction, 4),
        "top_features": top_feats,
        "alert_text"  : alert_text,
    }


def run_shap_explainer(sample_n: int = 2000) -> tuple:
    """Master function — compute SHAP, save plots, return artifacts."""
    model, features = load_lgb_model()

    # Load test data
    test_df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")
    shared   = [f for f in features if f in test_df.columns]
    X_test   = test_df[shared].fillna(0)

    explainer, shap_values, X_sample = compute_shap_values(
        model, X_test, sample_n=sample_n
    )

    # Global importance
    imp_df = get_top_features(shap_values, list(X_sample.columns))
    imp_df.to_csv(METADATA_DIR / "shap_importance.csv", index=False)
    log.info("SHAP importance saved -> shap_importance.csv")

    # Plots
    plot_summary(shap_values, X_sample)
    plot_bar(shap_values, X_sample)

    # Waterfall for first 3 samples
    for i in range(3):
        plot_waterfall(explainer, X_sample, shap_values, row_idx=i)

    # Sample individual explanations
    log.info("\nSample individual explanations:")
    for i in range(5):
        expl = explain_single_prediction(
            explainer, shap_values, X_sample, row_idx=i
        )
        log.info(f"\nRow {i} | score={expl['prediction']:.3f}")
        log.info(f"  {expl['alert_text']}")

    return explainer, shap_values, X_sample, imp_df


if __name__ == "__main__":
    explainer, shap_values, X_sample, imp_df = run_shap_explainer()
    print(f"\nTop 10 features by SHAP importance:")
    print(imp_df.head(10).to_string(index=False))
    print(f"\nPlots saved to: outputs/shap_plots/")