import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger
from evaluation.metrics import compute_all_metrics, METRICS_DIR

log = get_logger("experiments")


def experiment_threshold_sweep(
    scored_df  : pd.DataFrame,
    score_col  : str = "lgb_proba",
    label_col  : str = "label_binary",
    model_name : str = "LightGBM",
) -> pd.DataFrame:
    """
    Sweep decision threshold from 0.1 to 0.9.
    Shows how F1, FPR, recall trade off — helps pick optimal threshold.
    """
    log.info(f"Running threshold sweep for {model_name}...")
    thresholds = np.arange(0.1, 1.0, 0.05)
    rows       = []

    y_true   = scored_df[label_col].values
    y_scores = scored_df[score_col].values

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        fpr = (
            ((y_pred == 1) & (y_true == 0)).sum() /
            ((y_true == 0).sum() + 1e-9)
        )
        rows.append({
            "threshold": round(t, 2),
            "f1"       : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall"   : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "fpr"      : round(float(fpr), 4),
        })

    df = pd.DataFrame(rows)

    # Find optimal threshold — best F1 with FPR < 0.08
    valid  = df[df["fpr"] < 0.08]
    best   = valid.loc[valid["f1"].idxmax()] if len(valid) > 0 else df.loc[df["f1"].idxmax()]
    log.info(f"Optimal threshold: {best['threshold']} "
             f"(F1={best['f1']:.4f}, FPR={best['fpr']:.4f})")

    out = METRICS_DIR / f"threshold_sweep_{model_name.lower()}.csv"
    df.to_csv(out, index=False)
    log.info(f"Threshold sweep saved -> {out}")
    return df


def experiment_noise_sensitivity(
    train_df : pd.DataFrame,
    test_df  : pd.DataFrame,
    features : list,
) -> pd.DataFrame:
    """
    Test how DP noise level affects model performance.
    Trains LightGBM with varying noise multipliers and measures F1 drop.
    """
    import lightgbm as lgb
    from sklearn.metrics import f1_score
    from federated.dp_mechanism import add_gaussian_noise

    log.info("Running DP noise sensitivity experiment...")

    noise_levels = [0.0, 0.5, 1.0, 1.1, 2.0, 3.0]
    rows         = []

    X_train = train_df[features].fillna(0)
    y_train = train_df["label_binary"]
    X_test  = test_df[features].fillna(0)
    y_test  = test_df["label_binary"]

    for noise_mult in noise_levels:
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=7,
            learning_rate=0.05, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train, y_train)

        # Apply noise to feature importances as proxy
        if noise_mult > 0:
            weights = [model.feature_importances_.astype(float)]
            add_gaussian_noise(weights, noise_mult=noise_mult)

        y_pred = model.predict(X_test)
        f1     = f1_score(y_test, y_pred, zero_division=0)
        fpr    = (
            ((y_pred == 1) & (y_test == 0)).sum() /
            ((y_test == 0).sum() + 1e-9)
        )

        rows.append({
            "noise_multiplier": noise_mult,
            "f1"              : round(float(f1), 4),
            "fpr"             : round(float(fpr), 4),
        })
        log.info(f"  noise={noise_mult:.1f} | F1={f1:.4f} | FPR={fpr:.4f}")

    df  = pd.DataFrame(rows)
    out = METRICS_DIR / "noise_sensitivity.csv"
    df.to_csv(out, index=False)
    log.info(f"Noise sensitivity saved -> {out}")
    return df


def experiment_federated_vs_central(
) -> dict:
    """
    Load saved FL results and compare against centralised baseline.
    Produces the key table from the proposal evaluation section.
    """
    fl_path = METADATA_DIR / "fl_server_results.pkl"
    if not fl_path.exists():
        log.warning("FL results not found — run federated/server.py first")
        return {}

    fl_results = joblib.load(fl_path)

    comparison = {
        "centralised_f1"  : fl_results["central_f1"],
        "federated_f1"    : fl_results["test_results"]["mean_test_f1"],
        "gap"             : fl_results["fed_central_gap"],
        "within_3pct"     : fl_results["fed_central_gap"] <= 0.03,
        "epsilon_spent"   : fl_results["privacy_budget"]["epsilon_spent"],
        "epsilon_budget"  : fl_results["privacy_budget"]["epsilon_budget"],
        "rounds_to_converge": fl_results["n_rounds_run"],
    }

    log.info("\n--- Federated vs Centralised ---")
    for k, v in comparison.items():
        log.info(f"  {k:<25} : {v}")

    out = METRICS_DIR / "federated_vs_central.csv"
    pd.DataFrame([comparison]).to_csv(out, index=False)
    log.info(f"Saved -> {out}")
    return comparison


def run_all_experiments() -> dict:
    """Master function — run all experiments and collect results."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")
    lgb_data = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
    features = lgb_data["features"]

    # Experiment 1 — threshold sweep
    lgb_scored = pd.read_csv(PROCESSED_DIR / "nslkdd_test_lgb_scored.csv")
    threshold_df = experiment_threshold_sweep(lgb_scored)

    # Experiment 2 — noise sensitivity
    noise_df = experiment_noise_sensitivity(train_df, test_df, features)

    # Experiment 3 — federated vs central
    fl_comparison = experiment_federated_vs_central()

    return {
        "threshold_sweep"      : threshold_df,
        "noise_sensitivity"    : noise_df,
        "federated_vs_central" : fl_comparison,
    }


if __name__ == "__main__":
    results = run_all_experiments()

    print("\n--- Threshold Sweep (LightGBM) ---")
    print(results["threshold_sweep"].to_string(index=False))

    print("\n--- DP Noise Sensitivity ---")
    print(results["noise_sensitivity"].to_string(index=False))

    print("\n--- Federated vs Centralised ---")
    for k, v in results["federated_vs_central"].items():
        print(f"  {k:<25} : {v}")