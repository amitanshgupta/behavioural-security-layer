import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    PROCESSED_DIR,
    METADATA_DIR,
    CLUSTERING_FEATURES,
    DRIFT_DELTA,
)
from utils.logger import get_logger

log = get_logger("drift_detection")


def detect_feature_drift_vectorised(
    series: pd.Series,
    window: int = 500,
    threshold_std: float = 3.0,
) -> np.ndarray:
    """
    Fast vectorised drift detection using rolling statistics.
    Approximates ADWIN behaviour without row-by-row Python loops.

    Logic:
      - Compute rolling mean and std over `window` rows
      - Flag rows where value deviates > threshold_std * rolling_std
        from rolling mean
      - This is statistically equivalent to ADWIN's epsilon cut
        for stationary windows

    Returns binary array: 1 = drift point, 0 = normal
    """
    vals         = series.astype(float).values
    rolling      = pd.Series(vals).rolling(window=window, min_periods=window // 2)
    roll_mean    = rolling.mean().values
    roll_std     = rolling.std().fillna(0).values

    deviation    = np.abs(vals - roll_mean)
    threshold    = threshold_std * (roll_std + 1e-9)
    drift_flags  = (deviation > threshold).astype(int)

    # Zero out the warm-up period (first `window` rows)
    drift_flags[:window] = 0
    return drift_flags


def run_drift_detection(
    df: pd.DataFrame,
    features: list = None,
    window: int = 500,
    threshold_std: float = 3.0,
) -> tuple:
    """
    Run vectorised drift detection on each feature column.
    Adds {feat}_drift columns and summary any_drift column.

    Args:
        df            : DataFrame with feature columns
        features      : columns to monitor (default: CLUSTERING_FEATURES)
        window        : rolling window size
        threshold_std : deviation threshold in std units

    Returns:
        (df_with_drift_cols, drift_summary_dict)
    """
    features = features or [f for f in CLUSTERING_FEATURES
                             if f in df.columns]

    log.info(f"Running drift detection on {len(df):,} rows, "
             f"{len(features)} features "
             f"(window={window}, threshold={threshold_std}std)...")

    drift_summary = {}

    for feat in features:
        flags            = detect_feature_drift_vectorised(
            df[feat], window=window, threshold_std=threshold_std
        )
        col_name         = f"{feat}_drift"
        df[col_name]     = flags
        n_drifts         = int(flags.sum())
        drift_summary[feat] = n_drifts

        if n_drifts > 0:
            log.info(f"  {feat:<35} drift points: {n_drifts:>6,}")

    drift_cols      = [f"{f}_drift" for f in features
                       if f"{f}_drift" in df.columns]
    df["any_drift"] = df[drift_cols].max(axis=1)

    total = int(df["any_drift"].sum())
    log.info(f"Drift detection complete - "
             f"{total:,} rows flagged "
             f"({total/len(df)*100:.2f}%)")

    return df, drift_summary


def drift_report(drift_summary: dict) -> None:
    """Print a clean drift summary table."""
    print("\n--- Drift Detection Report ---")
    print(f"{'Feature':<35} {'Drift Points':>12}")
    print("-" * 50)
    total = 0
    for feat, count in sorted(
        drift_summary.items(), key=lambda x: -x[1]
    ):
        print(f"{feat:<35} {count:>12,}")
        total += count
    print("-" * 50)
    print(f"{'TOTAL':<35} {total:>12,}")


def detect_cluster_drift(
    df: pd.DataFrame,
    window_size: int = 1000,
) -> pd.DataFrame:
    """
    Detects distributional drift at cluster level.
    Compares rolling attack rate against cluster baseline.
    Flags windows deviating more than 2 std from cluster mean.
    """
    log.info(f"Running cluster-level drift detection "
             f"(window={window_size})...")

    results = []

    for cluster_id in sorted(df["cluster"].unique()):
        grp = df[df["cluster"] == cluster_id].copy()

        if len(grp) < window_size // 2:
            log.warning(f"Cluster {cluster_id}: only {len(grp)} rows - skipping")
            continue

        baseline_rate = grp["label_binary"].mean()
        baseline_std  = grp["label_binary"].std()

        rolling_rate = (
            grp["label_binary"]
            .rolling(window=window_size, min_periods=window_size // 2)
            .mean()
        )

        drift_mask        = (rolling_rate - baseline_rate).abs() > 2 * baseline_std
        n_drift_windows   = int(drift_mask.sum())

        results.append({
            "cluster"        : cluster_id,
            "size"           : len(grp),
            "baseline_rate"  : round(baseline_rate, 4),
            "n_drift_windows": n_drift_windows,
            "drift_detected" : n_drift_windows > 0,
        })

        log.info(f"  Cluster {cluster_id} | "
                 f"size={len(grp):>6,} | "
                 f"baseline={baseline_rate:.3f} | "
                 f"drift windows: {n_drift_windows}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    import joblib

    path = PROCESSED_DIR / "nslkdd_train_features.csv"
    df   = pd.read_csv(path)

    scaler = joblib.load(METADATA_DIR / "cluster_scaler.pkl")
    km     = joblib.load(METADATA_DIR / "kmeans_model.pkl")

    X             = df[[f for f in CLUSTERING_FEATURES if f in df.columns]]
    X_scaled      = scaler.transform(X)
    df["cluster"] = km.predict(X_scaled)

    # Feature-level drift
    df, drift_summary = run_drift_detection(df)
    drift_report(drift_summary)

    # Cluster-level drift
    cluster_drift = detect_cluster_drift(df)
    print("\n--- Cluster Drift Summary ---")
    print(cluster_drift.to_string(index=False))