import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

log = get_logger("peer_features")

# Features to compare against peer group means
PEER_COMPARISON_FEATURES = [
    "src_bytes",
    "dst_bytes",
    "duration",
    "num_failed_logins",
    "count",
    "srv_count",
]


def compute_peer_group_stats(
    df: pd.DataFrame,
    group_col: str = "protocol_type"
) -> pd.DataFrame:
    """
    For each connection, compute how it deviates from its
    peer group (same protocol_type) on key features.

    Adds two columns per feature:
      {feature}_peer_mean  — mean of that feature in the peer group
      {feature}_peer_zscore — (value - peer_mean) / (peer_std + 1e-9)

    Z-score > 3 or < -3 flags statistical outlier within peer group.
    """
    log.info(f"Computing peer group stats, grouping by '{group_col}'...")

    for feat in PEER_COMPARISON_FEATURES:
        if feat not in df.columns:
            log.warning(f"Feature '{feat}' not found — skipping peer stats")
            continue

        # Compute peer group mean and std
        peer_mean = df.groupby(group_col)[feat].transform("mean")
        peer_std  = df.groupby(group_col)[feat].transform("std").fillna(0)

        df[f"{feat}_peer_mean"]   = peer_mean
        df[f"{feat}_peer_zscore"] = (
            (df[feat].astype(float) - peer_mean) / (peer_std + 1e-9)
        )

    log.info(
        f"Peer stats computed for {len(PEER_COMPARISON_FEATURES)} features "
        f"across {df[group_col].nunique()} peer groups"
    )
    return df


def compute_peer_outlier_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all z-scores into a single peer outlier score.

    Score = mean of |z-score| across all peer-compared features.
    Higher score = more deviant from peer group = more suspicious.
    Normalised to [0, 1] by clipping at z=5 before averaging.
    """
    zscore_cols = [
        f"{feat}_peer_zscore"
        for feat in PEER_COMPARISON_FEATURES
        if f"{feat}_peer_zscore" in df.columns
    ]

    if not zscore_cols:
        log.warning("No z-score columns found — skipping outlier score")
        return df

    # Clip extreme z-scores, take absolute value, average
    clipped = df[zscore_cols].clip(-5, 5).abs()
    df["peer_outlier_score"] = clipped.mean(axis=1) / 5.0  # normalise to [0,1]

    high_pct = (df["peer_outlier_score"] > 0.5).mean() * 100
    log.info(
        f"Computed peer_outlier_score — "
        f"{high_pct:.1f}% of connections score above 0.5"
    )
    return df


def compute_bytes_peer_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Specific high-value signal: how many times above peer mean
    is this connection's src_bytes?

    ratio > 7.3x was cited in the proposal as a key alert trigger.
    We store the raw multiplier (clipped at 20x to prevent outlier dominance).
    """
    if "src_bytes_peer_mean" not in df.columns:
        log.warning("src_bytes_peer_mean not found — run compute_peer_group_stats first")
        return df

    raw = df["src_bytes"].astype(float) / (df["src_bytes_peer_mean"] + 1)
    df["bytes_above_peer_mean"] = raw.clip(upper=20.0)

    flagged_pct = (df["bytes_above_peer_mean"] > 7.3).mean() * 100
    log.info(
        f"Computed bytes_above_peer_mean — "
        f"{flagged_pct:.2f}% exceed 7.3x peer mean "
        f"(proposal alert threshold)"
    )
    return df


def build_peer_features(
    df: pd.DataFrame,
    group_col: str = "protocol_type"
) -> pd.DataFrame:
    """
    Master function — runs all peer feature computations in order.

    Args:
        df        : cleaned DataFrame (protocol_type must still be string
                    here — call this BEFORE network_features encodes it,
                    OR pass the encoded int col, both work for grouping)
        group_col : column to peer-group by (default: protocol_type)

    Returns:
        DataFrame with peer comparison features added
    """
    log.info(f"Building peer features on {len(df):,} rows...")

    df = compute_peer_group_stats(df, group_col)
    df = compute_peer_outlier_score(df)
    df = compute_bytes_peer_deviation(df)

    log.info(f"Peer features complete. New shape: {df.shape}")
    return df


if __name__ == "__main__":
    from utils.constants import PROCESSED_DIR

    for split in ("train", "test"):
        path = PROCESSED_DIR / f"nslkdd_{split}_clean.csv"
        df = pd.read_csv(path)
        df = build_peer_features(df)

        peer_cols = [
            "src_bytes_peer_zscore", "duration_peer_zscore",
            "peer_outlier_score", "bytes_above_peer_mean",
            "label_binary"
        ]
        print(f"\n{split} shape: {df.shape}")
        print(df[peer_cols].describe().to_string())

        # Show a sample of high peer outlier connections
        high_outliers = df[df["peer_outlier_score"] > 0.5][peer_cols].head(5)
        print(f"\nSample high peer outliers ({split}):")
        print(high_outliers.to_string())