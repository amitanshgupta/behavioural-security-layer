import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, INTERIM_DIR
from utils.logger import get_logger

log = get_logger("unify_schema")

# Columns present in BOTH datasets after renaming
# These form the unified network IDS feature set
UNIFIED_NETWORK_FEATURES = [
    # Core flow features (both datasets)
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    # Engineered features (both datasets)
    "bytes_ratio",
    "is_large_transfer",
    "error_rate_combined",
    "service_diversity_score",
    "peer_outlier_score",
    "auth_failure_rate",
    "genai_composite_score",
    "api_call_pattern_score",
    # CICIDS-specific (filled with 0 for NSL-KDD)
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_pkts_per_sec",
    "bwd_pkts_per_sec",
    "min_pkt_len",
    "max_pkt_len",
    "pkt_len_mean",
    "pkt_len_std",
    "syn_flag",
    "rst_flag",
    "fin_flag",
    "ack_flag",
    "avg_pkt_size",
    "init_win_fwd",
    "init_win_bwd",
    "active_mean",
    "idle_mean",
    # NSL-KDD-specific (filled with 0 for CICIDS)
    "serror_rate",
    "rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_suspicion_score",
    "same_service_ratio_delta",
    "src_bytes_peer_zscore",
    "dst_bytes_peer_zscore",
    "bytes_above_peer_mean",
    "privilege_abuse_score",
    "anomalous_login_flag",
]


def load_nslkdd_for_unification() -> pd.DataFrame:
    """
    Load NSL-KDD processed features and align to unified schema.
    CICIDS-specific columns filled with 0.
    """
    log.info("Loading NSL-KDD for unification...")
    train = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")
    test  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")
    df    = pd.concat([train, test], ignore_index=True)

    # Add CICIDS-specific columns with 0
    cicids_only = [
        "flow_bytes_per_sec", "flow_pkts_per_sec",
        "flow_iat_mean", "flow_iat_std",
        "fwd_pkts_per_sec", "bwd_pkts_per_sec",
        "min_pkt_len", "max_pkt_len",
        "pkt_len_mean", "pkt_len_std",
        "syn_flag", "rst_flag", "fin_flag", "ack_flag",
        "avg_pkt_size", "init_win_fwd", "init_win_bwd",
        "active_mean", "idle_mean",
    ]
    for col in cicids_only:
        if col not in df.columns:
            df[col] = 0.0

    df["dataset"]    = "nslkdd"
    df["attack_type"]= df.get("label_binary").map(
        {0: "BENIGN", 1: "attack"}
    )

    log.info(f"NSL-KDD unified: {len(df):,} rows")
    return df


def load_cicids_for_unification(
    sample_frac: float = 0.3,
) -> pd.DataFrame:
    """
    Load CICIDS interim and align to unified schema.
    NSL-KDD-specific columns filled with 0.
    Samples to balance dataset sizes.
    """
    log.info(f"Loading CICIDS for unification (sample={sample_frac})...")
    df = pd.read_csv(
        INTERIM_DIR / "cicids_clean.csv",
        low_memory=False
    )

    # Sample to avoid class imbalance from CICIDS dominating
    df = df.sample(frac=sample_frac, random_state=42)

    # Add NSL-KDD-specific columns with 0
    nslkdd_only = [
        "serror_rate", "rerror_rate",
        "same_srv_rate", "diff_srv_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_suspicion_score", "same_service_ratio_delta",
        "src_bytes_peer_zscore", "dst_bytes_peer_zscore",
        "bytes_above_peer_mean", "privilege_abuse_score",
        "anomalous_login_flag",
    ]
    for col in nslkdd_only:
        if col not in df.columns:
            df[col] = 0.0

    df["dataset"] = "cicids"
    log.info(f"CICIDS unified: {len(df):,} rows")
    return df


def unify_datasets(
    nsl_df  : pd.DataFrame,
    cic_df  : pd.DataFrame,
) -> tuple:
    """
    Align both DataFrames to UNIFIED_NETWORK_FEATURES and concatenate.

    Returns:
        (X, y, feature_list)
    """
    features = [f for f in UNIFIED_NETWORK_FEATURES
                if f in nsl_df.columns and f in cic_df.columns]

    log.info(f"Unified feature set: {len(features)} features")

    X_nsl = nsl_df[features].fillna(0)
    y_nsl = nsl_df["label_binary"]
    X_cic = cic_df[features].fillna(0)
    y_cic = cic_df["label_binary"]

    X = pd.concat([X_nsl, X_cic], ignore_index=True)
    y = pd.concat([y_nsl, y_cic], ignore_index=True)

    # Dataset source tags for analysis
    source = pd.concat([
        pd.Series(["nslkdd"] * len(X_nsl)),
        pd.Series(["cicids"] * len(X_cic)),
    ], ignore_index=True)

    log.info(f"Unified dataset: {len(X):,} rows | "
             f"attack rate: {y.mean()*100:.1f}%")
    log.info(f"NSL-KDD: {len(X_nsl):,} | CICIDS: {len(X_cic):,}")

    return X, y, features, source


def run_unification(sample_frac: float = 0.3) -> tuple:
    """Master function — unify NSL-KDD + CICIDS2017."""
    nsl_df = load_nslkdd_for_unification()
    cic_df = load_cicids_for_unification(sample_frac)

    X, y, features, source = unify_datasets(nsl_df, cic_df)

    # Save unified dataset
    out_df = X.copy()
    out_df["label_binary"] = y.values
    out_df["dataset"]      = source.values

    out_path = PROCESSED_DIR / "unified_network_features.csv"
    out_df.to_csv(out_path, index=False)
    log.info(f"Unified dataset saved -> {out_path}")

    return X, y, features, source


if __name__ == "__main__":
    X, y, features, source = run_unification(sample_frac=0.3)

    print(f"\nUnified Network Dataset")
    print(f"Shape          : {X.shape}")
    print(f"Attack rate    : {y.mean()*100:.1f}%")
    print(f"\nDataset breakdown:")
    print(source.value_counts().to_string())
    print(f"\nFeatures ({len(features)}):")
    for i, f in enumerate(features, 1):
        print(f"  {i:>3}. {f}")