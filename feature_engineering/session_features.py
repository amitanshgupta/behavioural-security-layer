import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

log = get_logger("session_features")


def compute_error_rate_combined(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted average of all four error rate columns:
      serror_rate, srv_serror_rate  — SYN errors (DoS signal)
      rerror_rate, srv_rerror_rate  — REJ errors (probe/scan signal)

    SYN errors weighted slightly higher as they are stronger DoS indicators.
    Result range: [0, 1]
    """
    df["error_rate_combined"] = (
        df["serror_rate"].astype(float)     * 0.30 +
        df["srv_serror_rate"].astype(float) * 0.25 +
        df["rerror_rate"].astype(float)     * 0.25 +
        df["srv_rerror_rate"].astype(float) * 0.20
    )
    log.info("Computed error_rate_combined")
    return df


def compute_service_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 - same_srv_rate gives us how diverse the services are
    in the connection window. High diversity = scanning behaviour.
    Adjusted by diff_srv_rate for cross-service spread.
    Result range: [0, 1]
    """
    df["service_diversity_score"] = (
        (1 - df["same_srv_rate"].astype(float)) * 0.6 +
        df["diff_srv_rate"].astype(float)       * 0.4
    )
    log.info("Computed service_diversity_score")
    return df


def compute_dst_host_suspicion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Destination host level suspicion score combining:
      - dst_host_serror_rate     (SYN errors at host level)
      - dst_host_rerror_rate     (REJ errors at host level)
      - dst_host_diff_srv_rate   (service diversity at host)

    High score = destination host is seeing anomalous traffic patterns.
    Result range: [0, 1]
    """
    df["dst_host_suspicion_score"] = (
        df["dst_host_serror_rate"].astype(float)   * 0.35 +
        df["dst_host_rerror_rate"].astype(float)   * 0.35 +
        df["dst_host_diff_srv_rate"].astype(float) * 0.30
    )
    log.info("Computed dst_host_suspicion_score")
    return df


def compute_same_service_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Difference between connection-window same_srv_rate and
    host-level dst_host_same_srv_rate.

    Large delta = local window behaviour diverges from host norm
                = possible targeted service probe or sweep.
    Absolute value taken so direction doesn't matter.
    """
    df["same_service_ratio_delta"] = (
        df["same_srv_rate"].astype(float) -
        df["dst_host_same_srv_rate"].astype(float)
    ).abs()
    log.info("Computed same_service_ratio_delta")
    return df


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — runs all session feature transformations in order.

    Args:
        df : cleaned DataFrame

    Returns:
        DataFrame with session features added
    """
    log.info(f"Building session features on {len(df):,} rows...")

    df = compute_error_rate_combined(df)
    df = compute_service_diversity(df)
    df = compute_dst_host_suspicion(df)
    df = compute_same_service_delta(df)

    new_cols = [
        "error_rate_combined", "service_diversity_score",
        "dst_host_suspicion_score", "same_service_ratio_delta"
    ]
    log.info(f"Session features added: {new_cols}")
    return df


if __name__ == "__main__":
    from utils.constants import PROCESSED_DIR

    for split in ("train", "test"):
        path = PROCESSED_DIR / f"nslkdd_{split}_clean.csv"
        df = pd.read_csv(path)
        df = build_session_features(df)
        print(f"\n{split} shape: {df.shape}")
        print(df[[
            "error_rate_combined", "service_diversity_score",
            "dst_host_suspicion_score", "same_service_ratio_delta",
            "label_binary"
        ]].describe().to_string())
        