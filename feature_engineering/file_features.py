import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

log = get_logger("file_features")

# Thresholds — configurable
FILE_CREATION_THRESHOLD = 2
FILE_ACCESS_THRESHOLD   = 5


def compute_file_access_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted combination of file-related activity columns:
      num_file_creations (weight 0.4) — creating files is more suspicious
      num_access_files   (weight 0.4) — accessing sensitive files
      num_outbound_cmds  (weight 0.2) — outbound commands suggest exfiltration

    Normalised to [0, 1] by max value.
    """
    raw = (
        df["num_file_creations"].astype(float) * 0.4 +
        df["num_access_files"].astype(float)   * 0.4 +
        df["num_outbound_cmds"].astype(float)  * 0.2
    )
    max_val = raw.max()
    df["file_access_intensity"] = raw / max_val if max_val > 0 else 0.0
    log.info("Computed file_access_intensity")
    return df


def compute_suspicious_file_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary: 1 if file creations > threshold OR file accesses > threshold.
    Either alone is worth flagging for analyst review.
    """
    df["is_suspicious_file_activity"] = (
        (df["num_file_creations"].astype(int) > FILE_CREATION_THRESHOLD) |
        (df["num_access_files"].astype(int)   > FILE_ACCESS_THRESHOLD)
    ).astype(int)

    pct = df["is_suspicious_file_activity"].mean() * 100
    log.info(
        f"Computed is_suspicious_file_activity "
        f"(creation>{FILE_CREATION_THRESHOLD}, "
        f"access>{FILE_ACCESS_THRESHOLD}) "
        f"— {pct:.2f}% flagged"
    )
    return df


def compute_access_to_creation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    num_access_files / (num_file_creations + 1)

    High ratio = reading many files but creating few
               = reconnaissance / data harvesting pattern.
    Clipped at 99th percentile to prevent extreme outliers
    distorting downstream scaling.
    """
    raw = df["num_access_files"].astype(float) / (
        df["num_file_creations"].astype(float) + 1
    )
    clip_val = raw.quantile(0.99)
    df["access_to_creation_ratio"] = raw.clip(upper=clip_val)
    log.info(
        f"Computed access_to_creation_ratio "
        f"(clipped at 99th pct={clip_val:.2f})"
    )
    return df


def compute_outbound_cmd_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary: 1 if num_outbound_cmds > 0.
    In NSL-KDD, any outbound command from a non-FTP session
    is anomalous by definition.
    """
    df["outbound_cmd_flag"] = (
        df["num_outbound_cmds"].astype(int) > 0
    ).astype(int)

    pct = df["outbound_cmd_flag"].mean() * 100
    log.info(f"Computed outbound_cmd_flag — {pct:.2f}% of connections flagged")
    return df


def build_file_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — runs all file feature transformations in order.

    Args:
        df : cleaned DataFrame

    Returns:
        DataFrame with file activity features added
    """
    log.info(f"Building file features on {len(df):,} rows...")

    df = compute_file_access_intensity(df)
    df = compute_suspicious_file_flag(df)
    df = compute_access_to_creation_ratio(df)
    df = compute_outbound_cmd_flag(df)

    new_cols = [
        "file_access_intensity", "is_suspicious_file_activity",
        "access_to_creation_ratio", "outbound_cmd_flag"
    ]
    log.info(f"File features added: {new_cols}")
    return df


if __name__ == "__main__":
    from utils.constants import PROCESSED_DIR

    for split in ("train", "test"):
        path = PROCESSED_DIR / f"nslkdd_{split}_clean.csv"
        df = pd.read_csv(path)
        df = build_file_features(df)
        print(f"\n{split} shape: {df.shape}")
        print(df[[
            "file_access_intensity", "is_suspicious_file_activity",
            "access_to_creation_ratio", "outbound_cmd_flag",
            "label_binary"
        ]].describe().to_string())