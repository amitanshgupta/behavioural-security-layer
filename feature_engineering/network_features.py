import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import NSLKDD_CATEGORICAL
from utils.logger import get_logger

log = get_logger("network_features")


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode protocol_type, service, flag.
    Stores the category codes and saves mapping to df.attrs
    so downstream modules can decode if needed.
    """
    mappings = {}
    for col in NSLKDD_CATEGORICAL:
        if col not in df.columns:
            log.warning(f"Column {col} not found — skipping encoding")
            continue
        cat = df[col].astype("category")
        mappings[col] = dict(enumerate(cat.cat.categories))
        df[col] = cat.cat.codes
        log.info(f"Encoded '{col}' -> {len(mappings[col])} unique values")

    df.attrs["categorical_mappings"] = mappings
    return df


def compute_bytes_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    src_bytes / (dst_bytes + 1) — the +1 avoids division by zero.
    High ratio = more data sent than received = possible exfiltration.
    """
    df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    log.info("Computed bytes_ratio")
    return df


def compute_large_transfer_flag(
    df: pd.DataFrame,
    threshold: float = 10000.0
) -> pd.DataFrame:
    """
    Binary flag: 1 if src_bytes > threshold, else 0.
    Threshold default = 10KB, configurable.
    """
    df["is_large_transfer"] = (df["src_bytes"] > threshold).astype(int)
    pct = df["is_large_transfer"].mean() * 100
    log.info(f"Computed is_large_transfer (threshold={threshold}) "
             f"— {pct:.1f}% of connections flagged")
    return df


def compute_connection_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalised connection activity score:
    average of count and srv_count, scaled to [0, 1].
    Reflects how many connections share the same host/service context.
    """
    raw = (df["count"] + df["srv_count"]) / 2.0
    max_val = raw.max()
    df["connection_rate"] = raw / max_val if max_val > 0 else 0.0
    log.info("Computed connection_rate")
    return df


def build_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — runs all network feature transformations in order.

    Args:
        df : cleaned DataFrame from cleaner.py

    Returns:
        DataFrame with network features added/transformed
    """
    log.info(f"Building network features on {len(df):,} rows...")

    df = encode_categoricals(df)
    df = compute_bytes_ratio(df)
    df = compute_large_transfer_flag(df)
    df = compute_connection_rate(df)

    new_cols = ["bytes_ratio", "is_large_transfer", "connection_rate"]
    log.info(f"Network features added: {new_cols}")
    return df


if __name__ == "__main__":
    from utils.constants import PROCESSED_DIR

    for split in ("train", "test"):
        path = PROCESSED_DIR / f"nslkdd_{split}_clean.csv"
        df = pd.read_csv(path)
        df = build_network_features(df)
        print(f"\n{split} shape: {df.shape}")
        print(df[["protocol_type", "service", "flag",
                   "bytes_ratio", "is_large_transfer",
                   "connection_rate", "label_binary"]].head(5).to_string())