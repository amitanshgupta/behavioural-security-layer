import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    NSLKDD_CATEGORICAL,
    INTERIM_DIR,
    PROCESSED_DIR,
)
from utils.logger import get_logger

log = get_logger("cleaner")

# Columns that must stay as strings
CATEGORICAL_COLS = NSLKDD_CATEGORICAL  # ["protocol_type", "service", "flag"]

# Columns to drop before modelling
DROP_COLS = ["label", "attack_type"]


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-categorical feature columns to numeric."""
    log.info("Fixing dtypes...")
    for col in df.columns:
        if col in CATEGORICAL_COLS or col in DROP_COLS or col == "label_binary":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip whitespace from categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].str.strip()

    return df


def report_nulls(df: pd.DataFrame) -> None:
    """Log any null counts per column."""
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        log.info("No null values found.")
    else:
        log.warning(f"Null values detected:\n{null_counts.to_string()}")


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw label columns — keep only label_binary as target."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info(f"Dropped columns: {cols_to_drop}")
    return df


def clean(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Full cleaning pipeline for one split.

    Args:
        df    : raw interim DataFrame from load_nslkdd
        split : "train" or "test" — used for output filename

    Returns:
        Cleaned DataFrame saved to data/processed/
    """
    log.info(f"--- Cleaning {split} split ({len(df):,} rows) ---")

    df = fix_dtypes(df)
    report_nulls(df)
    df = drop_columns(df)

    # Sanity check
    log.info(f"Final shape: {df.shape}")
    log.info(f"Columns: {list(df.columns)}")
    log.info(f"Dtypes:\n{df.dtypes.to_string()}")

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"nslkdd_{split}_clean.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved cleaned file -> {out_path}")

    return df


if __name__ == "__main__":
    # Load interim files produced by load_nslkdd.py
    for split in ("train", "test"):
        interim_path = INTERIM_DIR / f"nslkdd_{split}.csv"
        log.info(f"Reading {interim_path}")
        df = pd.read_csv(interim_path)
        cleaned = clean(df, split)
        print(f"\n{split} cleaned shape: {cleaned.shape}")
        print(cleaned.head(3).to_string())