import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import NSLKDD_CATEGORICAL
from utils.logger import get_logger

log = get_logger("validator")

# Minimum required columns after cleaning
REQUIRED_COLS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "label_binary",
]

# Columns that must never be negative
NON_NEGATIVE_COLS = [
    "duration", "src_bytes", "dst_bytes", "num_failed_logins",
    "num_compromised", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "count", "srv_count",
]


def check_required_columns(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        log.error(f"Missing required columns: {missing}")
        return False
    log.info("Required columns check: PASSED")
    return True


def check_nulls(df: pd.DataFrame) -> bool:
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        log.error(f"Null values found:\n{null_counts.to_string()}")
        return False
    log.info("Null check: PASSED")
    return True


def check_label_integrity(df: pd.DataFrame) -> bool:
    if "label_binary" not in df.columns:
        log.error("label_binary column missing — skipping label check")
        return False
    unique_vals = set(df["label_binary"].unique())
    if not unique_vals.issubset({0, 1}):
        log.error(f"label_binary contains unexpected values: {unique_vals}")
        return False
    counts = df["label_binary"].value_counts().to_dict()
    log.info(f"Label integrity check: PASSED — distribution: {counts}")
    return True


def check_non_negative(df: pd.DataFrame) -> bool:
    failed = []
    for col in NON_NEGATIVE_COLS:
        if col not in df.columns:
            continue
        if (df[col] < 0).any():
            failed.append(col)
    if failed:
        log.error(f"Negative values found in: {failed}")
        return False
    log.info("Non-negative check: PASSED")
    return True


def validate(df: pd.DataFrame, split: str = "") -> bool:
    """
    Run all validation checks on a cleaned DataFrame.

    Args:
        df    : cleaned DataFrame
        split : label for logging ("train" / "test")

    Returns:
        True if all checks pass, False otherwise.
    """
    tag = f"[{split}] " if split else ""
    log.info(f"{tag}Running validation on {len(df):,} rows...")

    results = [
        check_required_columns(df),
        check_nulls(df),
        check_label_integrity(df),
        check_non_negative(df),
    ]

    if all(results):
        log.info(f"{tag}All validation checks PASSED")
        return True
    else:
        failed_count = results.count(False)
        log.error(f"{tag}{failed_count}/4 validation checks FAILED")
        return False


if __name__ == "__main__":
    from utils.constants import PROCESSED_DIR

    for split in ("train", "test"):
        path = PROCESSED_DIR / f"nslkdd_{split}_clean.csv"
        log.info(f"Loading {path}")
        df = pd.read_csv(path)
        passed = validate(df, split)
        print(f"\n{split}: {'PASSED' if passed else 'FAILED'}")