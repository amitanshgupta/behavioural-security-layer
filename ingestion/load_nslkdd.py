import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    NSLKDD_TRAIN,
    NSLKDD_TEST,
    NSLKDD_COLUMNS,
    NSLKDD_CATEGORICAL,
    NSLKDD_ATTACK_MAP,
    INTERIM_DIR,
)
from utils.logger import get_logger

log = get_logger("load_nslkdd")


def load_arff(filepath: Path) -> pd.DataFrame:
    """Parse .arff by reading only the @DATA section as plain CSV."""
    log.info(f"Loading ARFF file: {filepath}")

    rows = []
    in_data = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.upper() == "@DATA":
                in_data = True
                continue
            if in_data:
                rows.append(line.split(","))

    df = pd.DataFrame(rows)
    log.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def assign_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Assign readable column names."""
    if len(df.columns) == len(NSLKDD_COLUMNS):
        df.columns = NSLKDD_COLUMNS
    else:
        log.warning(
            f"Column count mismatch: got {len(df.columns)}, "
            f"expected {len(NSLKDD_COLUMNS)}. Skipping rename."
        )
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns:
      - label_binary : 0 = normal, 1 = attack
      - attack_type  : original string label (kept for analysis)
    """
    df["attack_type"] = df["label"].str.strip().str.lower()
    df["label_binary"] = df["attack_type"].map(NSLKDD_ATTACK_MAP)

    unknown = df["label_binary"].isna().sum()
    if unknown > 0:
        log.warning(f"{unknown} rows have unrecognised labels — setting to 1 (attack)")
        df["label_binary"] = df["label_binary"].fillna(1).astype(int)
    else:
        df["label_binary"] = df["label_binary"].astype(int)

    return df


def load_nslkdd(split: str = "train") -> pd.DataFrame:
    """
    Main entry point.

    Args:
        split: "train" or "test"

    Returns:
        Clean DataFrame ready for preprocessing.
    """
    assert split in ("train", "test"), "split must be 'train' or 'test'"

    filepath = NSLKDD_TRAIN if split == "train" else NSLKDD_TEST

    df = load_arff(filepath)
    df = assign_columns(df)
    df = encode_labels(df)

    # Save to interim
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / f"nslkdd_{split}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved interim file -> {out_path}")

    # Quick summary
    log.info(f"Label distribution:\n{df['label_binary'].value_counts().to_string()}")
    log.info(f"Attack types found: {sorted(df['attack_type'].unique())}")

    return df


if __name__ == "__main__":
    train_df = load_nslkdd("train")
    test_df  = load_nslkdd("test")
    print("\nTrain shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("\nSample rows:")
    print(train_df.head(3).to_string())