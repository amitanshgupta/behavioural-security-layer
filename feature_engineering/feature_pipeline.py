import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from feature_engineering.network_features import build_network_features
from feature_engineering.auth_features    import build_auth_features
from feature_engineering.session_features import build_session_features
from feature_engineering.file_features    import build_file_features
from feature_engineering.peer_features    import build_peer_features
from feature_engineering.genai_features   import build_genai_features
from preprocessing.validator              import validate
from utils.constants                      import PROCESSED_DIR
from utils.logger                         import get_logger

log = get_logger("feature_pipeline")

# Final feature columns passed to models
# (excludes raw originals that have been superseded by engineered features)
DROP_AFTER_ENGINEERING = [
    # These are superseded by peer z-score versions
    "src_bytes_peer_mean",
    "dst_bytes_peer_mean",
    "duration_peer_mean",
    "num_failed_logins_peer_mean",
    "count_peer_mean",
    "srv_count_peer_mean",
]


def run_feature_pipeline(
    df: pd.DataFrame,
    split: str = "",
    group_col: str = "protocol_type",
    save: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Runs all 6 feature modules in the correct order and saves output.

    Order matters:
      1. peer_features  — must run BEFORE network_features encodes
                          protocol_type as int (we need string for groupby)
      2. network_features — encodes categoricals, adds network signals
      3. auth_features
      4. session_features
      5. file_features
      6. genai_features

    Args:
        df        : cleaned DataFrame from cleaner.py
        split     : "train" or "test" for logging and filename
        group_col : peer grouping column (default: protocol_type)
        save      : if True, saves output to data/processed/

    Returns:
        Fully feature-engineered DataFrame ready for modelling
    """
    tag = f"[{split}] " if split else ""
    log.info(f"{tag}Starting feature pipeline on {len(df):,} rows...")

    # Step 1 — peer features first (needs raw protocol_type strings)
    df = build_peer_features(df, group_col)

    # Step 2 — network features (encodes protocol_type → int)
    df = build_network_features(df)

    # Step 3 — auth features
    df = build_auth_features(df)

    # Step 4 — session features
    df = build_session_features(df)

    # Step 5 — file features
    df = build_file_features(df)

    # Step 6 — genai features
    df = build_genai_features(df)

    # Drop superseded intermediate columns
    cols_to_drop = [c for c in DROP_AFTER_ENGINEERING if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log.info(f"{tag}Dropped intermediate columns: {cols_to_drop}")

    log.info(f"{tag}Feature pipeline complete. Final shape: {df.shape}")
    log.info(f"{tag}Total features: {df.shape[1] - 1} + 1 label")

    # Validate before saving
    valid = validate(df, split)
    if not valid:
        log.error(f"{tag}Validation failed after feature engineering!")

    if save:
        out_path = PROCESSED_DIR / f"nslkdd_{split}_features.csv"
        df.to_csv(out_path, index=False)
        log.info(f"{tag}Saved -> {out_path}")

    return df


if __name__ == "__main__":
    for split in ("train", "test"):
        path = PROCESSED_DIR / f"nslkdd_{split}_clean.csv"
        log.info(f"Loading {path}")
        df = pd.read_csv(path)
        df = run_feature_pipeline(df, split=split)

        print(f"\n{'='*60}")
        print(f"{split.upper()} — Final engineered dataset")
        print(f"Shape : {df.shape}")
        print(f"Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:>3}. {col}")