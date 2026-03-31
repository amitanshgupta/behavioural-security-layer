import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

log = get_logger("auth_features")


def compute_auth_failure_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    failed_logins / (logged_in + failed_logins + 1)
    Ranges 0-1. High value = repeated failed attempts = brute force signal.
    """
    total_attempts = df["logged_in"] + df["num_failed_logins"] + 1
    df["auth_failure_rate"] = df["num_failed_logins"] / total_attempts
    log.info("Computed auth_failure_rate")
    return df


def compute_root_escalation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary: 1 if root_shell=1 OR su_attempted=1.
    Either condition alone is a strong privilege escalation signal.
    """
    df["is_root_escalation"] = (
        (df["root_shell"].astype(int) == 1) |
        (df["su_attempted"].astype(int) == 1)
    ).astype(int)

    pct = df["is_root_escalation"].mean() * 100
    log.info(f"Computed is_root_escalation — {pct:.2f}% of connections flagged")
    return df


def compute_privilege_abuse_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted sum of privilege-related signals:
      - num_root            (weight 0.4) — root-owned processes
      - num_compromised     (weight 0.3) — compromised conditions
      - num_shells          (weight 0.2) — shell prompts obtained
      - num_file_creations  (weight 0.1) — files created (lateral movement signal)

    Raw score normalised by its max to produce [0, 1] range.
    """
    raw_score = (
        df["num_root"].astype(float)           * 0.4 +
        df["num_compromised"].astype(float)    * 0.3 +
        df["num_shells"].astype(float)         * 0.2 +
        df["num_file_creations"].astype(float) * 0.1
    )
    max_val = raw_score.max()
    df["privilege_abuse_score"] = raw_score / max_val if max_val > 0 else 0.0
    log.info("Computed privilege_abuse_score")
    return df


def compute_anomalous_login_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary: 1 if is_guest_login=1 OR is_host_login=1.
    Both represent non-standard login contexts worth flagging.
    """
    df["anomalous_login_flag"] = (
        (df["is_guest_login"].astype(int) == 1) |
        (df["is_host_login"].astype(int) == 1)
    ).astype(int)

    pct = df["anomalous_login_flag"].mean() * 100
    log.info(f"Computed anomalous_login_flag — {pct:.2f}% of connections flagged")
    return df


def build_auth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — runs all auth feature transformations in order.

    Args:
        df : cleaned DataFrame (after network_features is optional)

    Returns:
        DataFrame with auth features added
    """
    log.info(f"Building auth features on {len(df):,} rows...")

    df = compute_auth_failure_rate(df)
    df = compute_root_escalation(df)
    df = compute_privilege_abuse_score(df)
    df = compute_anomalous_login_flag(df)

    new_cols = [
        "auth_failure_rate", "is_root_escalation",
        "privilege_abuse_score", "anomalous_login_flag"
    ]
    log.info(f"Auth features added: {new_cols}")
    return df


if __name__ == "__main__":
    from utils.constants import PROCESSED_DIR

    for split in ("train", "test"):
        path = PROCESSED_DIR / f"nslkdd_{split}_clean.csv"
        df = pd.read_csv(path)
        df = build_auth_features(df)
        print(f"\n{split} shape: {df.shape}")
        print(df[["auth_failure_rate", "is_root_escalation",
                   "privilege_abuse_score", "anomalous_login_flag",
                   "label_binary"]].describe().to_string())