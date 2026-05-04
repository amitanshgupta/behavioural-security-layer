import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    INTERIM_DIR, PROCESSED_DIR,
    CERT_USERS_PATH, CERT_PSYCHOMETRIC_PATH,
    CERT_ROLE_TIER,
)
from utils.logger import get_logger

log = get_logger("cert_features")


def load_cleaned(name: str) -> pd.DataFrame:
    path = INTERIM_DIR / f"cert_{name}_clean.csv"
    log.info(f"Loading {path.name}...")
    df = pd.read_csv(path, low_memory=False)
    df["date_only"] = pd.to_datetime(df["date_only"]).dt.date
    return df


def build_logon_features(logon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate logon events per user per day.
    Features:
      - total_logons, total_logoffs
      - after_hours_logons
      - weekend_logons
      - unique_pcs            (how many machines logged into)
      - logon_hour_mean       (average hour of logon)
      - logon_hour_std        (consistency of logon times)
    """
    log.info("Building logon features...")
    logons  = logon_df[logon_df["is_logon"] == 1]
    logoffs = logon_df[logon_df["is_logon"] == 0]

    agg = logon_df.groupby(["user", "date_only"]).agg(
        total_events      = ("id", "count"),
        after_hours_logons= ("is_after_hours", "sum"),
        weekend_logons    = ("is_weekend", "sum"),
        unique_pcs        = ("pc", "nunique"),
        logon_hour_mean   = ("hour", "mean"),
        logon_hour_std    = ("hour", "std"),
    ).reset_index()

    logon_counts = logons.groupby(
        ["user", "date_only"]
    ).size().reset_index(name="total_logons")

    logoff_counts = logoffs.groupby(
        ["user", "date_only"]
    ).size().reset_index(name="total_logoffs")

    agg = agg.merge(logon_counts,  on=["user", "date_only"], how="left")
    agg = agg.merge(logoff_counts, on=["user", "date_only"], how="left")
    agg["logon_hour_std"] = agg["logon_hour_std"].fillna(0)
    agg[["total_logons", "total_logoffs"]] = \
        agg[["total_logons", "total_logoffs"]].fillna(0)

    log.info(f"Logon features: {len(agg):,} user-day records")
    return agg


def build_device_features(device_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate USB/device events per user per day.
    Features:
      - usb_connects         (number of USB connect events)
      - usb_after_hours      (connects outside business hours)
      - usb_weekend          (weekend connects)
      - unique_devices       (distinct file_tree roots)
    """
    log.info("Building device features...")
    agg = device_df.groupby(["user", "date_only"]).agg(
        usb_connects   = ("is_connect", "sum"),
        usb_after_hours= ("is_after_hours", "sum"),
        usb_weekend    = ("is_weekend", "sum"),
        unique_devices = ("file_tree", "nunique"),
    ).reset_index()

    log.info(f"Device features: {len(agg):,} user-day records")
    return agg


def build_file_features(file_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate file activity per user per day.
    Features:
      - file_events          (total file operations)
      - exfil_attempts       (writes to removable media)
      - executable_access    (exe/bat/ps1 files touched)
      - after_hours_files    (file ops outside hours)
      - unique_files         (distinct filenames)
    """
    log.info("Building file features...")
    agg = file_df.groupby(["user", "date_only"]).agg(
        file_events       = ("id", "count"),
        exfil_attempts    = ("is_exfil_attempt", "sum"),
        executable_access = ("is_executable", "sum"),
        after_hours_files = ("is_after_hours", "sum"),
        unique_files      = ("filename", "nunique"),
    ).reset_index()

    log.info(f"File features: {len(agg):,} user-day records")
    return agg


def build_http_features(http_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate web activity per user per day.
    Features:
      - http_requests        (total web visits)
      - suspicious_domains   (cloud storage, paste sites)
      - job_site_visits      (LinkedIn, Monster etc — pre-resignation signal)
      - cloud_storage_visits (Dropbox, GDrive — exfiltration vector)
      - after_hours_http     (browsing outside hours)
      - unique_domains       (breadth of web activity)
    """
    log.info("Building HTTP features...")
    agg = http_df.groupby(["user", "date_only"]).agg(
        http_requests       = ("id", "count"),
        suspicious_domains  = ("is_suspicious_domain", "sum"),
        job_site_visits     = ("is_job_site", "sum"),
        cloud_storage_visits= ("is_cloud_storage", "sum"),
        after_hours_http    = ("is_after_hours", "sum"),
        unique_domains      = ("domain", "nunique"),
    ).reset_index()

    log.info(f"HTTP features: {len(agg):,} user-day records")
    return agg


def build_email_features(email_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate email activity per user per day.
    Features:
      - emails_sent          (outbound emails)
      - external_emails      (to non-dtaa.com)
      - high_risk_emails     (external + large + attachment)
      - avg_email_size       (mean size of outbound emails)
      - after_hours_emails   (sent outside hours)
    """
    log.info("Building email features...")
    sent = email_df[email_df["activity"] == "Send"]

    agg = sent.groupby(["user", "date_only"]).agg(
        emails_sent      = ("id", "count"),
        external_emails  = ("is_external", "sum"),
        high_risk_emails = ("is_high_risk_email", "sum"),
        avg_email_size   = ("size", "mean"),
        after_hours_emails=("is_after_hours", "sum"),
    ).reset_index()

    log.info(f"Email features: {len(agg):,} user-day records")
    return agg


def build_user_context(users_df: pd.DataFrame,
                        psycho_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge user roster with psychometric scores.
    Adds role tier and OCEAN-based risk score.
    """
    users_df["role_tier"] = users_df["role"].map(
        CERT_ROLE_TIER
    ).fillna(0).astype(int)

    psycho_df["psych_risk_score"] = (
        psycho_df["N"].astype(float) * 0.5 +
        (100 - psycho_df["C"].astype(float)) * 0.3 +
        (100 - psycho_df["A"].astype(float)) * 0.2
    ) / 100.0

    master = users_df.merge(
        psycho_df[["user_id", "psych_risk_score",
                   "O", "C", "E", "A", "N"]],
        on="user_id", how="left"
    )
    log.info(f"User context built: {len(master):,} employees")
    return master


def build_peer_deviation(
    user_day_df: pd.DataFrame,
    role_col   : str = "role_tier",
) -> pd.DataFrame:
    """
    Compute z-scores relative to role-based peer group.
    This is the CERT version of peer_features.py —
    grouping by role tier instead of protocol type.

    Features added per numeric column:
      {col}_role_zscore
    """
    log.info("Computing role-based peer deviations...")

    numeric_cols = [
        "total_logons", "after_hours_logons", "unique_pcs",
        "usb_connects", "exfil_attempts", "file_events",
        "http_requests", "suspicious_domains", "job_site_visits",
        "external_emails",
    ]
    numeric_cols = [c for c in numeric_cols if c in user_day_df.columns]

    for col in numeric_cols:
        peer_mean = user_day_df.groupby(role_col)[col].transform("mean")
        peer_std  = user_day_df.groupby(role_col)[col].transform("std").fillna(1)
        user_day_df[f"{col}_role_zscore"] = (
            (user_day_df[col] - peer_mean) / (peer_std + 1e-9)
        )

    log.info(f"Role peer z-scores added for {len(numeric_cols)} features")
    return user_day_df


def build_label_column(
    user_day_df: pd.DataFrame,
    file_df    : pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ground truth label for each user-day record.
    label_binary = 1 if user had ANY exfil attempt that day.

    This is a weak label — real CERT ground truth requires
    the insider threat scenario file (not always available).
    Using exfil_attempts > 0 as proxy for malicious activity.
    """
    user_day_df["label_binary"] = (
        user_day_df["exfil_attempts"] > 0
    ).astype(int)

    attack_pct = user_day_df["label_binary"].mean() * 100
    log.info(f"Labels built — {attack_pct:.1f}% user-days flagged as malicious")
    return user_day_df


def run_cert_feature_pipeline(save: bool = True) -> pd.DataFrame:
    """
    Master function — builds complete CERT feature matrix.

    Output: one row per user per day with all behavioural features.
    """
    log.info("Starting CERT feature pipeline...")

    # Load cleaned tables
    logon_df  = load_cleaned("logon")
    device_df = load_cleaned("device")
    file_df   = load_cleaned("file")
    http_df   = load_cleaned("http")
    email_df  = load_cleaned("email")

    # Load user context
    users_df  = pd.read_csv(CERT_USERS_PATH)
    psycho_df = pd.read_csv(CERT_PSYCHOMETRIC_PATH)
    user_ctx  = build_user_context(users_df, psycho_df)

    # Build per-table aggregations
    logon_feat  = build_logon_features(logon_df)
    device_feat = build_device_features(device_df)
    file_feat   = build_file_features(file_df)
    http_feat   = build_http_features(http_df)
    email_feat  = build_email_features(email_df)

    # Merge all on user + date_only
    log.info("Merging all feature tables...")
    df = logon_feat
    for feat_df in [device_feat, file_feat, http_feat, email_feat]:
        df = df.merge(feat_df, on=["user", "date_only"], how="left")

    # Merge user context
    df = df.merge(
        user_ctx[["user_id", "role", "role_tier",
                  "department", "psych_risk_score",
                  "O", "C", "E", "A", "N"]],
        left_on="user", right_on="user_id", how="left"
    ).drop(columns=["user_id"], errors="ignore")

    # Fill missing activity with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Role peer deviation
    df = build_peer_deviation(df)

    # Label
    df = build_label_column(df, file_df)

    log.info(f"CERT feature matrix: {df.shape[0]:,} user-day rows, "
             f"{df.shape[1]} columns")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / "cert_features.csv"
        df.to_csv(out, index=False)
        log.info(f"Saved -> {out}")

    return df


if __name__ == "__main__":
    df = run_cert_feature_pipeline(save=True)

    print(f"\nCERT Feature Matrix Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>3}. {col}")
    print(f"\nLabel distribution:")
    print(df["label_binary"].value_counts().to_string())
    print(f"\nSample rows:")
    print(df.head(3).to_string())