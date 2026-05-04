import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import INTERIM_DIR, PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger

log = get_logger("genai_detector_cert")

MODEL_PATH = METADATA_DIR / "genai_detector_cert.pkl"

# Domains associated with GenAI-adjacent or high-risk behaviour
# In 2010 context: freelancing, media, non-work browsing
GENAI_PROXY_DOMAINS = {
    # Freelancing / job seeking (pre-resignation)
    "elance.com", "freelancer.com", "odesk.com",
    "linkedin.com", "monster.com", "indeed.com",
    "glassdoor.com", "careerbuilder.com",
    # Cloud storage / file sharing (exfiltration)
    "dropbox.com", "drive.google.com", "wetransfer.com",
    "mediafire.com", "4shared.com", "box.com",
    # Paste / code sharing (data leakage)
    "pastebin.com", "github.com", "gist.github.com",
    # High-bandwidth media (policy violation)
    "youtube.com", "vimeo.com", "netflix.com",
    # Anonymisation
    "anonymizer.com", "hidemyass.com", "vpnbook.com",
}


def build_http_genai_features(
    http_df   : pd.DataFrame,
    cert_feat : pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-user-day GenAI proxy features from http activity.
    Merges with existing CERT feature matrix.

    New features:
      - genai_proxy_visits     : visits to GenAI-adjacent domains
      - freelance_visits       : freelancing/job site visits
      - cloud_visits           : cloud storage visits
      - media_visits           : video streaming visits
      - non_work_ratio         : non-work domains / total requests
      - browsing_entropy       : diversity of domains visited
      - peak_hour_browsing     : browsing during work peak hours
      - after_hours_ratio      : after-hours browsing fraction
      - rapid_request_score    : many requests in short time
    """
    log.info("Building HTTP GenAI proxy features...")

    # Flag domain categories
    http_df["is_genai_proxy"]  = http_df["domain"].isin(
        GENAI_PROXY_DOMAINS
    ).astype(int)

    http_df["is_freelance"]    = http_df["domain"].isin({
        "elance.com", "freelancer.com", "odesk.com",
        "linkedin.com", "monster.com", "indeed.com",
    }).astype(int)

    http_df["is_cloud"]        = http_df["domain"].isin({
        "dropbox.com", "drive.google.com", "wetransfer.com",
        "mediafire.com", "4shared.com", "box.com",
    }).astype(int)

    http_df["is_media"]        = http_df["domain"].isin({
        "youtube.com", "vimeo.com", "netflix.com",
        "dailymotion.com", "twitch.tv",
    }).astype(int)

    http_df["is_peak_hour"]    = (
        (http_df["hour"] >= 9) & (http_df["hour"] <= 17)
    ).astype(int)

    # Aggregate per user per day
    agg = http_df.groupby(["user", "date_only"]).agg(
        total_http        = ("id", "count"),
        genai_proxy_visits= ("is_genai_proxy", "sum"),
        freelance_visits  = ("is_freelance", "sum"),
        cloud_visits      = ("is_cloud", "sum"),
        media_visits      = ("is_media", "sum"),
        after_hours_http  = ("is_after_hours", "sum"),
        peak_hour_http    = ("is_peak_hour", "sum"),
        unique_domains    = ("domain", "nunique"),
    ).reset_index()

    # Derived ratios
    agg["non_work_ratio"] = (
        agg["genai_proxy_visits"] /
        (agg["total_http"] + 1)
    )
    agg["after_hours_ratio"] = (
        agg["after_hours_http"] /
        (agg["total_http"] + 1)
    )
    agg["rapid_request_score"] = (
        agg["total_http"] / (agg["unique_domains"] + 1)
    ).clip(upper=50) / 50.0

    # Browsing entropy — higher = more diverse (less focused attacks)
    agg["browsing_entropy"] = np.where(
        agg["total_http"] == 0, 0.0,
        -(1.0 / (agg["unique_domains"] + 1)) *
        np.log2(1.0 / (agg["unique_domains"] + 1) + 1e-9) *
        agg["unique_domains"]
    )

    log.info(f"HTTP features built: {len(agg):,} user-day records")
    return agg


def build_genai_feature_matrix(
    cert_feat : pd.DataFrame,
    http_agg  : pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge HTTP GenAI features into CERT feature matrix.
    """
    log.info("Merging GenAI features into CERT matrix...")
    merged = cert_feat.merge(
        http_agg,
        on=["user", "date_only"],
        how="left",
        suffixes=("", "_http")
    )

    # Fill missing HTTP activity with 0
    http_cols = [
        "genai_proxy_visits", "freelance_visits",
        "cloud_visits", "media_visits",
        "non_work_ratio", "after_hours_ratio",
        "rapid_request_score", "browsing_entropy",
        "total_http", "peak_hour_http",
    ]
    for col in http_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    log.info(f"Merged matrix: {merged.shape}")
    return merged


# Features for the GenAI one-class SVM
GENAI_CERT_FEATURES = [
    # HTTP proxy signals
    "genai_proxy_visits",
    "freelance_visits",
    "cloud_visits",
    "media_visits",
    "non_work_ratio",
    "after_hours_ratio",
    "rapid_request_score",
    "browsing_entropy",
    # USB / device signals (exfiltration proxy)
    "usb_connects",
    "usb_after_hours",
    # File signals
    "exfil_attempts_role_zscore",
    "executable_access",
    # Email signals
    "external_emails",
    "high_risk_emails",
    # Logon anomaly
    "after_hours_logons",
    "logon_hour_std",
    # Psychometric
    "psych_risk_score",
    "N",   # Neuroticism
    "C",   # Conscientiousness (low = higher risk)
]


def train_genai_cert(
    X_normal : pd.DataFrame,
    nu       : float = 0.05,
) -> tuple:
    """
    Train One-Class SVM on normal user behaviour.
    Learns what benign GenAI-adjacent activity looks like.
    """
    log.info(
        f"Training GenAI One-Class SVM on "
        f"{len(X_normal):,} normal user-days (nu={nu})..."
    )
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    model.fit(X_scaled)
    log.info("Training complete")
    return model, scaler


def evaluate_genai(
    model  : OneClassSVM,
    scaler : StandardScaler,
    X      : pd.DataFrame,
    y      : pd.Series,
    name   : str,
) -> dict:
    X_scaled = scaler.transform(X)
    raw      = model.predict(X_scaled)
    y_pred   = np.where(raw == -1, 1, 0)
    scores   = -model.decision_function(X_scaled)
    scores   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    f1  = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, scores)
    fpr = (
        ((y_pred == 1) & (y == 0)).sum() /
        ((y == 0).sum() + 1e-9)
    )

    log.info(f"\n--- {name} ---")
    log.info(f"F1={f1:.4f} | AUC={auc:.4f} | FPR={fpr:.4f}")
    log.info(f"\n{classification_report(y, y_pred, target_names=['normal','attack'])}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    return {
        "model" : name,
        "f1"    : round(f1, 4),
        "auc"   : round(auc, 4),
        "fpr"   : round(fpr, 4),
        "scores": scores,
        "y_pred": y_pred,
    }


def run_genai_cert() -> dict:
    """Master function."""
    # Load HTTP cleaned data
    log.info("Loading HTTP data...")
    http_df = pd.read_csv(
        INTERIM_DIR / "cert_http_clean.csv",
        low_memory=False
    )
    http_df["date_only"] = pd.to_datetime(
        http_df["date_only"]
    ).dt.date.astype(str)

    # Load CERT feature matrix — sample to 200k rows for speed
    log.info("Loading CERT feature matrix (sampled)...")
    cert_df = pd.read_csv(
        PROCESSED_DIR / "cert_features.csv",
        low_memory=False,
        nrows=200_000          # ← ADD THIS
    )
    cert_df["date_only"] = pd.to_datetime(
        cert_df["date_only"]
    ).dt.date.astype(str)

    # Build GenAI features
    http_agg = build_http_genai_features(http_df, cert_df)
    http_agg["date_only"] = pd.to_datetime(
        http_agg["date_only"]
    ).dt.date.astype(str)

    merged   = build_genai_feature_matrix(cert_df, http_agg)

    # Feature matrix
    features = [f for f in GENAI_CERT_FEATURES if f in merged.columns]
    log.info(f"Using {len(features)} GenAI features: {features}")

    X = merged[features].fillna(0)
    y = merged["label_binary"]

    # Temporal train/test split
    split_idx   = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train on normal only
    X_normal = X_train[y_train == 0]
    model, scaler = train_genai_cert(X_normal)

    # Evaluate
    train_results = evaluate_genai(
        model, scaler, X_train, y_train,
        "CERT GenAI Detector [train]"
    )
    test_results = evaluate_genai(
        model, scaler, X_test, y_test,
        "CERT GenAI Detector [test]"
    )

    # Save
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "features": features},
        MODEL_PATH
    )
    log.info(f"Model saved -> {MODEL_PATH}")

    # Save scored output
    out_df = X_test.copy()
    out_df["label_binary"]      = y_test.values
    out_df["genai_score"]       = test_results["scores"]
    out_df["genai_prediction"]  = test_results["y_pred"]
    out_df.to_csv(
        PROCESSED_DIR / "cert_genai_scored.csv", index=False
    )
    log.info("Scored output saved -> cert_genai_scored.csv")

    return test_results


if __name__ == "__main__":
    results = run_genai_cert()
    print(f"\nCERT GenAI Detector Results:")
    print(f"  F1      : {results['f1']:.4f}")
    print(f"  AUC     : {results['auc']:.4f}")
    print(f"  FPR     : {results['fpr']*100:.1f}%")