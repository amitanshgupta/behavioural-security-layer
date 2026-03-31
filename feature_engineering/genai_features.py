import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

log = get_logger("genai_features")

# Ports commonly associated with LLM API calls
# 443 = HTTPS (OpenAI, Anthropic, Gemini APIs all use this)
# In NSL-KDD we approximate via service column numeric codes
# These are placeholder signals — real values come from CERT/live data
GENAI_SUSPICIOUS_SERVICES = {
    "http",
    "https",
    "http_443",
    "ssl",
}

# Thresholds for GenAI-like behaviour simulation
HIGH_OUTBOUND_BYTES  = 5000    # large prompt = large src_bytes
HIGH_INBOUND_BYTES   = 50000   # large response = large dst_bytes
RAPID_CONN_THRESHOLD = 50      # many connections in window = API polling


def compute_api_call_pattern_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates detection of LLM API call patterns using available
    NSL-KDD proxy signals:

    - High src_bytes (large prompt being sent)
    - High dst_bytes (large response being received)
    - High count (rapid repeated connections = API polling loop)

    Each component normalised independently then averaged.
    Real GenAI detection will use actual API call logs from CERT.
    """
    src_score   = (df["src_bytes"].astype(float) /
                   (df["src_bytes"].max() + 1)).clip(0, 1)
    dst_score   = (df["dst_bytes"].astype(float) /
                   (df["dst_bytes"].max() + 1)).clip(0, 1)
    count_score = (df["count"].astype(float) /
                   (df["count"].max() + 1)).clip(0, 1)

    df["api_call_pattern_score"] = (
        src_score   * 0.35 +
        dst_score   * 0.35 +
        count_score * 0.30
    )
    log.info("Computed api_call_pattern_score")
    return df


def compute_data_exfil_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy signal for data exfiltration via LLM prompts:
    unusually high outbound bytes relative to session duration.

    exfil_proxy = src_bytes / (duration + 1)
    High value = large data sent in short time = possible prompt stuffing.
    Clipped at 99th percentile, normalised to [0, 1].
    """
    raw = df["src_bytes"].astype(float) / (df["duration"].astype(float) + 1)
    clip_val = raw.quantile(0.99)
    df["data_exfil_proxy"] = (raw.clip(upper=clip_val) /
                               (clip_val + 1e-9))
    log.info(f"Computed data_exfil_proxy (99th pct clip={clip_val:.2f})")
    return df


def compute_rapid_request_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary: 1 if count > RAPID_CONN_THRESHOLD.
    Many connections in a short window mimics API polling /
    automated LLM query loops.
    """
    df["rapid_request_flag"] = (
        df["count"].astype(int) > RAPID_CONN_THRESHOLD
    ).astype(int)

    pct = df["rapid_request_flag"].mean() * 100
    log.info(
        f"Computed rapid_request_flag "
        f"(threshold={RAPID_CONN_THRESHOLD}) "
        f"— {pct:.1f}% flagged"
    )
    return df


def compute_genai_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final composite GenAI suspicion score combining all three signals.
    This is the feature the one-class SVM will be trained on.

    Score range: [0, 1]
    Higher = more GenAI-like behavioural signature.

    NOTE: On NSL-KDD this will be low across the board since the
    dataset predates LLM API usage patterns. The feature engineering
    logic is validated here; real signal separation will appear
    on CERT v6.2 and live telemetry data.
    """
    df["genai_composite_score"] = (
        df["api_call_pattern_score"] * 0.40 +
        df["data_exfil_proxy"]       * 0.40 +
        df["rapid_request_flag"]     * 0.20
    )
    high_pct = (df["genai_composite_score"] > 0.3).mean() * 100
    log.info(
        f"Computed genai_composite_score — "
        f"{high_pct:.2f}% of connections score above 0.3"
    )
    return df


def build_genai_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — runs all GenAI feature computations in order.

    Args:
        df : cleaned DataFrame

    Returns:
        DataFrame with GenAI proxy features added
    """
    log.info(f"Building GenAI features on {len(df):,} rows...")

    df = compute_api_call_pattern_score(df)
    df = compute_data_exfil_proxy(df)
    df = compute_rapid_request_flag(df)
    df = compute_genai_composite_score(df)

    new_cols = [
        "api_call_pattern_score", "data_exfil_proxy",
        "rapid_request_flag", "genai_composite_score"
    ]
    log.info(f"GenAI features added: {new_cols}")
    return df


if __name__ == "__main__":
    from utils.constants import PROCESSED_DIR

    for split in ("train", "test"):
        path = PROCESSED_DIR / f"nslkdd_{split}_clean.csv"
        df = pd.read_csv(path)
        df = build_genai_features(df)
        print(f"\n{split} shape: {df.shape}")
        print(df[[
            "api_call_pattern_score", "data_exfil_proxy",
            "rapid_request_flag", "genai_composite_score",
            "label_binary"
        ]].describe().to_string())