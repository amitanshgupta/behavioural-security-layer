import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import CICIDS_DIR, INTERIM_DIR
from utils.logger import get_logger

log = get_logger("load_cicids")

# CICIDS2017 column name cleanup map
# Raw headers have leading spaces — strip them
# Also rename Label -> label for consistency

BENIGN_LABEL  = "BENIGN"
ATTACK_LABELS = {
    "FTP-Patator"      : 1,
    "SSH-Patator"      : 1,
    "DoS slowloris"    : 1,
    "DoS Slowhttptest" : 1,
    "DoS Hulk"         : 1,
    "DoS GoldenEye"    : 1,
    "Heartbleed"       : 1,
    "Web Attack \x96 Brute Force": 1,
    "Web Attack \x96 XSS"        : 1,
    "Web Attack \x96 Sql Injection": 1,
    "Web Attack – Brute Force"   : 1,
    "Web Attack – XSS"           : 1,
    "Web Attack – Sql Injection" : 1,
    "Infiltration"     : 1,
    "Bot"              : 1,
    "PortScan"         : 1,
    "DDoS"             : 1,
    "BENIGN"           : 0,
}

# Features to keep — selected for overlap with NSL-KDD concepts
CICIDS_KEEP_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean",
    "Active Std",
    "Idle Mean",
    "Idle Std",
    "Label",
]


def load_single_file(path: Path) -> pd.DataFrame:
    """Load one CICIDS CSV file with cleaning."""
    log.info(f"Loading {path.name}...")
    df = pd.read_csv(path, low_memory=False)

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Keep only selected features + Label
    available = [c for c in CICIDS_KEEP_FEATURES if c in df.columns]
    df = df[available].copy()

    # Strip label
    df["Label"] = df["Label"].str.strip()

    # Source file tag for tracking
    df["source_file"] = path.stem

    log.info(f"  {len(df):,} rows | "
             f"attacks: {(df['Label'] != BENIGN_LABEL).sum():,}")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map string labels to binary 0/1."""
    df["attack_type"]   = df["Label"].str.strip()
    df["label_binary"]  = df["attack_type"].map(
        lambda x: ATTACK_LABELS.get(x, 1)  # unknown = attack
    )
    df = df.drop(columns=["Label"])
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all feature columns to numeric.
    Replace inf values with NaN then fill with column median.
    CICIDS has inf values in flow rate columns (division by zero
    in zero-duration flows).
    """
    feature_cols = [c for c in df.columns
                    if c not in ["attack_type", "label_binary",
                                 "source_file"]]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median
    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)

    return df


def rename_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename CICIDS columns to unified schema
    for compatibility with NSL-KDD feature names
    where concepts overlap.
    """
    rename_map = {
        "Flow Duration"              : "duration",
        "Total Fwd Packets"          : "count",
        "Total Backward Packets"     : "srv_count",
        "Total Length of Fwd Packets": "src_bytes",
        "Total Length of Bwd Packets": "dst_bytes",
        "Flow Bytes/s"               : "flow_bytes_per_sec",
        "Flow Packets/s"             : "flow_pkts_per_sec",
        "Flow IAT Mean"              : "flow_iat_mean",
        "Flow IAT Std"               : "flow_iat_std",
        "Fwd Packets/s"              : "fwd_pkts_per_sec",
        "Bwd Packets/s"              : "bwd_pkts_per_sec",
        "Min Packet Length"          : "min_pkt_len",
        "Max Packet Length"          : "max_pkt_len",
        "Packet Length Mean"         : "pkt_len_mean",
        "Packet Length Std"          : "pkt_len_std",
        "FIN Flag Count"             : "fin_flag",
        "SYN Flag Count"             : "syn_flag",
        "RST Flag Count"             : "rst_flag",
        "PSH Flag Count"             : "psh_flag",
        "ACK Flag Count"             : "ack_flag",
        "URG Flag Count"             : "urg_flag",
        "Down/Up Ratio"              : "down_up_ratio",
        "Average Packet Size"        : "avg_pkt_size",
        "Avg Fwd Segment Size"       : "avg_fwd_seg_size",
        "Avg Bwd Segment Size"       : "avg_bwd_seg_size",
        "Init_Win_bytes_forward"     : "init_win_fwd",
        "Init_Win_bytes_backward"    : "init_win_bwd",
        "Active Mean"                : "active_mean",
        "Active Std"                 : "active_std",
        "Idle Mean"                  : "idle_mean",
        "Idle Std"                   : "idle_std",
        "Destination Port"           : "dst_port",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                             if k in df.columns})
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features derived from CICIDS columns
    that mirror NSL-KDD engineered features.
    """
    # bytes_ratio — mirrors NSL-KDD network_features.py
    df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)

    # is_large_transfer
    df["is_large_transfer"] = (df["src_bytes"] > 10000).astype(int)

    # error_rate_combined — using flag counts as proxy
    total_flags = (
        df.get("syn_flag", 0) +
        df.get("rst_flag", 0) +
        df.get("fin_flag", 0) + 1
    )
    df["error_rate_combined"] = (
        df.get("rst_flag", 0) + df.get("syn_flag", 0)
    ) / total_flags

    # service_diversity_score — proxy using port range
    df["service_diversity_score"] = (
        df.get("dst_port", 0) / 65535.0
    ).clip(0, 1)

    # peer_outlier_score placeholder
    df["peer_outlier_score"] = 0.0

    # auth_failure_rate placeholder
    df["auth_failure_rate"] = 0.0

    # GenAI features — placeholders for CICIDS
    df["genai_composite_score"]  = 0.0
    df["api_call_pattern_score"] = 0.0

    return df


def load_cicids(
    sample_frac: float = 1.0,
    random_state: int  = 42,
) -> pd.DataFrame:
    """
    Master function — load all CICIDS2017 files,
    clean, encode labels, and return unified DataFrame.

    Args:
        sample_frac  : fraction of each file to sample (1.0 = all)
        random_state : for reproducibility

    Returns:
        Cleaned CICIDS DataFrame ready for feature pipeline
    """
    files = list(CICIDS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {CICIDS_DIR}. "
            "Download CICIDS2017 MachineLearningCSV.zip first."
        )

    log.info(f"Found {len(files)} CICIDS2017 files")
    frames = []

    for f in sorted(files):
        df = load_single_file(f)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total rows before cleaning: {len(combined):,}")

    combined = encode_labels(combined)
    combined = clean_numeric(combined)
    combined = rename_to_unified(combined)
    combined = add_engineered_features(combined)

    # Summary
    attack_rate = combined["label_binary"].mean() * 100
    log.info(f"CICIDS loaded: {len(combined):,} rows | "
             f"attack rate: {attack_rate:.1f}%")
    log.info(f"Attack types: "
             f"{combined['attack_type'].value_counts().to_dict()}")

    # Save interim
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "cicids_clean.csv"
    combined.to_csv(out, index=False)
    log.info(f"Saved -> {out}")

    return combined


if __name__ == "__main__":
    df = load_cicids(sample_frac=1.0)

    print(f"\nCICIDS2017 Dataset Summary")
    print(f"Shape     : {df.shape}")
    print(f"\nLabel distribution:")
    print(df["label_binary"].value_counts().to_string())
    print(f"\nAttack types:")
    print(df["attack_type"].value_counts().to_string())
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>3}. {col}")