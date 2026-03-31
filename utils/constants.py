from pathlib import Path

# ── Root ────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data Directories ────────────────────────────────────
RAW_DIR       = ROOT_DIR / "data" / "raw"
INTERIM_DIR   = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
METADATA_DIR  = ROOT_DIR / "data" / "metadata"

# ── Dataset-Specific Raw Paths ───────────────────────────
NSLKDD_DIR  = RAW_DIR / "nslkdd"
CICIDS_DIR  = RAW_DIR / "cicids2017"
CERT_DIR    = RAW_DIR / "cert"

# ── NSL-KDD File Paths ───────────────────────────────────
NSLKDD_TRAIN = NSLKDD_DIR / "KDDTrain+.arff"
NSLKDD_TEST  = NSLKDD_DIR / "KDDTest+.arff"

# ── NSL-KDD Label Mapping ────────────────────────────────
# All attack categories → binary: 0 = normal, 1 = attack
NSLKDD_ATTACK_MAP = {
    "normal": 0,
    "anomaly": 1,    # pre-aggregated attacks in this ARFF version
    # keep all the rest below as fallback
    "back": 1, "land": 1, "neptune": 1, "pod": 1,
    "smurf": 1, "teardrop": 1, "mailbomb": 1,
    "processtable": 1, "udpstorm": 1, "apache2": 1,
    "ipsweep": 1, "nmap": 1, "portsweep": 1, "satan": 1,
    "mscan": 1, "saint": 1,
    "ftp_write": 1, "guess_passwd": 1, "imap": 1,
    "multihop": 1, "phf": 1, "spy": 1, "warezclient": 1,
    "warezmaster": 1, "sendmail": 1, "named": 1,
    "snmpgetattack": 1, "snmpguess": 1, "xlock": 1,
    "xsnoop": 1, "httptunnel": 1,
    "buffer_overflow": 1, "loadmodule": 1, "perl": 1,
    "rootkit": 1, "ps": 1, "sqlattack": 1, "xterm": 1,
    "worm": 1,
}

# ── NSL-KDD Column Names (41 features + label + difficulty) ─
NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label",
]

# ── Categorical columns in NSL-KDD ───────────────────────
NSLKDD_CATEGORICAL = ["protocol_type", "service", "flag"]