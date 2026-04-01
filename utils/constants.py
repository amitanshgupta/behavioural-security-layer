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

# ── Context Engine Config ────────────────────────────────
N_CLUSTERS          = 8       # K-Means clusters (role groups)
ROLLING_WINDOWS     = [7, 30, 90]   # days — for baseline profiles
DRIFT_DELTA         = 0.002   # ADWIN sensitivity
ANOMALY_THRESHOLD_K = 3.0     # mean ± k*std for dynamic thresholds

# Features used for clustering (behaviour-rich, no label)
CLUSTERING_FEATURES = [
    "duration", "src_bytes", "dst_bytes",
    "count", "srv_count",
    "serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate",
    "dst_host_count", "dst_host_srv_count",
    "bytes_ratio", "connection_rate",
    "error_rate_combined", "service_diversity_score",
    "peer_outlier_score",
]

# ── CERT Dataset Paths ───────────────────────────────────
CERT_LOGON_PATH       = CERT_DIR / "logon.csv"
CERT_DEVICE_PATH      = CERT_DIR / "device.csv"
CERT_FILE_PATH        = CERT_DIR / "file.csv"
CERT_EMAIL_PATH       = CERT_DIR / "email.csv"
CERT_HTTP_PATH        = CERT_DIR / "http.csv"
CERT_USERS_PATH       = CERT_DIR / "users.csv"
CERT_PSYCHOMETRIC_PATH= CERT_DIR / "psychometric.csv"
CERT_DECOY_PATH       = CERT_DIR / "decoy_file.csv"
CERT_LDAP_DIR         = CERT_DIR / "LDAP"

# ── CERT Column Schemas ──────────────────────────────────
CERT_LOGON_COLS   = ["id", "date", "user", "pc", "activity"]
CERT_DEVICE_COLS  = ["id", "date", "user", "pc", "file_tree", "activity"]
CERT_FILE_COLS    = ["id", "date", "user", "pc", "filename",
                     "activity", "to_removable_media",
                     "from_removable_media", "content"]
CERT_USERS_COLS   = ["employee_name", "user_id", "email", "role",
                     "projects", "business_unit", "functional_unit",
                     "department", "team", "supervisor",
                     "start_date", "end_date"]
CERT_PSYCHO_COLS  = ["employee_name", "user_id", "O", "C", "E", "A", "N"]

# ── CERT Role Mapping ────────────────────────────────────
CERT_ROLE_TIER = {
    # Executives
    "President": 5, "VicePresident": 5, "Director": 4,
    # IT
    "ITAdmin": 4, "ITHelpDesk": 3, "WebDeveloper": 3,
    "SoftwareDeveloper": 3, "SoftwareEngineer": 3,
    "ComputerProgrammer": 3, "ComputerScientist": 3,
    "SoftwareQualityEngineer": 2,
    # Engineering
    "ChiefEngineer": 4, "SystemsEngineer": 3,
    "MechanicalEngineer": 2, "MaterialsEngineer": 2,
    "HardwareEngineer": 2, "ElectricalEngineer": 2,
    "IndustrialEngineer": 2, "FieldServiceEngineer": 2,
    "TestEngineer": 2, "Engineer": 2,
    # Management
    "Manager": 3, "ProjectManager": 3, "LabManager": 3,
    "Supervisor": 3,
    # Science
    "Scientist": 2, "Physicist": 2, "Mathematician": 2,
    "Statistician": 2, "Economist": 2,
    # Admin/Support
    "AdministrativeAssistant": 1, "PurchasingClerk": 1,
    "StockroomClerk": 1, "HumanResourceSpecialist": 1,
    "InstructionalCoordinator": 1, "TechnicalWriter": 1,
    # Other
    "SecurityGuard": 1, "Technician": 2,
    "NursePractitioner": 2, "Nurse": 1,
    "Salesman": 1, "AccountManager": 2,
    "ProductionLineWorker": 1, "Unknown": 0,
}