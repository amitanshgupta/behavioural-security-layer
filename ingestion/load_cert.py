import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    CERT_LOGON_PATH, CERT_DEVICE_PATH, CERT_FILE_PATH,
    CERT_USERS_PATH, CERT_PSYCHOMETRIC_PATH, CERT_DECOY_PATH,
    CERT_LDAP_DIR, CERT_ROLE_TIER, INTERIM_DIR,
)
from utils.logger import get_logger

log = get_logger("load_cert")


def load_logon(nrows: int = None) -> pd.DataFrame:
    """
    Load logon.csv — login/logoff events.
    Parses timestamps, extracts hour-of-day and is_after_hours flag.
    """
    log.info(f"Loading logon.csv (nrows={nrows or 'all'})...")
    df = pd.read_csv(CERT_LOGON_PATH, nrows=nrows)

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S")
    df["hour"]          = df["date"].dt.hour
    df["day_of_week"]   = df["date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_after_hours"]= ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)
    df["is_logon"]      = (df["activity"] == "Logon").astype(int)

    log.info(f"Logon loaded: {len(df):,} rows | "
             f"after-hours: {df['is_after_hours'].mean()*100:.1f}%")
    return df


def load_device(nrows: int = None) -> pd.DataFrame:
    """
    Load device.csv — USB/removable device connect/disconnect events.
    High USB activity = data exfiltration risk.
    """
    log.info(f"Loading device.csv (nrows={nrows or 'all'})...")
    df = pd.read_csv(CERT_DEVICE_PATH, nrows=nrows)

    df["date"]        = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S")
    df["hour"]        = df["date"].dt.hour
    df["is_weekend"]  = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_connect"]  = (df["activity"] == "Connect").astype(int)
    df["is_after_hours"] = (
        (df["hour"] < 7) | (df["hour"] > 19)
    ).astype(int)

    log.info(f"Device loaded: {len(df):,} rows | "
             f"connect events: {df['is_connect'].sum():,}")
    return df


def load_file(nrows: int = None) -> pd.DataFrame:
    """
    Load file.csv — file access events.
    Flags removable media transfers and executable file access.
    """
    log.info(f"Loading file.csv (nrows={nrows or 'all'})...")
    df = pd.read_csv(
        CERT_FILE_PATH, nrows=nrows,
        usecols=["id", "date", "user", "pc",
                 "filename", "activity",
                 "to_removable_media", "from_removable_media"]
    )

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S")
    df["hour"] = df["date"].dt.hour
    df["is_after_hours"] = (
        (df["hour"] < 7) | (df["hour"] > 19)
    ).astype(int)

    # Normalise boolean columns
    df["to_removable_media"]   = df["to_removable_media"].map(
        {"True": 1, "False": 0, True: 1, False: 0}
    ).fillna(0).astype(int)
    df["from_removable_media"] = df["from_removable_media"].map(
        {"True": 1, "False": 0, True: 1, False: 0}
    ).fillna(0).astype(int)

    # Flag executable files
    df["is_executable"] = df["filename"].str.lower().str.endswith(
        (".exe", ".bat", ".ps1", ".sh", ".cmd")
    ).astype(int)

    # Flag exfiltration: writing TO removable media
    df["is_exfil_attempt"] = (
        (df["to_removable_media"] == 1) &
        (df["activity"] == "File Write")
    ).astype(int)

    log.info(f"File loaded: {len(df):,} rows | "
             f"exfil attempts: {df['is_exfil_attempt'].sum():,} | "
             f"executables: {df['is_executable'].sum():,}")
    return df


def load_users() -> pd.DataFrame:
    """
    Load users.csv — employee roster with roles and departments.
    Maps role strings to privilege tier (1-4).
    """
    log.info("Loading users.csv...")
    df = pd.read_csv(CERT_USERS_PATH)

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"])

    df["role_tier"] = df["role"].map(CERT_ROLE_TIER).fillna(0).astype(int)

    # Employment duration in days
    df["tenure_days"] = (df["end_date"] - df["start_date"]).dt.days

    log.info(f"Users loaded: {len(df):,} employees | "
             f"roles: {df['role'].unique().tolist()}")
    return df


def load_psychometric() -> pd.DataFrame:
    """
    Load psychometric.csv — Big Five personality scores (OCEAN).
    O=Openness, C=Conscientiousness, E=Extraversion,
    A=Agreeableness, N=Neuroticism.

    High N (Neuroticism) + Low C (Conscientiousness) = elevated
    insider threat risk in academic literature.
    """
    log.info("Loading psychometric.csv...")
    df = pd.read_csv(CERT_PSYCHOMETRIC_PATH)

    # Composite risk score: high N, low C
    df["psych_risk_score"] = (
        df["N"].astype(float) * 0.5 +
        (100 - df["C"].astype(float)) * 0.3 +
        (100 - df["A"].astype(float)) * 0.2
    ) / 100.0

    log.info(f"Psychometric loaded: {len(df):,} employees | "
             f"mean risk score: {df['psych_risk_score'].mean():.3f}")
    return df


def load_decoy() -> pd.DataFrame:
    """
    Load decoy_file.csv — honeypot file access events.
    ANY access to a decoy file is guaranteed malicious.
    """
    log.info("Loading decoy_file.csv...")
    df = pd.read_csv(CERT_DECOY_PATH)
    df["is_malicious"] = 1
    log.info(f"Decoy loaded: {len(df):,} guaranteed malicious events | "
             f"columns: {list(df.columns)}")
    return df


def load_ldap() -> pd.DataFrame:
    """
    Load and concatenate all monthly LDAP org chart files.
    Tracks role changes over time — sudden role escalations
    are a key insider threat signal.
    """
    log.info("Loading LDAP monthly files...")
    frames = []
    for f in sorted(CERT_LDAP_DIR.glob("*.csv")):
        month_df = pd.read_csv(f)
        month_df["month"] = f.stem   # e.g. "2010-03"
        frames.append(month_df)

    df = pd.concat(frames, ignore_index=True)
    log.info(f"LDAP loaded: {len(df):,} records across "
             f"{len(frames)} months")
    return df


def build_user_master(
    users_df: pd.DataFrame,
    psycho_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge users + psychometric into a single user master table.
    This is the reference table all other CERT loaders join against.
    """
    master = users_df.merge(
        psycho_df[["user_id", "O", "C", "E", "A", "N", "psych_risk_score"]],
        on="user_id",
        how="left",
    )
    log.info(f"User master built: {len(master):,} employees, "
             f"{len(master.columns)} columns")
    return master


if __name__ == "__main__":
    # Load all tables with row limits for fast testing
    logon_df  = load_logon(nrows=50_000)
    device_df = load_device(nrows=50_000)
    file_df   = load_file(nrows=50_000)
    users_df  = load_users()
    psycho_df = load_psychometric()
    decoy_df  = load_decoy()
    ldap_df   = load_ldap()

    master_df = build_user_master(users_df, psycho_df)

    print("\n--- CERT Dataset Summary ---")
    print(f"Logon events   : {len(logon_df):>10,}")
    print(f"Device events  : {len(device_df):>10,}")
    print(f"File events    : {len(file_df):>10,}")
    print(f"Users          : {len(users_df):>10,}")
    print(f"Decoy events   : {len(decoy_df):>10,}")
    print(f"LDAP records   : {len(ldap_df):>10,}")
    print(f"\nUser master sample:")
    print(master_df[["user_id", "role", "role_tier",
                      "tenure_days", "psych_risk_score"]].head(5).to_string())

    # Save interim
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(INTERIM_DIR / "cert_user_master.csv", index=False)
    log.info("Saved cert_user_master.csv")