import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    CERT_LOGON_PATH, CERT_DEVICE_PATH, CERT_FILE_PATH,
    CERT_HTTP_PATH, CERT_EMAIL_PATH,
    CERT_USERS_PATH, CERT_PSYCHOMETRIC_PATH,
    CERT_DECOY_PATH, INTERIM_DIR,
)
from utils.logger import get_logger

log = get_logger("cert_cleaner")

# Sample sizes for large files
HTTP_SAMPLE    = 500_000
EMAIL_SAMPLE   = 200_000
DATE_FMT       = "%m/%d/%Y %H:%M:%S"


def clean_logon(nrows: int = None) -> pd.DataFrame:
    """Load and clean full logon file."""
    log.info("Loading logon.csv...")
    df = pd.read_csv(CERT_LOGON_PATH, nrows=nrows)
    df["date"]          = pd.to_datetime(df["date"], format=DATE_FMT)
    df["hour"]          = df["date"].dt.hour
    df["day_of_week"]   = df["date"].dt.dayofweek
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_after_hours"]= ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)
    df["is_logon"]      = (df["activity"] == "Logon").astype(int)
    df["date_only"]     = df["date"].dt.date
    df = df.dropna(subset=["user", "date"])
    log.info(f"Logon cleaned: {len(df):,} rows")
    return df


def clean_device(nrows: int = None) -> pd.DataFrame:
    """Load and clean device file."""
    log.info("Loading device.csv...")
    df = pd.read_csv(CERT_DEVICE_PATH, nrows=nrows)
    df["date"]          = pd.to_datetime(df["date"], format=DATE_FMT)
    df["hour"]          = df["date"].dt.hour
    df["is_after_hours"]= ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)
    df["is_weekend"]    = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_connect"]    = (df["activity"] == "Connect").astype(int)
    df["date_only"]     = df["date"].dt.date
    df = df.dropna(subset=["user", "date"])
    log.info(f"Device cleaned: {len(df):,} rows")
    return df


def clean_file(nrows: int = None) -> pd.DataFrame:
    """Load and clean file activity."""
    log.info("Loading file.csv...")
    df = pd.read_csv(
        CERT_FILE_PATH, nrows=nrows,
        usecols=["id", "date", "user", "pc",
                 "filename", "activity",
                 "to_removable_media", "from_removable_media"]
    )
    df["date"]          = pd.to_datetime(df["date"], format=DATE_FMT)
    df["hour"]          = df["date"].dt.hour
    df["is_after_hours"]= ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)
    df["date_only"]     = df["date"].dt.date

    for col in ["to_removable_media", "from_removable_media"]:
        df[col] = df[col].map(
            {"True": 1, "False": 0, True: 1, False: 0}
        ).fillna(0).astype(int)

    df["is_executable"]   = df["filename"].str.lower().str.endswith(
        (".exe", ".bat", ".ps1", ".sh", ".cmd")
    ).astype(int)
    df["is_exfil_attempt"]= (
        (df["to_removable_media"] == 1) &
        (df["activity"] == "File Write")
    ).astype(int)
    df = df.dropna(subset=["user", "date"])
    log.info(f"File cleaned: {len(df):,} rows | "
             f"exfil: {df['is_exfil_attempt'].sum():,}")
    return df


def clean_http(sample_n: int = HTTP_SAMPLE) -> pd.DataFrame:
    """
    Sample and clean http.csv.
    Extracts domain from URL as key signal.
    Flags known cloud storage / file sharing domains.
    """
    log.info(f"Sampling http.csv ({sample_n:,} rows)...")

    # Use chunked reading to sample efficiently
    chunks  = []
    chunk_size = 50_000
    total   = 0
    for chunk in pd.read_csv(
        CERT_HTTP_PATH,
        chunksize = chunk_size,
        usecols   = ["id", "date", "user", "pc", "url", "activity"],
        on_bad_lines = "skip",
    ):
        chunks.append(chunk)
        total += len(chunk)
        if total >= sample_n:
            break

    df = pd.concat(chunks, ignore_index=True).head(sample_n)
    df["date"]      = pd.to_datetime(df["date"], format=DATE_FMT, errors="coerce")
    df              = df.dropna(subset=["date", "user"])
    df["hour"]      = df["date"].dt.hour
    df["is_after_hours"] = ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)
    df["date_only"] = df["date"].dt.date

    # Extract domain
    df["domain"] = df["url"].str.extract(r"https?://([^/]+)")[0].str.lower()

    # Flag suspicious domains — cloud storage, paste sites, job boards
    SUSPICIOUS_DOMAINS = {
        "dropbox.com", "drive.google.com", "wetransfer.com",
        "pastebin.com", "mediafire.com", "4shared.com",
        "linkedin.com", "monster.com", "indeed.com",
        "glassdoor.com", "careerbuilder.com",
    }
    df["is_suspicious_domain"] = df["domain"].isin(SUSPICIOUS_DOMAINS).astype(int)
    df["is_job_site"] = df["domain"].isin(
        {"linkedin.com", "monster.com", "indeed.com",
         "glassdoor.com", "careerbuilder.com"}
    ).astype(int)
    df["is_cloud_storage"] = df["domain"].isin(
        {"dropbox.com", "drive.google.com", "wetransfer.com",
         "mediafire.com", "4shared.com"}
    ).astype(int)

    log.info(f"HTTP cleaned: {len(df):,} rows | "
             f"suspicious: {df['is_suspicious_domain'].sum():,}")
    return df


def clean_email(sample_n: int = EMAIL_SAMPLE) -> pd.DataFrame:
    """
    Sample and clean email.csv.
    Flags external recipients, large emails, and high attachment counts.
    """
    log.info(f"Sampling email.csv ({sample_n:,} rows)...")

    chunks     = []
    chunk_size = 50_000
    total      = 0
    for chunk in pd.read_csv(
        CERT_EMAIL_PATH,
        chunksize    = chunk_size,
        usecols      = ["id", "date", "user", "pc",
                        "to", "from", "activity", "size", "attachments"],
        on_bad_lines = "skip",
    ):
        chunks.append(chunk)
        total += len(chunk)
        if total >= sample_n:
            break

    df = pd.concat(chunks, ignore_index=True).head(sample_n)
    df["date"]       = pd.to_datetime(df["date"], format=DATE_FMT, errors="coerce")
    df               = df.dropna(subset=["date", "user"])
    df["hour"]       = df["date"].dt.hour
    df["is_after_hours"] = ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)
    df["date_only"]  = df["date"].dt.date
    df["size"]       = pd.to_numeric(df["size"], errors="coerce").fillna(0)
    df["attachments"]= pd.to_numeric(df["attachments"], errors="coerce").fillna(0)

    # External email = recipient not @dtaa.com
    df["is_external"] = (~df["to"].str.contains("@dtaa.com", na=False)).astype(int)
    df["is_large"]    = (df["size"] > 100_000).astype(int)
    df["has_attachment"] = (df["attachments"] > 0).astype(int)

    # High risk = external + large + attachment
    df["is_high_risk_email"] = (
        (df["is_external"] == 1) &
        (df["has_attachment"] == 1) &
        (df["is_large"] == 1)
    ).astype(int)

    log.info(f"Email cleaned: {len(df):,} rows | "
             f"high risk: {df['is_high_risk_email'].sum():,}")
    return df


def run_cert_cleaner(save: bool = True) -> dict:
    """
    Master function — clean all CERT tables and save to interim.
    Returns dict of cleaned DataFrames.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "logon" : clean_logon(),
        "device": clean_device(),
        "file"  : clean_file(),
        "http"  : clean_http(),
        "email" : clean_email(),
    }

    if save:
        for name, df in tables.items():
            out = INTERIM_DIR / f"cert_{name}_clean.csv"
            df.to_csv(out, index=False)
            log.info(f"Saved -> {out}")

    log.info("\nCERT Cleaner Summary:")
    for name, df in tables.items():
        log.info(f"  {name:<8} : {len(df):>9,} rows, "
                 f"{len(df.columns)} columns")

    return tables


if __name__ == "__main__":
    tables = run_cert_cleaner(save=True)
    print("\nDone. Interim files saved to data/interim/")