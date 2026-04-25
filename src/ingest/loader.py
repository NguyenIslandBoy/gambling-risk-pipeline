"""
src/ingest/loader.py — Load raw .dat files into pandas DataFrames.

Each function:
  - auto-detects the delimiter (tab or space-delimited .dat files vary)
  - normalises UserID column name (files use 'UserID' or 'USERID' inconsistently)
  - applies light type casting based on the codebook
  - returns a clean DataFrame ready to write to DuckDB

Run this file directly to test all loads:
    python src/ingest/loader.py
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    RAW_ANALYTIC,
    RAW_DEMOGRAPHICS,
    RAW_DAILY,
    RAW_RG_DETAILS,
    LABEL_COL,
    USER_ID_COL,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_delimiter(filepath: Path) -> str:
    """Peek at first line to determine delimiter (tab vs whitespace)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    if "\t" in first_line:
        return "\t"
    return r"\s+"  # fallback: any whitespace


def _load_dat(filepath: Path, **kwargs) -> pd.DataFrame:
    """Generic .dat loader with delimiter detection and column normalisation."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"\nFile not found: {filepath}"
            f"\nMake sure you placed the .dat files in: {filepath.parent}"
        )
    sep = _detect_delimiter(filepath)
    df = pd.read_csv(
        filepath,
        sep=sep,
        engine="python",       # required for regex sep
        encoding="utf-8",
        encoding_errors="replace",
        **kwargs,
    )
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Normalise UserID casing — raw2/raw3 use 'UserID', raw1/analytic use 'USERID'
    df.columns = [
        "USERID" if c.upper() == "USERID" else c
        for c in df.columns
    ]
    return df


def _parse_dates(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Parse date columns using format='mixed' to suppress dateutil warnings."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    return df


# ── Individual loaders ────────────────────────────────────────────────────────

def load_analytic() -> pd.DataFrame:
    """
    Braverman 2013 analytic dataset.
    4,056 rows. Pre-computed 31-day features + RG_case label.
    This is our primary modelling table.
    """
    df = _load_dat(RAW_ANALYTIC)

    df[USER_ID_COL] = pd.to_numeric(df[USER_ID_COL], errors="coerce")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype("Int8")

    df = _parse_dates(df, [
        "first_active_product1_31days", "first_active_product2_31days",
        "first_active_product4_31days", "first_active_games_31days",
        "first_active_poker_31days",
    ])

    print(f"[analytic]      shape={df.shape}  RG_cases={df[LABEL_COL].sum()}  "
          f"controls={(df[LABEL_COL] == 0).sum()}")
    return df


def load_demographics() -> pd.DataFrame:
    """
    Raw Demographics (shared across both studies).
    4,134 rows. User ID, country, gender, year of birth, registration date.
    """
    df = _load_dat(RAW_DEMOGRAPHICS)

    df[USER_ID_COL] = pd.to_numeric(df[USER_ID_COL], errors="coerce")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype("Int8")

    df = _parse_dates(df, ["Registration_date", "First_Deposit_Date"])

    if "YearofBirth" in df.columns:
        df["YearofBirth"] = pd.to_numeric(df["YearofBirth"], errors="coerce")

    print(f"[demographics]  shape={df.shape}  nulls_country="
          f"{df['CountryName'].isna().sum() if 'CountryName' in df.columns else 'N/A'}")
    return df


def load_daily_aggregates() -> pd.DataFrame:
    """
    Raw Daily Aggregates — the core time-series table.
    ~981,782 rows. One row per user x date x product.
    Columns: USERID, Date, ProductType, Turnover, Hold, NumberofBets.
    """
    df = _load_dat(RAW_DAILY)

    df[USER_ID_COL] = pd.to_numeric(df[USER_ID_COL], errors="coerce")
    df["ProductType"] = pd.to_numeric(df["ProductType"], errors="coerce")
    df["Turnover"] = pd.to_numeric(df["Turnover"], errors="coerce")
    df["Hold"] = pd.to_numeric(df["Hold"], errors="coerce")
    df["NumberofBets"] = pd.to_numeric(df["NumberofBets"], errors="coerce")

    df = _parse_dates(df, ["Date"])

    print(f"[daily_aggs]    shape={df.shape}  "
          f"date_range={df['Date'].min().date()} to {df['Date'].max().date()}  "
          f"unique_users={df[USER_ID_COL].nunique()}")
    return df


def load_rg_details() -> pd.DataFrame:
    """
    Responsible Gambling event details (Gray 2012, Raw Dataset III).
    2,068 rows — RG cases only.
    Columns: USERID, RGsumevents, RGFirst_Date, RGLast_date,
             Event_type_first, Interventiontype_first.
    """
    df = _load_dat(RAW_RG_DETAILS)

    df[USER_ID_COL] = pd.to_numeric(df[USER_ID_COL], errors="coerce")
    df["RGsumevents"] = pd.to_numeric(df["RGsumevents"], errors="coerce")
    df["Event_type_first"] = pd.to_numeric(df["Event_type_first"], errors="coerce")
    df["Interventiontype_first"] = pd.to_numeric(df["Interventiontype_first"], errors="coerce")

    df = _parse_dates(df, ["RGFirst_Date", "RGLast_date"])

    print(f"[rg_details]    shape={df.shape}  "
          f"event_types={sorted(df['Event_type_first'].dropna().unique().tolist())}")
    return df


# ── Quick smoke test ──────────────────────────────────────────────────────────

def load_all() -> dict:
    """Load all four files. Returns dict of DataFrames."""
    print("=" * 55)
    print("Loading raw datasets...")
    print("=" * 55)
    dfs = {
        "analytic":     load_analytic(),
        "demographics": load_demographics(),
        "daily":        load_daily_aggregates(),
        "rg_details":   load_rg_details(),
    }
    print("=" * 55)
    print("All files loaded successfully.")
    return dfs


if __name__ == "__main__":
    dfs = load_all()

    # Quick column inspection
    for name, df in dfs.items():
        print(f"\n-- {name} columns --")
        print(df.dtypes.to_string())