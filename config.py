"""
config.py — Central configuration for the gambling risk pipeline.
All paths, filenames, and constants live here. Nothing is hardcoded elsewhere.
"""

from pathlib import Path

# ── Project root (this file's directory) ─────────────────────────────────────
ROOT = Path(__file__).parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ── Raw input files (place your .dat files here) ──────────────────────────────
# Study B (Braverman 2013) — primary source
RAW_ANALYTIC    = DATA_RAW / "AnalyticDataset_Braverman_LaPlante_PAB_2013.dat"
RAW_DEMOGRAPHICS = DATA_RAW / "Raw Datset I.Demographics_Braverman_LaPlante_PAB_2013.dat"
RAW_DAILY       = DATA_RAW / "Raw Datset II.Daily aggregates_Braverman_LaPlante_PAB_2013.dat"

# Study A (Gray 2012) — RG event details
RAW_RG_DETAILS  = DATA_RAW / "Raw Datset III.Responsible gambling details_Gray_LaPlante_PAB_2012.dat"

# ── DuckDB database ───────────────────────────────────────────────────────────
DB_PATH = DATA_PROCESSED / "gambling_risk.duckdb"

# ── Table names in DuckDB ─────────────────────────────────────────────────────
TABLE_ANALYTIC     = "analytic"
TABLE_DEMOGRAPHICS = "demographics"
TABLE_DAILY        = "daily_aggregates"
TABLE_RG_DETAILS   = "rg_details"
TABLE_FEATURES     = "features"

# ── Expected row counts from codebook (used in validation) ────────────────────
EXPECTED_ROWS = {
    TABLE_ANALYTIC:     4056,
    TABLE_DEMOGRAPHICS: 4134,
    TABLE_DAILY:        981782,
    TABLE_RG_DETAILS:   2068,
}

# ── Target label ──────────────────────────────────────────────────────────────
LABEL_COL = "RG_case"     # 1 = triggered RG alert, 0 = control
USER_ID_COL = "USERID"

# ── Product type IDs we trust for wager/hold data (from codebook Appendix 1) ──
# Products 1, 2, 4, 8, 17 only — all others have invalid turnover/hold
VALID_PRODUCT_IDS = {1, 2, 4, 8, 17}

PRODUCT_NAMES = {
    1:  "fixed_odds",
    2:  "live_action",
    4:  "casino_boss",
    8:  "casino_chartwell",
    17: "mobile_casino",
}

# Group casino products together for feature engineering
PRODUCT_GROUPS = {
    "fixed_odds":  [1],
    "live_action": [2],
    "casino":      [4, 8, 17],
}

# ── First-month window (Braverman study definition) ───────────────────────────
FIRST_MONTH_DAYS = 31

# ── Train/validation split (already defined in analytic dataset) ──────────────
VALIDATION_SET_COL = "ValidationSet"  # 0 = train, 1 = validation

# ── Model output ─────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "src" / "models"
MODEL_PATH = MODELS_DIR / "lgbm_risk_model.pkl"