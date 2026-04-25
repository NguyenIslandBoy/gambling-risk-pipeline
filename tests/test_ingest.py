"""
tests/test_ingest.py — Unit tests for ingestion layer.
Tests use the live DuckDB database (integration-style).
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest
import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DB_PATH,
    TABLE_ANALYTIC, TABLE_DEMOGRAPHICS, TABLE_DAILY, TABLE_RG_DETAILS,
    LABEL_COL, USER_ID_COL, VALID_PRODUCT_IDS,
)


@pytest.fixture(scope="module")
def con():
    if not DB_PATH.exists():
        pytest.skip("Database not found — run src/ingest/database.py first")
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    yield conn
    conn.close()


# ── Row counts ────────────────────────────────────────────────────────────────

def test_analytic_row_count(con):
    n = con.execute(f"SELECT COUNT(*) FROM {TABLE_ANALYTIC}").fetchone()[0]
    assert 4050 <= n <= 4060, f"Expected ~4056 rows, got {n}"


def test_demographics_row_count(con):
    n = con.execute(f"SELECT COUNT(*) FROM {TABLE_DEMOGRAPHICS}").fetchone()[0]
    assert 4130 <= n <= 4140, f"Expected ~4134 rows, got {n}"


def test_daily_row_count(con):
    n = con.execute(f"SELECT COUNT(*) FROM {TABLE_DAILY}").fetchone()[0]
    assert n > 900_000, f"Expected >900k rows, got {n}"


def test_rg_details_row_count(con):
    n = con.execute(f"SELECT COUNT(*) FROM {TABLE_RG_DETAILS}").fetchone()[0]
    assert 2060 <= n <= 2070, f"Expected ~2068 rows, got {n}"


# ── Label integrity ───────────────────────────────────────────────────────────

def test_label_is_binary(con):
    vals = {
        r[0] for r in
        con.execute(f"SELECT DISTINCT {LABEL_COL} FROM {TABLE_ANALYTIC}").fetchall()
    }
    assert vals == {0, 1}, f"Expected {{0, 1}}, got {vals}"


def test_label_balance(con):
    result = con.execute(f"""
        SELECT {LABEL_COL}, COUNT(*) FROM {TABLE_ANALYTIC}
        GROUP BY {LABEL_COL}
    """).fetchall()
    counts = {r[0]: r[1] for r in result}
    ratio = counts[1] / counts[0]
    assert 0.90 <= ratio <= 1.10, f"Label ratio too skewed: {ratio:.2f}"


def test_rg_details_all_cases(con):
    """RG details table should contain only RG cases (label=1)."""
    rg_ids = con.execute(
        f"SELECT DISTINCT {USER_ID_COL} FROM {TABLE_RG_DETAILS}"
    ).df()[USER_ID_COL]
    analytic = con.execute(
        f"SELECT {USER_ID_COL}, {LABEL_COL} FROM {TABLE_ANALYTIC}"
    ).df()
    merged = analytic[analytic[USER_ID_COL].isin(rg_ids)]
    non_cases = (merged[LABEL_COL] == 0).sum()
    assert non_cases == 0, f"{non_cases} controls found in RG details table"


# ── Key columns present ───────────────────────────────────────────────────────

def test_daily_required_columns(con):
    cols = {r[0] for r in con.execute(f"DESCRIBE {TABLE_DAILY}").fetchall()}
    required = {"USERID", "Date", "ProductType", "Turnover", "Hold", "NumberofBets"}
    missing = required - cols
    assert not missing, f"Missing columns: {missing}"


def test_demographics_required_columns(con):
    cols = {r[0] for r in con.execute(f"DESCRIBE {TABLE_DEMOGRAPHICS}").fetchall()}
    required = {"USERID", "RG_case", "Gender", "YearofBirth", "First_Deposit_Date"}
    missing = required - cols
    assert not missing, f"Missing columns: {missing}"


# ── User ID integrity ─────────────────────────────────────────────────────────

def test_no_null_user_ids(con):
    for table in [TABLE_ANALYTIC, TABLE_DEMOGRAPHICS, TABLE_DAILY]:
        nulls = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {USER_ID_COL} IS NULL"
        ).fetchone()[0]
        assert nulls == 0, f"Null USERIDs in {table}: {nulls}"


def test_analytic_users_in_daily(con):
    overlap = con.execute(f"""
        SELECT COUNT(DISTINCT a.{USER_ID_COL})
        FROM {TABLE_ANALYTIC} a
        INNER JOIN {TABLE_DAILY} d ON a.{USER_ID_COL} = d.{USER_ID_COL}
    """).fetchone()[0]
    assert overlap >= 4000, f"Only {overlap} analytic users found in daily aggregates"


# ── Product types ─────────────────────────────────────────────────────────────

def test_trusted_product_ids_present(con):
    found = {
        r[0] for r in
        con.execute(f"SELECT DISTINCT ProductType FROM {TABLE_DAILY}").fetchall()
    }
    missing = VALID_PRODUCT_IDS - found
    assert not missing, f"Trusted product IDs missing from daily: {missing}"