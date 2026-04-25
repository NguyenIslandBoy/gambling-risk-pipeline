"""
src/ingest/database.py — Write raw DataFrames into DuckDB.

Creates one table per dataset. Idempotent: re-running drops and recreates tables.

Run:
    python src/ingest/database.py
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DB_PATH,
    TABLE_ANALYTIC,
    TABLE_DEMOGRAPHICS,
    TABLE_DAILY,
    TABLE_RG_DETAILS,
)
from src.ingest.loader import load_all


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to the project database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def write_table(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    """Drop-and-replace a table with the given DataFrame."""
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"  ✓ {table_name:<20} {count:>10,} rows written")


def build_database() -> None:
    """Load all raw files and write them into DuckDB."""
    print("=" * 55)
    print(f"Target DB: {DB_PATH}")
    print("=" * 55)

    dfs = load_all()

    print("\nWriting tables to DuckDB...")
    con = get_connection()

    write_table(con, dfs["analytic"],     TABLE_ANALYTIC)
    write_table(con, dfs["demographics"], TABLE_DEMOGRAPHICS)
    write_table(con, dfs["daily"],        TABLE_DAILY)
    write_table(con, dfs["rg_details"],   TABLE_RG_DETAILS)

    # Verify with a quick summary query
    print("\nTable summary in DB:")
    tables = con.execute(
        "SELECT table_name, estimated_size "
        "FROM duckdb_tables()"
    ).fetchall()
    for t in tables:
        print(f"  {t[0]}")

    con.close()
    print(f"\nDatabase ready at: {DB_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    build_database()