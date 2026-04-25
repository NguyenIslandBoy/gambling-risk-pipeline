"""
src/ingest/validate.py — Sanity checks on ingested data.

Checks row counts, label balance, null rates, and date ranges
against expected values from the codebook.

Run AFTER database.py:
    python src/ingest/validate.py
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DB_PATH,
    TABLE_ANALYTIC,
    TABLE_DEMOGRAPHICS,
    TABLE_DAILY,
    TABLE_RG_DETAILS,
    EXPECTED_ROWS,
    LABEL_COL,
    USER_ID_COL,
    VALID_PRODUCT_IDS,
)


def validate_all() -> bool:
    """
    Run all validation checks. Returns True if all pass.
    Prints a clear PASS / FAIL for each check so issues are easy to spot.
    """
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run src/ingest/database.py first.")
        return False

    con = duckdb.connect(str(DB_PATH), read_only=True)
    all_passed = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal all_passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
        if not passed:
            all_passed = False

    # ── 1. Row counts vs codebook ─────────────────────────────────────────────
    print("\n── Row counts ───────────────────────────────────────────")
    for table, expected in EXPECTED_ROWS.items():
        actual = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        # Allow ±5 rows tolerance (codebook notes minor exclusions)
        passed = abs(actual - expected) <= 5
        check(
            f"{table}: {actual:,} rows",
            passed,
            f"expected ~{expected:,}"
        )

    # ── 2. Label balance (analytic table) ─────────────────────────────────────
    print("\n── Label balance (analytic) ──────────────────────────────")
    result = con.execute(f"""
        SELECT {LABEL_COL}, COUNT(*) as n
        FROM {TABLE_ANALYTIC}
        GROUP BY {LABEL_COL}
        ORDER BY {LABEL_COL}
    """).fetchall()
    for label_val, count in result:
        pct = count / sum(r[1] for r in result) * 100
        print(f"  RG_case={label_val}: {count:,} ({pct:.1f}%)")
    # Should be roughly 50/50 (case-control matched design)
    counts = {r[0]: r[1] for r in result}
    ratio = counts.get(1, 0) / max(counts.get(0, 1), 1)
    check("Label balance ~50/50", 0.80 <= ratio <= 1.20, f"ratio={ratio:.2f}")

    # ── 3. Null rates on key columns ──────────────────────────────────────────
    print("\n── Null rates (analytic) ─────────────────────────────────")
    for col in [USER_ID_COL, LABEL_COL, "p1sumstake31days", "p2sumstake31days"]:
        nulls = con.execute(
            f"SELECT COUNT(*) FROM {TABLE_ANALYTIC} WHERE {col} IS NULL"
        ).fetchone()[0]
        pct = nulls / EXPECTED_ROWS[TABLE_ANALYTIC] * 100
        check(f"{col} nulls: {nulls} ({pct:.1f}%)", nulls == 0, "expected 0")

    # ── 4. User ID overlap (analytic ∩ daily aggregates) ─────────────────────
    print("\n── User ID overlap ───────────────────────────────────────")
    overlap = con.execute(f"""
        SELECT COUNT(DISTINCT a.{USER_ID_COL})
        FROM {TABLE_ANALYTIC} a
        INNER JOIN {TABLE_DAILY} d ON a.{USER_ID_COL} = d.{USER_ID_COL}
    """).fetchone()[0]
    check(
        f"analytic ∩ daily: {overlap:,} users",
        overlap >= 4000,
        "expected ~4,056"
    )

    # ── 5. RG details covers only RG cases ────────────────────────────────────
    print("\n── RG details integrity ──────────────────────────────────")
    rg_count = con.execute(
        f"SELECT COUNT(DISTINCT {USER_ID_COL}) FROM {TABLE_RG_DETAILS}"
    ).fetchone()[0]
    check(f"RG details unique users: {rg_count:,}", 2060 <= rg_count <= 2068,
          "expected ~2,068")

    # ── 6. Daily aggregates: valid product IDs ────────────────────────────────
    print("\n── Product types (daily aggregates) ──────────────────────")
    products = con.execute(
        f"SELECT DISTINCT ProductType FROM {TABLE_DAILY} ORDER BY ProductType"
    ).fetchall()
    product_ids = {r[0] for r in products}
    valid = {1, 2, 4, 8, 17}
    all_products = sorted(product_ids)
    trusted = sorted(product_ids & valid)
    print(f"  All product IDs found:    {all_products}")
    print(f"  Trusted (wager/hold ok):  {trusted}")
    check("Trusted product IDs present", valid.issubset(product_ids))

    # ── 7. Date range (daily aggregates) ─────────────────────────────────────
    print("\n── Date range (daily aggregates) ─────────────────────────")
    date_range = con.execute(
        f"SELECT MIN(Date), MAX(Date) FROM {TABLE_DAILY}"
    ).fetchone()
    print(f"  Min date: {date_range[0]}  Max date: {date_range[1]}")
    check("Date range sensible", date_range[0] is not None)

    # ── Summary ───────────────────────────────────────────────────────────────
    con.close()
    print("\n" + "=" * 55)
    if all_passed:
        print("ALL CHECKS PASSED — ready for feature engineering.")
    else:
        print("SOME CHECKS FAILED — review output above before proceeding.")
    print("=" * 55)

    return all_passed


if __name__ == "__main__":
    validate_all()