"""
src/features/engineer.py — Feature engineering for the gambling risk pipeline.

Two feature sets:
  A) From the analytic dataset (already aggregated over 31 days) — fast, pre-computed
  B) From raw daily aggregates (computed from scratch) — richer, time-series derived

We build BOTH and join them. This lets us:
  - use set A as a quick baseline
  - use set A+B as the full model
  - compare to validate our raw computations are correct

Output: a single 'features' table written to DuckDB, ready for modelling.

Run:
    python src/features/engineer.py
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DB_PATH,
    TABLE_ANALYTIC,
    TABLE_DAILY,
    TABLE_DEMOGRAPHICS,
    TABLE_RG_DETAILS,
    TABLE_FEATURES,
    LABEL_COL,
    USER_ID_COL,
    VALID_PRODUCT_IDS,
    PRODUCT_GROUPS,
    FIRST_MONTH_DAYS,
    VALIDATION_SET_COL,
)


# ── Connection ────────────────────────────────────────────────────────────────

def get_con() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}\n"
            "Run src/ingest/database.py first."
        )
    return duckdb.connect(str(DB_PATH))


# ── Feature Set A: from analytic dataset ─────────────────────────────────────

def build_features_analytic(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Pull the most informative pre-computed columns from the analytic dataset.
    These are 31-day aggregates already computed by the original researchers.
    """
    print("  Building feature set A (analytic dataset)...")

    df = con.execute(f"""
        SELECT
            {USER_ID_COL},
            {LABEL_COL},
            {VALIDATION_SET_COL},

            -- Fixed odds (product 1)
            p1sumstake31days        AS fo_total_stakes,
            p1sumbets31days         AS fo_total_bets,
            p1totalactivedays_31days AS fo_active_days,
            p1SDStakes31days        AS fo_sd_stakes,
            p1SDBets31day           AS fo_sd_bets,
            p1avgbetsize            AS fo_avg_bet_size,
            p1avgbetsperactiveday   AS fo_bets_per_active_day,

            -- Live action (product 2)
            p2sumstake31days        AS la_total_stakes,
            p2sumbets31days         AS la_total_bets,
            p2totalactivedays_31days AS la_active_days,
            p2SDStakes31days        AS la_sd_stakes,
            p2SDBets31days          AS la_sd_bets,
            p2avgbetsize            AS la_avg_bet_size,
            p2avgbetsperactiveday   AS la_bets_per_active_day,

            -- Casino (products 4, 8, 17 combined)
            pcsumstake31days        AS ca_total_stakes,
            pcsumbets31days         AS ca_total_bets,
            casino_totalactivedays_31days AS ca_active_days,
            pcSDStakes31days        AS ca_sd_stakes,
            pcSDBets31days          AS ca_sd_bets,
            pcavgbetsize            AS ca_avg_bet_size,
            pcavgbetsperactiveday   AS ca_bets_per_active_day,

            -- Cross-game behaviour
            NumberofGames31days     AS n_games_played,
            firstGamePlayed         AS first_game_type,
            mostFrequentGame        AS most_frequent_game,
            playedFO                AS played_fixed_odds,
            playedLA                AS played_live_action,
            playedPO                AS played_poker,
            playedCA                AS played_casino,
            playedGA                AS played_other_games,

            -- Overall activity
            totalactivedays_31days  AS total_active_days,
            wk1frequency            AS wk1_freq,
            wk2frequency            AS wk2_freq,
            wk3frequency            AS wk3_freq,
            wk4frequency            AS wk4_freq,

            -- Weekly trajectories (1=decreasing, 2=stable, 3=increasing)
            weekfrequencytraj       AS freq_trajectory,
            FOBetsWeeklyTraj        AS fo_bets_trajectory,
            FOstakesWeeklyTraj      AS fo_stakes_trajectory,
            LABetsWeeklyTraj        AS la_bets_trajectory,
            LAstakesWeeklyTraj      AS la_stakes_trajectory,
            CasinoBetsWeeklyTraj    AS ca_bets_trajectory,
            CasinostakesWeeklyTraj  AS ca_stakes_trajectory,

            -- Weekend vs weekday ratios
            p1wkendsumbetssratio    AS fo_weekend_bets_ratio,
            p2wkendsumbetsratio     AS la_weekend_bets_ratio,
            pcwkendsumbetsratio     AS ca_weekend_bets_ratio,
            p1wkendsumstakesratio   AS fo_weekend_stakes_ratio,
            p2wkendsumstakesratio   AS la_weekend_stakes_ratio,
            pcwkendsumstakesratio   AS ca_weekend_stakes_ratio,

            -- Risk groups — exclude from model input, keep for analysis only
            RiskGroup1,
            RiskGroup2,
            RiskGroupCombined,

            period_tillDeposit      AS days_registration_to_deposit

        FROM {TABLE_ANALYTIC}
    """).df()

    print(f"    shape={df.shape}")
    return df


# ── Feature Set B: engineered from raw daily aggregates ───────────────────────

def build_features_raw(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Engineer features directly from the raw daily aggregates.
    Restricted to the first 31 days from each user's first deposit date.
    Only trusted product IDs (1, 2, 4, 8, 17) used for Turnover/Hold.
    """
    print("  Building feature set B (raw daily aggregates)...")

    deposit_dates = con.execute(f"""
        SELECT USERID, First_Deposit_Date
        FROM {TABLE_DEMOGRAPHICS}
    """).df()

    daily = con.execute(f"""
        SELECT USERID, Date, ProductType, Turnover, Hold, NumberofBets
        FROM {TABLE_DAILY}
        WHERE ProductType IN ({','.join(str(p) for p in VALID_PRODUCT_IDS)})
    """).df()

    daily = daily.merge(deposit_dates, on="USERID", how="left")
    daily["days_since_deposit"] = (daily["Date"] - daily["First_Deposit_Date"]).dt.days
    daily_31 = daily[
        (daily["days_since_deposit"] >= 0) &
        (daily["days_since_deposit"] < FIRST_MONTH_DAYS)
    ].copy()

    print(f"    daily rows in 31-day window: {len(daily_31):,}  "
          f"unique users: {daily_31['USERID'].nunique():,}")

    grp = daily_31.groupby("USERID")

    agg = pd.DataFrame(index=grp.groups.keys())
    agg.index.name = "USERID"

    # Volume
    agg["raw_total_stakes"]      = grp["Turnover"].sum()
    agg["raw_total_bets"]        = grp["NumberofBets"].sum()
    agg["raw_total_active_days"] = grp["Date"].nunique()
    agg["raw_net_position"]      = -grp["Hold"].sum()

    # Intensity
    agg["raw_stakes_per_day"] = agg["raw_total_stakes"] / agg["raw_total_active_days"].clip(lower=1)
    agg["raw_bets_per_day"]   = agg["raw_total_bets"]   / agg["raw_total_active_days"].clip(lower=1)

    # Variability — CV of daily stakes
    daily_stakes_by_user = daily_31.groupby(["USERID", "Date"])["Turnover"].sum()
    agg["raw_cv_stakes"] = (
        daily_stakes_by_user.groupby("USERID").std() /
        daily_stakes_by_user.groupby("USERID").mean().clip(lower=1e-6)
    )

    # Number of distinct products
    agg["raw_n_products"] = grp["ProductType"].nunique()

    # ── Missingness flags (critical — tells model which zeros are real vs imputed) ──
    # A user needs at least 2 active days to compute slope or loss chasing
    active_days_count = grp["Date"].nunique()
    agg["had_enough_days_for_ts"] = (active_days_count >= 2).astype(int)

    # ── Stake escalation ──────────────────────────────────────────────────────
    def stake_slope(group: pd.DataFrame) -> float:
        daily_totals = (
            group.groupby("days_since_deposit")["Turnover"]
            .sum().reset_index()
        )
        if len(daily_totals) < 2:
            return np.nan   # not enough data — flag separately
        x = daily_totals["days_since_deposit"].values
        y = daily_totals["Turnover"].values
        x_mean, y_mean = x.mean(), y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    print("    Computing stake escalation slopes...")
    slopes = daily_31.groupby("USERID").apply(stake_slope)
    agg["raw_stake_escalation"] = slopes
    # Flag: 1 = slope was computable, 0 = insufficient data
    agg["raw_escalation_valid"] = agg["raw_stake_escalation"].notna().astype(int)
    # Normalise slope by mean stakes so it's scale-independent
    mean_stakes = agg["raw_stakes_per_day"].clip(lower=1e-6)
    agg["raw_stake_escalation_norm"] = agg["raw_stake_escalation"] / mean_stakes
    agg["raw_stake_escalation"] = agg["raw_stake_escalation"].fillna(0)
    agg["raw_stake_escalation_norm"] = agg["raw_stake_escalation_norm"].fillna(0)

    # ── Loss chasing index ────────────────────────────────────────────────────
    def loss_chasing(group: pd.DataFrame) -> float:
        daily_pnl = (
            group.groupby("days_since_deposit")
            .agg(hold=("Hold", "sum"), bets=("NumberofBets", "sum"))
            .reset_index().sort_values("days_since_deposit")
        )
        if len(daily_pnl) < 2:
            return np.nan
        daily_pnl["prev_hold"] = daily_pnl["hold"].shift(1)
        after_loss = daily_pnl.loc[daily_pnl["prev_hold"] > 0, "bets"].mean()
        after_win  = daily_pnl.loc[daily_pnl["prev_hold"] < 0, "bets"].mean()
        if pd.isna(after_win) or after_win == 0:
            return np.nan
        return float(after_loss / after_win)

    print("    Computing loss chasing index...")
    lc = daily_31.groupby("USERID").apply(loss_chasing)
    agg["raw_loss_chasing"] = lc
    agg["raw_loss_chasing_valid"] = agg["raw_loss_chasing"].notna().astype(int)
    agg["raw_loss_chasing"] = agg["raw_loss_chasing"].fillna(1.0)  # 1.0 = neutral (no chasing)

    agg = agg.reset_index()
    print(f"    shape={agg.shape}")
    return agg


# ── Combine and finalise ──────────────────────────────────────────────────────

def build_all_features() -> pd.DataFrame:
    con = get_con()

    feat_a = build_features_analytic(con)
    feat_b = build_features_raw(con)

    print("  Joining feature sets...")
    features = feat_a.merge(feat_b, on=USER_ID_COL, how="left")

    # ── Derived interaction features ──────────────────────────────────────────

    # Total cross-product volume
    features["total_stakes_31d"] = (
        features["fo_total_stakes"].fillna(0) +
        features["la_total_stakes"].fillna(0) +
        features["ca_total_stakes"].fillna(0)
    )
    features["total_bets_31d"] = (
        features["fo_total_bets"].fillna(0) +
        features["la_total_bets"].fillna(0) +
        features["ca_total_bets"].fillna(0)
    )
    features["overall_avg_bet_size"] = (
        features["total_stakes_31d"] /
        features["total_bets_31d"].clip(lower=1)
    )
    features["activity_ratio"] = features["total_active_days"] / FIRST_MONTH_DAYS

    # Escalation × loss chasing interaction — both high = strong risk signal
    features["escalation_x_chasing"] = (
        features["raw_stake_escalation_norm"].clip(lower=0) *
        (features["raw_loss_chasing"] - 1).clip(lower=0)
    )

    # Casino dominance — what fraction of bets are casino
    features["casino_bet_share"] = (
        features["ca_total_bets"].fillna(0) /
        features["total_bets_31d"].clip(lower=1)
    )

    # Live action dominance
    features["la_bet_share"] = (
        features["la_total_bets"].fillna(0) /
        features["total_bets_31d"].clip(lower=1)
    )

    # Volatility relative to volume — high CV + high stakes = erratic big bettor
    features["vol_x_stakes"] = (
        features["raw_cv_stakes"].fillna(0) *
        np.log1p(features["raw_total_stakes"].fillna(0))
    )

    # Net loss rate — how much of stakes is lost (only meaningful if stakes > 0)
    features["net_loss_rate"] = (
        features["raw_net_position"].fillna(0) /
        features["raw_total_stakes"].clip(lower=1e-6).fillna(1e-6)
    ).clip(lower=-5, upper=5)  # cap extreme values from tiny stakes

    # Any increasing trajectory
    traj_cols = [c for c in features.columns if "trajectory" in c]
    features["any_increasing_traj"] = (
        features[traj_cols].eq(3).any(axis=1).astype(int)
    )

    # ── Imputation ────────────────────────────────────────────────────────────
    # Activity/count: null = no activity = 0
    zero_impute_patterns = [
        "stakes", "bets", "active_days", "n_games", "played_",
        "raw_total", "raw_n_", "raw_net_position", "raw_cv",
        "raw_stakes_per_day", "raw_bets_per_day",
        "casino_bet_share", "la_bet_share", "vol_x_stakes",
        "escalation_x_chasing",
    ]
    for col in features.columns:
        if any(p in col for p in zero_impute_patterns):
            if pd.api.types.is_numeric_dtype(features[col]):
                features[col] = features[col].fillna(0)

    # Ratio/variability: null = insufficient data → median
    median_impute_patterns = ["ratio", "sd", "avg_bet", "net_loss_rate"]
    for col in features.columns:
        if any(p in col for p in median_impute_patterns):
            if pd.api.types.is_numeric_dtype(features[col]):
                med = features[col].median()
                features[col] = features[col].fillna(med)

    # Validity flags: users with no raw data get 0 (not computable)
    for col in ["raw_escalation_valid", "raw_loss_chasing_valid",
                "had_enough_days_for_ts", "raw_n_products",
                "raw_stake_escalation_norm"]:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # Explicit null fills for time-series features that slipped through
    # raw_stake_escalation: 0 = no trend (neutral)
    # raw_loss_chasing: 1.0 = neutral (equal bets after win/loss)
    if "raw_stake_escalation" in features.columns:
        features["raw_stake_escalation"] = features["raw_stake_escalation"].fillna(0)
    if "raw_loss_chasing" in features.columns:
        features["raw_loss_chasing"] = features["raw_loss_chasing"].fillna(1.0)

    # ── Write to DuckDB ───────────────────────────────────────────────────────
    print(f"  Writing {TABLE_FEATURES} to DuckDB...")
    con.execute(f"DROP TABLE IF EXISTS {TABLE_FEATURES}")
    con.execute(f"CREATE TABLE {TABLE_FEATURES} AS SELECT * FROM features")
    count = con.execute(f"SELECT COUNT(*) FROM {TABLE_FEATURES}").fetchone()[0]
    print(f"  Done. {count:,} rows written to '{TABLE_FEATURES}'.")

    con.close()
    return features


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Feature engineering")
    print("=" * 55)

    features = build_all_features()

    print("\n-- Feature summary --")
    print(f"Total features: {features.shape[1] - 3}")
    print(f"Total users:    {features.shape[0]:,}")
    print(f"RG cases:       {features[LABEL_COL].sum():,}")
    print(f"Controls:       {(features[LABEL_COL] == 0).sum():,}")

    print("\n-- Null check (should be 0 for all) --")
    null_counts = features.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("  No nulls remaining.")
    else:
        print(null_counts.to_string())

    print("\n-- New feature stats --")
    new_cols = [
        "raw_escalation_valid", "raw_loss_chasing_valid",
        "raw_stake_escalation_norm", "escalation_x_chasing",
        "casino_bet_share", "la_bet_share", "vol_x_stakes", "net_loss_rate",
    ]
    print(features[new_cols].describe().round(3).to_string())