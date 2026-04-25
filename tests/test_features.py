"""
tests/test_features.py — Unit tests for feature engineering output.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest
import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, TABLE_FEATURES, LABEL_COL, USER_ID_COL, VALIDATION_SET_COL


@pytest.fixture(scope="module")
def features():
    if not DB_PATH.exists():
        pytest.skip("Database not found")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {TABLE_FEATURES}").df()
    except Exception:
        pytest.skip("Features table not found — run src/features/engineer.py first")
    finally:
        con.close()
    return df


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_feature_row_count(features):
    assert 4050 <= len(features) <= 4060, f"Unexpected row count: {len(features)}"


def test_feature_col_count(features):
    assert features.shape[1] >= 70, f"Too few feature columns: {features.shape[1]}"


# ── No nulls ──────────────────────────────────────────────────────────────────

def test_no_nulls_in_features(features):
    null_counts = features.isnull().sum()
    nulls = null_counts[null_counts > 0]
    assert nulls.empty, f"Null values found:\n{nulls.to_string()}"


# ── Label and split integrity ─────────────────────────────────────────────────

def test_label_values(features):
    assert set(features[LABEL_COL].unique()) == {0, 1}


def test_validation_set_values(features):
    assert set(features[VALIDATION_SET_COL].unique()).issubset({0, 1})


def test_train_val_split_sizes(features):
    train = (features[VALIDATION_SET_COL] == 0).sum()
    val   = (features[VALIDATION_SET_COL] == 1).sum()
    assert train > val, "Train set should be larger than validation set"
    assert val >= 900, f"Validation set too small: {val}"


# ── Feature value ranges ──────────────────────────────────────────────────────

def test_activity_ratio_range(features):
    assert features["activity_ratio"].between(0, 1).all(), \
        "activity_ratio out of [0, 1] range"


def test_casino_bet_share_range(features):
    assert features["casino_bet_share"].between(0, 1).all(), \
        "casino_bet_share out of [0, 1] range"


def test_validity_flags_binary(features):
    for col in ["raw_escalation_valid", "raw_loss_chasing_valid", "had_enough_days_for_ts"]:
        assert features[col].isin([0, 1]).all(), f"{col} contains non-binary values"


def test_total_stakes_non_negative(features):
    assert (features["total_stakes_31d"] >= 0).all(), \
        "total_stakes_31d contains negative values"


def test_loss_chasing_neutral_default(features):
    """Users with insufficient data should have loss_chasing = 1.0 (neutral)."""
    no_data = features[features["raw_loss_chasing_valid"] == 0]
    if len(no_data) > 0:
        assert (no_data["raw_loss_chasing"] == 1.0).all(), \
            "Users with no LC data should have raw_loss_chasing=1.0"


# ── Engineered features exist ─────────────────────────────────────────────────

def test_interaction_features_present(features):
    required = [
        "escalation_x_chasing", "casino_bet_share", "la_bet_share",
        "vol_x_stakes", "net_loss_rate", "any_increasing_traj",
    ]
    missing = [c for c in required if c not in features.columns]
    assert not missing, f"Missing engineered features: {missing}"


def test_raw_features_present(features):
    required = [
        "raw_total_stakes", "raw_net_position", "raw_cv_stakes",
        "raw_stake_escalation", "raw_loss_chasing",
        "raw_escalation_valid", "raw_loss_chasing_valid",
    ]
    missing = [c for c in required if c not in features.columns]
    assert not missing, f"Missing raw features: {missing}"