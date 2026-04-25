"""
tests/test_api.py — Unit tests for the FastAPI inference endpoint.
Uses TestClient — no server needed.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip entire module if model not trained yet
from config import MODEL_PATH
if not MODEL_PATH.exists():
    pytest.skip("Model not found — run src/models/train.py first",
                allow_module_level=True)

from fastapi.testclient import TestClient
from src.api.app import app, load_model

# Force model load before any test runs — TestClient startup event
# doesn't always fire reliably depending on pytest collection order
load_model()

client = TestClient(app)

# ── Fixtures ──────────────────────────────────────────────────────────────────

HIGH_RISK_USER = {
    "fo_total_stakes": 450.0,
    "fo_total_bets": 120,
    "fo_active_days": 18,
    "la_total_stakes": 1200.0,
    "la_total_bets": 310,
    "la_active_days": 22,
    "ca_total_stakes": 3800.0,
    "ca_total_bets": 950,
    "ca_active_days": 25,
    "n_games_played": 3,
    "first_game_type": 1,
    "most_frequent_game": 4,
    "played_fixed_odds": 1,
    "played_live_action": 1,
    "played_casino": 1,
    "total_active_days": 26,
    "wk1_freq": 0.14,
    "wk2_freq": 0.21,
    "wk3_freq": 0.29,
    "wk4_freq": 0.29,
    "freq_trajectory": 3,
    "ca_bets_trajectory": 3,
    "raw_total_stakes": 5450.0,
    "raw_total_bets": 1380,
    "raw_total_active_days": 26,
    "raw_net_position": 820.0,
    "raw_stakes_per_day": 209.6,
    "raw_bets_per_day": 53.1,
    "raw_cv_stakes": 1.8,
    "raw_n_products": 3,
    "had_enough_days_for_ts": 1,
    "raw_stake_escalation": 45.2,
    "raw_escalation_valid": 1,
    "raw_stake_escalation_norm": 0.22,
    "raw_loss_chasing": 2.1,
    "raw_loss_chasing_valid": 1,
}

LOW_RISK_USER = {
    "fo_total_stakes": 25.0,
    "fo_total_bets": 10,
    "fo_active_days": 3,
    "total_active_days": 3,
    "raw_total_stakes": 25.0,
    "raw_total_bets": 10,
    "raw_total_active_days": 3,
    "raw_net_position": -5.0,
    "raw_loss_chasing": 0.8,
}


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_health_has_feature_count():
    r = client.get("/health")
    assert r.json()["n_features"] > 0


# ── Model info ────────────────────────────────────────────────────────────────

def test_model_info_returns_features():
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "feature_cols" in body
    assert len(body["feature_cols"]) > 0
    assert body["training_auroc"] > 0.5


# ── Single prediction ─────────────────────────────────────────────────────────

def test_predict_returns_200():
    r = client.post("/predict", json=HIGH_RISK_USER)
    assert r.status_code == 200


def test_predict_response_schema():
    r = client.post("/predict", json=HIGH_RISK_USER)
    body = r.json()
    assert "risk_score" in body
    assert "risk_label" in body
    assert "risk_tier" in body
    assert "confidence" in body
    assert "latency_ms" in body


def test_predict_score_range():
    for payload in [HIGH_RISK_USER, LOW_RISK_USER]:
        r = client.post("/predict", json=payload)
        score = r.json()["risk_score"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_predict_risk_label_valid():
    for payload in [HIGH_RISK_USER, LOW_RISK_USER]:
        r = client.post("/predict", json=payload)
        assert r.json()["risk_label"] in {"HIGH", "MEDIUM", "LOW"}


def test_predict_risk_tier_valid():
    for payload in [HIGH_RISK_USER, LOW_RISK_USER]:
        r = client.post("/predict", json=payload)
        assert r.json()["risk_tier"] in {1, 2, 3}


def test_high_risk_user_scores_higher(  ):
    high = client.post("/predict", json=HIGH_RISK_USER).json()["risk_score"]
    low  = client.post("/predict", json=LOW_RISK_USER).json()["risk_score"]
    assert high > low, f"High risk user ({high}) should score above low risk ({low})"


def test_high_risk_label_is_high():
    r = client.post("/predict", json=HIGH_RISK_USER)
    assert r.json()["risk_label"] == "HIGH"


def test_low_risk_label_is_low():
    r = client.post("/predict", json=LOW_RISK_USER)
    assert r.json()["risk_label"] == "LOW"


def test_empty_payload_uses_defaults():
    """Empty body should not crash — all fields have defaults."""
    r = client.post("/predict", json={})
    assert r.status_code == 200
    assert 0.0 <= r.json()["risk_score"] <= 1.0


def test_label_tier_consistency():
    """risk_label and risk_tier must be consistent."""
    for payload in [HIGH_RISK_USER, LOW_RISK_USER]:
        body = client.post("/predict", json=payload).json()
        label, tier = body["risk_label"], body["risk_tier"]
        expected = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        assert expected[label] == tier, f"Label/tier mismatch: {label} vs {tier}"


# ── Batch prediction ──────────────────────────────────────────────────────────

def test_batch_predict_returns_200():
    r = client.post("/predict/batch", json={"users": [HIGH_RISK_USER, LOW_RISK_USER]})
    assert r.status_code == 200


def test_batch_result_count():
    r = client.post("/predict/batch", json={"users": [HIGH_RISK_USER, LOW_RISK_USER]})
    assert len(r.json()["results"]) == 2


def test_batch_aggregate_counts():
    r = client.post("/predict/batch", json={"users": [HIGH_RISK_USER, LOW_RISK_USER]})
    body = r.json()
    total = body["n_high_risk"] + body["n_medium_risk"] + body["n_low_risk"]
    assert total == 2


def test_batch_empty_raises_422():
    r = client.post("/predict/batch", json={"users": []})
    assert r.status_code in {400, 422}


def test_batch_single_user():
    r = client.post("/predict/batch", json={"users": [HIGH_RISK_USER]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1