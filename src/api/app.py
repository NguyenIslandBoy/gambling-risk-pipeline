"""
src/api/app.py — FastAPI inference endpoint for the gambling risk classifier.

Endpoints:
  GET  /health          — liveness check
  GET  /model/info      — model metadata (features, training AUROC)
  POST /predict         — score a single user from raw 31-day behaviour
  POST /predict/batch   — score multiple users at once

Run locally:
    uvicorn src.api.app:app --reload --port 8000

Then test:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d @examples/single_user.json
"""

import pickle
import sys
from pathlib import Path
from typing import Optional
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MODEL_PATH, LABEL_COL

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gambling Risk Classifier API",
    description=(
        "Predicts the probability that a new subscriber will trigger a "
        "Responsible Gambling intervention, based on their first 31 days "
        "of betting behaviour. Built on the bwin/Harvard Division on Addiction dataset."
    ),
    version="1.0.0",
)

# ── Load model at startup ─────────────────────────────────────────────────────

_model = None
_feature_cols = None


def load_model():
    global _model, _feature_cols
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run src/models/train.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    _model = payload["model"]
    _feature_cols = payload["feature_cols"]


@app.on_event("startup")
def startup_event():
    load_model()
    print(f"Model loaded. Features: {len(_feature_cols)}")


# ── Request / Response schemas ────────────────────────────────────────────────

class UserFeatures(BaseModel):
    """
    Raw 31-day betting behaviour for a single user.
    All monetary values in Euros. All counts as integers.
    Fields mirror the feature names in the model.
    Missing fields default to 0 (no activity).
    """
    # Volume — fixed odds
    fo_total_stakes: float = Field(default=0.0, ge=0, description="Total staked on fixed odds (€)")
    fo_total_bets: float = Field(default=0.0, ge=0, description="Total bets placed on fixed odds")
    fo_active_days: float = Field(default=0.0, ge=0, le=31)
    fo_sd_stakes: float = Field(default=0.0, ge=0)
    fo_sd_bets: float = Field(default=0.0, ge=0)
    fo_avg_bet_size: float = Field(default=0.0, ge=0)
    fo_bets_per_active_day: float = Field(default=0.0, ge=0)

    # Volume — live action
    la_total_stakes: float = Field(default=0.0, ge=0)
    la_total_bets: float = Field(default=0.0, ge=0)
    la_active_days: float = Field(default=0.0, ge=0, le=31)
    la_sd_stakes: float = Field(default=0.0, ge=0)
    la_sd_bets: float = Field(default=0.0, ge=0)
    la_avg_bet_size: float = Field(default=0.0, ge=0)
    la_bets_per_active_day: float = Field(default=0.0, ge=0)

    # Volume — casino
    ca_total_stakes: float = Field(default=0.0, ge=0)
    ca_total_bets: float = Field(default=0.0, ge=0)
    ca_active_days: float = Field(default=0.0, ge=0, le=31)
    ca_sd_stakes: float = Field(default=0.0, ge=0)
    ca_sd_bets: float = Field(default=0.0, ge=0)
    ca_avg_bet_size: float = Field(default=0.0, ge=0)
    ca_bets_per_active_day: float = Field(default=0.0, ge=0)

    # Cross-game
    n_games_played: float = Field(default=1.0, ge=0, le=5)
    first_game_type: float = Field(default=1.0, ge=1, le=5,
        description="1=FO, 2=LA, 3=Poker, 4=Casino, 5=Other")
    most_frequent_game: float = Field(default=1.0, ge=1, le=5)
    played_fixed_odds: float = Field(default=0.0, ge=0, le=1)
    played_live_action: float = Field(default=0.0, ge=0, le=1)
    played_poker: float = Field(default=0.0, ge=0, le=1)
    played_casino: float = Field(default=0.0, ge=0, le=1)
    played_other_games: float = Field(default=0.0, ge=0, le=1)

    # Activity
    total_active_days: float = Field(default=0.0, ge=0, le=31)
    wk1_freq: float = Field(default=0.0, ge=0, le=1)
    wk2_freq: float = Field(default=0.0, ge=0, le=1)
    wk3_freq: float = Field(default=0.0, ge=0, le=1)
    wk4_freq: float = Field(default=0.0, ge=0, le=1)

    # Trajectories (1=decreasing, 2=stable, 3=increasing)
    freq_trajectory: float = Field(default=2.0, ge=1, le=3)
    fo_bets_trajectory: float = Field(default=2.0, ge=1, le=3)
    fo_stakes_trajectory: float = Field(default=2.0, ge=1, le=3)
    la_bets_trajectory: float = Field(default=2.0, ge=1, le=3)
    la_stakes_trajectory: float = Field(default=2.0, ge=1, le=3)
    ca_bets_trajectory: float = Field(default=2.0, ge=1, le=3)
    ca_stakes_trajectory: float = Field(default=2.0, ge=1, le=3)

    # Weekend ratios
    fo_weekend_bets_ratio: float = Field(default=0.0, ge=0, le=1)
    la_weekend_bets_ratio: float = Field(default=0.0, ge=0, le=1)
    ca_weekend_bets_ratio: float = Field(default=0.0, ge=0, le=1)
    fo_weekend_stakes_ratio: float = Field(default=0.0, ge=0, le=1)
    la_weekend_stakes_ratio: float = Field(default=0.0, ge=0, le=1)
    ca_weekend_stakes_ratio: float = Field(default=0.0, ge=0, le=1)

    # Registration to deposit lag
    days_registration_to_deposit: float = Field(default=0.0, ge=0)

    # Raw engineered features
    raw_total_stakes: float = Field(default=0.0, ge=0)
    raw_total_bets: float = Field(default=0.0, ge=0)
    raw_total_active_days: float = Field(default=0.0, ge=0, le=31)
    raw_net_position: float = Field(default=0.0,
        description="Net position in €. Positive = net loss, negative = net win.")
    raw_stakes_per_day: float = Field(default=0.0, ge=0)
    raw_bets_per_day: float = Field(default=0.0, ge=0)
    raw_cv_stakes: float = Field(default=0.0, ge=0,
        description="Coefficient of variation of daily stakes")
    raw_n_products: float = Field(default=1.0, ge=0)
    had_enough_days_for_ts: float = Field(default=0.0, ge=0, le=1)
    raw_stake_escalation: float = Field(default=0.0,
        description="Linear slope of daily stakes over 31 days")
    raw_escalation_valid: float = Field(default=0.0, ge=0, le=1)
    raw_stake_escalation_norm: float = Field(default=0.0,
        description="Escalation slope normalised by mean daily stakes")
    raw_loss_chasing: float = Field(default=1.0, ge=0,
        description="Ratio of bets after losing vs winning days. >1 = chasing losses.")
    raw_loss_chasing_valid: float = Field(default=0.0, ge=0, le=1)

    @field_validator("raw_loss_chasing")
    @classmethod
    def cap_loss_chasing(cls, v):
        return min(v, 20.0)  # cap extreme outliers


class PredictionResponse(BaseModel):
    risk_score: float = Field(description="Probability of triggering RG intervention (0–1)")
    risk_label: str = Field(description="HIGH / MEDIUM / LOW based on score thresholds")
    risk_tier: int = Field(description="1=LOW, 2=MEDIUM, 3=HIGH")
    confidence: str = Field(description="Model confidence note")
    latency_ms: float


class BatchRequest(BaseModel):
    users: list[UserFeatures] = Field(max_length=500)


class BatchResponse(BaseModel):
    results: list[dict]
    n_high_risk: int
    n_medium_risk: int
    n_low_risk: int
    latency_ms: float


# ── Inference logic ───────────────────────────────────────────────────────────

def _derive_features(data: dict) -> dict:
    """
    Compute derived features from raw inputs — mirrors engineer.py post-join logic.
    Called at inference time so the API accepts raw behaviour, not pre-engineered features.
    """
    fo_stakes = data.get("fo_total_stakes", 0)
    la_stakes = data.get("la_total_stakes", 0)
    ca_stakes = data.get("ca_total_stakes", 0)
    fo_bets   = data.get("fo_total_bets", 0)
    la_bets   = data.get("la_total_bets", 0)
    ca_bets   = data.get("ca_total_bets", 0)

    total_stakes = fo_stakes + la_stakes + ca_stakes
    total_bets   = fo_bets + la_bets + ca_bets

    data["total_stakes_31d"]    = total_stakes
    data["total_bets_31d"]      = total_bets
    data["overall_avg_bet_size"] = total_stakes / max(total_bets, 1)
    data["activity_ratio"]      = data.get("total_active_days", 0) / 31

    # Interaction features
    esc_norm  = data.get("raw_stake_escalation_norm", 0)
    lc        = data.get("raw_loss_chasing", 1.0)
    data["escalation_x_chasing"] = max(esc_norm, 0) * max(lc - 1, 0)
    data["casino_bet_share"]     = ca_bets / max(total_bets, 1)
    data["la_bet_share"]         = la_bets / max(total_bets, 1)
    data["vol_x_stakes"]         = (
        data.get("raw_cv_stakes", 0) * np.log1p(data.get("raw_total_stakes", 0))
    )
    raw_stakes = data.get("raw_total_stakes", 0)
    raw_net    = data.get("raw_net_position", 0)
    data["net_loss_rate"] = float(
        np.clip(raw_net / max(raw_stakes, 1e-6), -5, 5)
    )

    traj_cols = [
        "freq_trajectory", "fo_bets_trajectory", "fo_stakes_trajectory",
        "la_bets_trajectory", "la_stakes_trajectory",
        "ca_bets_trajectory", "ca_stakes_trajectory",
    ]
    data["any_increasing_traj"] = int(any(data.get(c, 2) == 3 for c in traj_cols))

    return data


def _score(features_dict: dict) -> float:
    """Build a single-row DataFrame aligned to model features and return risk score."""
    features_dict = _derive_features(features_dict)
    row = pd.DataFrame([features_dict])

    # Align to model feature columns — fill missing with 0
    for col in _feature_cols:
        if col not in row.columns:
            row[col] = 0.0
    row = row[_feature_cols]

    prob = float(_model.predict_proba(row)[0, 1])
    return prob


def _label(score: float) -> tuple[str, int]:
    """Convert probability to risk tier. Thresholds from validation set distribution."""
    if score >= 0.65:
        return "HIGH", 3
    elif score >= 0.45:
        return "MEDIUM", 2
    else:
        return "LOW", 1


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "n_features": len(_feature_cols) if _feature_cols else 0,
    }


@app.get("/model/info")
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(_model).__name__,
        "n_features": len(_feature_cols),
        "feature_cols": _feature_cols,
        "best_iteration": getattr(_model, "best_iteration_", None),
        "training_auroc": 0.746,
        "note": (
            "Trained on bwin/Harvard Division on Addiction dataset. "
            "Predicts probability of triggering a Responsible Gambling intervention "
            "within the first 31 days. AUROC 0.746 on held-out validation set."
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(user: UserFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        score = _score(user.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    label, tier = _label(score)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    confidence = (
        "High confidence" if score < 0.3 or score > 0.7
        else "Moderate confidence — score near decision boundary"
    )

    return PredictionResponse(
        risk_score=round(score, 4),
        risk_label=label,
        risk_tier=tier,
        confidence=confidence,
        latency_ms=latency,
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not batch.users:
        raise HTTPException(status_code=400, detail="Empty batch")

    t0 = time.perf_counter()
    results = []
    for user in batch.users:
        try:
            score = _score(user.model_dump())
            label, tier = _label(score)
            results.append({
                "risk_score": round(score, 4),
                "risk_label": label,
                "risk_tier": tier,
            })
        except Exception as e:
            results.append({"error": str(e)})

    latency = round((time.perf_counter() - t0) * 1000, 2)
    labels = [r.get("risk_label", "") for r in results]

    return BatchResponse(
        results=results,
        n_high_risk=labels.count("HIGH"),
        n_medium_risk=labels.count("MEDIUM"),
        n_low_risk=labels.count("LOW"),
        latency_ms=latency,
    )