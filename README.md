# Gambling Risk Pipeline

![CI](https://github.com/NguyenIslandBoy/gambling-risk-pipeline/actions/workflows/ci.yml/badge.svg)


Early warning system that predicts whether a new online gambling subscriber will trigger a Responsible Gambling (RG) intervention, based on their first 31 days of betting behaviour.

**AUROC 0.746** on held-out validation set, beating the original paper's logistic regression baseline (~0.70).

---

## Background

Built on the [Harvard Division on Addiction / bwin Transparency Project](http://www.thetransparencyproject.org/) dataset - real betting records from 4,134 bwin subscribers (2,068 who triggered RG interventions, 2,066 matched controls). Directly replicates and extends the methodology of:

> Braverman, J., LaPlante, D. A., Nelson, S. E, & Shaffer, H. J. (2013). *Using Cross-game Behavioral Markers for Early Identification of High-risk Internet Gamblers.* Psychology of Addictive Behaviors.

The problem structure maps directly to **fraud/AML detection** in fintech: binary classification from behavioural time series with an imbalanced, noisy label.

---

## Project structure

```
gambling-risk-pipeline/
├── config.py                  # All paths and constants
├── data/
│   ├── raw/                   # Original .dat files (not committed)
│   └── processed/             # DuckDB database (generated)
├── src/
│   ├── ingest/
│   │   ├── loader.py          # Load raw .dat files → DataFrames
│   │   ├── database.py        # Write DataFrames → DuckDB
│   │   └── validate.py        # Sanity checks vs codebook expectations
│   ├── features/
│   │   └── engineer.py        # Feature engineering (two sets: analytic + raw)
│   ├── models/
│   │   └── train.py           # LightGBM + hyperparameter tuning + SHAP
│   └── api/
│       └── app.py             # FastAPI inference endpoint
├── tests/
│   ├── test_gambling_pipeline.py  # 64 unit tests (pytest, no DB required)
│   ├── test_ingest.py             # Integration tests (requires live DB)
│   ├── test_features.py           # Integration tests (requires live DB)
│   └── test_api.py                # Integration tests (requires trained model)
├── examples/
│   ├── single_user.json       # Example high-risk user payload
│   └── test_api.py            # Live API smoke test
└── requirements.txt
```

---

## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Place raw data files in `data/raw/`**
```
AnalyticDataset_Braverman_LaPlante_PAB_2013.dat
Raw Datset I.Demographics_Braverman_LaPlante_PAB_2013.dat
Raw Datset II.Daily aggregates_Braverman_LaPlante_PAB_2013.dat
Raw Datset III.Responsible gambling details_Gray_LaPlante_PAB_2012.dat
```
Download from [The Transparency Project](http://www.thetransparencyproject.org/download_index.php).

**3. Run the pipeline**
```bash
python src/ingest/loader.py       # verify files load correctly
python src/ingest/database.py     # write to DuckDB
python src/ingest/validate.py     # sanity checks

python src/features/engineer.py   # build feature table (~3 min)
python src/models/train.py        # train + evaluate + save model (~5 min)
```

**4. Start the API**
```bash
uvicorn src.api.app:app --reload --port 8000
```

**5. Test**
```bash
pytest tests/test_gambling_pipeline.py -v   # unit tests (no data needed)
pytest tests/ -v                            # full suite (requires DB + model)
```

Swagger UI available at `http://localhost:8000/docs`.

---

## Feature engineering

Features are derived from two sources and joined per user:

**Set A - analytic dataset (pre-computed by original researchers)**
- Staking volume, bet counts, active days per product (fixed odds, live action, casino)
- Standard deviation of daily stakes and bets (within-product variability)
- Weekly frequency trajectories (increasing / stable / decreasing)
- Weekend vs weekday betting ratios

**Set B - engineered from raw daily aggregates (first 31 days)**
| Feature | Description |
|---|---|
| `raw_stake_escalation` | Linear slope of daily stakes - are bets growing? |
| `raw_stake_escalation_norm` | Escalation normalised by mean stakes - scale-independent |
| `raw_loss_chasing` | Ratio of bets after losing vs winning days. >1 = chasing |
| `raw_cv_stakes` | Coefficient of variation of daily stakes |
| `raw_escalation_valid` | Flag: was slope computable? (≥2 active days required) |
| `raw_loss_chasing_valid` | Flag: was LC ratio computable? |
| `escalation_x_chasing` | Interaction: escalating AND chasing = strong risk signal |
| `casino_bet_share` | Fraction of total bets placed on casino products |
| `net_loss_rate` | Net position / total stakes (capped at ±5) |
| `vol_x_stakes` | Volatility × log(volume) - erratic high-volume bettor signal |

**Missingness flags** are critical: `raw_loss_chasing_valid=0` means the user had only 1 active day (insufficient data), not that they didn't chase losses. Imputing 0 without a flag would mislead the model.

---

## Model

**Algorithm:** LightGBM binary classifier  
**Split:** Pre-defined train/validation split from Braverman (2013) - `ValidationSet` column  
**Tuning:** 5-fold stratified CV grid search on train set only  
**Class weighting:** balanced (slight 51/49 imbalance)

| Metric | Value |
|---|---|
| AUROC | 0.746 |
| Accuracy | 0.70 |
| Precision (RG case) | 0.75 |
| Recall (RG case) | 0.61 |

**Top predictors (SHAP):** total staking volume, net position, bets per day, number of games played, net loss rate, stake escalation, loss chasing.

**Honest limitation:** `raw_total_stakes` and `raw_net_position` dominate SHAP (~50% of total importance). This reflects a known confound - high-volume bettors have more platform interactions and are more visible to RG systems, independent of genuine risk. A production system would normalise behavioural features by volume. The AUROC ceiling on this dataset in the literature is approximately 0.78.

---

## API

```
GET  /health          liveness + model status
GET  /model/info      feature list, AUROC, metadata
POST /predict         score a single user → probability + HIGH/MEDIUM/LOW
POST /predict/batch   score up to 500 users at once
```

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @examples/single_user.json
```

**Example response:**
```json
{
  "risk_score": 0.8058,
  "risk_label": "HIGH",
  "risk_tier": 3,
  "confidence": "High confidence",
  "latency_ms": 10.45
}
```

Risk thresholds: `HIGH ≥ 0.65`, `MEDIUM 0.45–0.65`, `LOW < 0.45`.

---

## Fintech relevance

The architecture is domain-agnostic - replacing the gambling dataset with transaction data produces an AML/fraud early warning system with identical structure:

| Gambling | Fintech |
|---|---|
| RG alert triggered | Fraud flag / SAR filed |
| Stakes variability | Transaction amount variability |
| Loss chasing | Repeated failed transactions |
| Cross-game diversity | Cross-product activity |
| 31-day observation window | 30/60-day onboarding window |