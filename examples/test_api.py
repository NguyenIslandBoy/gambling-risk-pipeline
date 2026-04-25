"""
examples/test_api.py — Quick API smoke test.
Run AFTER starting the server:
    uvicorn src.api.app:app --port 8000
Then:
    python examples/test_api.py
"""

import json
import sys
import urllib.request
import urllib.error

BASE = "http://localhost:8000"


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}") as r:
        return json.loads(r.read())


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def run():
    print("=" * 50)
    print("API smoke test")
    print("=" * 50)

    # 1. Health
    print("\n[1] GET /health")
    r = get("/health")
    assert r["status"] == "ok", f"Health check failed: {r}"
    print(f"    status={r['status']}  model_loaded={r['model_loaded']}  "
          f"n_features={r['n_features']}")

    # 2. Model info
    print("\n[2] GET /model/info")
    r = get("/model/info")
    print(f"    model_type={r['model_type']}  "
          f"n_features={r['n_features']}  "
          f"training_auroc={r['training_auroc']}")

    # 3. Single prediction — high risk user
    print("\n[3] POST /predict — high risk user")
    with open("examples/single_user.json") as f:
        payload = json.load(f)
    r = post("/predict", payload)
    print(f"    risk_score={r['risk_score']}  "
          f"risk_label={r['risk_label']}  "
          f"latency={r['latency_ms']}ms")
    print(f"    confidence: {r['confidence']}")

    # 4. Single prediction — low risk user (minimal activity)
    print("\n[4] POST /predict — low risk user (minimal activity)")
    low_risk = {
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
    r = post("/predict", low_risk)
    print(f"    risk_score={r['risk_score']}  "
          f"risk_label={r['risk_label']}  "
          f"latency={r['latency_ms']}ms")

    # 5. Batch prediction
    print("\n[5] POST /predict/batch — 2 users")
    batch = {"users": [payload, low_risk]}
    r = post("/predict/batch", batch)
    print(f"    n_high={r['n_high_risk']}  "
          f"n_medium={r['n_medium_risk']}  "
          f"n_low={r['n_low_risk']}  "
          f"latency={r['latency_ms']}ms")
    for i, res in enumerate(r["results"]):
        print(f"    user[{i}]: score={res['risk_score']}  label={res['risk_label']}")

    print("\n" + "=" * 50)
    print("All tests passed.")
    print("=" * 50)


if __name__ == "__main__":
    try:
        run()
    except urllib.error.URLError:
        print("ERROR: Could not connect. Is the server running?")
        print("  Start with: uvicorn src.api.app:app --port 8000")
        sys.exit(1)