"""
pipeline.py — Prefect orchestration for the gambling risk pipeline.

Stages:
  1. ingest    — load .dat files → DuckDB (loader + database)
  2. validate  — sanity checks vs codebook expectations
  3. engineer  — build feature table from analytic + raw daily aggregates
  4. train     — LightGBM hyperparameter tuning + final model + SHAP

Run locally (one-off):
    python pipeline.py

Run with Prefect UI (open http://localhost:4200 after starting server):
    prefect server start    # terminal 1
    python pipeline.py      # terminal 2

Skip flags for faster iteration:
    python pipeline.py --skip-ingest             # skip .dat file loading
    python pipeline.py --skip-ingest --skip-engineering  # jump straight to training
"""

import sys
import time
from pathlib import Path

from prefect import flow, task, get_run_logger

sys.path.insert(0, str(Path(__file__).parent))

from src.ingest.database import build_database
from src.ingest.validate import validate_all
from src.features.engineer import build_all_features
from src.models.train import (
    load_features, get_feature_cols, split,
    tune_hyperparams, train_model, evaluate,
    shap_importance, save_model,
)


# ===========================================================================
# Tasks
# ===========================================================================

@task(name="ingest", retries=2, retry_delay_seconds=10)
def task_ingest() -> dict:
    """Load raw .dat files into DuckDB. Retries twice on transient IO errors."""
    logger = get_run_logger()
    logger.info("Stage 1 — Ingesting raw .dat files into DuckDB")
    t0 = time.perf_counter()

    build_database()

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(f"Ingest complete in {elapsed}s")
    return {"elapsed_s": elapsed}


@task(name="validate", retries=0)
def task_validate() -> dict:
    """
    Run sanity checks vs codebook expectations.
    Zero retries — a validation failure means data is wrong, not transient.
    Fails the pipeline immediately if any check fails.
    """
    logger = get_run_logger()
    logger.info("Stage 2 — Validating ingested data")
    t0 = time.perf_counter()

    passed = validate_all()
    if not passed:
        raise RuntimeError(
            "Data validation failed — check output above for FAIL items. "
            "Fix before proceeding to feature engineering."
        )

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(f"Validation passed in {elapsed}s")
    return {"elapsed_s": elapsed, "passed": True}


@task(name="engineer_features", retries=1, retry_delay_seconds=15)
def task_engineer() -> dict:
    """
    Build feature table from analytic dataset + raw daily aggregates.
    Includes stake escalation slopes and loss chasing index (~3 min).
    Retries once in case of transient DuckDB lock.
    """
    logger = get_run_logger()
    logger.info("Stage 3 — Engineering features")
    t0 = time.perf_counter()

    features = build_all_features()
    n_rows = len(features)
    n_cols = features.shape[1]

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(
        f"Feature engineering complete in {elapsed}s | "
        f"{n_rows:,} users x {n_cols} features"
    )
    return {"elapsed_s": elapsed, "n_rows": n_rows, "n_features": n_cols}


@task(name="train_model", retries=1, retry_delay_seconds=30)
def task_train() -> dict:
    """
    Hyperparameter tune + train final LightGBM model + SHAP.
    Uses pre-defined train/val split from Braverman (2013).
    (~5 min including CV tuning over 5 param combinations).
    """
    logger = get_run_logger()
    logger.info("Stage 4 — Training LightGBM risk classifier")
    t0 = time.perf_counter()

    df           = load_features()
    feature_cols = get_feature_cols(df)

    X_train, y_train, X_val, y_val = split(df, feature_cols)

    best_params = tune_hyperparams(X_train, y_train)
    model       = train_model(X_train, y_train, X_val, y_val, best_params)
    results     = evaluate(model, X_val, y_val)

    shap_importance(model, X_val)
    save_model(model, feature_cols)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(
        f"Training complete in {elapsed}s | "
        f"AUROC={results['auroc']:.4f} | "
        f"Best params={best_params}"
    )
    return {
        "elapsed_s":  elapsed,
        "auroc":      results["auroc"],
        "best_params": best_params,
        "n_features": len(feature_cols),
    }


# ===========================================================================
# Flow
# ===========================================================================

@flow(
    name="gambling-risk-pipeline",
    description=(
        "Early warning pipeline for gambling harm detection. "
        "Ingest → validate → engineer features → train LightGBM classifier."
    ),
    log_prints=True,
)
def gambling_pipeline(
    skip_ingest: bool = False,
    skip_engineering: bool = False,
) -> dict:
    """
    Main pipeline flow.

    Args:
        skip_ingest:      Skip .dat file ingestion (use existing DuckDB).
        skip_engineering: Skip feature engineering (use existing features table).

    Returns:
        Summary dict with timing and key metrics per stage.
    """
    logger = get_run_logger()
    logger.info("=" * 55)
    logger.info("Gambling Risk Pipeline — starting")
    logger.info("=" * 55)

    summary = {}
    pipeline_start = time.perf_counter()

    # Stage 1 — Ingest
    if not skip_ingest:
        ingest_result = task_ingest()
        summary["ingest"] = ingest_result
    else:
        logger.info("Stage 1 — Skipped (skip_ingest=True)")

    # Stage 2 — Validate (always runs — enforces data contract)
    if not skip_ingest:
        validate_result = task_validate(wait_for=[ingest_result])
    else:
        validate_result = task_validate()
    summary["validate"] = validate_result

    # Stage 3 — Feature engineering
    if not skip_engineering:
        engineer_result = task_engineer(wait_for=[validate_result])
        summary["engineer"] = engineer_result
    else:
        logger.info("Stage 3 — Skipped (skip_engineering=True)")
        engineer_result = validate_result  # pass through for wait_for

    # Stage 4 — Train
    train_result = task_train(wait_for=[engineer_result])
    summary["train"] = train_result

    total = round(time.perf_counter() - pipeline_start, 1)
    summary["total_elapsed_s"] = total

    logger.info("=" * 55)
    logger.info(f"Pipeline complete in {total}s")
    for stage, result in summary.items():
        if isinstance(result, dict) and "elapsed_s" in result:
            logger.info(f"  {stage:<20} {result['elapsed_s']}s")
    if "train" in summary:
        logger.info(f"  Final AUROC: {summary['train']['auroc']:.4f}")
    logger.info("=" * 55)

    return summary


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gambling risk pipeline")
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip .dat file ingestion (use existing DuckDB)"
    )
    parser.add_argument(
        "--skip-engineering", action="store_true",
        help="Skip feature engineering (use existing features table)"
    )
    args = parser.parse_args()

    result = gambling_pipeline(
        skip_ingest=args.skip_ingest,
        skip_engineering=args.skip_engineering,
    )
    print("\nSummary:")
    for k, v in result.items():
        print(f"  {k}: {v}")