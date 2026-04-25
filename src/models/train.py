"""
src/models/train.py — Train and evaluate the LightGBM risk classifier.

Pipeline:
  1. Load features from DuckDB
  2. Split using the pre-defined ValidationSet column (from Braverman 2013)
  3. Hyperparameter tune via cross-validation on train split
  4. Retrain best model on full train split
  5. Evaluate on held-out validation split
  6. SHAP feature importance
  7. Save model to disk

Run:
    python src/models/train.py
"""

import sys
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DB_PATH, TABLE_FEATURES, LABEL_COL, USER_ID_COL,
    VALIDATION_SET_COL, MODEL_PATH, MODELS_DIR,
)

NON_FEATURE_COLS = {
    USER_ID_COL, LABEL_COL, VALIDATION_SET_COL,
    "RiskGroup1", "RiskGroup2", "RiskGroupCombined",
}


# ── Load ──────────────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}.")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"SELECT * FROM {TABLE_FEATURES}").df()
    con.close()
    print(f"Loaded features: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"Model input features: {len(cols)}")
    return cols


# ── Split ─────────────────────────────────────────────────────────────────────

def split(df: pd.DataFrame, feature_cols: list):
    train = df[df[VALIDATION_SET_COL] == 0]
    val   = df[df[VALIDATION_SET_COL] == 1]
    X_train = train[feature_cols]
    y_train = train[LABEL_COL].astype(int)
    X_val   = val[feature_cols]
    y_val   = val[LABEL_COL].astype(int)
    print(f"Train: {len(train):,}  (RG={y_train.sum()}, ctrl={(y_train==0).sum()})")
    print(f"Val:   {len(val):,}  (RG={y_val.sum()}, ctrl={(y_val==0).sum()})")
    return X_train, y_train, X_val, y_val


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Grid search over key LightGBM params using 5-fold stratified CV on train set.
    We keep the grid small and principled — no blind large grids.
    """
    print("\nHyperparameter tuning (5-fold CV on train set)...")

    param_grid = [
        {"max_depth": 3, "num_leaves": 7,  "min_child_samples": 30, "learning_rate": 0.05},
        {"max_depth": 4, "num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.05},
        {"max_depth": 5, "num_leaves": 20, "min_child_samples": 20, "learning_rate": 0.05},
        {"max_depth": 4, "num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.02},
        {"max_depth": 4, "num_leaves": 31, "min_child_samples": 15, "learning_rate": 0.05},
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1
    best_params = None

    for i, params in enumerate(param_grid):
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            m = lgb.LGBMClassifier(
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                **params,
            )
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            score = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
            fold_scores.append(score)

        mean_auc = np.mean(fold_scores)
        std_auc  = np.std(fold_scores)
        print(f"  [{i+1}/{len(param_grid)}] params={params}  "
              f"CV AUC={mean_auc:.4f} ± {std_auc:.4f}")

        if mean_auc > best_score:
            best_score = mean_auc
            best_params = params

    print(f"\n  Best CV AUC: {best_score:.4f}  params={best_params}")
    return best_params


# ── Train final model ─────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: dict,
) -> lgb.LGBMClassifier:
    """Retrain on full train split with best params, early stop on val."""
    print("\nTraining final model with best params...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **best_params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"Best iteration: {model.best_iteration_}")
    return model


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(
    model: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    auroc  = roc_auc_score(y_val, y_prob)

    print("\n── Validation results ───────────────────────────────────")
    print(f"  AUROC:  {auroc:.4f}")
    print("\n  Classification report (threshold=0.5):")
    print(classification_report(y_val, y_pred,
                                target_names=["Control", "RG Case"]))
    cm = confusion_matrix(y_val, y_pred)
    print("  Confusion matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_val, y_prob, ax=ax, name="LightGBM")
    ax.set_title("ROC Curve — Gambling Risk Classifier")
    roc_path = MODELS_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ROC curve saved: {roc_path}")

    return {"auroc": auroc, "y_prob": y_prob, "y_pred": y_pred}


# ── SHAP importance ───────────────────────────────────────────────────────────

def shap_importance(model: lgb.LGBMClassifier, X_val: pd.DataFrame) -> None:
    print("\n── SHAP feature importance ──────────────────────────────")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_shap = pd.Series(
        np.abs(sv).mean(axis=0), index=X_val.columns
    ).sort_values(ascending=False)

    print("\n  Top 20 features by mean |SHAP|:")
    for feat, val in mean_shap.head(20).items():
        bar = "█" * int(val / mean_shap.iloc[0] * 20)
        print(f"  {feat:<45} {val:.4f}  {bar}")

    fig, ax = plt.subplots(figsize=(8, 7))
    mean_shap.head(20).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 20 Features — Mean |SHAP| Value")
    ax.set_xlabel("Mean |SHAP|")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    shap_path = MODELS_DIR / "shap_importance.png"
    fig.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  SHAP plot saved: {shap_path}")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_model(model: lgb.LGBMClassifier, feature_cols: list) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "feature_cols": feature_cols}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Model saved: {MODEL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Model training")
    print("=" * 55)

    df           = load_features()
    feature_cols = get_feature_cols(df)

    X_train, y_train, X_val, y_val = split(df, feature_cols)

    best_params = tune_hyperparams(X_train, y_train)
    model       = train_model(X_train, y_train, X_val, y_val, best_params)
    results     = evaluate(model, X_val, y_val)

    shap_importance(model, X_val)
    save_model(model, feature_cols)

    print("\n" + "=" * 55)
    print(f"Done. Final AUROC: {results['auroc']:.4f}")
    print("=" * 55)