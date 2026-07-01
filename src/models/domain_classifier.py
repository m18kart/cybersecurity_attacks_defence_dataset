"""
Model 1 — Domain Threat-Severity Classifier

Target  : Binary — Low vs Elevated (Medium + High collapsed)
Rationale: 3_malicious_domains.csv has only 162 rows: 145 Low, 10 Medium, 7 High.
           7-10 examples per class is insufficient for 3-class learning.
           Collapsing Medium+High into "Elevated" gives 145 vs 17 — tractable
           as a binary problem with scale_pos_weight balancing.
Approach: XGBoost binary, with baseline comparison, stratified CV,
          RandomizedSearch tuning, and SHAP interpretability.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features.domain_features import CATEGORICAL, NUMERIC, TARGET

CV_FOLDS     = 5
RANDOM_STATE = 42

SEARCH_SPACE = {
    "model__max_depth":        [3, 4, 5, 6],
    "model__learning_rate":    [0.03, 0.05, 0.08, 0.12],
    "model__subsample":        [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5],
    "model__n_estimators":     [100, 200, 300, 400],
}


def _binarise(series: pd.Series) -> np.ndarray:
    """Low -> 0,  Medium/High -> 1  (Elevated)."""
    return (series.str.strip().str.lower() != "low").astype(int).to_numpy()


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])


def _xgb_pipe(pos_weight: float = 1.0) -> Pipeline:
    return Pipeline([
        ("preprocess", _preprocessor()),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )),
    ])


# ── Baseline comparison ───────────────────────────────────────────────────────

def compare_baselines(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compares Logistic Regression, Decision Tree, and XGBoost on binary target.
    Prints class distribution so the data story is transparent.
    """
    data  = features.dropna(subset=NUMERIC + [TARGET])
    y     = _binarise(data[TARGET])
    X     = data[CATEGORICAL + NUMERIC]

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  Class distribution (binary):")
    print(f"    Low (0):      {n_neg}  ({n_neg/len(y)*100:.1f}%)")
    print(f"    Elevated (1): {n_pos}  ({n_pos/len(y)*100:.1f}%)  <- Medium + High collapsed")
    print(f"  Total samples: {len(y)}")

    pos_weight = n_neg / max(n_pos, 1)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree":       DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                                       random_state=RANDOM_STATE),
        "xgboost":             XGBClassifier(
                                   n_estimators=300, max_depth=4, learning_rate=0.05,
                                   scale_pos_weight=pos_weight,
                                   eval_metric="logloss", random_state=RANDOM_STATE,
                               ),
    }

    rows = []
    for name, clf in candidates.items():
        pipe   = Pipeline([("preprocess", _preprocessor()), ("model", clf)])
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1")
        rows.append({
            "model":   name,
            "f1_mean": round(scores.mean(), 4),
            "f1_std":  round(scores.std(),  4),
        })

    return (pd.DataFrame(rows)
              .sort_values("f1_mean", ascending=False)
              .reset_index(drop=True))


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune(features: pd.DataFrame, n_iter: int = 40) -> RandomizedSearchCV:
    data       = features.dropna(subset=NUMERIC + [TARGET])
    y          = _binarise(data[TARGET])
    X          = data[CATEGORICAL + NUMERIC]
    pos_weight = (len(y) - int(y.sum())) / max(int(y.sum()), 1)

    skf  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([
        ("preprocess", _preprocessor()),
        ("model", XGBClassifier(
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )),
    ])
    search = RandomizedSearchCV(
        pipe, SEARCH_SPACE, n_iter=n_iter, cv=skf,
        scoring="f1", random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    search.fit(X, y)
    print(f"Best F1 (tuned): {search.best_score_:.4f}")
    print(f"Best params:     {search.best_params_}")
    return search


# ── Train final model ─────────────────────────────────────────────────────────

def train(features: pd.DataFrame):
    """
    Binary classifier: Low vs Elevated.
    Reports F1, ROC-AUC, and PR-AUC.
    """
    data       = features.dropna(subset=NUMERIC + [TARGET])
    y          = _binarise(data[TARGET])
    X          = data[CATEGORICAL + NUMERIC]
    pos_weight = (len(y) - int(y.sum())) / max(int(y.sum()), 1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = _xgb_pipe(pos_weight=pos_weight)
    pipe.fit(X_tr, y_tr)

    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    return pipe, {
        "f1":      round(f1_score(y_te, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_te, y_proba), 4),
        "pr_auc":  round(average_precision_score(y_te, y_proba), 4),
        "report":  classification_report(
                       y_te, y_pred,
                       target_names=["Low", "Elevated"],
                       output_dict=True, zero_division=0,
                   ),
        "note": "Elevated = Medium + High collapsed (n=10, n=7 — too few for 3-class).",
    }


# ── SHAP interpretability ─────────────────────────────────────────────────────

def shap_ranking(pipe: Pipeline, X_sample: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    X_t       = pipe.named_steps["preprocess"].transform(X_sample)
    xgb       = pipe.named_steps["model"]
    ohe_names = (pipe.named_steps["preprocess"]
                     .named_transformers_["cat"]
                     .get_feature_names_out(CATEGORICAL).tolist())
    feat_names = ohe_names + NUMERIC

    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_t)
    mean_abs    = np.abs(shap_values).mean(axis=0)

    return (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def save(pipe, path: str = "models/domain_classifier.joblib"):
    joblib.dump({"pipeline": pipe}, path)

def load(path: str = "models/domain_classifier.joblib"):
    return joblib.load(path)["pipeline"]


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(pipe: Pipeline, record: dict) -> dict:
    row   = pd.DataFrame([record])[CATEGORICAL + NUMERIC]
    proba = float(pipe.predict_proba(row)[0, 1])
    return {
        "predicted":   "Elevated" if proba >= 0.5 else "Low",
        "probability": round(proba, 4),
    }
