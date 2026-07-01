"""
Model 2 — CVE Ransomware-Weaponization Risk Predictor

Target  : knownRansomwareCampaignUse == 'Known'  (binary)
Approach: XGBoost + TF-IDF hybrid pipeline.
          Class imbalance handled via scale_pos_weight.
          Evaluation on PR-AUC (not accuracy) — positives are ~20% of data.
          Algorithm comparison: LR-L1, Random Forest, LightGBM, XGBoost.
          SHAP for feature-level interpretability.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.features.cve_features import CATEGORICAL, NUMERIC, TARGET, TEXT_COL

RANDOM_STATE   = 42
CV_FOLDS       = 5
TFIDF_FEATURES = 300


class RansomwareRiskModel:
    """
    Combines structured tabular features with TF-IDF over shortDescription.
    Handles class imbalance with scale_pos_weight.
    """

    def __init__(self, max_tfidf: int = TFIDF_FEATURES):
        self.ohe          = OneHotEncoder(handle_unknown="ignore")
        self.tfidf        = TfidfVectorizer(max_features=max_tfidf, stop_words="english")
        self.model        = None
        self._pos_weight  = 1.0
        self._best_params = {}

    # ── internal helpers ──────────────────────────────────────────────────

    def _assemble(self, df: pd.DataFrame, fit: bool) -> csr_matrix:
        cat  = df[CATEGORICAL].astype(str)
        num  = csr_matrix(df[NUMERIC].fillna(0).to_numpy())
        text = df[TEXT_COL].astype(str)
        if fit:
            cat_enc  = self.ohe.fit_transform(cat)
            text_enc = self.tfidf.fit_transform(text)
        else:
            cat_enc  = self.ohe.transform(cat)
            text_enc = self.tfidf.transform(text)
        return hstack([cat_enc, num, text_enc]).tocsr()

    def _binarise(self, series: pd.Series) -> np.ndarray:
        return (series == "Known").astype(int).to_numpy()

    # ── algorithm comparison ──────────────────────────────────────────────

    def compare_algorithms(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Benchmarks four algorithms on identical train/test split with full
        TF-IDF + structured feature matrix. All handle class imbalance the
        same way so the comparison is fair.

        Algorithms:
          - Logistic Regression (L1) — sparse, interpretable baseline
          - Random Forest            — low-variance ensemble, no boosting
          - LightGBM                 — faster boosting, native sparse support
          - XGBoost                  — current production model

        Primary metric: PR-AUC (correct for imbalanced binary problems).
        """
        data = features.dropna(subset=NUMERIC).copy()
        y    = self._binarise(data[TARGET])
        pos  = int(y.sum())
        neg  = len(y) - pos
        spw  = neg / max(pos, 1)

        X_tr_df, X_te_df, y_tr, y_te = train_test_split(
            data, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        X_tr = self._assemble(X_tr_df, fit=True)
        X_te = self._assemble(X_te_df, fit=False)

        candidates = {
            "logistic_regression_l1": LogisticRegression(
                penalty="l1", solver="liblinear", C=0.1,
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300, max_depth=8,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=spw, random_state=RANDOM_STATE, verbose=-1,
            ),
            "xgboost": XGBClassifier(
                **{k: v for k, v in {
                    "n_estimators": 300, "max_depth": 5, "learning_rate": 0.08,
                    "subsample": 0.9, "colsample_bytree": 0.9,
                    **self._best_params,
                }.items()},
                scale_pos_weight=spw,
                eval_metric="logloss", random_state=RANDOM_STATE,
            ),
        }

        rows = []
        for name, clf in candidates.items():
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            rows.append({
                "model":   name,
                "pr_auc":  round(average_precision_score(y_te, proba), 4),
                "roc_auc": round(roc_auc_score(y_te, proba), 4),
            })

        return (pd.DataFrame(rows)
                  .sort_values("pr_auc", ascending=False)
                  .reset_index(drop=True))

    # ── baseline cross-validation ─────────────────────────────────────────

    def cv_score(self, features: pd.DataFrame) -> dict:
        """5-fold CV on structured features only (no TF-IDF) — fast baseline."""
        data = features.dropna(subset=NUMERIC).copy()
        y    = self._binarise(data[TARGET])
        pos  = int(y.sum())
        neg  = len(y) - pos
        spw  = neg / max(pos, 1)

        cat_enc = self.ohe.fit_transform(data[CATEGORICAL].astype(str))
        num_arr = data[NUMERIC].fillna(0).to_numpy()
        X       = hstack([cat_enc, csr_matrix(num_arr)]).tocsr()

        clf    = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            scale_pos_weight=spw, eval_metric="logloss", random_state=RANDOM_STATE,
        )
        skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
        return {
            "roc_auc_mean": round(float(scores.mean()), 4),
            "roc_auc_std":  round(float(scores.std()),  4),
        }

    # ── hyperparameter tuning ─────────────────────────────────────────────

    SEARCH_SPACE = {
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.03, 0.05, 0.08, 0.12],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "n_estimators":     [150, 250, 350, 500],
    }

    def tune(self, features: pd.DataFrame, n_iter: int = 40) -> dict:
        """RandomizedSearch on structured features; best params used in fit()."""
        data = features.dropna(subset=NUMERIC).copy()
        y    = self._binarise(data[TARGET])
        pos  = int(y.sum())
        neg  = len(y) - pos
        spw  = neg / max(pos, 1)

        cat_enc = self.ohe.fit_transform(data[CATEGORICAL].astype(str))
        num_arr = data[NUMERIC].fillna(0).to_numpy()
        X       = hstack([cat_enc, csr_matrix(num_arr)]).tocsr()

        clf    = XGBClassifier(
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            clf, self.SEARCH_SPACE, n_iter=n_iter, cv=skf,
            scoring="average_precision",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
        )
        search.fit(X, y)

        self._best_params = search.best_params_
        print(f"  Best PR-AUC (CV, structured only): {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")
        return {
            "best_pr_auc_cv": round(search.best_score_, 4),
            "best_params":    search.best_params_,
        }

    # ── full fit ──────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame) -> dict:
        """
        Fits full pipeline (structured + TF-IDF).
        PR-AUC is the primary metric — accuracy misleads under imbalance.
        """
        data = features.dropna(subset=NUMERIC).copy()
        y    = self._binarise(data[TARGET])
        pos  = int(y.sum())
        neg  = len(y) - pos
        self._pos_weight = neg / max(pos, 1)
        print(f"  Class balance — positives: {pos} ({pos/len(y)*100:.1f}%), "
              f"negatives: {neg}  →  scale_pos_weight={self._pos_weight:.2f}")

        X_tr_df, X_te_df, y_tr, y_te = train_test_split(
            data, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        X_tr = self._assemble(X_tr_df, fit=True)
        X_te = self._assemble(X_te_df, fit=False)

        self.model = XGBClassifier(
            **{k: v for k, v in {
                "n_estimators": 300, "max_depth": 5, "learning_rate": 0.08,
                "subsample": 0.9, "colsample_bytree": 0.9,
                **self._best_params,
            }.items()},
            scale_pos_weight=self._pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )
        self.model.fit(X_tr, y_tr)

        proba = self.model.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)

        return {
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
            "pr_auc":  round(average_precision_score(y_te, proba), 4),
            "report":  classification_report(y_te, preds, output_dict=True),
            "note":    "PR-AUC is the primary metric given class imbalance.",
        }

    # ── SHAP ─────────────────────────────────────────────────────────────

    def shap_ranking(self, features: pd.DataFrame,
                     n_sample: int = 200, top_n: int = 15) -> pd.DataFrame:
        """Top-N features ranked by mean |SHAP| value."""
        sample = features.dropna(subset=NUMERIC).sample(
            min(n_sample, len(features)), random_state=RANDOM_STATE
        )
        X = self._assemble(sample, fit=False)

        ohe_names   = self.ohe.get_feature_names_out(CATEGORICAL).tolist()
        tfidf_names = self.tfidf.get_feature_names_out().tolist()
        feat_names  = ohe_names + NUMERIC + tfidf_names

        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        mean_abs    = np.abs(shap_values).mean(axis=0)

        return (
            pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ── inference ────────────────────────────────────────────────────────

    def predict(self, record: dict) -> dict:
        row   = pd.DataFrame([record])
        X     = self._assemble(row, fit=False)
        proba = float(self.model.predict_proba(X)[0, 1])
        return {
            "ransomware_risk_score": round(proba, 4),
            "risk_tier": "High" if proba >= 0.66 else "Medium" if proba >= 0.33 else "Low",
        }

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "models/ransomware_risk.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/ransomware_risk.joblib") -> "RansomwareRiskModel":
        return joblib.load(path)
