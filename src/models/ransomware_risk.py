"""
Model 2 — CVE Ransomware-Weaponization Risk Predictor

Target  : knownRansomwareCampaignUse == 'Known'  (binary)
Approach: XGBoost + TF-IDF hybrid pipeline.
          Class imbalance handled via scale_pos_weight.
          Evaluation on PR-AUC (not accuracy) because positives are ~30% of data.
          SHAP for feature-level interpretability.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score, classification_report,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.features.cve_features import CATEGORICAL, NUMERIC, TARGET, TEXT_COL

RANDOM_STATE   = 42
CV_FOLDS       = 5
TFIDF_FEATURES = 300


class RansomwareRiskModel:
    """
    Combines structured tabular features with TF-IDF over shortDescription.
    Handles class imbalance with scale_pos_weight (ratio of negatives to positives).
    """

    def __init__(self, max_tfidf: int = TFIDF_FEATURES):
        self.ohe   = OneHotEncoder(handle_unknown="ignore")
        self.tfidf = TfidfVectorizer(max_features=max_tfidf, stop_words="english")
        self.model: XGBClassifier | None = None
        self._pos_weight: float = 1.0

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

    # ── baseline cross-validation ─────────────────────────────────────────

    def cv_score(self, features: pd.DataFrame) -> dict:
        """
        Quick 5-fold CV on structured features only (no TF-IDF) to get a
        stable ROC-AUC estimate before fitting the full pipeline.
        """
        data = features.dropna(subset=NUMERIC).copy()
        y    = self._binarise(data[TARGET])
        pos  = y.sum()
        neg  = len(y) - pos
        spw  = neg / max(pos, 1)

        cat_enc  = self.ohe.fit_transform(data[CATEGORICAL].astype(str))
        num_arr  = data[NUMERIC].fillna(0).to_numpy()
        X        = hstack([cat_enc, csr_matrix(num_arr)]).tocsr()

        clf    = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            scale_pos_weight=spw, eval_metric="logloss", random_state=RANDOM_STATE,
        )
        skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
        return {"roc_auc_mean": round(scores.mean(), 4), "roc_auc_std": round(scores.std(), 4)}

    # ── full fit ──────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame) -> dict:
        """
        Fits the full pipeline (structured + TF-IDF).
        Reports ROC-AUC, PR-AUC, and full classification report.
        PR-AUC is the primary metric — accuracy is misleading under imbalance.
        """
        data = features.dropna(subset=NUMERIC).copy()
        y    = self._binarise(data[TARGET])
        pos  = y.sum()
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
            **getattr(self, "_best_params", {}),
            **{k: v for k, v in {   # defaults, overridden by tuned params
                "n_estimators": 300, "max_depth": 5, "learning_rate": 0.08,
                "subsample": 0.9, "colsample_bytree": 0.9,
            }.items() if k not in getattr(self, "_best_params", {})},
            scale_pos_weight=self._pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )
        self.model.fit(X_tr, y_tr)

        proba = self.model.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)

        precision, recall, _ = precision_recall_curve(y_te, proba)
        return {
            "roc_auc":  round(roc_auc_score(y_te, proba), 4),
            "pr_auc":   round(average_precision_score(y_te, proba), 4),
            "report":   classification_report(y_te, preds, output_dict=True),
            "note":     "PR-AUC is the primary metric given class imbalance.",
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
        """
        RandomizedSearch over XGBoost hyperparameters on structured features
        only (faster than including TF-IDF in every CV fold).
        Applies best params to self.model before the full fit() call.
        """
        data = features.dropna(subset=NUMERIC).copy()
        y    = self._binarise(data[TARGET])
        pos  = int(y.sum())
        neg  = len(y) - pos
        spw  = neg / max(pos, 1)

        # fit OHE on full data so CV folds can transform consistently
        cat_enc = self.ohe.fit_transform(data[CATEGORICAL].astype(str))
        num_arr = data[NUMERIC].fillna(0).to_numpy()
        X       = hstack([cat_enc, csr_matrix(num_arr)]).tocsr()

        clf = XGBClassifier(
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            clf, self.SEARCH_SPACE, n_iter=n_iter, cv=skf,
            scoring="average_precision",   # PR-AUC as tuning objective
            random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
        )
        search.fit(X, y)

        best = search.best_params_
        print(f"  Best PR-AUC (CV, structured only): {search.best_score_:.4f}")
        print(f"  Best params: {best}")

        # store best params so fit() picks them up
        self._best_params = best
        return {"best_pr_auc_cv": round(search.best_score_, 4), "best_params": best}

    # ── SHAP ─────────────────────────────────────────────────────────────

    def shap_ranking(self, features: pd.DataFrame, n_sample: int = 200, top_n: int = 15) -> pd.DataFrame:
        """Returns top-N features ranked by mean |SHAP| value."""
        sample  = features.dropna(subset=NUMERIC).sample(
            min(n_sample, len(features)), random_state=RANDOM_STATE
        )
        X       = self._assemble(sample, fit=False)
        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        ohe_names  = self.ohe.get_feature_names_out(CATEGORICAL).tolist()
        tfidf_names = self.tfidf.get_feature_names_out().tolist()
        feat_names  = ohe_names + NUMERIC + tfidf_names

        mean_abs = np.abs(shap_values).mean(axis=0)
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
