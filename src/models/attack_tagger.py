"""
Model 3 — MITRE ATT&CK Technique Tagger

Target  : Top-15 most frequent Attack_IDs in 1_otx_threat_intel.csv (multi-label)
Approach: TF-IDF over Title + Description, one-vs-rest XGBoost per technique.
Metrics : macro F1, hamming loss, per-label precision/recall.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, f1_score, hamming_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from src.features.otx_features import build_otx_features

RANDOM_STATE   = 42
TFIDF_FEATURES = 2000


class AttackTechniqueTagger:

    def __init__(self, top_n: int = 15, max_tfidf: int = TFIDF_FEATURES):
        self.top_n      = top_n
        self.tfidf      = TfidfVectorizer(max_features=max_tfidf, stop_words="english")
        self.model      = MultiOutputClassifier(
            XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", random_state=RANDOM_STATE,
            )
        )
        self.techniques: list[str] = []

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, otx: pd.DataFrame) -> dict:
        text, labels, self.techniques = build_otx_features(otx, self.top_n)

        X = self.tfidf.fit_transform(text)
        y = labels.to_numpy()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_te)

        # per-label report
        per_label = {}
        for i, tech in enumerate(self.techniques):
            per_label[tech] = classification_report(
                y_te[:, i], y_pred[:, i], output_dict=True, zero_division=0
            )

        return {
            "macro_f1":    round(f1_score(y_te, y_pred, average="macro",   zero_division=0), 4),
            "micro_f1":    round(f1_score(y_te, y_pred, average="micro",   zero_division=0), 4),
            "hamming_loss": round(hamming_loss(y_te, y_pred), 4),
            "per_label_f1": {
                tech: round(per_label[tech]["1"]["f1-score"], 4)
                for tech in self.techniques
            },
            "techniques":  self.techniques,
            "note":        "hamming_loss = fraction of labels incorrectly predicted; lower is better.",
        }

    # ── inference ────────────────────────────────────────────────────────

    def predict(self, text: str, threshold: float = 0.4) -> list[dict]:
        """Returns techniques above confidence threshold, sorted descending."""
        X      = self.tfidf.transform([text])
        probas = self.model.predict_proba(X)   # list of (1,2) per label
        return sorted(
            [{"technique": t, "confidence": round(float(p[0, 1]), 4)}
             for t, p in zip(self.techniques, probas)
             if float(p[0, 1]) >= threshold],
            key=lambda r: -r["confidence"],
        )

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "models/attack_tagger.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/attack_tagger.joblib") -> "AttackTechniqueTagger":
        return joblib.load(path)
