"""
Model 3 (replacing domain classifier) — IP Anomaly Detector

Approach : Isolation Forest trained exclusively on the 79 clean IPs.
           Scores all 200 IPs; malware IPs should surface as most anomalous.
Evaluation: Since we have ground-truth labels (clean / malware / unrated),
            we validate with ROC-AUC: anomaly score vs (malware == 1).
            This is unsupervised — labels are used only for post-hoc validation,
            not during training.
Why IF?   : Supervised classification has only 14 malware examples — statistically
            insufficient. Isolation Forest learns the "normal" manifold from 79
            clean IPs and flags deviations, which is how real SOC tools work.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features.ip_features import NUMERIC, TARGET, CLEAN_LABEL, split_by_label

RANDOM_STATE = 42

# contamination = expected fraction of outliers in the full dataset
# 14 malware / 200 total ≈ 0.07; we add a small buffer
CONTAMINATION = 0.10


class IPAnomalyDetector:
    """
    Isolation Forest anomaly detector for malicious IP scoring.
    Trained only on clean IPs; validated against malware ground truth.
    """

    def __init__(self, contamination: float = CONTAMINATION, n_estimators: int = 200):
        self.contamination = contamination
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=RANDOM_STATE,
            )),
        ])

    def fit(self, features: pd.DataFrame) -> dict:
        """
        Trains on clean IPs only. Evaluates anomaly scores against
        malware ground truth using ROC-AUC and PR-AUC.
        """
        X_clean, X_all, y_all = split_by_label(features)

        print(f"  Training on {len(X_clean)} clean IPs")
        print(f"  Scoring all {len(X_all)} IPs (clean={len(X_clean)}, "
              f"malware={(y_all=='malware').sum()}, "
              f"unrated={(y_all=='unrated').sum()})")

        # fit on clean only — this is what makes it unsupervised
        self.pipe.fit(X_clean)

        # decision_function: higher = more normal, lower = more anomalous
        # negate so that higher score = more anomalous (matches convention)
        scores_all = -self.pipe.decision_function(X_all)

        # validate against binary malware label
        y_binary = (y_all == "malware").astype(int).to_numpy()

        roc_auc = roc_auc_score(y_binary, scores_all)
        pr_auc  = average_precision_score(y_binary, scores_all)

        # rank all IPs by anomaly score for inspection
        ranking = pd.DataFrame({
            "anomaly_score":  scores_all.round(4),
            TARGET:           y_all.to_numpy(),
        }).sort_values("anomaly_score", ascending=False).reset_index(drop=True)

        # how well does top-K anomaly ranking recover malware?
        k = int((y_binary == 1).sum())   # top-K = number of malware IPs
        top_k_precision = (ranking.head(k)[TARGET] == "malware").mean()

        return {
            "roc_auc":          round(roc_auc, 4),
            "pr_auc":           round(pr_auc, 4),
            "top_k_precision":  round(top_k_precision, 4),
            "k":                k,
            "ranking_sample":   ranking.head(20),
            "note": (
                "Unsupervised: trained on clean IPs only. "
                "Labels used for post-hoc validation, not training."
            ),
        }

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns anomaly scores (higher = more suspicious) for a feature matrix."""
        return -self.pipe.decision_function(X[NUMERIC])

    def predict_record(self, record: dict) -> dict:
        """Score a single IP record dict."""
        row   = pd.DataFrame([record])[NUMERIC]
        score = float(-self.pipe.decision_function(row)[0])
        return {
            "anomaly_score": round(score, 4),
            "verdict":       "Suspicious" if score > 0 else "Normal",
        }

    def save(self, path: str = "models/ip_anomaly.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/ip_anomaly.joblib") -> "IPAnomalyDetector":
        return joblib.load(path)
