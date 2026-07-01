"""
Feature engineering for IP anomaly detection.

Why network features only (not vote/reputation columns):
    Some clean IPs have Malicious_Votes > 0 and negative Reputation_Score
    (e.g. Reputation_Score=-26 for a labelled-clean IP). Including vote-based
    features teaches Isolation Forest that high malicious votes are "sometimes
    normal", which poisons the clean training distribution and hurts recall.
    Network metadata (ASN, TOR status, country, owner frequency,
    times_submitted) is structurally independent of analyst votes and gives
    a cleaner "what does a normal IP look like" signal.
"""

import numpy as np
import pandas as pd

NATION_STATE_COUNTRIES = {"RU", "CN", "KP", "IR"}
PERMISSIVE_COUNTRIES   = {"NL", "DE", "BG", "VG", "PA"}

NUMERIC = [
    "ASN_filled",
    "is_tor",
    "country_risk_score",
    "owner_freq",
    "Times_Submitted",    # behavioural: frequently submitted = under scrutiny
]

TARGET      = "Threat_Category"
CLEAN_LABEL = "clean"


def _country_risk(country: str) -> int:
    c = str(country).upper()
    if c in NATION_STATE_COUNTRIES: return 2
    if c in PERMISSIVE_COUNTRIES:   return 1
    return 0


def build_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["ASN"]         = pd.to_numeric(d["ASN"],             errors="coerce")
    d["ASN_filled"]  = d["ASN"].fillna(d["ASN"].median())
    d["is_tor"]      = (d["TOR_Node"].astype(str).str.lower().str.strip() == "yes").astype(int)
    d["country_risk_score"] = d["Country"].apply(_country_risk)

    owner_counts    = d["Owner"].value_counts()
    d["owner_freq"] = d["Owner"].map(owner_counts).fillna(0)

    d["Times_Submitted"] = pd.to_numeric(d["Times_Submitted"], errors="coerce").fillna(0)

    out = d[NUMERIC].copy()
    out[TARGET] = d[TARGET].fillna("unrated")
    return out


def split_by_label(features: pd.DataFrame):
    """Train on clean IPs only; evaluate on all 200."""
    X_all   = features[NUMERIC]
    y_all   = features[TARGET]
    X_clean = features.loc[features[TARGET] == CLEAN_LABEL, NUMERIC]
    return X_clean, X_all, y_all
