"""
Feature engineering for the domain severity classifier.

Actual columns in 3_malicious_domains.csv (verified from cyber_threat.py):
    Domain, Domain_Length, TLD, Has_Numbers, Has_Hyphen, Reputation, Threat_Severity
"""

import numpy as np
import pandas as pd

# TLD buckets — same classification logic as cyber_threat.py
NATION_STATE_TLDS  = {"ru", "cn", "ir", "kp"}
BULLETPROOF_TLDS   = {"top", "xyz", "online", "site", "tk", "ml", "ga", "cf", "gq"}
LEGACY_TLDS        = {"com", "net", "org", "ch"}

CATEGORICAL = ["tld_bucket"]
NUMERIC     = [
    "Domain_Length", "reputation_filled",
    "has_numbers", "has_hyphen",
    "entropy", "digit_ratio", "is_long_domain",
]
TARGET = "Threat_Severity"


def _tld_bucket(tld: str) -> str:
    tld = str(tld).lower()
    if tld in NATION_STATE_TLDS:  return "nation_state"
    if tld in BULLETPROOF_TLDS:   return "bulletproof"
    if tld in LEGACY_TLDS:        return "legacy"
    return "other"


def _shannon_entropy(s: str) -> float:
    """High entropy → likely DGA-generated domain."""
    s = str(s)
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return float(-sum(p * np.log2(p) for p in probs))


def build_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : raw 3_malicious_domains.csv DataFrame
    Output: feature DataFrame with TARGET column intact
    """
    d = df.copy()

    # ── coerce types ──────────────────────────────────────────────────────
    d["Domain_Length"] = pd.to_numeric(d["Domain_Length"], errors="coerce")
    d["Reputation"]    = pd.to_numeric(d["Reputation"],    errors="coerce")

    # ── binary flags (Has_Numbers / Has_Hyphen are 'Yes'/'No' strings) ───
    d["has_numbers"] = d["Has_Numbers"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    d["has_hyphen"]  = d["Has_Hyphen"].astype(str).str.strip().str.lower().eq("yes").astype(int)

    # ── TLD bucket ────────────────────────────────────────────────────────
    d["tld_bucket"] = d["TLD"].apply(_tld_bucket)

    # ── text-derived features from Domain column ──────────────────────────
    d["entropy"]     = d["Domain"].astype(str).apply(_shannon_entropy)
    d["digit_ratio"] = d["Domain"].astype(str).apply(
        lambda s: sum(c.isdigit() for c in s) / max(len(s), 1)
    )
    d["is_long_domain"] = (d["Domain_Length"] > 30).astype(int)

    # ── fill reputation NaN with median ──────────────────────────────────
    med = d["Reputation"].median()
    d["reputation_filled"] = d["Reputation"].fillna(med)

    out = d[CATEGORICAL + NUMERIC].copy()
    out[TARGET] = d[TARGET].fillna("Unknown")
    return out
