"""
Feature engineering for the CVE ransomware-risk predictor.

Actual columns in 2_cve_vulnerabilities.csv (verified from cyber_threat.py):
    dateAdded, dueDate, vendorProject, cwes, knownRansomwareCampaignUse, shortDescription
"""

import re
import numpy as np
import pandas as pd

# CWE classes — same taxonomy as the EDA CWE_NAMES dict in cyber_threat.py
MEMORY_CWE      = {"CWE-416", "CWE-119", "CWE-787"}
INJECTION_CWE   = {"CWE-78", "CWE-94", "CWE-89", "CWE-77", "CWE-20"}
AUTH_CWE        = {"CWE-287", "CWE-306", "CWE-798"}
PATH_CWE        = {"CWE-22", "CWE-434"}
DESERIAL_CWE    = {"CWE-502"}

# Keyword sets — same HIGH_RISK / MEDIUM_RISK from cyber_threat.py's NLP plot
HIGH_RISK_KW = {
    "injection", "overflow", "execution", "escalation", "bypass",
    "traversal", "deserialization", "command", "spoofing", "disclosure",
}
MEDIUM_RISK_KW = {
    "authentication", "authorization", "credentials", "password",
    "privilege", "improper", "uncontrolled", "insufficient", "missing",
}

CATEGORICAL = ["vendor_bucket", "cwe_class"]
NUMERIC     = ["days_to_remediate", "high_risk_kw", "medium_risk_kw", "desc_word_count"]
TEXT_COL    = "shortDescription"
TARGET      = "knownRansomwareCampaignUse"


def _cwe_class(cwe: str) -> str:
    cwe = str(cwe).strip()
    if cwe in MEMORY_CWE:    return "memory_safety"
    if cwe in INJECTION_CWE: return "injection"
    if cwe in AUTH_CWE:      return "auth_bypass"
    if cwe in PATH_CWE:      return "path_traversal"
    if cwe in DESERIAL_CWE:  return "deserialization"
    return "other"


def _kw_count(text: str, vocab: set) -> int:
    words = set(re.findall(r"[a-z]{4,}", str(text).lower()))
    return len(words & vocab)


def build_cve_features(df: pd.DataFrame, top_vendors: int = 20) -> pd.DataFrame:
    """
    Input : raw 2_cve_vulnerabilities.csv DataFrame (dates already parsed)
    Output: feature DataFrame with TARGET column and raw shortDescription intact
    """
    d = df.copy()

    # ── date parse if not already done ───────────────────────────────────
    for col in ("dateAdded", "dueDate"):
        if not np.issubdtype(d[col].dtype, np.datetime64):
            d[col] = pd.to_datetime(d[col], errors="coerce")

    # ── days between CISA add date and remediation deadline ───────────────
    d["days_to_remediate"] = (d["dueDate"] - d["dateAdded"]).dt.days

    # ── CWE class ─────────────────────────────────────────────────────────
    d["cwe_class"] = d["cwes"].apply(_cwe_class)

    # ── vendor bucket: keep top-N, lump rest as 'Other' ──────────────────
    top = d["vendorProject"].value_counts().head(top_vendors).index
    d["vendor_bucket"] = d["vendorProject"].where(d["vendorProject"].isin(top), other="Other")

    # ── NLP keyword counts (same vocab as cyber_threat.py plot 9.1) ───────
    desc = d[TEXT_COL].fillna("")
    d["high_risk_kw"]   = desc.apply(lambda t: _kw_count(t, HIGH_RISK_KW))
    d["medium_risk_kw"] = desc.apply(lambda t: _kw_count(t, MEDIUM_RISK_KW))
    d["desc_word_count"] = desc.apply(lambda t: len(str(t).split()))

    out = d[CATEGORICAL + NUMERIC].copy()
    out[TEXT_COL] = desc          # kept for TF-IDF in the model pipeline
    out[TARGET]   = d[TARGET]
    return out
