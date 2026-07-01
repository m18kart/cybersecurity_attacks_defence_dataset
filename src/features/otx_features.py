"""
Feature engineering for the MITRE ATT&CK technique tagger.

Actual columns in 1_otx_threat_intel.csv (verified from cyber_threat.py):
    Pulse_ID, Created, Modified, Indicators_Count, Subscribers, Industries,
    Countries, Malware_Families, Tags, Description, Title, Attack_IDs
"""

import re
from collections import Counter

import pandas as pd

TEXT_COL    = "pulse_text"   # synthesised from Title + Description
TARGET_COL  = "Attack_IDs"


def build_otx_features(df: pd.DataFrame, top_n: int = 15):
    """
    Input : raw 1_otx_threat_intel.csv DataFrame
    Output: (text_series, label_df, technique_list)
        text_series   — Title + Description concatenated per row
        label_df      — binary columns, one per top-N technique
        technique_list — ordered list of technique IDs used as labels
    """
    d = df.copy()

    # ── synthesise text field (Title + Description, same fields EDA uses) ─
    d[TEXT_COL] = (
        d["Title"].fillna("") + " " + d["Description"].fillna("")
    ).astype(str)

    # ── extract T-IDs from Attack_IDs  ────────────────────────────────────
    def _parse_ids(val) -> list:
        return [t.strip() for t in str(val).split(",")
                if re.match(r"T\d{4}", t.strip())]

    d["technique_ids"] = d[TARGET_COL].apply(_parse_ids)

    # ── pick top-N most frequent techniques ──────────────────────────────
    counter = Counter(tid for ids in d["technique_ids"] for tid in ids)
    techniques = [t for t, _ in counter.most_common(top_n)]

    # ── keep only rows that have at least one top-N technique ────────────
    mask = d["technique_ids"].apply(lambda ids: any(t in techniques for t in ids))
    d = d[mask].copy()

    text   = d[TEXT_COL]
    labels = pd.DataFrame(
        {t: d["technique_ids"].apply(lambda ids: int(t in ids)) for t in techniques},
        index=d.index,
    )
    return text, labels, techniques
