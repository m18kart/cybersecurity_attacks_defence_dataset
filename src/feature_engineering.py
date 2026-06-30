"""
Feature Engineering Module

Transforms cybersecurity threat intelligence datasets into a machine-learning
ready feature matrix.

Author: Karthik Maheswaran
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class FeatureEngineer:
    """
    Creates ML-ready features from threat intelligence datasets.
    """

    def __init__(self):
        self.attack_encoder = MultiLabelBinarizer()
        self.tag_encoder = MultiLabelBinarizer()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def build_features(
        self,
        otx_df: pd.DataFrame,
        cve_df: pd.DataFrame,
        domains_df: pd.DataFrame,
        ips_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates one consolidated feature dataframe.
        """

        features = pd.DataFrame()

        features["cvss_score"] = cve_df.get("cvss_score", pd.Series(dtype=float))
        features["severity"] = cve_df.get("severity", pd.Series(dtype=str))

        features["vendor"] = cve_df.get("vendor", pd.Series(dtype=str))
        features["product"] = cve_df.get("product", pd.Series(dtype=str))

        features["malware_family"] = otx_df.get(
            "malware_family", pd.Series(dtype=str)
        )

        features["industry"] = otx_df.get(
            "target_industry", pd.Series(dtype=str)
        )

        features["country"] = otx_df.get(
            "country", pd.Series(dtype=str)
        )

        features["mitre_count"] = (
            otx_df["mitre_attack"]
            .fillna("")
            .apply(self.count_attack_techniques)
        )

        features["tag_count"] = (
            otx_df["tags"]
            .fillna("")
            .apply(self.count_tags)
        )

        features["description_length"] = (
            cve_df["description"]
            .fillna("")
            .str.len()
        )

        features["keyword_count"] = (
            cve_df["description"]
            .fillna("")
            .apply(self.keyword_count)
        )

        features["exploit_keyword"] = (
            cve_df["description"]
            .fillna("")
            .str.contains(
                "exploit",
                case=False,
                regex=True,
            )
            .astype(int)
        )

        features["ransomware_keyword"] = (
            cve_df["description"]
            .fillna("")
            .str.contains(
                "ransom",
                case=False,
                regex=True,
            )
            .astype(int)
        )

        features["ip_reputation"] = ips_df.get(
            "reputation_score",
            pd.Series(dtype=float),
        )

        features["domain_age"] = domains_df.get(
            "domain_age_days",
            pd.Series(dtype=float),
        )

        return features.fillna(0)

    # ---------------------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------------------

    @staticmethod
    def count_attack_techniques(value: str) -> int:
        if pd.isna(value):
            return 0

        return len(
            [
                x
                for x in str(value).split(",")
                if x.strip()
            ]
        )

    @staticmethod
    def count_tags(value: str) -> int:
        if pd.isna(value):
            return 0

        return len(
            [
                x
                for x in str(value).split(",")
                if x.strip()
            ]
        )

    @staticmethod
    def keyword_count(text: str) -> int:
        if pd.isna(text):
            return 0

        return len(
            re.findall(
                r"[A-Za-z]{4,}",
                text,
            )
        )