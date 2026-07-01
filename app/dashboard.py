"""
Cyber Defense Intelligence Dashboard

Run: streamlit run app/dashboard.py
Requires: models/ directory populated by  python -m src.train --model all
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="Cyber Defense Intelligence", layout="wide")
st.title("🛡️ Cyber Defense Intelligence Engine")
st.caption("Live scoring: domain severity · CVE ransomware risk · ATT&CK technique tagging")

tab1, tab2, tab3 = st.tabs(["Domain Severity", "CVE Ransomware Risk", "ATT&CK Tagger"])

# ── Tab 1: Domain Severity ────────────────────────────────────────────────────
with tab1:
    st.subheader("Malicious Domain Threat-Severity Scorer")
    col1, col2 = st.columns(2)

    with col1:
        domain_name  = st.text_input("Domain", "paypal-secure-login.xyz")
        reputation   = st.slider("Reputation score (VirusTotal-style)", -100, 100, -40)
        tld          = domain_name.rsplit(".", 1)[-1] if "." in domain_name else "com"

    with col2:
        has_numbers = st.checkbox("Contains numbers")
        has_hyphen  = st.checkbox("Contains hyphen", value=True)

    if st.button("Score Domain", key="domain_btn"):
        try:
            from src.models.domain_classifier import load, predict
            from src.features.domain_features import _tld_bucket, _shannon_entropy

            pipe, le = load("models/domain_classifier.joblib")
            record = {
                "tld_bucket":        _tld_bucket(tld),
                "Domain_Length":     len(domain_name),
                "reputation_filled": reputation,
                "has_numbers":       int(has_numbers),
                "has_hyphen":        int(has_hyphen),
                "entropy":           _shannon_entropy(domain_name),
                "digit_ratio":       sum(c.isdigit() for c in domain_name) / max(len(domain_name), 1),
                "is_long_domain":    int(len(domain_name) > 30),
            }
            result = predict(pipe, le, record)
            st.metric("Predicted severity", result["predicted_severity"])
            st.bar_chart(result["probabilities"])
        except FileNotFoundError:
            st.warning("Run `python -m src.train --model domain` first.")

# ── Tab 2: CVE Ransomware Risk ────────────────────────────────────────────────
with tab2:
    st.subheader("CVE Ransomware-Weaponization Risk")
    vendor     = st.text_input("Vendor / project", "Ivanti")
    cwe_class  = st.selectbox("CWE class",
        ["injection", "memory_safety", "auth_bypass", "path_traversal", "deserialization", "other"])
    days       = st.slider("Days to remediate (CISA deadline)", 0, 60, 14)
    description = st.text_area(
        "Short description",
        "A heap-based buffer overflow allows a remote attacker to execute arbitrary code without authentication.",
    )

    if st.button("Score CVE", key="cve_btn"):
        try:
            from src.models.ransomware_risk import RansomwareRiskModel
            import re

            model = RansomwareRiskModel.load("models/ransomware_risk.joblib")
            HIGH_RISK  = {"injection","overflow","execution","bypass","traversal","deserialization"}
            MED_RISK   = {"authentication","privilege","credentials"}
            record = {
                "vendor_bucket":     vendor,
                "cwe_class":         cwe_class,
                "days_to_remediate": days,
                "high_risk_kw":      sum(1 for w in HIGH_RISK if w in description.lower()),
                "medium_risk_kw":    sum(1 for w in MED_RISK  if w in description.lower()),
                "desc_word_count":   len(description.split()),
                "shortDescription":  description,
            }
            result = model.predict(record)
            st.metric("Risk score",  result["ransomware_risk_score"])
            st.metric("Risk tier",   result["risk_tier"])
        except FileNotFoundError:
            st.warning("Run `python -m src.train --model ransomware` first.")

# ── Tab 3: ATT&CK Tagger ──────────────────────────────────────────────────────
with tab3:
    st.subheader("MITRE ATT&CK Technique Tagger")
    pulse_text = st.text_area(
        "Threat intelligence text (paste pulse title + description)",
        "Adversaries used obfuscated PowerShell scripts to establish C2 over HTTPS and exfiltrate credentials.",
        height=120,
    )
    threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.4)

    if st.button("Tag Techniques", key="attack_btn"):
        try:
            from src.models.attack_tagger import AttackTechniqueTagger

            tagger  = AttackTechniqueTagger.load("models/attack_tagger.joblib")
            results = tagger.predict(pulse_text, threshold=threshold)
            if results:
                for r in results:
                    st.write(f"**{r['technique']}** — confidence {r['confidence']:.2f}")
            else:
                st.info("No techniques above threshold. Try lowering it.")
        except FileNotFoundError:
            st.warning("Run `python -m src.train --model attack` first.")

st.divider()
st.caption("Data: AlienVault OTX · CISA KEV · Malicious domain/IP feeds")
