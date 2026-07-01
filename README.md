# Cyber Defense Intelligence Engine

End-to-end ML pipeline on four real-world threat intelligence datasets:
AlienVault OTX pulses, CISA Known Exploited Vulnerabilities (KEV),
malicious domains, and malicious IPs.

`cyber_threat.py` profiles the threat landscape (EDA + visualization).
This layer turns that analysis into three trained, evaluated, and
interpreted models across three distinct ML paradigms.

---

## Models & Results

| # | Model | Paradigm | Target | Primary Metric | Result |
|---|-------|----------|--------|----------------|--------|
| 1 | IP Anomaly Detector | Unsupervised | Malicious IP detection | Top-14 Precision | **0.50 (7× lift over base rate)** |
| 2 | CVE Ransomware-Risk Predictor | Supervised binary | `knownRansomwareCampaignUse` | PR-AUC | **0.40 (ROC-AUC 0.72)** |
| 3 | ATT&CK Technique Tagger | Multi-label NLP | Top-15 ATT&CK techniques | Macro F1 | **0.32 (Hamming loss 0.28)** |

---

## Key findings

### CVE Ransomware-Risk Predictor — SHAP analysis
SHAP revealed that **all top-15 predictive features are TF-IDF text tokens**,
not structured metadata (vendor, CWE class, remediation window):

| Rank | Token | Mean \|SHAP\| | Interpretation |
|------|-------|--------------|----------------|
| 1 | `ios` | 0.254 | iOS CVEs disproportionately weaponized by ransomware |
| 2 | `server` | 0.198 | Server-side vulnerabilities — high-value targets |
| 3 | `crafted` | 0.164 | "specially crafted input" — classic exploit pattern |
| 4 | `unauthenticated` | 0.097 | No-auth RCE = immediate weaponization risk |
| 5 | `arbitrary` | 0.096 | "arbitrary code execution" — strongest RCE signal |
| 6 | `remote` | 0.077 | Remote exploitability — network-accessible attack surface |

The model learned to detect the **RCE language pattern**:
*"allows a remote unauthenticated attacker to execute arbitrary code"* —
which is the highest-risk CVE archetype in ransomware campaigns.

**Implication:** structured CVE metadata (CWE class, vendor, remediation
deadline) adds marginal value over description text alone. Future work
should focus on richer NLP features (bigrams, BERT embeddings) rather
than additional metadata columns.

### IP Anomaly Detector — feature engineering insight
Adding VirusTotal vote/reputation columns to the Isolation Forest
**degraded** performance (ROC-AUC: 0.73 → 0.64). Root cause: some
labelled-clean IPs carry negative reputation scores and non-zero
malicious votes — analyst labels and VirusTotal reputation disagree
on ~15% of the dataset. Including these features teaches Isolation
Forest that high malicious votes are "sometimes normal", poisoning
the clean training distribution.

Final model uses structurally independent network metadata only:
ASN, TOR status, country risk, owner frequency.

### ATT&CK Technique Tagger — per-label variance
Technique detection quality varies significantly by data density:

| Technique | F1 | Why |
|-----------|-----|-----|
| T1027 (Obfuscation) | 0.60 | High frequency, distinct vocabulary |
| T1566 (Phishing) | 0.52 | Strong lexical signal ("phishing", "lure") |
| T1190 (Exploit Public App) | 0.51 | CVE references, exploit terminology |
| T1059 (Command Interpreter) | 0.13 | Overlaps with many other techniques |
| T1071.001 (C2 over HTTP) | 0.18 | Generic HTTP vocabulary, low discriminability |

Low-F1 techniques share vocabulary with adjacent techniques — a known
limitation of bag-of-words TF-IDF for multi-label NLP. Bigrams or
BERT-based embeddings would help distinguish overlapping technique families.

---

## DS/ML methodology

**Baseline comparison before XGBoost.**
Every supervised model benchmarked against Logistic Regression and
Decision Tree baselines through identical stratified CV. Lift is
quantified, not assumed.

**Stratified k-fold cross-validation.**
Reported as mean ± std across 5 folds — not a single train/test split.

**Hyperparameter tuning.**
`RandomizedSearchCV` (40 iterations) over `max_depth`, `learning_rate`,
`subsample`, `colsample_bytree`, `min_child_weight`, `n_estimators`.
Tuning objective is PR-AUC (not accuracy) for the imbalanced CVE problem.

**Class imbalance handling.**
CVE dataset: 20% positive rate. Model uses `scale_pos_weight=4.0` and
is evaluated on PR-AUC. Accuracy would show 80%+ while the model predicts
nothing useful — PR-AUC exposes this.

**SHAP interpretability.**
`models/ransomware_shap_ranking.csv` — features ranked by mean |SHAP|
value, generated automatically on every training run.

**Data-driven model selection.**
Domain severity classification (162 rows, 7 High / 10 Medium) was
abandoned for supervised learning and replaced with unsupervised anomaly
detection on the IP dataset. The domain dataset is insufficient for
reliable 3-class learning at this sample size.

---

## Project layout

```
cyber_threat.py              EDA + threat landscape visualization
src/
  features/
    cve_features.py          CWE class encoding, vendor bucketing, NLP keyword counts
    ip_features.py           ASN, TOR, country risk, owner frequency
    otx_features.py          Title + Description concat, ATT&CK ID extraction
  models/
    ransomware_risk.py       XGBoost + TF-IDF hybrid, SHAP ranking, tune()
    ip_anomaly.py            Isolation Forest, clean-only training, top-K precision
    attack_tagger.py         Multi-label XGBoost, per-label F1 + hamming loss
  train.py                   Unified CLI + MLflow logging + SHAP CSV auto-save
app/
  dashboard.py               Streamlit live-scoring UI (all 3 models)
models/
  ransomware_risk.joblib
  ip_anomaly.joblib
  attack_tagger.joblib
  ransomware_shap_ranking.csv   ← auto-generated on every train run
```

---

## Quickstart

```bash
pip install -r requirements.txt

python -m src.train --model all                    # train all 3 models
python -m src.train --model ransomware --tune      # with hyperparameter search
streamlit run app/dashboard.py                     # live scoring UI
```

---

## Roadmap

- [ ] SHAP for ATT&CK tagger (per-technique token importance)
- [ ] Bigram TF-IDF or BERT embeddings for the ATT&CK tagger
- [ ] Optuna for Bayesian hyperparameter search on CVE model
- [ ] C++ inference layer via ONNX export (CVE model priority)
