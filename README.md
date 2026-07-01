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

### OTX Pulse Clustering — K-Means discovered attack campaign archetypes

K-Means (k=14, TF-IDF bigrams + LSA) on 2,365 OTX pulses **recovered
meaningful attack campaign taxonomies without any label supervision**.
Cluster validity was confirmed post-hoc by checking alignment with
`Malware_Families` — a field not used during training.

| Cluster | Size | Archetype | Dominant Malware |
|---------|------|-----------|-----------------|
| 0 | 160 | Multi-stage RAT delivery | XWorm, AsyncRAT, Remcos RAT |
| 1 | 135 | 2025 CVE exploitation | Cobalt Strike, VShell, XMRig |
| 2 | 241 | Phishing infrastructure | Rhadamanthys, Tycoon2FA, ValleyRAT |
| 3 | 158 | Social engineering / ClickFix | AsyncRAT, NetSupport RAT |
| 4 | 123 | 2023–24 CVE exploitation | Cobalt Strike, Akira, Mirai |
| 5 | 86 | **DPRK / North Korea ops** | BeaverTail, InvisibleFerret, OtterCookie |
| 6 | 180 | Ransomware / RaaS | LockBit, SystemBC |
| 7 | 67 | IoT botnets / DDoS | Mirai, BADBOX, XMRig |
| 8 | 104 | Supply chain (npm/PyPI) | Shai-Hulud, plain-crypto-js |
| 9 | 111 | Android banking trojans | SparkCat, NGate, SpyNote |
| 10 | 285 | APT espionage | PlugX, POISONPLUG.SHADOW |
| 11 | 82 | Spear phishing / APT | ROKRAT, XenoRAT, CozyCar |
| 12 | 473 | General malware → *sub-clustered* | XMRig, Cobalt Strike, AsyncRAT |
| 13 | 160 | Infostealer / MaaS | Lumma Stealer, Vidar, StealC |

**Cluster 12 sub-clustered (k=8, silhouette-selected):**

| Sub-cluster | Size | Archetype | Dominant Malware |
|-------------|------|-----------|-----------------|
| 12.0 | 49 | Web skimming / Magecart | LummaC2, Latrodectus |
| 12.1 | 22 | Malicious browser extensions | VoidStealer, SpyMax |
| 12.2 | 57 | RAT infrastructure / access brokers | AsyncRAT, VenomRAT, Remcos |
| 12.3 | 30 | Cloud cryptomining (Docker/K8s) | XMRig, GSocket, Sliver |
| 12.4 | 140 | General C2 infrastructure *(residual)* | Cobalt Strike, Latrodectus |
| 12.5 | 96 | Loaders / malware staging | Zloader, LummaC2 |
| 12.6 | 45 | **Linux rootkits + AI-targeted malware** | VoidLink, BCObserver |
| 12.7 | 34 | SEO poisoning | BadIIS, GotoHTTP |

**Total: 21 meaningful attack archetypes discovered without labels.**

**Notable findings:**
- **Cluster 5 (DPRK)** is the tightest cluster — nation-state actor campaigns
  have sufficiently distinct vocabulary that K-Means isolated them completely
  without country or actor labels.
- **Clusters 1 and 4** separated 2023–24 vs 2025 CVE campaigns purely on
  temporal vocabulary (`cve 2024` vs `cve 2025`), even though the attack
  pattern is identical. Demonstrates temporal drift in threat intel corpus.
- **Sub-cluster 12.6 (Linux + AI + rootkits)** reflects an emerging 2025 threat
  pattern not visible in older datasets — AI infrastructure being targeted
  alongside traditional kernel-level persistence.
- **Sub-cluster 12.7 (SEO poisoning)** cleanly isolated from just 34 pulses,
  confirming BadIIS as the dominant indicator.
- Malware families were **not used as input** — their alignment with discovered
  clusters validates the unsupervised approach.


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
