"""
Post-training visualization — generates result plots into reports/figures/.

Run after `python -m src.train --model all`:
    python -m src.visualize

Plots generated:
    01_class_distributions.png   — class imbalance across all 4 datasets
    02_shap_cve.png              — SHAP token importance for ransomware model
    03_algo_comparison.png       — LR vs RF vs LightGBM vs XGBoost (PR-AUC + ROC-AUC)
    04_attack_f1_perlabel.png    — per-technique F1 for ATT&CK tagger
    05_ip_anomaly_scores.png     — anomaly score distribution by threat category
    06_kmeans_elbow.png          — elbow + silhouette curve for k selection
    07_cluster_sizes.png         — cluster sizes with archetype labels
    08_shap_attack_heatmap.png   — top SHAP tokens per ATT&CK technique (heatmap)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_DIR    = os.environ.get("CYBERDEFENSE_DATA", str(_REPO_ROOT / "data" / "raw"))
MODEL_DIR   = str(_REPO_ROOT / "models")
FIGURES_DIR = str(_REPO_ROOT / "reports" / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL   = "#0D9488"
NAVY   = "#1B2A4A"
AMBER  = "#F59E0B"
RED    = "#EF4444"
SLATE  = "#475569"
MUTED  = "#94A3B8"
BG     = "#F8FAFC"
GREEN  = "#10B981"
PURPLE = "#8B5CF6"

PALETTE = [TEAL, AMBER, RED, PURPLE, GREEN, NAVY, "#F97316", "#06B6D4", "#EC4899"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.color":        "#E2E8F0",
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "text.color":        NAVY,
    "axes.labelcolor":   SLATE,
    "xtick.color":       SLATE,
    "ytick.color":       SLATE,
})

def _save(name: str):
    path = f"{FIGURES_DIR}/{name}"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓  {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Class distributions across datasets
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Class Distributions — What the Models Face", fontsize=14,
                 fontweight="bold", color=NAVY, y=1.02)

    # CVE ransomware
    cve = pd.read_csv(f"{DATA_DIR}/2_cve_vulnerabilities.csv")
    counts = cve["knownRansomwareCampaignUse"].value_counts()
    axes[0].bar(counts.index, counts.values, color=[RED, TEAL], width=0.5)
    axes[0].set_title("CVE Ransomware Label", fontweight="bold", color=NAVY)
    axes[0].set_ylabel("Count")
    for i, (v) in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontsize=11, color=NAVY, fontweight="bold")

    # IP threat category
    ips = pd.read_csv(f"{DATA_DIR}/4_malicious_ips.csv")
    counts2 = ips["Threat_Category"].value_counts()
    axes[1].bar(counts2.index, counts2.values, color=PALETTE[:len(counts2)], width=0.5)
    axes[1].set_title("IP Threat Category", fontweight="bold", color=NAVY)
    for i, v in enumerate(counts2.values):
        axes[1].text(i, v + 1, str(v), ha="center", fontsize=11, color=NAVY, fontweight="bold")

    # ATT&CK top-10 technique frequency
    otx = pd.read_csv(f"{DATA_DIR}/1_otx_threat_intel.csv")
    import re
    from collections import Counter
    all_ids = []
    for val in otx["Attack_IDs"].dropna():
        all_ids.extend([t.strip() for t in str(val).split(",") if re.match(r"T\d{4}", t.strip())])
    top = pd.Series(Counter(all_ids)).nlargest(10)
    axes[2].barh(top.index[::-1], top.values[::-1], color=TEAL, height=0.6)
    axes[2].set_title("Top-10 ATT&CK Technique Frequency", fontweight="bold", color=NAVY)
    axes[2].set_xlabel("Pulse count")

    plt.tight_layout()
    _save("01_class_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 02 — SHAP feature importance (CVE ransomware model)
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_cve():
    path = f"{MODEL_DIR}/ransomware_shap_ranking.csv"
    if not os.path.exists(path):
        print("  ⚠  ransomware_shap_ranking.csv not found — run src.train first")
        return

    df = pd.read_csv(path).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [RED if i < 5 else TEAL for i in range(len(df))]
    bars = ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1], color=colors[::-1], height=0.65)
    ax.set_title("CVE Ransomware Risk — SHAP Feature Importance (Top 15)",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Mean |SHAP| value")

    legend = [mpatches.Patch(color=RED, label="Top 5 — text tokens dominate"),
              mpatches.Patch(color=TEAL, label="Remaining features")]
    ax.legend(handles=legend, loc="lower right", framealpha=0.7)
    ax.text(0.98, 0.02, "All top-15 features are TF-IDF tokens,\nnot structured metadata",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=SLATE, style="italic")

    plt.tight_layout()
    _save("02_shap_cve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Algorithm comparison (CVE model)
# ─────────────────────────────────────────────────────────────────────────────
def plot_algo_comparison():
    path = f"{MODEL_DIR}/ransomware_algo_comparison.csv"
    if not os.path.exists(path):
        print("  ⚠  ransomware_algo_comparison.csv not found — run src.train first")
        return

    df = pd.read_csv(path)
    x   = np.arange(len(df))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, df["pr_auc"],  width=w, color=TEAL,  label="PR-AUC (primary)", zorder=3)
    b2 = ax.bar(x + w/2, df["roc_auc"], width=w, color=AMBER, label="ROC-AUC",           zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_title("CVE Ransomware Risk — Algorithm Comparison (same train/test split)",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.axhline(0.4, color=RED, linewidth=1, linestyle="--", alpha=0.5, label="PR-AUC baseline")

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, color=NAVY, fontweight="bold")

    plt.tight_layout()
    _save("03_algo_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 04 — ATT&CK per-label F1
# ─────────────────────────────────────────────────────────────────────────────
def plot_attack_f1():
    from src.models.attack_tagger import AttackTechniqueTagger
    path = f"{MODEL_DIR}/attack_tagger.joblib"
    if not os.path.exists(path):
        print("  ⚠  attack_tagger.joblib not found — run src.train first")
        return

    tagger = AttackTechniqueTagger.load(path)
    otx    = pd.read_csv(f"{DATA_DIR}/1_otx_threat_intel.csv")
    metrics = tagger.fit(otx)
    per_f1  = metrics["per_label_f1"]

    labels = list(per_f1.keys())
    values = list(per_f1.values())
    colors = [GREEN if v >= 0.45 else TEAL if v >= 0.30 else AMBER for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)
    ax.set_title("ATT&CK Technique Tagger — Per-Label F1 Score",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 0.75)
    ax.axhline(metrics["macro_f1"], color=RED, linewidth=1.5,
               linestyle="--", label=f"Macro F1 = {metrics['macro_f1']:.2f}")
    ax.legend()
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=8.5, color=NAVY)

    legend = [mpatches.Patch(color=GREEN, label="F1 ≥ 0.45 (strong)"),
              mpatches.Patch(color=TEAL,  label="F1 0.30–0.45 (moderate)"),
              mpatches.Patch(color=AMBER, label="F1 < 0.30 (weak — vocab overlap)")]
    ax.legend(handles=legend, loc="upper right", framealpha=0.8, fontsize=9)

    plt.tight_layout()
    _save("04_attack_f1_perlabel.png")


# ─────────────────────────────────────────────────────────────────────────────
# 05 — IP anomaly score distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_ip_anomaly_scores():
    from src.models.ip_anomaly import IPAnomalyDetector
    from src.features.ip_features import build_ip_features, NUMERIC, TARGET

    path = f"{MODEL_DIR}/ip_anomaly.joblib"
    if not os.path.exists(path):
        print("  ⚠  ip_anomaly.joblib not found — run src.train first")
        return

    ips      = pd.read_csv(f"{DATA_DIR}/4_malicious_ips.csv")
    features = build_ip_features(ips)
    model    = IPAnomalyDetector.load(path)
    scores   = -model.pipe.decision_function(features[NUMERIC])

    df = features.copy()
    df["anomaly_score"] = scores

    fig, ax = plt.subplots(figsize=(10, 5))
    cat_colors = {"malware": RED, "clean": TEAL, "unrated": MUTED}
    for cat, grp in df.groupby(TARGET):
        ax.scatter(range(len(grp)), grp["anomaly_score"].values,
                   label=f"{cat} (n={len(grp)})",
                   color=cat_colors.get(cat, NAVY),
                   alpha=0.7, s=40, zorder=3)

    ax.axhline(0, color=NAVY, linewidth=1.2, linestyle="--", alpha=0.5, label="Decision boundary")
    ax.set_title("IP Anomaly Detector — Score Distribution by Threat Category",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("IP index (sorted by category)")
    ax.set_ylabel("Anomaly score (higher = more suspicious)")
    ax.legend(framealpha=0.8)
    ax.text(0.98, 0.97, "Trained on clean IPs only\nLabels used for post-hoc validation",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=SLATE, style="italic")

    plt.tight_layout()
    _save("05_ip_anomaly_scores.png")


# ─────────────────────────────────────────────────────────────────────────────
# 06 — K-Means elbow + silhouette
# ─────────────────────────────────────────────────────────────────────────────
def plot_kmeans_elbow():
    path = f"{MODEL_DIR}/clustering_k_sweep.csv"
    if not os.path.exists(path):
        print("  ⚠  clustering_k_sweep.csv not found — run src.train --model cluster first")
        return

    df  = pd.read_csv(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("K-Means Cluster Selection — Elbow + Silhouette", fontsize=13,
                 fontweight="bold", color=NAVY)

    ax1.plot(df["k"], df["inertia"], marker="o", color=TEAL, linewidth=2, markersize=7)
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Curve", fontweight="bold", color=NAVY)
    ax1.set_xticks(df["k"])

    best_k = df.loc[df["silhouette"].idxmax(), "k"]
    ax2.plot(df["k"], df["silhouette"], marker="o", color=AMBER, linewidth=2, markersize=7)
    ax2.axvline(best_k, color=RED, linewidth=1.5, linestyle="--",
                label=f"Optimal k={int(best_k)}")
    ax2.set_xlabel("k (number of clusters)")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette Score", fontweight="bold", color=NAVY)
    ax2.set_xticks(df["k"])
    ax2.legend()

    plt.tight_layout()
    _save("06_kmeans_elbow.png")


# ─────────────────────────────────────────────────────────────────────────────
# 07 — Cluster sizes with archetype labels
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_sizes():
    path = f"{MODEL_DIR}/otx_cluster_assignments.csv"
    if not os.path.exists(path):
        print("  ⚠  otx_cluster_assignments.csv not found — run src.train --model cluster first")
        return

    ARCHETYPES = {
        0: "Multi-stage RAT",     1: "2025 CVE Exploit",
        2: "Phishing Infra",      3: "Social Eng / ClickFix",
        4: "2023-24 CVE Exploit", 5: "DPRK / North Korea",
        6: "Ransomware / RaaS",   7: "IoT Botnet / DDoS",
        8: "Supply Chain",        9: "Android Banking",
        10: "APT Espionage",      11: "Spear Phishing / APT",
        12: "General Malware*",   13: "Infostealer / MaaS",
    }

    df     = pd.read_csv(path)
    counts = df["cluster"].value_counts().sort_index()
    labels = [f"C{i}: {ARCHETYPES.get(i, str(i))}" for i in counts.index]
    colors = [RED if i == 12 else TEAL for i in counts.index]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(counts)), counts.values, color=colors, width=0.65, zorder=3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_title("OTX Pulse Clustering — Cluster Sizes & Archetypes (k=14)",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylabel("Number of pulses")

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(int(bar.get_height())), ha="center", fontsize=8.5, color=NAVY)

    ax.text(0.98, 0.97, "* Cluster 12 sub-clustered into 8 archetypes",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=SLATE, style="italic")

    plt.tight_layout()
    _save("07_cluster_sizes.png")


# ─────────────────────────────────────────────────────────────────────────────
# 08 — SHAP heatmap: top tokens per ATT&CK technique
# ─────────────────────────────────────────────────────────────────────────────
def plot_attack_shap_heatmap():
    path = f"{MODEL_DIR}/attack_tagger_shap_summary.csv"
    if not os.path.exists(path):
        print("  ⚠  attack_tagger_shap_summary.csv not found — run src.train first")
        return

    df = pd.read_csv(path)

    # Parse top tokens into a presence matrix
    all_tokens = []
    for tokens in df["top_8_tokens"]:
        all_tokens.extend([t.strip() for t in str(tokens).split(",")])
    unique_tokens = list(dict.fromkeys(all_tokens))[:30]  # top 30 unique

    matrix = []
    for _, row in df.iterrows():
        row_tokens = [t.strip() for t in str(row["top_8_tokens"]).split(",")]
        matrix.append([1 if t in row_tokens else 0 for t in unique_tokens])

    mat = np.array(matrix)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(unique_tokens)))
    ax.set_xticklabels(unique_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["technique"], fontsize=9)
    ax.set_title("ATT&CK Tagger — SHAP Top Tokens per Technique",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Top TF-IDF tokens (by SHAP value)")
    ax.set_ylabel("ATT&CK Technique")

    fig.colorbar(im, ax=ax, shrink=0.6, label="Token in top-8")
    plt.tight_layout()
    _save("08_shap_attack_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating result visualizations → {FIGURES_DIR}\n")

    plot_class_distributions()
    plot_shap_cve()
    plot_algo_comparison()
    plot_attack_f1()
    plot_ip_anomaly_scores()
    plot_kmeans_elbow()
    plot_cluster_sizes()
    plot_attack_shap_heatmap()

    print(f"\nDone — {len(list(Path(FIGURES_DIR).glob('*.png')))} figures saved.")
