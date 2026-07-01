"""
Train any or all models. Logs metrics to MLflow.

Usage:
    python -m src.train --model all
    python -m src.train --model ip
    python -m src.train --model ransomware
    python -m src.train --model attack
    python -m src.train --model ransomware --tune
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = os.environ.get("CYBERDEFENSE_DATA", str(_REPO_ROOT / "data" / "raw"))
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Model 1: IP Anomaly Detector ─────────────────────────────────────────────

def train_ip():
    from src.features.ip_features import build_ip_features
    from src.models.ip_anomaly import IPAnomalyDetector

    print("\n── IP Anomaly Detector (Isolation Forest) ───────────────────────")
    ips      = pd.read_csv(f"{DATA_DIR}/4_malicious_ips.csv")
    features = build_ip_features(ips)

    with mlflow.start_run(run_name="ip_anomaly_detector"):
        model   = IPAnomalyDetector()
        metrics = model.fit(features)

        mlflow.log_metric("roc_auc",         metrics["roc_auc"])
        mlflow.log_metric("pr_auc",          metrics["pr_auc"])
        mlflow.log_metric("top_k_precision", metrics["top_k_precision"])
        mlflow.log_param("model_type",    "isolation_forest_unsupervised")
        mlflow.log_param("trained_on",    "clean_ips_only")
        mlflow.log_param("contamination", IPAnomalyDetector().contamination)

        model.save(f"{MODEL_DIR}/ip_anomaly.joblib")

        print(f"ROC-AUC: {metrics['roc_auc']:.4f}  |  "
              f"PR-AUC: {metrics['pr_auc']:.4f}  |  "
              f"Top-{metrics['k']} Precision: {metrics['top_k_precision']:.4f}")
        print(f"  ({metrics['note']})")
        print("\n  Top-20 anomalous IPs:")
        print(metrics["ranking_sample"].to_string(index=False))


# ── Model 2: CVE Ransomware-Risk ─────────────────────────────────────────────

def train_ransomware(tune: bool = False):
    from src.features.cve_features import build_cve_features
    from src.models.ransomware_risk import RansomwareRiskModel

    print("\n── CVE Ransomware-Risk Predictor ────────────────────────────────")
    cve = pd.read_csv(f"{DATA_DIR}/2_cve_vulnerabilities.csv")
    cve["dateAdded"] = pd.to_datetime(cve["dateAdded"], errors="coerce")
    cve["dueDate"]   = pd.to_datetime(cve["dueDate"],   errors="coerce")
    features = build_cve_features(cve)

    model = RansomwareRiskModel()
    print("CV ROC-AUC (structured features only):")
    print(f"  {model.cv_score(features)}")

    print("\nAlgorithm comparison (TF-IDF + structured, same split):")
    comparison = model.compare_algorithms(features)
    print(comparison.to_string(index=False))

    if tune:
        print("Running RandomizedSearch (n_iter=40, scoring=PR-AUC)...")
        tune_metrics = model.tune(features, n_iter=40)

    with mlflow.start_run(run_name="ransomware_risk_predictor"):
        metrics = model.fit(features)
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("pr_auc",  metrics["pr_auc"])
        mlflow.log_param("model_type",     "xgboost_tfidf_hybrid")
        mlflow.log_param("primary_metric", "pr_auc")
        mlflow.log_param("tuned", str(tune))
        if tune:
            mlflow.log_metric("tune_pr_auc_cv", tune_metrics["best_pr_auc_cv"])
            mlflow.log_params({f"best_{k}": v for k, v in tune_metrics["best_params"].items()})

        # ── algorithm comparison CSV ──────────────────────────────────────
        comp_path = f"{MODEL_DIR}/ransomware_algo_comparison.csv"
        comparison.to_csv(comp_path, index=False)
        mlflow.log_artifact(comp_path)

        # ── SHAP ranking — saved as CSV artifact ──────────────────────────
        shap_df = model.shap_ranking(features, top_n=15)
        shap_path = f"{MODEL_DIR}/ransomware_shap_ranking.csv"
        shap_df.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path)
        print("\n  SHAP feature ranking (top 15):")
        print(shap_df.to_string(index=False))

        model.save(f"{MODEL_DIR}/ransomware_risk.joblib")
        print(f"\nROC-AUC: {metrics['roc_auc']:.4f}  |  PR-AUC: {metrics['pr_auc']:.4f}")
        print("  (PR-AUC is the primary metric — accuracy misleads under imbalance)")


# ── Model 4: OTX Pulse Clustering ────────────────────────────────────────────

def train_clustering(k: int = 0):
    """
    k=0 means auto-select: run elbow+silhouette sweep first, then fit best k.
    Pass k=N to skip the sweep and fit directly.
    """
    from src.models.otx_clusterer import OTXClusterer

    print("\n── OTX Pulse Clustering (K-Means + LSA + TF-IDF) ───────────────")
    otx = pd.read_csv(f"{DATA_DIR}/1_otx_threat_intel.csv")
    print(f"  Corpus: {len(otx)} pulses")

    clusterer = OTXClusterer()

    if k == 0:
        print(f"  Sweeping k=2..15 (elbow + silhouette):")
        sweep = clusterer.find_optimal_k(otx)
        best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
        print(f"\n  → Optimal k by silhouette: {best_k}")
        sweep_path = f"{MODEL_DIR}/clustering_k_sweep.csv"
        sweep.to_csv(sweep_path, index=False)
    else:
        best_k = k

    print(f"\n  Fitting K-Means with k={best_k}...")
    with mlflow.start_run(run_name="otx_pulse_clustering"):
        metrics = clusterer.fit(otx, k=best_k)

        mlflow.log_metric("silhouette",  metrics["silhouette"])
        mlflow.log_param("k",            metrics["k"])
        mlflow.log_param("model_type",   "kmeans_lsa_tfidf")
        mlflow.log_param("tfidf_ngrams", "(1,2)")

        # save cluster assignments CSV
        cluster_df = metrics["df"][["Pulse_ID", "Title", "cluster", "pca_x", "pca_y",
                                     "Malware_Families", "Industries", "Attack_IDs"]]
        cluster_path = f"{MODEL_DIR}/otx_cluster_assignments.csv"
        cluster_df.to_csv(cluster_path, index=False)
        mlflow.log_artifact(cluster_path)

        clusterer.save(f"{MODEL_DIR}/otx_clusterer.joblib")

        print(f"\n  Silhouette score: {metrics['silhouette']:.4f}")
        print(f"  Cluster sizes: {metrics['cluster_sizes']}")
        print("\n  Cluster profiles:")
        for p in metrics["profiles"]:
            print(f"\n  Cluster {p['cluster']} (n={p['size']})")
            print(f"    Top terms:      {', '.join(p['top_terms'][:6])}")
            print(f"    Top malware:    {', '.join(p['top_malware']) or 'Unknown'}")
            print(f"    Top industries: {', '.join(p['top_industries']) or 'Unknown'}")
            print(f"    Sample titles:")
            for t in p["sample_titles"]:
                print(f"      · {t[:80]}")

def train_attack():
    from src.models.attack_tagger import AttackTechniqueTagger

    print("\n── ATT&CK Technique Tagger ──────────────────────────────────────")
    otx = pd.read_csv(f"{DATA_DIR}/1_otx_threat_intel.csv")

    with mlflow.start_run(run_name="attack_technique_tagger"):
        tagger  = AttackTechniqueTagger(top_n=15)
        metrics = tagger.fit(otx)
        mlflow.log_metric("macro_f1",     metrics["macro_f1"])
        mlflow.log_metric("hamming_loss", metrics["hamming_loss"])
        mlflow.log_param("n_techniques", len(metrics["techniques"]))
        mlflow.log_param("model_type",   "xgboost_multilabel_tfidf")
        tagger.save(f"{MODEL_DIR}/attack_tagger.joblib")
        print(f"Macro F1: {metrics['macro_f1']:.4f}  |  "
              f"Hamming loss: {metrics['hamming_loss']:.4f}")
        print("Per-label F1:")
        for tech, f1 in metrics["per_label_f1"].items():
            print(f"  {tech}: {f1:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["ip", "ransomware", "attack", "cluster", "all"],
        default="all",
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--k", type=int, default=0,
                        help="Number of clusters (0 = auto via silhouette sweep)")
    args = parser.parse_args()

    if args.model in ("ip",         "all"): train_ip()
    if args.model in ("ransomware", "all"): train_ransomware(tune=args.tune)
    if args.model in ("attack",     "all"): train_attack()
    if args.model in ("cluster",    "all"): train_clustering(k=args.k)
