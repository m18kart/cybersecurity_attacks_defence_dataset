"""
OTX Pulse Clustering — K-Means on TF-IDF

Discovers natural attack campaign groupings from 2,365 OTX pulses
without using any labels. Clusters are interpreted post-hoc using
top TF-IDF terms, dominant Malware_Families, and Industries.

Pipeline:
    1. TF-IDF on Title + Description + Tags
    2. Elbow method + Silhouette score → optimal k
    3. K-Means clustering
    4. PCA (2D) for visualization
    5. Cluster profiling: top terms + dominant metadata
    6. Cross-reference with ATT&CK tagger output (optional)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

RANDOM_STATE   = 42
TFIDF_FEATURES = 1000
MIN_K          = 2
MAX_K          = 15


def _build_corpus(df: pd.DataFrame) -> pd.Series:
    """Concatenate Title + Description + Tags into a single text field per pulse."""
    return (
        df["Title"].fillna("") + " " +
        df["Description"].fillna("") + " " +
        df["Tags"].fillna("")
    ).astype(str)


class OTXClusterer:
    """
    K-Means clustering on OTX pulse text.
    Trained unsupervised; validated via silhouette score and
    cluster coherence against Malware_Families ground truth.
    """

    def __init__(self, max_tfidf: int = TFIDF_FEATURES):
        self.tfidf      = TfidfVectorizer(
            max_features=max_tfidf,
            stop_words="english",
            ngram_range=(1, 2),      # bigrams capture "lateral movement", "command execution"
            min_df=3,                # ignore very rare terms
            sublinear_tf=True,       # log(1+tf) dampens high-frequency tokens
        )
        self.kmeans: KMeans | None = None
        self.svd: TruncatedSVD | None = None   # for dense projection (needed for silhouette on sparse)
        self.pca: PCA | None = None
        self.k: int = 0
        self._X_tfidf = None
        self._X_reduced = None

    # ── fit TF-IDF + reduce dimensionality ───────────────────────────────

    def _fit_transform(self, corpus: pd.Series):
        X = self.tfidf.fit_transform(corpus)
        # LSA via TruncatedSVD: reduces sparse TF-IDF to 100 dense dims
        # improves K-Means performance on high-dimensional sparse data
        self.svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
        X_reduced = self.svd.fit_transform(X)
        X_reduced = normalize(X_reduced)   # L2 normalise after SVD
        return X, X_reduced

    # ── elbow + silhouette to pick k ─────────────────────────────────────

    def find_optimal_k(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sweeps k from MIN_K to MAX_K.
        Returns DataFrame with inertia and silhouette score per k.
        Use the elbow in inertia + peak in silhouette to pick k.
        """
        corpus  = _build_corpus(df)
        _, X    = self._fit_transform(corpus)

        rows = []
        for k in range(MIN_K, MAX_K + 1):
            km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X)
            sil    = silhouette_score(X, labels, sample_size=min(1000, len(X)),
                                      random_state=RANDOM_STATE)
            rows.append({
                "k":          k,
                "inertia":    round(km.inertia_, 2),
                "silhouette": round(sil, 4),
            })
            print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

        return pd.DataFrame(rows)

    # ── fit final model ───────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, k: int) -> dict:
        """
        Fits K-Means with chosen k. Returns cluster assignments and profiles.
        """
        self.k      = k
        corpus      = _build_corpus(df)
        X_sparse, X = self._fit_transform(corpus)
        self._X_tfidf   = X_sparse
        self._X_reduced = X

        self.kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels      = self.kmeans.fit_predict(X)

        sil = silhouette_score(X, labels, sample_size=min(1000, len(X)),
                               random_state=RANDOM_STATE)

        # PCA → 2D for visualization
        self.pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords   = self.pca.fit_transform(X)

        df_out = df.copy()
        df_out["cluster"]  = labels
        df_out["pca_x"]    = coords[:, 0]
        df_out["pca_y"]    = coords[:, 1]

        profiles = self._profile_clusters(df_out, X_sparse)

        return {
            "k":             k,
            "silhouette":    round(sil, 4),
            "cluster_sizes": df_out["cluster"].value_counts().sort_index().to_dict(),
            "profiles":      profiles,
            "df":            df_out,
        }

    # ── cluster profiling ─────────────────────────────────────────────────

    def _top_terms(self, X_sparse, mask, n: int = 10) -> list[str]:
        """Top TF-IDF terms for rows in mask — characterises the cluster topic."""
        centroid   = X_sparse[mask].mean(axis=0)
        centroid   = np.asarray(centroid).flatten()
        top_idx    = centroid.argsort()[::-1][:n]
        vocab      = self.tfidf.get_feature_names_out()
        return vocab[top_idx].tolist()

    def _dominant(self, series: pd.Series, n: int = 3) -> list[str]:
        counts = (
            series.fillna("Unknown")
            .str.split(",").explode()
            .str.strip()
            .replace("", "Unknown")
            .value_counts()
        )
        counts = counts[counts.index != "Unknown"]
        return counts.head(n).index.tolist()

    def _profile_clusters(self, df: pd.DataFrame, X_sparse) -> list[dict]:
        profiles = []
        for c in range(self.k):
            mask     = (df["cluster"] == c).to_numpy()
            top_terms = self._top_terms(X_sparse, mask)
            profile   = {
                "cluster":          c,
                "size":             int(mask.sum()),
                "top_terms":        top_terms,
                "top_malware":      self._dominant(df.loc[mask, "Malware_Families"]),
                "top_industries":   self._dominant(df.loc[mask, "Industries"]),
                "sample_titles":    df.loc[mask, "Title"].head(3).tolist(),
            }
            profiles.append(profile)
        return profiles

    # ── sub-cluster the catch-all bucket ─────────────────────────────────

    def subcluster(self, df: pd.DataFrame, cluster_id: int, k: int = 0) -> dict:
        """
        Re-clusters a single cluster in isolation — used to break open the
        catch-all bucket (typically the largest cluster) into sub-groups.
        If k=0, runs a mini silhouette sweep (k=2..8) to find optimal k.
        """
        mask    = df["cluster"] == cluster_id
        sub_df  = df[mask].copy().reset_index(drop=True)
        corpus  = _build_corpus(sub_df)
        print(f"  Sub-clustering cluster {cluster_id} ({len(sub_df)} pulses)...")

        X = self.tfidf.transform(corpus)
        X_svd = normalize(self.svd.transform(X))

        if k == 0:
            best_k, best_sil = 2, -1
            for kk in range(2, min(9, len(sub_df) // 10 + 2)):
                km  = KMeans(n_clusters=kk, random_state=RANDOM_STATE, n_init=10)
                lbl = km.fit_predict(X_svd)
                sil = silhouette_score(X_svd, lbl, random_state=RANDOM_STATE)
                print(f"    k={kk}  silhouette={sil:.4f}")
                if sil > best_sil:
                    best_sil, best_k = sil, kk
            k = best_k
            print(f"  → Best sub-k: {k}")

        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_svd)
        sub_df["sub_cluster"] = labels

        profiles = []
        for c in range(k):
            cmask = (sub_df["sub_cluster"] == c).to_numpy()
            profiles.append({
                "sub_cluster":    c,
                "size":           int(cmask.sum()),
                "top_terms":      self._top_terms(X, cmask, n=8),
                "top_malware":    self._dominant(sub_df.loc[cmask, "Malware_Families"]),
                "top_industries": self._dominant(sub_df.loc[cmask, "Industries"]),
                "sample_titles":  sub_df.loc[cmask, "Title"].head(3).tolist(),
            })
        return {"parent_cluster": cluster_id, "sub_k": k, "profiles": profiles}

    # ── inference ────────────────────────────────────────────────────────

    def predict_cluster(self, title: str, description: str = "", tags: str = "") -> dict:
        """Assign a new pulse to a cluster."""
        text    = f"{title} {description} {tags}"
        X       = self.tfidf.transform([text])
        X_svd   = normalize(self.svd.transform(X))
        cluster = int(self.kmeans.predict(X_svd)[0])
        return {"cluster": cluster}

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "models/otx_clusterer.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/otx_clusterer.joblib") -> "OTXClusterer":
        return joblib.load(path)
