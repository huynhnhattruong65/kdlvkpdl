"""
Module: clustering.py
Thành viên 2 - K-Means và DBSCAN Clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.load_data import get_project_root, save_processed_data


# Feature mặc định dùng cho clustering
DEFAULT_FEATURES = [
    "danceability_scaled",
    "energy_scaled",
    "loudness_scaled",
    "speechiness_scaled",
    "acousticness_scaled",
    "instrumentalness_scaled",
    "liveness_scaled",
    "valence_scaled",
    "tempo_scaled",
]


class ClusteringPipeline:
    """Pipeline clustering hoàn chỉnh cho nhạc Spotify."""

    def __init__(self, feature_cols: List[str] = None):
        self.feature_cols = feature_cols or DEFAULT_FEATURES
        self.kmeans = None
        self.dbscan = None
        self.pca = None
        self.best_k = None
        self.cluster_names = {}

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Lấy ma trận feature, tự động chọn cột có sẵn."""
        available = [c for c in self.feature_cols if c in df.columns]
        if not available:
            # Fallback: dùng cột gốc nếu chưa scale
            fallback = [
                "danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness", "valence", "tempo"
            ]
            available = [c for c in fallback if c in df.columns]

        if not available:
            raise ValueError("Không tìm thấy feature nào phù hợp để clustering.")

        self.feature_cols = available
        print(f"Sử dụng {len(available)} features: {available}")
        return df[available].values

    def find_optimal_k(
        self,
        X: np.ndarray,
        k_range: range = range(2, 11),
        random_state: int = 42,
    ) -> Tuple[int, Dict]:
        """
        Tìm k tối ưu bằng Elbow (inertia) + Silhouette Score.
        """
        inertias = []
        silhouettes = []

        print("Đang tìm k tối ưu...")
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sil = silhouette_score(X, labels)
            silhouettes.append(sil)
            print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.4f}")

        # Chọn k có silhouette cao nhất
        best_idx = int(np.argmax(silhouettes))
        best_k = list(k_range)[best_idx]
        self.best_k = best_k

        results = {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouettes": silhouettes,
            "best_k": best_k,
        }
        print(f"\n✅ k tối ưu theo Silhouette: {best_k}")
        return best_k, results

    def fit_kmeans(
        self,
        X: np.ndarray,
        n_clusters: int = None,
        random_state: int = 42,
    ) -> np.ndarray:
        """Train K-Means."""
        if n_clusters is None:
            if self.best_k is None:
                n_clusters, _ = self.find_optimal_k(X)
            else:
                n_clusters = self.best_k

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        labels = self.kmeans.fit_predict(X)
        print(f"✅ K-Means đã train với k={n_clusters}")
        return labels

    def fit_dbscan(
        self,
        X: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> np.ndarray:
        """Train DBSCAN."""
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = self.dbscan.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"✅ DBSCAN: {n_clusters} cụm, {n_noise} điểm noise (eps={eps}, min_samples={min_samples})")
        return labels

    def evaluate(self, X: np.ndarray, labels: np.ndarray, name: str = "Model") -> Dict:
        """Tính các metric clustering; silhouette được ước lượng trên tối đa 10.000 điểm."""
        # Bỏ qua nếu chỉ có 1 cụm hoặc toàn noise
        unique = set(labels)
        if -1 in unique:
            unique.remove(-1)
        if len(unique) < 2:
            print(f"⚠️ {name}: Không đủ cụm để đánh giá.")
            return {}

        # Chỉ lấy điểm không phải noise để tính metric
        mask = labels != -1
        if mask.sum() < 2:
            return {}

        metrics = {
            "silhouette": silhouette_score(
                X[mask],
                labels[mask],
                sample_size=min(10_000, int(mask.sum())),
                random_state=42,
            ),
            "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
            "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
            "n_clusters": len(unique),
            "n_noise": int((labels == -1).sum()),
        }
        print(f"\n📊 Đánh giá {name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return metrics

    def name_clusters(self, df: pd.DataFrame, labels: np.ndarray, feature_cols: List[str] = None) -> Dict:
        """
        Tự động đặt tên cụm dựa trên đặc trưng trung bình.
        """
        if feature_cols is None:
            feature_cols = self.feature_cols

        df_temp = df.copy()
        df_temp["cluster"] = labels

        # Chỉ lấy các cụm hợp lệ
        valid_clusters = [c for c in df_temp["cluster"].unique() if c != -1]
        names = {}

        for c in sorted(valid_clusters):
            subset = df_temp[df_temp["cluster"] == c]
            means = subset[feature_cols].mean()

            # Heuristic đơn giản để đặt tên
            energy = means.get("energy_scaled", means.get("energy", 0))
            valence = means.get("valence_scaled", means.get("valence", 0))
            dance = means.get("danceability_scaled", means.get("danceability", 0))
            acoustic = means.get("acousticness_scaled", means.get("acousticness", 0))

            if energy > 0.5 and dance > 0.5:
                name = "High Energy Dance"
            elif energy > 0.5 and valence < 0:
                name = "Intense / Aggressive"
            elif energy < -0.3 and valence > 0.3:
                name = "Calm / Peaceful"
            elif acoustic > 0.5:
                name = "Acoustic / Soft"
            elif valence < -0.3:
                name = "Sad / Melancholic"
            else:
                name = f"Cluster_{c}"

            names[c] = name
            print(f"  Cụm {c} → {name} (n={len(subset)})")

        self.cluster_names = names
        return names

    def save_models(self):
        """Lưu model đã train."""
        root = get_project_root()
        model_dir = root / "results" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.kmeans is not None:
            joblib.dump(self.kmeans, model_dir / "kmeans_model.joblib")
            print("💾 Đã lưu kmeans_model.joblib")
        if self.dbscan is not None:
            joblib.dump(self.dbscan, model_dir / "dbscan_model.joblib")
            print("💾 Đã lưu dbscan_model.joblib")

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        k: int = None,
        eps: float = 0.8,
        min_samples: int = 10,
    ) -> pd.DataFrame:
        """
        Chạy toàn bộ pipeline clustering và trả về df đã gắn nhãn.
        """
        print("=" * 50)
        print("CLUSTERING PIPELINE (K-Means + DBSCAN)")
        print("=" * 50)

        X = self.prepare_features(df)

        # K-Means
        if k is None:
            k, _ = self.find_optimal_k(X)
        kmeans_labels = self.fit_kmeans(X, n_clusters=k)
        self.evaluate(X, kmeans_labels, "K-Means")
        self.name_clusters(df, kmeans_labels)

        # DBSCAN
        dbscan_labels = self.fit_dbscan(X, eps=eps, min_samples=min_samples)
        self.evaluate(X, dbscan_labels, "DBSCAN")

        # Gắn nhãn vào dataframe
        df_out = df.copy()
        df_out["kmeans_cluster"] = kmeans_labels
        df_out["dbscan_cluster"] = dbscan_labels
        df_out["cluster_name"] = df_out["kmeans_cluster"].map(self.cluster_names)

        # Lưu
        save_processed_data(df_out, "clustered_data.csv")
        self.save_models()

        print("=" * 50)
        print("✅ HOÀN TẤT CLUSTERING")
        print("=" * 50)
        return df_out


if __name__ == "__main__":
    from ..data.load_data import load_processed_data

    try:
        df = load_processed_data("scaled_features.csv")
        pipeline = ClusteringPipeline()
        result = pipeline.run_full_pipeline(df, k=6)
        print(result[["track_name", "kmeans_cluster", "cluster_name"]].head(10))
    except FileNotFoundError as e:
        print(e)
