"""
Module: content_based.py
Thành viên 3 - Content-based Filtering sử dụng Spotify audio features
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Union
import joblib
from pathlib import Path

from ..data.load_data import get_project_root


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


class ContentBasedRecommender:
    """
    Content-based Filtering dựa trên độ tương đồng audio features.
    """

    def __init__(self, feature_cols: List[str] = None, metric: str = "cosine"):
        """
        Parameters
        ----------
        metric : str
            "cosine" hoặc "euclidean"
        """
        self.feature_cols = feature_cols or DEFAULT_FEATURES
        self.metric = metric
        self.nn_model = None
        self.df = None
        self.feature_matrix = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str] = None):
        """
        Fit model với dữ liệu.
        """
        self.df = df.copy().reset_index(drop=True)

        if feature_cols:
            self.feature_cols = feature_cols

        available = [c for c in self.feature_cols if c in self.df.columns]
        if not available:
            # Fallback
            fallback = [
                "danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness", "valence", "tempo"
            ]
            available = [c for c in fallback if c in self.df.columns]

        self.feature_cols = available
        self.feature_matrix = self.df[available].values

        # Dùng NearestNeighbors để tìm kiếm nhanh
        algorithm = "brute" if self.metric == "cosine" else "auto"
        self.nn_model = NearestNeighbors(
            n_neighbors=min(50, len(self.df)),
            metric=self.metric,
            algorithm=algorithm,
            n_jobs=-1,
        )
        self.nn_model.fit(self.feature_matrix)

        print(f"✅ ContentBasedRecommender đã fit với {len(self.df):,} bài hát")
        print(f"   Features: {self.feature_cols}")
        print(f"   Metric: {self.metric}")
        return self

    def recommend_by_index(
        self,
        idx: int,
        n: int = 10,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """
        Gợi ý dựa trên index của bài hát trong dataframe.
        """
        if self.nn_model is None:
            raise RuntimeError("Model chưa được fit. Hãy gọi .fit() trước.")

        n_neighbors = n + 1 if exclude_self else n
        distances, indices = self.nn_model.kneighbors(
            [self.feature_matrix[idx]],
            n_neighbors=n_neighbors,
        )

        indices = indices[0]
        distances = distances[0]

        if exclude_self:
            # Bỏ chính nó (thường ở vị trí đầu)
            mask = indices != idx
            indices = indices[mask][:n]
            distances = distances[mask][:n]

        result = self.df.iloc[indices].copy()
        result["similarity_score"] = 1 - distances if self.metric == "cosine" else 1 / (1 + distances)
        result["distance"] = distances
        return result.reset_index(drop=True)

    def recommend_by_track_id(
        self,
        track_id: str,
        n: int = 10,
    ) -> pd.DataFrame:
        """Gợi ý dựa trên track_id."""
        if "track_id" not in self.df.columns:
            raise ValueError("DataFrame không có cột track_id")

        matches = self.df.index[self.df["track_id"] == track_id].tolist()
        if not matches:
            raise ValueError(f"Không tìm thấy track_id: {track_id}")

        return self.recommend_by_index(matches[0], n=n)

    def recommend_by_name(
        self,
        track_name: str,
        artist_name: str = None,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Gợi ý dựa trên tên bài hát (và nghệ sĩ nếu có).
        Tìm kiếm gần đúng.
        """
        mask = self.df["track_name"].str.contains(track_name, case=False, na=False)
        if artist_name and "artist_name" in self.df.columns:
            mask &= self.df["artist_name"].str.contains(artist_name, case=False, na=False)

        matches = self.df[mask]
        if matches.empty:
            raise ValueError(f"Không tìm thấy bài hát: {track_name}")

        # Lấy bài đầu tiên tìm được
        idx = matches.index[0]
        print(f"Đã chọn: {self.df.loc[idx, 'track_name']} - {self.df.loc[idx].get('artist_name', '')}")
        return self.recommend_by_index(idx, n=n)

    def recommend_by_vector(
        self,
        feature_vector: np.ndarray,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Gợi ý dựa trên một vector feature bất kỳ (ví dụ: vector trung bình sở thích user).
        """
        if self.nn_model is None:
            raise RuntimeError("Model chưa được fit.")

        distances, indices = self.nn_model.kneighbors(
            [feature_vector],
            n_neighbors=n,
        )
        result = self.df.iloc[indices[0]].copy()
        result["similarity_score"] = 1 - distances[0] if self.metric == "cosine" else 1 / (1 + distances[0])
        return result.reset_index(drop=True)

    def recommend_by_user_profile(
        self,
        liked_track_ids: List[str],
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Xây dựng profile user từ danh sách bài hát yêu thích,
        sau đó gợi ý các bài tương tự profile đó.
        """
        if "track_id" not in self.df.columns:
            raise ValueError("Cần cột track_id")

        liked = self.df[self.df["track_id"].isin(liked_track_ids)]
        if liked.empty:
            raise ValueError("Không tìm thấy bài hát nào trong danh sách yêu thích")

        # Vector trung bình sở thích
        profile_vector = liked[self.feature_cols].mean().values
        return self.recommend_by_vector(profile_vector, n=n)

    def save(self, filename: str = "content_based_model.joblib"):
        """Lưu model."""
        root = get_project_root()
        path = root / "results" / "models" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"💾 Đã lưu ContentBasedRecommender → {path}")

    @staticmethod
    def load(filename: str = "content_based_model.joblib"):
        """Load model đã lưu."""
        root = get_project_root()
        path = root / "results" / "models" / filename
        return joblib.load(path)


if __name__ == "__main__":
    from ..data.load_data import load_processed_data

    try:
        df = load_processed_data("clustered_data.csv")
        model = ContentBasedRecommender(metric="cosine")
        model.fit(df)

        # Demo: lấy 1 bài ngẫu nhiên
        sample = df.sample(1).iloc[0]
        print(f"\n🎵 Bài gốc: {sample.get('track_name', '')} - {sample.get('artist_name', '')}")
        recs = model.recommend_by_index(sample.name, n=5)
        print(recs[["track_name", "artist_name", "similarity_score"]])
    except FileNotFoundError as e:
        print(e)
