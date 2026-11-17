"""
Module: recommender.py
Thành viên 3 - Hệ thống gợi ý hoàn chỉnh
Kết hợp: Content-based + Valence-Arousal + Clustering
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

from ..models.content_based import ContentBasedRecommender
from ..data.load_data import get_project_root, load_processed_data


class MusicRecommender:
    """
    Hệ thống gợi ý nhạc tích hợp:
    - Content-based Filtering
    - Lọc theo cảm xúc (Russell’s Valence-Arousal)
    - Lọc theo cluster
    """

    def __init__(self):
        self.df = None
        self.content_model = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame = None, data_file: str = "clustered_data.csv"):
        """
        Load dữ liệu và fit content-based model.
        Ưu tiên dùng clustered_data.csv (đã có emotion + cluster).
        """
        if df is None:
            try:
                df = load_processed_data(data_file)
            except FileNotFoundError:
                # Fallback
                for fallback in ["valence_arousal_mapped.csv", "cleaned_data.csv"]:
                    try:
                        df = load_processed_data(fallback)
                        break
                    except FileNotFoundError:
                        continue
                else:
                    raise FileNotFoundError("Không tìm thấy dữ liệu processed nào.")

        self.df = df.reset_index(drop=True)
        self.content_model = ContentBasedRecommender(metric="cosine")
        self.content_model.fit(self.df)
        self.is_fitted = True
        print("✅ MusicRecommender đã sẵn sàng!")
        return self

    def _ensure_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Hãy gọi .fit() trước khi sử dụng.")

    def recommend_by_track(
        self,
        track_id: str = None,
        track_name: str = None,
        artist_name: str = None,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Gợi ý bài hát tương tự một bài cụ thể.
        """
        self._ensure_fitted()

        if track_id:
            return self.content_model.recommend_by_track_id(track_id, n=n)
        elif track_name:
            return self.content_model.recommend_by_name(track_name, artist_name, n=n)
        else:
            raise ValueError("Cần cung cấp track_id hoặc track_name")

    def recommend_by_emotion(
        self,
        emotion: str,
        n: int = 10,
        min_popularity: int = 20,
    ) -> pd.DataFrame:
        """
        Gợi ý theo cảm xúc (Russell’s model).

        emotion có thể là:
        - "Happy/Excited"
        - "Calm/Relaxed"
        - "Angry/Tense"
        - "Sad/Depressed"
        """
        self._ensure_fitted()

        if "emotion_label" not in self.df.columns:
            raise ValueError("Dữ liệu chưa có emotion_label. Hãy chạy Valence-Arousal mapping trước.")

        subset = self.df[self.df["emotion_label"] == emotion].copy()

        if "popularity" in subset.columns:
            subset = subset[subset["popularity"] >= min_popularity]
            subset = subset.sort_values("popularity", ascending=False)

        if subset.empty:
            print(f"⚠️ Không tìm thấy bài nào với emotion={emotion}")
            return pd.DataFrame()

        return subset.head(n).reset_index(drop=True)

    def recommend_by_cluster(
        self,
        cluster_id: int = None,
        cluster_name: str = None,
        n: int = 10,
    ) -> pd.DataFrame:
        """Gợi ý theo cụm (K-Means)."""
        self._ensure_fitted()

        if "kmeans_cluster" not in self.df.columns:
            raise ValueError("Dữ liệu chưa có kmeans_cluster.")

        if cluster_name and "cluster_name" in self.df.columns:
            subset = self.df[self.df["cluster_name"] == cluster_name]
        elif cluster_id is not None:
            subset = self.df[self.df["kmeans_cluster"] == cluster_id]
        else:
            raise ValueError("Cần cluster_id hoặc cluster_name")

        if "popularity" in subset.columns:
            subset = subset.sort_values("popularity", ascending=False)

        return subset.head(n).reset_index(drop=True)

    def recommend_by_user_profile(
        self,
        liked_track_ids: List[str],
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Gợi ý dựa trên danh sách bài hát yêu thích của user.
        """
        self._ensure_fitted()
        return self.content_model.recommend_by_user_profile(liked_track_ids, n=n)

    def recommend_hybrid(
        self,
        track_id: str = None,
        track_name: str = None,
        emotion: str = None,
        cluster_name: str = None,
        n: int = 10,
        content_weight: float = 0.7,
    ) -> pd.DataFrame:
        """
        Gợi ý lai: Content-based + lọc theo emotion/cluster.

        Cách hoạt động:
        1. Lấy top candidates từ content-based (nhiều hơn n)
        2. Ưu tiên những bài cùng emotion / cluster với bài gốc
        """
        self._ensure_fitted()

        # Bước 1: Content-based
        if track_id or track_name:
            candidates = self.recommend_by_track(
                track_id=track_id,
                track_name=track_name,
                n=n * 3,  # lấy nhiều hơn để lọc
            )
        else:
            candidates = self.df.sample(min(n * 5, len(self.df)))

        # Bước 2: Lọc / boost theo emotion
        if emotion and "emotion_label" in candidates.columns:
            same_emotion = candidates["emotion_label"] == emotion
            candidates.loc[same_emotion, "similarity_score"] = (
                candidates.loc[same_emotion, "similarity_score"] * 1.2
            )

        # Bước 3: Lọc / boost theo cluster
        if cluster_name and "cluster_name" in candidates.columns:
            same_cluster = candidates["cluster_name"] == cluster_name
            candidates.loc[same_cluster, "similarity_score"] = (
                candidates.loc[same_cluster, "similarity_score"] * 1.15
            )

        # Sắp xếp lại và lấy top n
        if "similarity_score" in candidates.columns:
            candidates = candidates.sort_values("similarity_score", ascending=False)

        return candidates.head(n).reset_index(drop=True)

    def get_available_emotions(self) -> List[str]:
        """Liệt kê các emotion có trong dữ liệu."""
        self._ensure_fitted()
        if "emotion_label" in self.df.columns:
            return sorted(self.df["emotion_label"].unique().tolist())
        return []

    def get_available_clusters(self) -> Dict:
        """Liệt kê các cluster."""
        self._ensure_fitted()
        result = {}
        if "kmeans_cluster" in self.df.columns:
            counts = self.df["kmeans_cluster"].value_counts().to_dict()
            if "cluster_name" in self.df.columns:
                names = self.df.groupby("kmeans_cluster")["cluster_name"].first().to_dict()
                result = {k: {"name": names.get(k, f"Cluster_{k}"), "count": v} for k, v in counts.items()}
            else:
                result = {k: {"name": f"Cluster_{k}", "count": v} for k, v in counts.items()}
        return result

    def summary(self):
        """In thông tin tổng quan hệ thống."""
        self._ensure_fitted()
        print("=" * 50)
        print("MUSIC RECOMMENDER SYSTEM - SUMMARY")
        print("=" * 50)
        print(f"Tổng số bài hát : {len(self.df):,}")
        if "emotion_label" in self.df.columns:
            print("\nPhân bố cảm xúc:")
            print(self.df["emotion_label"].value_counts().to_string())
        if "kmeans_cluster" in self.df.columns:
            print("\nPhân bố cluster:")
            print(self.df["kmeans_cluster"].value_counts().sort_index().to_string())
        print("=" * 50)


# ====================== DEMO NHANH ======================
if __name__ == "__main__":
    recommender = MusicRecommender()

    try:
        recommender.fit()
        recommender.summary()

        print("\n--- Demo: Gợi ý theo cảm xúc Happy/Excited ---")
        recs = recommender.recommend_by_emotion("Happy/Excited", n=5)
        cols = [c for c in ["track_name", "artist_name", "popularity", "emotion_label"] if c in recs.columns]
        print(recs[cols])

    except Exception as e:
        print(f"Lỗi: {e}")
        print("Hãy chắc chắn đã chạy preprocessing + valence-arousal + clustering trước.")
