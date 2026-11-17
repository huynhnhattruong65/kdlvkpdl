"""
run_pipeline.py
Script chạy toàn bộ pipeline từ đầu đến cuối.
Chỉ cần đặt file SpotifyAudioFeatures.csv vào data/raw/ rồi chạy:

    python run_pipeline.py
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.load_data import load_raw_data, load_processed_data
from src.data.preprocess import full_preprocess
from src.features.valence_arousal import create_valence_arousal_dataset
from src.models.clustering import ClusteringPipeline
from src.recommendation.recommender import MusicRecommender
from src.utils.visualization import (
    plot_valence_arousal,
    plot_emotion_distribution,
    plot_correlation_heatmap,
    plot_cluster_pca,
)


def main():
    print("\n" + "=" * 60)
    print("🎵 MUSIC RECOMMENDATION SYSTEM - FULL PIPELINE")
    print("=" * 60 + "\n")

    # -------------------------------------------------
    # BƯỚC 1: Load + Preprocess (Thành viên 1)
    # -------------------------------------------------
    print(">>> BƯỚC 1: Load & Preprocess dữ liệu")
    try:
        raw_df = load_raw_data("SpotifyAudioFeatures.csv")
    except FileNotFoundError as e:
        print(e)
        print("\n⚠️  Hãy tải dataset từ Kaggle:")
        print("    https://www.kaggle.com/datasets/tomigelo/spotify-audio-features")
        print("    và đặt file vào: data/raw/SpotifyAudioFeatures.csv")
        return

    processed_df = full_preprocess(raw_df, remove_outliers=False, scale_method="standard")

    # -------------------------------------------------
    # BƯỚC 2: Valence-Arousal Mapping (Thành viên 1)
    # -------------------------------------------------
    print("\n>>> BƯỚC 2: Russell’s Valence-Arousal Mapping")
    va_df = create_valence_arousal_dataset(processed_df, threshold=0.5)

    # Visualization
    plot_correlation_heatmap(va_df)
    plot_emotion_distribution(va_df)
    plot_valence_arousal(va_df, sample=8000)

    # -------------------------------------------------
    # BƯỚC 3: Clustering (Thành viên 2)
    # -------------------------------------------------
    print("\n>>> BƯỚC 3: Clustering (K-Means + DBSCAN)")
    clustering = ClusteringPipeline()
    clustered_df = clustering.run_full_pipeline(
        va_df,
        k=6,           # có thể để None để tự tìm
        eps=0.9,
        min_samples=15,
    )

    # Vẽ cluster
    feature_cols = [c for c in clustering.feature_cols if c in clustered_df.columns]
    plot_cluster_pca(clustered_df, feature_cols, labels_col="kmeans_cluster")

    # -------------------------------------------------
    # BƯỚC 4: Hệ thống gợi ý (Thành viên 3)
    # -------------------------------------------------
    print("\n>>> BƯỚC 4: Khởi tạo hệ thống gợi ý")
    recommender = MusicRecommender()
    recommender.fit(clustered_df)
    recommender.summary()

    # Demo một số gợi ý
    print("\n" + "-" * 40)
    print("DEMO GỢI Ý")
    print("-" * 40)

    print("\n1. Gợi ý theo cảm xúc 'Happy/Excited':")
    recs = recommender.recommend_by_emotion("Happy/Excited", n=5)
    cols = [c for c in ["track_name", "artist_name", "popularity", "emotion_label"] if c in recs.columns]
    if not recs.empty:
        print(recs[cols].to_string(index=False))

    print("\n2. Gợi ý theo cảm xúc 'Sad/Depressed':")
    recs = recommender.recommend_by_emotion("Sad/Depressed", n=5)
    if not recs.empty:
        print(recs[cols].to_string(index=False))

    # Lấy 1 bài ngẫu nhiên để demo content-based
    if "track_name" in clustered_df.columns:
        sample = clustered_df.sample(1).iloc[0]
        print(f"\n3. Gợi ý tương tự bài: {sample.get('track_name', 'N/A')} - {sample.get('artist_name', '')}")
        try:
            recs = recommender.recommend_by_track(track_name=str(sample["track_name"]), n=5)
            cols2 = [c for c in ["track_name", "artist_name", "similarity_score", "emotion_label"] if c in recs.columns]
            print(recs[cols2].to_string(index=False))
        except Exception as e:
            print(f"   (Bỏ qua demo content-based: {e})")

    print("\n" + "=" * 60)
    print("✅ PIPELINE HOÀN TẤT!")
    print("Các file kết quả nằm trong:")
    print("  - data/processed/")
    print("  - results/models/")
    print("  - results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
