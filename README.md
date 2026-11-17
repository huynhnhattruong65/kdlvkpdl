# kdlvkpdl
Kho du lieu va khai pha du lieu

---

# 🎵 Gợi ý chọn nhạc theo sở thích cá nhân
**Personalized Music Recommendation System**

Hệ thống gợi ý nhạc kết hợp:
- **Russell’s Valence-Arousal Mapping**
- **Clustering (K-Means / DBSCAN)**
- **Content-based Filtering** với Spotify audio features

---

## 📊 Dataset

[Spotify Audio Features – Kaggle](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features)

Tải về và đặt file vào:
```
data/raw/SpotifyAudioFeatures.csv
```

---

## 🚀 Cài đặt & Chạy nhanh

```bash
# 1. Clone repo
git clone <your-repo-url>
cd Music_Recommendation_Project

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Đặt dataset vào data/raw/

# 4. Chạy toàn bộ pipeline
python run_pipeline.py

# 5. (Tùy chọn) Chạy giao diện demo
streamlit run app_streamlit.py
```

---

## 👥 Phân công nhóm 3 người

| Thành viên | Vai trò | Module chính |
|------------|---------|--------------|
| **Thành viên 1** | Data + Emotion Mapping | `src/data/`, `src/features/valence_arousal.py` |
| **Thành viên 2** | Clustering | `src/models/clustering.py` |
| **Thành viên 3** | Recommendation | `src/models/content_based.py`, `src/recommendation/` |

Chi tiết checklist xem trong thư mục `docs/`.

---

## 📁 Cấu trúc project

```
Music_Recommendation_Project/
├── data/
│   ├── raw/                  # Đặt SpotifyAudioFeatures.csv vào đây
│   └── processed/            # Dữ liệu đã xử lý (tự sinh)
├── src/
│   ├── data/                 # load + preprocess
│   ├── features/             # valence_arousal
│   ├── models/               # clustering + content_based
│   ├── recommendation/       # recommender chính
│   └── utils/                # visualization
├── results/
│   ├── figures/              # Biểu đồ
│   ├── models/               # Model đã train
│   └── evaluations/
├── run_pipeline.py           # Chạy toàn bộ pipeline
├── app_streamlit.py          # Demo giao diện
├── requirements.txt
└── README.md
```

---

## 🛠️ Pipeline chính

1. **Preprocess** → làm sạch + chuẩn hóa feature
2. **Valence-Arousal Mapping** → gán nhãn cảm xúc (Happy/Excited, Calm/Relaxed, Angry/Tense, Sad/Depressed)
3. **Clustering** → K-Means + DBSCAN, đặt tên cụm
4. **Content-based Filtering** → Cosine Similarity / NearestNeighbors
5. **Recommender** → kết hợp cả 3 phương pháp

---

## 📌 Cách sử dụng hệ thống gợi ý (code)

```python
from src.recommendation.recommender import MusicRecommender

rec = MusicRecommender()
rec.fit()                          # load dữ liệu processed

# Gợi ý theo cảm xúc
recs = rec.recommend_by_emotion("Happy/Excited", n=10)

# Gợi ý bài tương tự
recs = rec.recommend_by_track(track_name="Shape of You", n=10)

# Gợi ý theo cluster
recs = rec.recommend_by_cluster(cluster_id=2, n=10)

# Gợi ý theo profile user
recs = rec.recommend_by_user_profile(liked_track_ids=["id1", "id2"], n=10)
```

---

## 📅 Timeline đề xuất

| Tuần | Công việc |
|------|-----------|
| 1–2  | Thành viên 1: Preprocess + Valence-Arousal |
| 2–3  | Thành viên 2: Clustering |
| 3–5  | Thành viên 3: Content-based + Tích hợp + Demo |
| 5–6  | Cả nhóm: Báo cáo + hoàn thiện |

---

**Chúc nhóm làm việc vui vẻ và ra được sản phẩm chất lượng!** 🎶
