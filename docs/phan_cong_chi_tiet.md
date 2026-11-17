# 🎵 Dự án: Gợi ý chọn nhạc theo sở thích cá nhân
**Personalized Music Recommendation System**

---

## 📋 1. Mô tả dự án

Hệ thống gợi ý nhạc dựa trên sở thích cá nhân người dùng, kết hợp 3 thành phần chính:

| Thuật toán | Mô tả |
|-----------|-------|
| **Russell’s Valence-Arousal Mapping** | Ánh xạ cảm xúc âm nhạc trên mặt phẳng 2D (Valence – tích cực/tiêu cực, Arousal – năng lượng cao/thấp) |
| **Clustering (K-Means / DBSCAN)** | Phân cụm bài hát theo đặc trưng âm thanh Spotify |
| **Content-based Filtering** | Gợi ý bài hát dựa trên độ tương đồng của các audio features |

### Nguồn dữ liệu
- **Dataset**: [Spotify Audio Features](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features) (Kaggle)
- Số lượng: ~116.000 bài hát
- Các feature chính: `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `instrumentalness`, `liveness`, `speechiness`, `loudness`, `key`, `mode`, `popularity`...

---

## 👥 2. Phân công công việc nhóm 3 người

| Thành viên | Vai trò chính | Module phụ trách | Deadline đề xuất |
|------------|---------------|------------------|------------------|
| **Thành viên 1** | Data Engineer + Emotion Mapping | Tiền xử lý dữ liệu + Russell’s Valence-Arousal | Tuần 1–2 |
| **Thành viên 2** | ML Engineer (Clustering) | K-Means / DBSCAN + Phân tích cụm | Tuần 2–3 |
| **Thành viên 3** | Recommendation Engineer | Content-based Filtering + Hệ thống gợi ý | Tuần 3–5 |

---

## 📁 3. Cấu trúc thư mục dự án (Skeleton)

```
Music_Recommendation_Project/
│
├── data/
│   ├── raw/
│   │   └── SpotifyAudioFeatures.csv          # Dữ liệu gốc từ Kaggle
│   └── processed/
│       ├── cleaned_data.csv                  # Dữ liệu đã làm sạch
│       ├── scaled_features.csv               # Feature đã chuẩn hóa
│       ├── valence_arousal_mapped.csv        # Đã map cảm xúc
│       └── clustered_data.csv                # Đã gắn nhãn cụm
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb        # Thành viên 1
│   ├── 02_Valence_Arousal_Mapping.ipynb      # Thành viên 1
│   ├── 03_Clustering_KMeans_DBSCAN.ipynb     # Thành viên 2
│   ├── 04_Content_Based_Filtering.ipynb      # Thành viên 3
│   └── 05_Full_Recommendation_System.ipynb   # Cả nhóm (tích hợp)
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── valence_arousal.py
│   ├── models/
│   │   ├── clustering.py
│   │   └── content_based.py
│   ├── recommendation/
│   │   └── recommender.py
│   └── utils/
│       ├── visualization.py
│       └── metrics.py
│
├── results/
│   ├── figures/                              # Biểu đồ, heatmap, cluster plots...
│   ├── models/                               # Model đã train (.joblib / .pkl)
│   └── evaluations/                          # Kết quả đánh giá
│
├── docs/
│   ├── thanh_vien_1.md
│   ├── thanh_vien_2.md
│   └── thanh_vien_3.md
│
├── reports/
│   └── final_report.md
│
├── requirements.txt
└── README.md
```

---

## 👤 4. Chi tiết công việc từng thành viên

### 4.1. Thành viên 1 – Data Engineer & Emotion Mapping

**Vai trò**: Thu thập, tiền xử lý dữ liệu và thực hiện **Russell’s Valence-Arousal Mapping**.

#### Checklist công việc

**1. Thu thập & Chuẩn bị dữ liệu**
- [ ] Tải dataset từ Kaggle
- [ ] Đặt file vào `data/raw/SpotifyAudioFeatures.csv`
- [ ] Kiểm tra số dòng, cột, kiểu dữ liệu, missing values
- [ ] Viết hàm `src/data/load_data.py`

**2. Tiền xử lý dữ liệu**
- [ ] Xử lý missing values, duplicates
- [ ] Lọc outlier (nếu cần)
- [ ] Chuẩn hóa (StandardScaler / MinMaxScaler)
- [ ] Tạo `data/processed/cleaned_data.csv` và `scaled_features.csv`
- [ ] Viết module `src/data/preprocess.py`

**3. Russell’s Valence-Arousal Mapping**
- [ ] Sử dụng `valence` (trục Valence) + `energy` (trục Arousal)
- [ ] Ánh xạ mỗi bài hát thành điểm (valence, arousal)
- [ ] Phân loại 4 góc phần tư theo mô hình Russell:
  - High Valence + High Arousal → **Happy / Excited**
  - High Valence + Low Arousal → **Calm / Relaxed**
  - Low Valence + High Arousal → **Angry / Tense**
  - Low Valence + Low Arousal → **Sad / Depressed**
- [ ] Tạo cột `emotion_label` và `va_quadrant`
- [ ] Lưu `data/processed/valence_arousal_mapped.csv`
- [ ] Viết module `src/features/valence_arousal.py`

**4. Exploratory Data Analysis (EDA)**
- [ ] Phân bố các feature quan trọng
- [ ] Correlation heatmap
- [ ] Scatter plot Valence-Arousal plane (màu theo emotion)
- [ ] Top artists / tracks theo từng góc phần tư
- [ ] Lưu biểu đồ vào `results/figures/`

**5. Notebooks phụ trách**
- `notebooks/01_EDA_and_Preprocessing.ipynb`
- `notebooks/02_Valence_Arousal_Mapping.ipynb`

**Deliverables cuối cùng**
1. Dữ liệu sạch + đã chuẩn hóa
2. File đã map Valence-Arousal + emotion label
3. 2 notebook hoàn chỉnh
4. Module Python có thể import
5. Bộ biểu đồ EDA + Valence-Arousal plane

---

### 4.2. Thành viên 2 – Clustering Specialist (K-Means / DBSCAN)

**Vai trò**: Phân cụm bài hát bằng K-Means và DBSCAN.

#### Checklist công việc

**1. Chuẩn bị feature cho Clustering**
- [ ] Sử dụng dữ liệu từ Thành viên 1
- [ ] Chọn bộ feature: energy, valence, danceability, tempo, acousticness, instrumentalness, speechiness, loudness...
- [ ] (Tùy chọn) Giảm chiều bằng PCA để visualization

**2. K-Means Clustering**
- [ ] Xác định số cụm tối ưu bằng **Elbow Method** + **Silhouette Score**
- [ ] Chạy K-Means với k tối ưu
- [ ] Phân tích cluster centroids
- [ ] Gán tên ý nghĩa cho từng cụm (vd: “High Energy Dance”, “Chill Acoustic”, “Sad Ballad”...)
- [ ] Lưu model: `results/models/kmeans_model.joblib`

**3. DBSCAN Clustering**
- [ ] Thử nghiệm các giá trị `eps` và `min_samples`
- [ ] So sánh với K-Means (số cụm, noise points, silhouette)
- [ ] Phân tích các điểm noise
- [ ] Lưu model: `results/models/dbscan_model.joblib`

**4. Đánh giá & Visualization**
- [ ] Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz
- [ ] Scatter plot 2D (PCA hoặc Valence-Arousal) tô màu theo cluster
- [ ] Radar chart / bar chart so sánh đặc trưng giữa các cụm
- [ ] Heatmap cluster centers
- [ ] Lưu hình vào `results/figures/`

**5. Tích hợp với Valence-Arousal**
- [ ] Kiểm tra mối quan hệ giữa cluster và emotion quadrant

**6. Notebooks & Module**
- `notebooks/03_Clustering_KMeans_DBSCAN.ipynb`
- `src/models/clustering.py`

**Deliverables cuối cùng**
1. Model K-Means + DBSCAN đã train
2. Dữ liệu đã gắn nhãn cụm
3. Phân tích ý nghĩa từng cụm
4. Notebook + module Python
5. Bộ visualization + bảng so sánh metric

---

### 4.3. Thành viên 3 – Recommendation Engineer

**Vai trò**: Content-based Filtering + xây dựng hệ thống gợi ý hoàn chỉnh.

#### Checklist công việc

**1. Content-based Filtering**
- [ ] Tính độ tương đồng bằng Cosine Similarity / Euclidean Distance
- [ ] Sử dụng `sklearn.neighbors.NearestNeighbors` hoặc ma trận similarity
- [ ] Hàm gợi ý: input 1 bài hát → top-N bài tương tự

**2. Kết hợp với Valence-Arousal & Clustering**
- [ ] Cho phép lọc theo cảm xúc mong muốn (emotion quadrant)
- [ ] Cho phép lọc theo cluster
- [ ] Hỗ trợ input danh sách bài hát yêu thích → tính vector sở thích trung bình

**3. Xây dựng hệ thống gợi ý**
- [ ] Class `MusicRecommender` trong `src/recommendation/recommender.py`
- [ ] Các method chính:
  - `recommend_by_track(track_id, n=10)`
  - `recommend_by_emotion(emotion, n=10)`
  - `recommend_by_user_profile(liked_tracks, n=10)`
  - `recommend_hybrid(...)`
- [ ] Xử lý edge cases

**4. Đánh giá hệ thống**
- [ ] Precision@K, Recall@K (nếu có)
- [ ] Diversity, Coverage
- [ ] Qualitative evaluation (nghe thử)
- [ ] Lưu kết quả vào `results/evaluations/`

**5. Demo**
- [ ] Notebook demo tương tác
- [ ] (Khuyến khích) Streamlit app đơn giản

**6. Notebooks phụ trách**
- `notebooks/04_Content_Based_Filtering.ipynb`
- `notebooks/05_Full_Recommendation_System.ipynb`

**Deliverables cuối cùng**
1. Module Content-based Filtering
2. Hệ thống gợi ý tích hợp (Valence-Arousal + Clustering + Content-based)
3. Notebook demo đầy đủ
4. Đánh giá định lượng + định tính
5. (Khuyến khích) Demo tương tác

---

## 📅 5. Timeline đề xuất (4–6 tuần)

| Tuần | Công việc chính | Người thực hiện |
|------|-----------------|-----------------|
| **Tuần 1** | Thu thập + tiền xử lý dữ liệu | Thành viên 1 |
| **Tuần 2** | Valence-Arousal Mapping + EDA → bàn giao | Thành viên 1 |
| **Tuần 2–3** | Clustering (K-Means + DBSCAN) | Thành viên 2 |
| **Tuần 3–4** | Content-based Filtering | Thành viên 3 |
| **Tuần 4–5** | Tích hợp hệ thống + đánh giá | Thành viên 3 + cả nhóm |
| **Tuần 5–6** | Viết báo cáo + chuẩn bị demo | Cả nhóm |

---

## 🛠️ 6. Công nghệ sử dụng

- **Ngôn ngữ**: Python 3.9+
- **Thư viện chính**:
  - `pandas`, `numpy`
  - `scikit-learn` (KMeans, DBSCAN, NearestNeighbors, preprocessing)
  - `matplotlib`, `seaborn`, `plotly`
  - `joblib` (lưu model)
- **Tùy chọn**: `streamlit` / `gradio` (demo), `yellowbrick` (visualization clustering)

### Gợi ý file `requirements.txt`
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0
joblib>=1.2.0
streamlit>=1.20.0
yellowbrick>=1.5
```

---

## 📌 7. Hướng dẫn sử dụng nhanh

1. Clone repository về máy
2. Tải dataset từ Kaggle và đặt vào `data/raw/SpotifyAudioFeatures.csv`
3. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
4. Chạy notebook theo thứ tự từ `01` → `05`
5. Mỗi thành viên theo dõi checklist trong phần 4 tương ứng

---

## 📝 8. Ghi chú quan trọng

- Thành viên 1 nên hoàn thành **trước** để Thành viên 2 và 3 có dữ liệu sạch.
- Nên họp nhóm ngắn mỗi tuần để đồng bộ tiến độ.
- Tất cả code nên có docstring và comment rõ ràng.
- Commit thường xuyên lên GitHub theo từng module.

---

**Chúc nhóm làm việc hiệu quả và có sản phẩm chất lượng!** 🎶
