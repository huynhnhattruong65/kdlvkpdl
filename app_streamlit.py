"""
Giao diện web cho hệ thống gợi ý nhạc.

Chạy:
    streamlit run app_streamlit.py
"""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.recommendation.recommender import MusicRecommender


EMOTION_COLORS = {
    "Happy/Excited": "#f4b942",
    "Calm/Relaxed": "#42c6a5",
    "Angry/Tense": "#ef6a62",
    "Sad/Depressed": "#7188e8",
}

st.set_page_config(
    page_title="Moodify | Gợi ý âm nhạc",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(88, 75, 181, .20), transparent 34%),
            radial-gradient(ellipse at 88% 8%, rgba(37, 173, 149, .12), transparent 30%);
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, .045);
        border: 1px solid rgba(255, 255, 255, .09);
        border-radius: 14px;
        padding: 16px 18px;
    }
    .hero {
        border: 1px solid rgba(255, 255, 255, .12);
        border-radius: 22px;
        padding: 30px 34px;
        margin: 4px 0 22px;
        background: linear-gradient(115deg, rgba(91, 72, 179, .35), rgba(31, 159, 141, .18));
    }
    .hero-kicker {
        color: #b6b0ff;
        font-size: .82rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 {
        font-size: clamp(2rem, 5vw, 3.35rem);
        margin: .45rem 0 .55rem;
    }
    .hero p {
        color: rgba(255, 255, 255, .76);
        font-size: 1.05rem;
        margin: 0;
        max-width: 760px;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(255, 255, 255, .10);
        border-radius: 16px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_recommender():
    return MusicRecommender().fit()


def chart_layout(fig, height=420):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=55, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e7e8f4"),
        legend_title_text="",
    )
    return fig


def show_recommendations(results, title):
    st.subheader(title)
    if results.empty:
        st.info("Không tìm thấy bài hát phù hợp. Hãy thử đổi điều kiện lọc.")
        return

    st.success(f"Tìm thấy {len(results)} bài hát phù hợp.")
    preferred_columns = [
        "track_name",
        "artist_name",
        "popularity",
        "similarity_score",
        "emotion_label",
        "cluster_name",
        "valence",
        "energy",
        "danceability",
        "acousticness",
        "tempo",
    ]
    visible_columns = [column for column in preferred_columns if column in results.columns]
    chart_column = "similarity_score" if "similarity_score" in results.columns else "popularity"

    chart_col, table_col = st.columns([1, 1.35], gap="large")
    with chart_col:
        if chart_column in results.columns:
            chart_data = results.copy()
            label_column = "track_name" if "track_name" in chart_data.columns else chart_data.index.astype(str)
            chart_data["Bài hát"] = label_column
            fig = px.bar(
                chart_data.iloc[::-1],
                x=chart_column,
                y="Bài hát",
                color="emotion_label" if "emotion_label" in chart_data.columns else None,
                color_discrete_map=EMOTION_COLORS,
                orientation="h",
                title="Độ tương đồng" if chart_column == "similarity_score" else "Độ phổ biến",
                labels={chart_column: "Điểm", "Bài hát": ""},
                hover_data=[
                    column
                    for column in ["artist_name", "popularity", "emotion_label", "similarity_score"]
                    if column in chart_data.columns
                ],
            )
            st.plotly_chart(
                chart_layout(fig, height=max(320, min(520, len(results) * 42))),
                width="stretch",
            )
    with table_col:
        st.dataframe(
            results[visible_columns],
            hide_index=True,
            width="stretch",
            height=440,
        )

    with st.expander("Xem toàn bộ thông tin và audio features"):
        st.dataframe(results, hide_index=True, width="stretch")


try:
    recommender = load_recommender()
except FileNotFoundError as error:
    st.error(f"Chưa tìm thấy dữ liệu đã xử lý: {error}")
    st.info("Hãy đặt dataset vào `data/raw/SpotifyAudioFeatures.csv` rồi chạy `python run_pipeline.py`.")
    st.stop()

df = recommender.df

st.sidebar.markdown("## 🎧 MOODIFY")
st.sidebar.caption("Âm nhạc phù hợp với cảm xúc của bạn")
page = st.sidebar.radio(
    "Điều hướng",
    ["Tổng quan", "Gợi ý âm nhạc", "Khám phá dữ liệu"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown("**Nguồn dữ liệu**")
st.sidebar.caption(f"{len(df):,} bài hát đã qua xử lý")
st.sidebar.caption("Spotify Audio Features · Kaggle")

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Personalized music discovery</div>
      <h1>Âm nhạc đúng với<br>tâm trạng của bạn.</h1>
      <p>Khám phá những giai điệu mới qua cảm xúc, đặc trưng âm thanh và sở thích nghe nhạc của bạn.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if page == "Tổng quan":
    st.subheader("Toàn cảnh thư viện âm nhạc")
    metric_columns = st.columns(4)
    artist_count = df["artist_name"].nunique() if "artist_name" in df.columns else 0
    emotion_count = df["emotion_label"].nunique() if "emotion_label" in df.columns else 0
    cluster_count = df["kmeans_cluster"].nunique() if "kmeans_cluster" in df.columns else 0
    mean_popularity = df["popularity"].mean() if "popularity" in df.columns else 0
    metric_columns[0].metric("Bài hát", f"{len(df):,}")
    metric_columns[1].metric("Nghệ sĩ", f"{artist_count:,}")
    metric_columns[2].metric("Nhóm cảm xúc", f"{emotion_count}")
    metric_columns[3].metric("Độ phổ biến TB", f"{mean_popularity:.1f}/100")

    chart_tabs = st.tabs(["Cảm xúc & cụm nhạc", "Bản đồ Valence – Arousal", "Đặc trưng âm thanh"])
    with chart_tabs[0]:
        emotion_col, cluster_col = st.columns(2, gap="large")
        if "emotion_label" in df.columns:
            emotion_counts = (
                df["emotion_label"]
                .value_counts()
                .rename_axis("Cảm xúc")
                .reset_index(name="Số bài hát")
            )
            fig = px.bar(
                emotion_counts,
                x="Cảm xúc",
                y="Số bài hát",
                color="Cảm xúc",
                color_discrete_map=EMOTION_COLORS,
                title="Phân bố bài hát theo cảm xúc",
            )
            emotion_col.plotly_chart(chart_layout(fig), width="stretch")
        if "kmeans_cluster" in df.columns:
            cluster_data = (
                df.groupby(["kmeans_cluster", "cluster_name"], dropna=False)
                .size()
                .reset_index(name="Số bài hát")
            )
            cluster_data["Tên cụm"] = cluster_data["cluster_name"].fillna("Chưa đặt tên")
            fig = px.bar(
                cluster_data,
                x="Tên cụm",
                y="Số bài hát",
                color="Tên cụm",
                hover_data=["kmeans_cluster"],
                title="Quy mô các cụm K-Means",
            )
            fig.update_layout(showlegend=False)
            cluster_col.plotly_chart(chart_layout(fig), width="stretch")

        if "dbscan_cluster" in df.columns:
            noise_count = int((df["dbscan_cluster"] == -1).sum())
            st.caption(
                f"DBSCAN phát hiện {df['dbscan_cluster'].nunique() - 1:,} cụm "
                f"và {noise_count:,} bài thuộc nhóm nhiễu."
            )

    with chart_tabs[1]:
        required = {"valence", "energy", "emotion_label"}
        if required.issubset(df.columns):
            scatter_data = df.sample(min(8000, len(df)), random_state=42)
            hover_columns = [
                column
                for column in ["track_name", "artist_name", "popularity", "cluster_name"]
                if column in scatter_data.columns
            ]
            fig = px.scatter(
                scatter_data,
                x="valence",
                y="energy",
                color="emotion_label",
                color_discrete_map=EMOTION_COLORS,
                hover_data=hover_columns,
                opacity=0.62,
                title="Mỗi điểm là một bài hát · mẫu hiển thị tối đa 8.000 bài",
                labels={"valence": "Valence · tích cực", "energy": "Arousal · năng lượng"},
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(255,255,255,.55)")
            fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,.55)")
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(chart_layout(fig, height=550), width="stretch")
            st.caption(
                "Russell’s Valence-Arousal: Valence biểu thị mức độ tích cực; "
                "Energy được dùng làm đại diện cho Arousal."
            )
        else:
            st.info("Dữ liệu chưa có đủ valence, energy và emotion_label để vẽ biểu đồ.")

    with chart_tabs[2]:
        audio_features = [
            column
            for column in ["valence", "energy", "danceability", "acousticness", "instrumentalness", "speechiness"]
            if column in df.columns
        ]
        if audio_features and "emotion_label" in df.columns:
            feature_means = (
                df.groupby("emotion_label")[audio_features]
                .mean()
                .rename_axis("Cảm xúc")
                .reset_index()
                .melt(id_vars="Cảm xúc", var_name="Đặc trưng", value_name="Giá trị trung bình")
            )
            fig = px.bar(
                feature_means,
                x="Đặc trưng",
                y="Giá trị trung bình",
                color="Cảm xúc",
                barmode="group",
                color_discrete_map=EMOTION_COLORS,
                title="Audio features trung bình theo cảm xúc",
            )
            st.plotly_chart(chart_layout(fig), width="stretch")

elif page == "Gợi ý âm nhạc":
    st.subheader("Tìm giai điệu tiếp theo")
    st.caption("Chọn một cách khám phá. Kết quả có thể mở rộng để xem toàn bộ metadata và audio features.")
    mode = st.radio(
        "Phương thức gợi ý",
        [
            "Theo cảm xúc",
            "Bài hát tương tự",
            "Theo cụm nhạc",
            "Theo sở thích",
            "Gợi ý kết hợp",
        ],
        horizontal=True,
    )
    n_recommendations = st.slider("Số bài muốn xem", 5, 20, 10)
    result_key = "recommendation_result"
    result_title_key = "recommendation_title"
    result_mode_key = "recommendation_mode"

    if mode == "Theo cảm xúc":
        with st.form("emotion_recommendation_form"):
            emotions = recommender.get_available_emotions()
            emotion = st.selectbox("Chọn tâm trạng", emotions)
            min_popularity = st.slider("Độ phổ biến tối thiểu", 0, 100, 30)
            submitted = st.form_submit_button("✨ Gợi ý bài hát", type="primary")
        if submitted:
            results = recommender.recommend_by_emotion(
                emotion,
                n=n_recommendations,
                min_popularity=min_popularity,
            )
            st.session_state[result_key] = results
            st.session_state[result_title_key] = f"Giai điệu cho tâm trạng: {emotion}"
            st.session_state[result_mode_key] = mode

    elif mode == "Bài hát tương tự":
        with st.form("track_recommendation_form"):
            track_name = st.text_input("Tên bài hát", placeholder="Ví dụ: Shape of You")
            artist_name = st.text_input("Nghệ sĩ (không bắt buộc)", placeholder="Ví dụ: Ed Sheeran")
            submitted = st.form_submit_button("🎵 Tìm bài tương tự", type="primary")
        if submitted:
            if not track_name.strip():
                st.warning("Hãy nhập tên bài hát để bắt đầu.")
            else:
                try:
                    results = recommender.recommend_by_track(
                        track_name=track_name.strip(),
                        artist_name=artist_name.strip() or None,
                        n=n_recommendations,
                    )
                    st.session_state[result_key] = results
                    st.session_state[result_title_key] = f"Bài hát tương tự: {track_name.strip()}"
                    st.session_state[result_mode_key] = mode
                except ValueError as error:
                    st.error(str(error))

    elif mode == "Theo cụm nhạc":
        clusters = recommender.get_available_clusters()
        if not clusters:
            st.info("Dữ liệu hiện tại chưa có nhãn cụm. Hãy chạy lại pipeline clustering.")
        else:
            cluster_options = {
                f"{details['name']} · {details['count']:,} bài · ID {cluster_id}": cluster_id
                for cluster_id, details in sorted(clusters.items())
            }
            with st.form("cluster_recommendation_form"):
                selected_cluster = st.selectbox("Chọn cụm nhạc", list(cluster_options))
                submitted = st.form_submit_button("🎼 Khám phá cụm nhạc", type="primary")
            if submitted:
                cluster_id = cluster_options[selected_cluster]
                results = recommender.recommend_by_cluster(
                    cluster_id=cluster_id,
                    n=n_recommendations,
                )
                st.session_state[result_key] = results
                st.session_state[result_title_key] = f"Bài hát trong cụm: {selected_cluster}"
                st.session_state[result_mode_key] = mode

    elif mode == "Theo sở thích":
        if "track_id" not in df.columns:
            st.info("Dataset không có track_id để tạo hồ sơ sở thích.")
        else:
            popular_tracks = (
                df.nlargest(300, "popularity")
                if "popularity" in df.columns
                else df.head(300)
            )
            track_options = {}
            for _, row in popular_tracks.iterrows():
                track_id = str(row["track_id"])
                track_name = str(row.get("track_name", "Không rõ tên"))
                artist_name = str(row.get("artist_name", "Không rõ nghệ sĩ"))
                track_options[f"{track_name} — {artist_name} [{track_id[:8]}]"] = track_id
            with st.form("profile_recommendation_form"):
                liked_tracks = st.multiselect(
                    "Chọn tối đa 10 bài bạn yêu thích",
                    list(track_options),
                    max_selections=10,
                    placeholder="Tìm và chọn ít nhất một bài hát",
                )
                submitted = st.form_submit_button("💜 Tạo gợi ý cá nhân", type="primary")
            if submitted:
                if not liked_tracks:
                    st.warning("Chọn ít nhất một bài hát để tạo hồ sơ sở thích.")
                else:
                    try:
                        results = recommender.recommend_by_user_profile(
                            [track_options[label] for label in liked_tracks],
                            n=n_recommendations,
                        )
                        st.session_state[result_key] = results
                        st.session_state[result_title_key] = "Gợi ý dựa trên những bài bạn yêu thích"
                        st.session_state[result_mode_key] = mode
                    except ValueError as error:
                        st.error(str(error))

    else:
        emotions = ["Không lọc theo cảm xúc"] + recommender.get_available_emotions()
        cluster_names = sorted(
            name
            for name in df["cluster_name"].dropna().unique().tolist()
        ) if "cluster_name" in df.columns else []
        with st.form("hybrid_recommendation_form"):
            seed_track = st.text_input("Bài hát làm điểm bắt đầu", placeholder="Ví dụ: Blinding Lights")
            target_emotion = st.selectbox("Ưu tiên cảm xúc", emotions)
            target_cluster = st.selectbox("Ưu tiên cụm nhạc", ["Không lọc theo cụm"] + cluster_names)
            submitted = st.form_submit_button("⚡ Tạo gợi ý kết hợp", type="primary")
        if submitted:
            if not seed_track.strip():
                st.warning("Nhập tên bài hát làm điểm bắt đầu.")
            else:
                try:
                    results = recommender.recommend_hybrid(
                        track_name=seed_track.strip(),
                        emotion=None if target_emotion == emotions[0] else target_emotion,
                        cluster_name=None if target_cluster == "Không lọc theo cụm" else target_cluster,
                        n=n_recommendations,
                    )
                    st.session_state[result_key] = results
                    st.session_state[result_title_key] = f"Gợi ý kết hợp từ: {seed_track.strip()}"
                    st.session_state[result_mode_key] = mode
                except ValueError as error:
                    st.error(str(error))

    if (
        st.session_state.get(result_key) is not None
        and st.session_state.get(result_mode_key) == mode
    ):
        show_recommendations(
            st.session_state[result_key],
            st.session_state.get(result_title_key, "Kết quả gợi ý"),
        )
    else:
        st.info("Thiết lập tiêu chí rồi nhấn nút để xem danh sách gợi ý.")

elif page == "Khám phá dữ liệu":
    st.subheader("Khám phá thư viện")
    st.caption("Tìm kiếm bài hát và xem các thông tin audio, cảm xúc, độ phổ biến và cụm nhạc.")
    query = st.text_input("Tìm theo tên bài hát hoặc nghệ sĩ", placeholder="Nhập từ khóa...")
    if query.strip():
        track_mask = (
            df["track_name"].astype(str).str.contains(query.strip(), case=False, na=False)
            if "track_name" in df.columns
            else False
        )
        artist_mask = (
            df["artist_name"].astype(str).str.contains(query.strip(), case=False, na=False)
            if "artist_name" in df.columns
            else False
        )
        matches = df[track_mask | artist_mask].head(100)
        st.caption(f"Hiển thị tối đa 100 kết quả đầu tiên · tìm thấy {len(matches):,} bài.")
        if matches.empty:
            st.info("Không tìm thấy kết quả. Hãy thử từ khóa khác.")
        else:
            st.dataframe(matches, hide_index=True, width="stretch")
    else:
        st.info("Nhập tên bài hát hoặc nghệ sĩ để xem thông tin chi tiết.")
        with st.expander("Xem 50 dòng dữ liệu mẫu"):
            st.dataframe(df.head(50), hide_index=True, width="stretch")

    with st.expander("Tóm tắt các đặc trưng số"):
        numeric_columns = [
            column
            for column in [
                "popularity",
                "danceability",
                "energy",
                "valence",
                "acousticness",
                "instrumentalness",
                "speechiness",
                "liveness",
                "loudness",
                "tempo",
            ]
            if column in df.columns
        ]
        if numeric_columns:
            st.dataframe(df[numeric_columns].describe().T, width="stretch")

st.divider()
st.caption(
    "Moodify · Russell’s Valence-Arousal · K-Means / DBSCAN · Content-based Filtering"
)
