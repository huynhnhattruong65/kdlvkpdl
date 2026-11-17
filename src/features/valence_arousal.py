"""
Module: valence_arousal.py
Thành viên 1 - Russell’s Valence-Arousal Mapping

Mô hình Russell (1980):
- Trục ngang: Valence (positive ↔ negative)
- Trục dọc  : Arousal / Energy (high ↔ low)

Spotify cung cấp sẵn:
- valence  : 0.0 → 1.0 (tiêu cực → tích cực)
- energy   : 0.0 → 1.0 (thấp → cao)  ← dùng làm Arousal
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

from ..data.load_data import get_project_root, save_processed_data


def map_emotion(valence: float, arousal: float, threshold: float = 0.5) -> str:
    """
    Phân loại cảm xúc theo 4 góc phần tư của Russell.

    Parameters
    ----------
    valence : float
        Giá trị valence (0-1)
    arousal : float
        Giá trị energy/arousal (0-1)
    threshold : float
        Ngưỡng chia (mặc định 0.5)

    Returns
    -------
    str : nhãn cảm xúc
    """
    high_valence = valence >= threshold
    high_arousal = arousal >= threshold

    if high_valence and high_arousal:
        return "Happy/Excited"
    elif high_valence and not high_arousal:
        return "Calm/Relaxed"
    elif not high_valence and high_arousal:
        return "Angry/Tense"
    else:
        return "Sad/Depressed"


def map_quadrant(valence: float, arousal: float, threshold: float = 0.5) -> str:
    """Trả về tên góc phần tư."""
    high_valence = valence >= threshold
    high_arousal = arousal >= threshold

    if high_valence and high_arousal:
        return "Q1_HighValence_HighArousal"
    elif high_valence and not high_arousal:
        return "Q2_HighValence_LowArousal"
    elif not high_valence and high_arousal:
        return "Q3_LowValence_HighArousal"
    else:
        return "Q4_LowValence_LowArousal"


def add_valence_arousal_features(
    df: pd.DataFrame,
    valence_col: str = "valence",
    arousal_col: str = "energy",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Thêm các cột liên quan đến Valence-Arousal vào DataFrame.
    """
    df = df.copy()

    if valence_col not in df.columns or arousal_col not in df.columns:
        raise ValueError(f"Thiếu cột {valence_col} hoặc {arousal_col}")

    # Sao chép để rõ ràng
    df["va_valence"] = df[valence_col]
    df["va_arousal"] = df[arousal_col]

    # Gán nhãn
    df["emotion_label"] = df.apply(
        lambda row: map_emotion(row["va_valence"], row["va_arousal"], threshold),
        axis=1,
    )
    df["va_quadrant"] = df.apply(
        lambda row: map_quadrant(row["va_valence"], row["va_arousal"], threshold),
        axis=1,
    )

    # Thống kê nhanh
    print("\n📊 Phân bố cảm xúc (Russell’s Valence-Arousal):")
    print(df["emotion_label"].value_counts())
    print("\n📊 Phân bố góc phần tư:")
    print(df["va_quadrant"].value_counts())

    return df


def create_valence_arousal_dataset(
    df: pd.DataFrame,
    threshold: float = 0.5,
    save: bool = True,
) -> pd.DataFrame:
    """
    Pipeline đầy đủ: thêm feature Valence-Arousal và lưu file.
    """
    print("=" * 50)
    print("RUSSELL’S VALENCE-AROUSAL MAPPING")
    print("=" * 50)

    df_va = add_valence_arousal_features(df, threshold=threshold)

    if save:
        save_processed_data(df_va, "valence_arousal_mapped.csv")

    print("=" * 50)
    print("✅ HOÀN TẤT VALENCE-AROUSAL MAPPING")
    print("=" * 50)
    return df_va


def get_tracks_by_emotion(
    df: pd.DataFrame,
    emotion: str,
    n: int = 10,
    popularity_min: int = 0,
) -> pd.DataFrame:
    """
    Lấy top n bài hát theo cảm xúc, ưu tiên popularity cao.
    """
    if "emotion_label" not in df.columns:
        raise ValueError("DataFrame chưa có cột emotion_label. Hãy chạy mapping trước.")

    subset = df[df["emotion_label"] == emotion].copy()

    if "popularity" in subset.columns:
        subset = subset[subset["popularity"] >= popularity_min]
        subset = subset.sort_values("popularity", ascending=False)

    return subset.head(n)


if __name__ == "__main__":
    from ..data.load_data import load_processed_data

    try:
        df = load_processed_data("cleaned_data.csv")
        df_va = create_valence_arousal_dataset(df)
        print(df_va[["track_name", "artist_name", "valence", "energy", "emotion_label"]].head(10))
    except FileNotFoundError as e:
        print(e)
