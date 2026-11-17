"""
Module: preprocess.py
Thành viên 1 - Tiền xử lý dữ liệu Spotify Audio Features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Optional
import joblib
from pathlib import Path

from .load_data import get_project_root, save_processed_data


# Các cột audio features chính của Spotify
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Cột dùng cho clustering / recommendation (bỏ key, mode vì categorical)
NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch cơ bản:
    - Xóa dòng trùng track_id
    - Xóa dòng thiếu giá trị quan trọng
    - Reset index
    """
    df = df.copy()
    print(f"Số dòng ban đầu: {len(df):,}")

    # Xóa trùng lặp theo track_id nếu có
    if "track_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["track_id"], keep="first")
        print(f"Đã xóa {before - len(df):,} dòng trùng track_id")

    # Xóa dòng thiếu các feature quan trọng
    essential = [c for c in NUMERIC_FEATURES if c in df.columns]
    before = len(df)
    df = df.dropna(subset=essential)
    print(f"Đã xóa {before - len(df):,} dòng thiếu giá trị")

    df = df.reset_index(drop=True)
    print(f"Số dòng sau cleaning: {len(df):,}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """Loại bỏ outlier bằng phương pháp IQR (tùy chọn)."""
    df = df.copy()
    mask = pd.Series(True, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask &= (df[col] >= lower) & (df[col] <= upper)

    removed = (~mask).sum()
    print(f"Đã loại bỏ {removed:,} outlier bằng IQR")
    return df[mask].reset_index(drop=True)


def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    method: str = "standard",
    save_scaler: bool = True,
) -> Tuple[pd.DataFrame, object]:
    """
    Chuẩn hóa các feature số.

    Parameters
    ----------
    method : str
        "standard" (StandardScaler) hoặc "minmax" (MinMaxScaler)
    """
    if feature_cols is None:
        feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    df = df.copy()
    X = df[feature_cols].values

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method phải là 'standard' hoặc 'minmax'")

    X_scaled = scaler.fit_transform(X)

    # Tạo dataframe mới với tên cột _scaled
    scaled_df = df.copy()
    for i, col in enumerate(feature_cols):
        scaled_df[f"{col}_scaled"] = X_scaled[:, i]

    if save_scaler:
        root = get_project_root()
        model_dir = root / "results" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, model_dir / "scaler.joblib")
        print("💾 Đã lưu scaler vào results/models/scaler.joblib")

    return scaled_df, scaler


def full_preprocess(
    df: pd.DataFrame,
    remove_outliers: bool = False,
    scale_method: str = "standard",
) -> pd.DataFrame:
    """
    Pipeline tiền xử lý đầy đủ.
    """
    print("=" * 50)
    print("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 50)

    df = basic_cleaning(df)

    if remove_outliers:
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        df = remove_outliers_iqr(df, cols)

    df, scaler = scale_features(df, method=scale_method)

    # Lưu cleaned data
    save_processed_data(df, "cleaned_data.csv")

    # Lưu chỉ các cột scaled
    scaled_cols = [c for c in df.columns if c.endswith("_scaled")]
    meta_cols = [c for c in ["track_id", "track_name", "artist_name", "popularity"] if c in df.columns]
    scaled_only = df[meta_cols + scaled_cols]
    save_processed_data(scaled_only, "scaled_features.csv")

    print("=" * 50)
    print("✅ HOÀN TẤT TIỀN XỬ LÝ")
    print("=" * 50)
    return df


if __name__ == "__main__":
    from .load_data import load_raw_data

    try:
        raw = load_raw_data()
        processed = full_preprocess(raw, remove_outliers=False)
        print(processed.head())
    except FileNotFoundError as e:
        print(e)
