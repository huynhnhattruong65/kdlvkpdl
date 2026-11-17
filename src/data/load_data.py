"""
Module: load_data.py
Thành viên 1 - Load dữ liệu Spotify Audio Features
"""

import pandas as pd
import os
from pathlib import Path


def get_project_root() -> Path:
    """Trả về đường dẫn gốc của project."""
    current = Path(__file__).resolve()
    # src/data/load_data.py → lên 3 cấp
    return current.parent.parent.parent


def load_raw_data(filename: str = "SpotifyAudioFeatures.csv") -> pd.DataFrame:
    """
    Load dữ liệu thô từ thư mục data/raw.

    Parameters
    ----------
    filename : str
        Tên file csv (mặc định SpotifyAudioFeatures.csv)

    Returns
    -------
    pd.DataFrame
    """
    root = get_project_root()
    filepath = root / "data" / "raw" / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {filepath}\n"
            f"Hãy tải dataset từ Kaggle và đặt vào data/raw/"
        )

    df = pd.read_csv(filepath)
    print(f"✅ Đã load {len(df):,} dòng từ {filepath.name}")
    print(f"   Số cột: {df.shape[1]}")
    return df


def load_processed_data(filename: str) -> pd.DataFrame:
    """Load dữ liệu đã xử lý từ data/processed."""
    root = get_project_root()
    filepath = root / "data" / "processed" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file processed: {filepath}")

    df = pd.read_csv(filepath)
    print(f"✅ Đã load processed data: {filename} ({len(df):,} dòng)")
    return df


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Lưu DataFrame vào data/processed."""
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    filepath = processed_dir / filename
    df.to_csv(filepath, index=False)
    print(f"💾 Đã lưu: {filepath}")


if __name__ == "__main__":
    # Test nhanh
    try:
        df = load_raw_data()
        print(df.head())
        print(df.columns.tolist())
    except FileNotFoundError as e:
        print(e)
