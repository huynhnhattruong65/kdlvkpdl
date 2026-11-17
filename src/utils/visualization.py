"""
Module: visualization.py
Các hàm vẽ biểu đồ dùng chung
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

from ..data.load_data import get_project_root


def set_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11


def save_fig(name: str, dpi: int = 150):
    """Lưu figure vào results/figures/"""
    root = get_project_root()
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"💾 Đã lưu biểu đồ: {path}")
    plt.close()


def plot_valence_arousal(df: pd.DataFrame, sample: int = 5000, save: bool = True):
    """Vẽ mặt phẳng Valence-Arousal."""
    set_style()
    data = df.sample(min(sample, len(df))) if len(df) > sample else df

    plt.figure(figsize=(10, 8))
    if "emotion_label" in data.columns:
        sns.scatterplot(
            data=data,
            x="valence",
            y="energy",
            hue="emotion_label",
            alpha=0.5,
            s=15,
        )
        plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.scatter(data["valence"], data["energy"], alpha=0.4, s=10)

    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Valence (Negative → Positive)")
    plt.ylabel("Arousal / Energy (Low → High)")
    plt.title("Russell’s Valence-Arousal Plane")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    if save:
        save_fig("valence_arousal_plane.png")
    else:
        plt.show()


def plot_cluster_pca(df: pd.DataFrame, feature_cols: List[str], labels_col: str = "kmeans_cluster", save: bool = True):
    """Giảm chiều PCA và vẽ cluster."""
    from sklearn.decomposition import PCA

    set_style()
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": df[labels_col].values,
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", palette="tab10", alpha=0.6, s=20)
    plt.title(f"PCA Visualization of Clusters ({labels_col})")
    plt.tight_layout()

    if save:
        save_fig(f"cluster_pca_{labels_col}.png")
    else:
        plt.show()


def plot_emotion_distribution(df: pd.DataFrame, save: bool = True):
    """Biểu đồ phân bố cảm xúc."""
    set_style()
    if "emotion_label" not in df.columns:
        print("Không có cột emotion_label")
        return

    plt.figure(figsize=(8, 5))
    order = df["emotion_label"].value_counts().index
    sns.countplot(data=df, y="emotion_label", order=order, palette="viridis")
    plt.title("Phân bố cảm xúc (Russell’s Model)")
    plt.xlabel("Số lượng bài hát")
    plt.ylabel("")
    plt.tight_layout()

    if save:
        save_fig("emotion_distribution.png")
    else:
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, cols: List[str] = None, save: bool = True):
    """Heatmap tương quan các audio features."""
    set_style()
    if cols is None:
        cols = [
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"
        ]
    cols = [c for c in cols if c in df.columns]

    plt.figure(figsize=(10, 8))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap - Spotify Audio Features")
    plt.tight_layout()

    if save:
        save_fig("correlation_heatmap.png")
    else:
        plt.show()
