"""
Data loading utilities.

- anime.csv    — читается локально (лежит в репо)
- ratings.csv  — скачивается с Google Drive при первом запуске, кешируется
"""

import os
import io
import pandas as pd
import numpy as np
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
# /tmp persists for the lifetime of the server and won't be wiped by git pulls
CACHE_DIR = "/tmp/anime_recommender_cache"

# Google Drive file ID для ratings.csv
RATINGS_GDRIVE_ID = "1ICeM6mzu4HZme8V_CLj5JO8fU3-qYINC"

RATINGS_SAMPLE = 200_000


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _download_gdrive(file_id: str, dest: str, label: str) -> None:
    """Скачать файл с Google Drive через gdown."""
    import gdown
    progress = st.info(f"📥 Скачивание {label} с Google Drive…")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest, quiet=False)
    progress.empty()


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_cache = _cache_path(f"ratings_{RATINGS_SAMPLE}.csv")

    # ── anime.csv — берём прямо из репо ───────────────────────────────────────
    anime_path = os.path.join(_HERE, "anime.csv")
    anime_df = pd.read_csv(anime_path)
    anime_df = anime_df.dropna(subset=["name", "genre"])
    anime_df["rating"] = pd.to_numeric(anime_df["rating"], errors="coerce").fillna(0)
    anime_df["genre"]  = anime_df["genre"].str.strip()
    anime_df["type"]   = anime_df["type"].fillna("Unknown")

    # ── ratings.csv — скачиваем с Google Drive, сэмплируем, кешируем ─────────
    if not os.path.exists(ratings_cache):
        raw_cache = _cache_path("ratings_raw.csv")
        if not os.path.exists(raw_cache):
            _download_gdrive(RATINGS_GDRIVE_ID, raw_cache, "ratings.csv")

        st.info("⚙️ Обработка ratings.csv (один раз)…")
        chunks = []
        for chunk in pd.read_csv(raw_cache, chunksize=500_000):
            chunk = chunk[chunk["rating"].between(1, 10)]
            chunks.append(chunk)
        full = pd.concat(chunks, ignore_index=True)

        user_counts = full["user_id"].value_counts()
        active_users = user_counts[user_counts >= 5].index
        full = full[full["user_id"].isin(active_users)]
        sampled = full.sample(n=min(RATINGS_SAMPLE, len(full)), random_state=42)
        sampled.to_csv(ratings_cache, index=False)

        try:
            os.remove(raw_cache)
        except OSError:
            pass

    ratings_df = pd.read_csv(ratings_cache)
    return anime_df, ratings_df


def get_all_genres(anime_df: pd.DataFrame) -> list[str]:
    genres: set[str] = set()
    for s in anime_df["genre"].dropna():
        for g in s.split(","):
            genres.add(g.strip())
    return sorted(genres)


def get_all_types(anime_df: pd.DataFrame) -> list[str]:
    return sorted(anime_df["type"].dropna().unique().tolist())


def get_anime_names(anime_df: pd.DataFrame) -> list[str]:
    return sorted(anime_df["name"].dropna().unique().tolist())
