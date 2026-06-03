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

        # Pass 1: count user ratings without loading everything into memory
        user_counts = {}
        for chunk in pd.read_csv(raw_cache, chunksize=200_000,
                                  usecols=["user_id", "rating"]):
            chunk = chunk[chunk["rating"].between(1, 10)]
            for uid, cnt in chunk["user_id"].value_counts().items():
                user_counts[uid] = user_counts.get(uid, 0) + cnt

        active_users = {uid for uid, cnt in user_counts.items() if cnt >= 5}

        # Pass 2: collect valid rows, reservoir-sample to RATINGS_SAMPLE
        import random
        random.seed(42)
        reservoir = []
        seen = 0
        for chunk in pd.read_csv(raw_cache, chunksize=200_000):
            chunk = chunk[chunk["rating"].between(1, 10)]
            chunk = chunk[chunk["user_id"].isin(active_users)]
            for row in chunk.itertuples(index=False):
                seen += 1
                if len(reservoir) < RATINGS_SAMPLE:
                    reservoir.append(row)
                else:
                    j = random.randint(0, seen - 1)
                    if j < RATINGS_SAMPLE:
                        reservoir[j] = row

        sampled = pd.DataFrame(reservoir, columns=["user_id", "anime_id", "rating"])
        sampled.to_csv(ratings_cache, index=False)
        del reservoir

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
