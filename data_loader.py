import os
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    anime_df = pd.read_csv(os.path.join(_HERE, "anime.csv"))
    anime_df = anime_df.dropna(subset=["name", "genre"])
    anime_df["rating"] = pd.to_numeric(anime_df["rating"], errors="coerce").fillna(0)
    anime_df["genre"]  = anime_df["genre"].str.strip()
    anime_df["type"]   = anime_df["type"].fillna("Unknown")

    ratings_df = pd.read_csv(os.path.join(_HERE, "ratings.csv"))
    ratings_df = ratings_df[ratings_df["rating"].between(1, 10)]

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
