"""
Collaborative Filtering — sparse implementation for low-memory environments.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringRecommender:
    def __init__(self, anime_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.anime_df = anime_df.copy()
        self.ratings_df = ratings_df.copy()
        self._build()

    def _build(self):
        # Map user/anime ids to integer indices
        users = self.ratings_df["user_id"].unique()
        items = self.ratings_df["anime_id"].unique()

        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = {i: it for it, i in self.item2idx.items()}

        rows = self.ratings_df["user_id"].map(self.user2idx)
        cols = self.ratings_df["anime_id"].map(self.item2idx)
        vals = self.ratings_df["rating"].astype(np.float32)

        self.sparse_matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(users), len(items)),
            dtype=np.float32,
        )

    def recommend_item_based(self, user_ratings, top_n=10,
                              genre_filter=None, type_filter=None):
        name_to_id = dict(zip(self.anime_df["name"], self.anime_df["anime_id"]))

        rated_items = {}
        for name, rating in user_ratings.items():
            aid = name_to_id.get(name)
            if aid is not None and aid in self.item2idx:
                rated_items[self.item2idx[aid]] = rating / 10.0

        if not rated_items:
            return pd.DataFrame()

        # For each rated item compute similarity with all other items
        score_dict = {}
        weight_total = sum(rated_items.values())

        for item_idx, weight in rated_items.items():
            item_vec = self.sparse_matrix[:, item_idx].T  # (1, n_users)
            # Compute cosine similarity between this item and all items
            sims = cosine_similarity(item_vec, self.sparse_matrix.T)[0]
            for i, sim in enumerate(sims):
                if i not in rated_items:
                    score_dict[i] = score_dict.get(i, 0) + weight * float(sim)

        if weight_total > 0:
            score_dict = {k: v / weight_total for k, v in score_dict.items()}

        if not score_dict:
            return pd.DataFrame()

        top = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_n * 3]
        anime_ids = [self.idx2item[i] for i, _ in top]
        scores = [s for _, s in top]

        result = pd.DataFrame({"anime_id": anime_ids, "cf_score": scores})
        result = result.merge(self.anime_df, on="anime_id")

        if genre_filter and genre_filter != "All":
            result = result[result["genre"].str.contains(genre_filter, na=False, case=False)]
        if type_filter and type_filter != "All":
            result = result[result["type"] == type_filter]

        result = result.sort_values("cf_score", ascending=False).head(top_n)
        return result[["name", "genre", "type", "rating", "cf_score"]].rename(
            columns={"cf_score": "score", "rating": "mal_rating"}
        )

    def recommend_user_based(self, user_ratings, top_n=10,
                              genre_filter=None, type_filter=None, n_neighbors=20):
        name_to_id = dict(zip(self.anime_df["name"], self.anime_df["anime_id"]))

        virtual = np.zeros(self.sparse_matrix.shape[1], dtype=np.float32)
        rated_idxs = set()
        for name, rating in user_ratings.items():
            aid = name_to_id.get(name)
            if aid is not None and aid in self.item2idx:
                idx = self.item2idx[aid]
                virtual[idx] = float(rating)
                rated_idxs.add(idx)

        if not rated_idxs:
            return pd.DataFrame()

        virtual_sparse = csr_matrix(virtual.reshape(1, -1))
        sims = cosine_similarity(virtual_sparse, self.sparse_matrix)[0]

        top_users = np.argsort(sims)[::-1][:n_neighbors]
        weight_total = sims[top_users].sum()

        score_dict = {}
        for u_idx in top_users:
            sim = float(sims[u_idx])
            if sim <= 0:
                continue
            user_row = self.sparse_matrix[u_idx].toarray()[0]
            for i, r in enumerate(user_row):
                if r > 0 and i not in rated_idxs:
                    score_dict[i] = score_dict.get(i, 0) + sim * float(r)

        if weight_total > 0:
            score_dict = {k: v / weight_total for k, v in score_dict.items()}

        if not score_dict:
            return pd.DataFrame()

        top = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_n * 3]
        anime_ids = [self.idx2item[i] for i, _ in top]
        scores = [s for _, s in top]

        result = pd.DataFrame({"anime_id": anime_ids, "cf_score": scores})
        result = result.merge(self.anime_df, on="anime_id")

        if genre_filter and genre_filter != "All":
            result = result[result["genre"].str.contains(genre_filter, na=False, case=False)]
        if type_filter and type_filter != "All":
            result = result[result["type"] == type_filter]

        result = result.sort_values("cf_score", ascending=False).head(top_n)
        return result[["name", "genre", "type", "rating", "cf_score"]].rename(
            columns={"cf_score": "score", "rating": "mal_rating"}
        )

    def get_stats(self):
        n_ratings = self.sparse_matrix.nnz
        n_users, n_items = self.sparse_matrix.shape
        return {
            "n_users": n_users,
            "n_anime": n_items,
            "n_ratings": n_ratings,
            "sparsity": round(1 - n_ratings / (n_users * n_items), 4),
        }
