"""
src/recommender.py
-------------------
Content-Based Recommendation Engine for Netflix titles.

HOW CONTENT-BASED FILTERING WORKS:
  1. Each title is represented as a "feature vector" (a list of numbers).
  2. The vector combines: TF-IDF of description + encoded genre + encoded cast/director.
  3. When you ask for recommendations for title X:
       a. Find X's feature vector
       b. Compute cosine similarity between X and ALL other titles
       c. Return top-N most similar titles (highest cosine score)

COSINE SIMILARITY:
  - Measures the angle between two vectors in N-dimensional space
  - Score = 1.0 means identical content; 0.0 means no overlap
  - Formula: cos(θ) = (A · B) / (‖A‖ × ‖B‖)
  - Why cosine? It's length-independent — a long description and a short one
    about the same topic still score high (unlike dot product).

USAGE:
  rec = NetflixRecommender()
  rec.fit(df)
  rec.recommend("Stranger Things")        # → list of similar shows
  rec.recommend("The Dark Knight", n=10)  # → top 10 similar movies
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")


class NetflixRecommender:
    """
    Content-based recommendation engine.

    Attributes:
      feature_matrix : combined sparse matrix (TF-IDF + genre + cast)
      title_index    : dict mapping lowercase title → row index
      df             : cleaned DataFrame
    """

    def __init__(self):
        self.feature_matrix = None
        self.title_index     = {}
        self.df              = None
        self._vectorizer     = None

    # ─── FIT ─────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame,
            weight_desc: float = 1.0,
            weight_genre: float = 0.8,
            weight_cast: float = 0.5,
            weight_director: float = 0.4) -> "NetflixRecommender":
        """
        Build the feature matrix from the DataFrame.

        Parameters:
          df             : cleaned Netflix DataFrame (from data_cleaner.py)
          weight_desc    : importance of description similarity
          weight_genre   : importance of genre match
          weight_cast    : importance of shared actors
          weight_director: importance of same director

        Why weights?
          Description is most informative (rich text).
          Genre is strong but coarse (many titles share a genre).
          Cast/director add precision for fans of specific people.
        """
        self.df = df.reset_index(drop=True)

        # ── Feature 1: TF-IDF on descriptions ────────────────────────────────
        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            sublinear_tf=True,
        )
        desc_text = (df["description"].fillna("") + " " + df["listed_in"].fillna(""))
        desc_matrix = self._vectorizer.fit_transform(desc_text) * weight_desc

        # ── Feature 2: Genre bag-of-words ─────────────────────────────────────
        genre_vec = TfidfVectorizer(max_features=500, tokenizer=lambda x: x.split(", "),
                                    token_pattern=None)
        genre_matrix = genre_vec.fit_transform(df["listed_in"].fillna("")) * weight_genre

        # ── Feature 3: Cast bag-of-words ──────────────────────────────────────
        cast_vec = TfidfVectorizer(max_features=3000, tokenizer=lambda x: x.split(", "),
                                   token_pattern=None)
        cast_matrix = cast_vec.fit_transform(df["cast"].fillna("Unknown")) * weight_cast

        # ── Feature 4: Director ───────────────────────────────────────────────
        dir_vec = TfidfVectorizer(max_features=1000, tokenizer=lambda x: x.split(", "),
                                  token_pattern=None)
        dir_matrix = dir_vec.fit_transform(df["director"].fillna("Unknown")) * weight_director

        # ── Combine horizontally into one big feature matrix ──────────────────
        # Each row = one title; columns = all features stacked side by side
        self.feature_matrix = sp.hstack(
            [desc_matrix, genre_matrix, cast_matrix, dir_matrix]
        ).tocsr()  # CSR = efficient row slicing

        # ── Build lookup index ────────────────────────────────────────────────
        self.title_index = {
            title.lower().strip(): idx
            for idx, title in enumerate(df["title"])
        }

        print(f"Recommender fitted on {len(df):,} titles.")
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        return self

    # ─── RECOMMEND ────────────────────────────────────────────────────────────
    def recommend(self, title: str, n: int = 10,
                  same_type: bool = False) -> pd.DataFrame:
        """
        Return top-N recommendations for a given title.

        Parameters:
          title     : exact or approximate title string
          n         : number of recommendations to return
          same_type : if True, only recommend same content type (Movie/TV Show)

        Returns:
          DataFrame with columns: title, type, genres, similarity_score, description
        """
        if self.feature_matrix is None:
            raise RuntimeError("Call .fit(df) before .recommend()")

        # ── Fuzzy title lookup ────────────────────────────────────────────────
        query = title.lower().strip()
        idx = self._find_title_index(query)
        if idx is None:
            suggestions = self._suggest_titles(query, n=5)
            raise ValueError(
                f"Title '{title}' not found.\n"
                f"Did you mean one of these?\n  " + "\n  ".join(suggestions)
            )

        # ── Cosine similarity for this one title vs all others ────────────────
        query_vec = self.feature_matrix[idx]
        # dot product / (norm_query × norm_all)  — computed efficiently on sparse
        sim_scores = cosine_similarity(query_vec, self.feature_matrix).flatten()

        # ── Sort by score, exclude the title itself (score=1.0) ───────────────
        sim_scores[idx] = -1  # exclude self
        top_indices = sim_scores.argsort()[::-1]

        # ── Optional: filter to same type (Movie/Movie or TV/TV) ─────────────
        if same_type:
            target_type = self.df.iloc[idx]["type"]
            mask = self.df["type"] == target_type
            top_indices = [i for i in top_indices if mask.iloc[i]]

        top_indices = top_indices[:n]

        # ── Build result DataFrame ────────────────────────────────────────────
        result = self.df.iloc[top_indices][
            ["title", "type", "listed_in", "release_year",
             "primary_country", "rating", "description", "director", "cast"]
        ].copy()
        result["similarity_score"] = sim_scores[top_indices].round(4)
        result = result.sort_values("similarity_score", ascending=False)
        return result.reset_index(drop=True)

    # ─── GET INFO ─────────────────────────────────────────────────────────────
    def get_title_info(self, title: str) -> pd.Series:
        """Return the row for a specific title."""
        idx = self._find_title_index(title.lower().strip())
        if idx is None:
            raise ValueError(f"'{title}' not found in dataset.")
        return self.df.iloc[idx]

    # ─── SEARCH ───────────────────────────────────────────────────────────────
    def search(self, query: str, n: int = 10) -> pd.DataFrame:
        """
        Free-text search across titles + descriptions using TF-IDF similarity.
        Different from recommend(): recommend() uses a known title as query;
        search() uses a text string like "space adventure robots".
        """
        query_vec = self._vectorizer.transform([query])
        # Compare only against description portion of feature matrix
        desc_cols = self._vectorizer.get_feature_names_out().shape[0]
        desc_only = self.feature_matrix[:, :desc_cols]
        sim = cosine_similarity(query_vec, desc_only).flatten()
        top_idx = sim.argsort()[::-1][:n]
        result = self.df.iloc[top_idx][["title", "type", "listed_in", "description"]].copy()
        result["score"] = sim[top_idx].round(4)
        return result.reset_index(drop=True)

    # ─── PRIVATE HELPERS ──────────────────────────────────────────────────────
    def _find_title_index(self, query: str):
        """Exact match first, then partial match."""
        if query in self.title_index:
            return self.title_index[query]
        # Partial match
        for stored_title, idx in self.title_index.items():
            if query in stored_title or stored_title in query:
                return idx
        return None

    def _suggest_titles(self, query: str, n: int = 5) -> list:
        """Return titles containing any word from the query."""
        words = set(query.lower().split())
        matches = [
            title for title in self.df["title"]
            if any(w in title.lower() for w in words)
        ]
        return matches[:n]


# ─── Convenience function ─────────────────────────────────────────────────────
def build_recommender(df: pd.DataFrame) -> NetflixRecommender:
    """Build and return a fitted recommender in one line."""
    rec = NetflixRecommender()
    rec.fit(df)
    return rec


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_cleaner import get_clean_df

    df = get_clean_df()
    rec = build_recommender(df)

    print("\n" + "="*60)
    print("Recommendations for: 'Stranger Things'")
    print("="*60)
    results = rec.recommend("Stranger Things", n=5)
    print(results[["title", "type", "listed_in", "similarity_score"]].to_string())

    print("\n" + "="*60)
    print("Recommendations for: 'The Dark Knight'")
    print("="*60)
    results2 = rec.recommend("The Dark Knight", n=5)
    print(results2[["title", "type", "listed_in", "similarity_score"]].to_string())
