"""
src/data_cleaner.py
--------------------
All data loading and cleaning logic for the Netflix dataset.
Import this in notebooks and the Streamlit app to avoid repeating code.

HOW IT WORKS:
  - load_data()       → reads CSV, returns raw DataFrame
  - clean_data()      → handles nulls, type casts, feature engineering
  - get_clean_df()    → convenience wrapper: load + clean in one call
"""

import os
import pandas as pd
import numpy as np
import re

# ─── Path helpers ────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "netflix_titles.csv")


# ─── 1. LOAD ─────────────────────────────────────────────────────────────────
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """
    Read the raw CSV into a DataFrame.
    Raises FileNotFoundError with a helpful message if missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Run:  python data/download_data.py\n"
            "Or download manually from: https://www.kaggle.com/datasets/shivamb/netflix-shows"
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ─── 2. CLEAN ────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a new DataFrame — never mutates input.

    Steps:
      1. Drop duplicates
      2. Normalise column names (strip whitespace)
      3. Fill / flag missing values
      4. Parse date_added → proper datetime
      5. Extract year_added, month_added, day_of_week
      6. Cast release_year to int
      7. Parse duration into numeric minutes / seasons
      8. Explode listed_in (genres) into a helper column
      9. Explode country to handle co-productions
    """
    df = df.copy()

    # ── 2.1 Column names ──────────────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── 2.2 Drop full duplicates ──────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset="show_id", inplace=True)
    print(f"Dropped {before - len(df)} duplicate rows")

    # ── 2.3 Fill missing values ───────────────────────────────────────────────
    fill_map = {
        "director"    : "Unknown",
        "cast"        : "Unknown",
        "country"     : "Unknown",
        "rating"      : "Not Rated",
        "duration"    : "0 min",
        "description" : "",
    }
    df.fillna(fill_map, inplace=True)
    df["date_added"].fillna("January 1, 2000", inplace=True)

    # ── 2.4 Parse dates ───────────────────────────────────────────────────────
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
    df["year_added"]        = df["date_added"].dt.year.astype("Int64")
    df["month_added"]       = df["date_added"].dt.month.astype("Int64")
    df["month_name"]        = df["date_added"].dt.strftime("%b")
    df["day_of_week"]       = df["date_added"].dt.day_name()

    # ── 2.5 release_year ─────────────────────────────────────────────────────
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")

    # ── 2.6 Duration parsing ──────────────────────────────────────────────────
    # Movies: "98 min"  →  98
    # TV:     "2 Seasons" → 2
    def parse_duration(row):
        dur = str(row["duration"])
        if row["type"] == "Movie":
            m = re.search(r"(\d+)", dur)
            return int(m.group(1)) if m else np.nan
        else:
            m = re.search(r"(\d+)", dur)
            return int(m.group(1)) if m else np.nan

    df["duration_numeric"] = df.apply(parse_duration, axis=1)

    # ── 2.7 Genre list (keep original; also expose first genre) ──────────────
    df["listed_in_clean"] = df["listed_in"].str.strip()
    df["primary_genre"]   = df["listed_in_clean"].str.split(",").str[0].str.strip()

    # ── 2.8 Primary country (first listed) ───────────────────────────────────
    df["primary_country"] = df["country"].str.split(",").str[0].str.strip()

    # ── 2.9 Content age (years since release) ────────────────────────────────
    df["content_age"] = df["year_added"] - df["release_year"]

    print("Cleaning complete ✅")
    print(df.dtypes[["type", "year_added", "duration_numeric", "primary_genre"]].to_string())
    return df


# ─── 3. Convenience wrapper ───────────────────────────────────────────────────
def get_clean_df(path: str = CSV_PATH) -> pd.DataFrame:
    """One-liner: load + clean."""
    return clean_data(load_data(path))


# ─── 4. Explode helpers ───────────────────────────────────────────────────────
def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a long-form DataFrame where each row = one title × one genre.
    Useful for genre frequency counts and heatmaps.
    """
    df2 = df.copy()
    df2["genre"] = df2["listed_in_clean"].str.split(", ")
    return df2.explode("genre").reset_index(drop=True)


def explode_countries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a long-form DataFrame where each row = one title × one country.
    Handles co-productions like 'United States, United Kingdom'.
    """
    df2 = df.copy()
    df2["country_single"] = df2["country"].str.split(", ")
    return df2.explode("country_single").reset_index(drop=True)


def explode_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per actor."""
    df2 = df.copy()
    df2["actor"] = df2["cast"].str.split(", ")
    return df2.explode("actor").reset_index(drop=True)


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = get_clean_df()
    print("\nSample rows:")
    print(df[["title", "type", "year_added", "duration_numeric", "primary_genre", "primary_country"]].head())
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
