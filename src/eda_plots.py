"""
src/eda_plots.py
-----------------
All EDA visualization functions.
Each function takes a cleaned DataFrame and returns a figure.

WHY SEPARATE FROM NOTEBOOKS?
  - DRY principle: call the same function from notebook AND Streamlit app
  - Easier to test / iterate
  - Clean notebooks that focus on narrative, not code

SECTIONS:
  A. Distribution plots
  B. Trend / time-series plots
  C. Country analysis
  D. Genre heatmaps
  E. Duration analysis
  F. Plotly interactive charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Global style ─────────────────────────────────────────────────────────────
NETFLIX_RED   = "#E50914"
NETFLIX_DARK  = "#141414"
NETFLIX_WHITE = "#FFFFFF"
PALETTE       = [NETFLIX_RED, "#4a9eff", "#1db954", "#ff9500", "#9b59b6",
                 "#e74c3c", "#1abc9c", "#f39c12", "#2980b9", "#8e44ad"]

def _set_style():
    """Apply Netflix-inspired dark theme."""
    plt.rcParams.update({
        "figure.facecolor"    : "#1a1a1a",
        "axes.facecolor"      : "#1a1a1a",
        "axes.edgecolor"      : "#333",
        "axes.labelcolor"     : "#ccc",
        "axes.titlecolor"     : "#fff",
        "xtick.color"         : "#888",
        "ytick.color"         : "#888",
        "text.color"          : "#ccc",
        "grid.color"          : "#2a2a2a",
        "grid.linestyle"      : "--",
        "font.family"         : "DejaVu Sans",
        "axes.spines.top"     : False,
        "axes.spines.right"   : False,
    })

_set_style()


# ══════════════════════════════════════════════════════════════════════════════
# A. DISTRIBUTION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_type_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Donut chart: Movie vs TV Show split.
    Shows the overall catalog composition.
    """
    counts = df["type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[NETFLIX_RED, "#4a9eff"],
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"width": 0.55, "edgecolor": "#1a1a1a", "linewidth": 3},
    )
    for t in texts:
        t.set_color("#fff")
        t.set_fontsize(14)
    for a in autotexts:
        a.set_color("#fff")
        a.set_fontsize(12)
        a.set_fontweight("bold")

    ax.set_title("Movie vs TV Show Distribution", fontsize=16, color="#fff", pad=20)
    centre = plt.Circle((0, 0), 0.45, color="#1a1a1a")
    ax.add_patch(centre)
    ax.text(0, 0, f"{counts.sum():,}\nTitles", ha="center", va="center",
            color="#fff", fontsize=13, fontweight="bold", linespacing=1.8)
    fig.tight_layout()
    return fig


def plot_ratings_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of content ratings (TV-MA, PG-13, etc.)
    grouped by content type.
    """
    rating_order = ["TV-MA", "TV-14", "TV-PG", "TV-Y7", "TV-Y", "TV-G",
                    "R", "PG-13", "PG", "G", "NR", "Not Rated"]
    pivot = df[df["rating"].isin(rating_order)].groupby(
        ["rating", "type"]).size().unstack(fill_value=0)
    pivot = pivot.reindex([r for r in rating_order if r in pivot.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", ax=ax, color=[NETFLIX_RED, "#4a9eff"], edgecolor="none")
    ax.set_title("Content Ratings Distribution", fontsize=15, color="#fff")
    ax.set_xlabel("Number of Titles", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#ccc")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# B. TREND / TIME-SERIES PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_content_over_time(df: pd.DataFrame) -> plt.Figure:
    """
    Stacked area chart: Movies vs TV Shows added per year.
    Shows how Netflix shifted its strategy over time.
    """
    yearly = (df[df["year_added"].notna() & (df["year_added"] >= 2008)]
              .groupby(["year_added", "type"])
              .size().unstack(fill_value=0)
              .reset_index())
    yearly.columns = [str(c) for c in yearly.columns]
    yearly["year_added"] = yearly["year_added"].astype(int)

    fig, ax = plt.subplots(figsize=(14, 6))
    years = yearly["year_added"].values
    movies = yearly.get("Movie", pd.Series(0, index=yearly.index)).values
    tvs    = yearly.get("TV Show", pd.Series(0, index=yearly.index)).values

    ax.fill_between(years, 0, movies, alpha=0.85, color=NETFLIX_RED,   label="Movies")
    ax.fill_between(years, movies, movies+tvs, alpha=0.85, color="#4a9eff", label="TV Shows")
    ax.plot(years, movies+tvs, color="#fff", lw=1.5, alpha=0.6)

    # Annotate peak year
    peak_y = (movies+tvs).max()
    peak_x = years[(movies+tvs).argmax()]
    ax.annotate(f"Peak: {int(peak_y):,} titles ({peak_x})",
                xy=(peak_x, peak_y), xytext=(peak_x-3, peak_y+80),
                arrowprops={"arrowstyle": "->", "color": "#fff"},
                color="#fff", fontsize=10)

    ax.set_title("Content Added to Netflix Per Year", fontsize=15, color="#fff")
    ax.set_xlabel("Year", color="#aaa")
    ax.set_ylabel("Titles Added", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#ccc", framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tv_shows_growth(df: pd.DataFrame) -> plt.Figure:
    """
    Line chart comparing movie vs TV show growth rate (indexed to 2012=100).
    INSIGHT: TV shows grew much faster after 2016.
    """
    yearly = (df[df["year_added"].between(2012, 2021)]
              .groupby(["year_added", "type"])
              .size().unstack(fill_value=0))
    # Index to 100 at 2012
    indexed = (yearly / yearly.iloc[0] * 100).reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    for col, color, label in [("Movie", NETFLIX_RED, "Movies"),
                               ("TV Show", "#4a9eff", "TV Shows")]:
        if col in indexed:
            ax.plot(indexed["year_added"], indexed[col],
                    marker="o", color=color, lw=2.5, label=label)
            last = indexed[col].iloc[-1]
            ax.annotate(f"{last:.0f}x", xy=(indexed["year_added"].iloc[-1], last),
                        xytext=(8, 0), textcoords="offset points",
                        color=color, fontsize=10, fontweight="bold")

    ax.axvline(2016, color="#fff", lw=1, ls="--", alpha=0.5)
    ax.text(2016.1, 20, "TV surge\nbegins", color="#ccc", fontsize=9)
    ax.set_title("Growth Index: Movies vs TV Shows (2012 = 100)", fontsize=14, color="#fff")
    ax.set_ylabel("Index (2012 = 100)", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#ccc")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monthly_releases(df: pd.DataFrame) -> plt.Figure:
    """
    Heatmap: year × month — shows seasonal release patterns.
    Netflix tends to release more content in Q4 (Oct–Dec).
    """
    monthly = (df[df["year_added"].between(2015, 2021)]
               .groupby(["year_added", "month_added"])
               .size().unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(14, 5))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly.columns = [month_names[i-1] for i in monthly.columns]
    sns.heatmap(monthly, ax=ax, cmap="YlOrRd", linewidths=0.5,
                linecolor="#333", annot=True, fmt="d",
                cbar_kws={"label": "Titles Added"})
    ax.set_title("Monthly Release Heatmap (titles added per month)", fontsize=14, color="#fff")
    ax.set_xlabel("")
    ax.set_ylabel("Year", color="#aaa")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# C. COUNTRY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_countries(df: pd.DataFrame, n: int = 15) -> plt.Figure:
    """
    Horizontal bar: top N content-producing countries.
    Uses primary_country (first listed country per title).
    """
    top = (df[df["primary_country"] != "Unknown"]
           ["primary_country"].value_counts().head(n))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [NETFLIX_RED if i == 0 else "#333" for i in range(n)]
    bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="none")
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", color="#ccc", fontsize=10)
    ax.set_title(f"Top {n} Content-Producing Countries", fontsize=14, color="#fff")
    ax.set_xlabel("Number of Titles", color="#aaa")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_country_type_heatmap(df: pd.DataFrame, n: int = 12) -> plt.Figure:
    """
    Heatmap: top countries × content type.
    Shows which countries specialise in movies vs TV.
    """
    from src.data_cleaner import explode_countries
    df_exp = explode_countries(df)
    top_countries = (df_exp[df_exp["country_single"] != "Unknown"]
                     ["country_single"].value_counts().head(n).index.tolist())
    pivot = (df_exp[df_exp["country_single"].isin(top_countries)]
             .groupby(["country_single", "type"])
             .size().unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, ax=ax, cmap="Reds", annot=True, fmt="d",
                linewidths=0.5, linecolor="#333")
    ax.set_title("Country × Content Type Heatmap", fontsize=13, color="#fff")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_country_genre_heatmap(df: pd.DataFrame, n_countries: int = 10, n_genres: int = 10) -> plt.Figure:
    """
    Heatmap: top countries × top genres.
    INSIGHT: certain countries specialise in specific genres.
    e.g. South Korea → Dramas, India → International Movies.
    """
    from src.data_cleaner import explode_genres, explode_countries
    # Explode both genres and countries
    df_g = explode_genres(df).rename(columns={"genre": "genre_single"})
    df_g["country_single"] = df_g["country"].str.split(", ").str[0].str.strip()

    top_c = df_g[df_g["country_single"] != "Unknown"]["country_single"].value_counts().head(n_countries).index
    top_g = df_g["genre_single"].value_counts().head(n_genres).index
    pivot = (df_g[df_g["country_single"].isin(top_c) & df_g["genre_single"].isin(top_g)]
             .groupby(["country_single", "genre_single"])
             .size().unstack(fill_value=0))

    # Normalise to % per country row
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot_pct, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.5, linecolor="#222",
                cbar_kws={"label": "% of country's content"})
    ax.set_title("Country × Genre Specialisation Heatmap (%)", fontsize=14, color="#fff")
    ax.set_xlabel("Genre", color="#aaa")
    ax.set_ylabel("Country", color="#aaa")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# D. GENRE POPULARITY
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_genres(df: pd.DataFrame, n: int = 15) -> plt.Figure:
    """Bar chart of most common genres across all content."""
    from src.data_cleaner import explode_genres
    genre_counts = explode_genres(df)["genre"].value_counts().head(n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
                   color=NETFLIX_RED, edgecolor="none", alpha=0.85)
    for bar, val in zip(bars, genre_counts.values[::-1]):
        ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
                f"{val:,}", va="center", color="#ccc", fontsize=10)
    ax.set_title(f"Top {n} Genres on Netflix", fontsize=14, color="#fff")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_genre_over_time(df: pd.DataFrame, top_n: int = 6) -> plt.Figure:
    """
    Line chart: top N genres over years.
    Shows genre trends — e.g. International Movies rose sharply after 2016.
    """
    from src.data_cleaner import explode_genres
    df_g = explode_genres(df)
    top_genres = df_g["genre"].value_counts().head(top_n).index.tolist()
    df_filtered = df_g[df_g["genre"].isin(top_genres) & df_g["year_added"].between(2013, 2021)]
    trend = df_filtered.groupby(["year_added", "genre"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, genre in enumerate(top_genres):
        if genre in trend.columns:
            ax.plot(trend.index, trend[genre], marker="o",
                    color=PALETTE[i], lw=2, label=genre)
    ax.set_title("Top Genre Trends Over Time", fontsize=14, color="#fff")
    ax.set_xlabel("Year", color="#aaa")
    ax.set_ylabel("Titles Added", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#ccc", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E. DURATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_movie_duration_dist(df: pd.DataFrame) -> plt.Figure:
    """
    Histogram + KDE of movie runtimes.
    Shows the most common movie length (~90 min) and outliers.
    """
    movies = df[(df["type"] == "Movie") &
                df["duration_numeric"].between(30, 300)]

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(movies["duration_numeric"], bins=50, kde=True, ax=ax,
                 color=NETFLIX_RED, edgecolor="#1a1a1a", alpha=0.7,
                 line_kws={"color": "#fff", "lw": 2})

    median = movies["duration_numeric"].median()
    ax.axvline(median, color="#4a9eff", lw=2, ls="--")
    ax.text(median+2, ax.get_ylim()[1]*0.85, f"Median: {median:.0f} min",
            color="#4a9eff", fontsize=11)

    ax.set_title("Movie Duration Distribution", fontsize=14, color="#fff")
    ax.set_xlabel("Duration (minutes)", color="#aaa")
    ax.set_ylabel("Count", color="#aaa")
    fig.tight_layout()
    return fig


def plot_duration_by_genre(df: pd.DataFrame, n: int = 12) -> plt.Figure:
    """
    Box plot: movie duration grouped by primary genre.
    INSIGHT: Documentaries tend to be shorter; Action longer.
    """
    movies = df[(df["type"] == "Movie") & df["duration_numeric"].between(30, 300)]
    top_genres = movies["primary_genre"].value_counts().head(n).index
    filtered = movies[movies["primary_genre"].isin(top_genres)]

    order = (filtered.groupby("primary_genre")["duration_numeric"]
             .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=filtered, x="primary_genre", y="duration_numeric",
                order=order, ax=ax, palette="Reds_r",
                flierprops={"marker": ".", "color": "#666"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color="#aaa")
    ax.set_title("Movie Duration by Genre", fontsize=14, color="#fff")
    ax.set_xlabel("")
    ax.set_ylabel("Duration (minutes)", color="#aaa")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tv_seasons_dist(df: pd.DataFrame) -> plt.Figure:
    """
    Count plot: TV show season counts.
    INSIGHT: Most shows on Netflix have only 1–2 seasons.
    """
    shows = df[(df["type"] == "TV Show") & df["duration_numeric"].between(1, 15)]
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = shows["duration_numeric"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color="#4a9eff", edgecolor="none", alpha=0.85)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
                str(val), ha="center", va="bottom", color="#ccc", fontsize=9)
    ax.set_title("TV Show Season Count Distribution", fontsize=14, color="#fff")
    ax.set_xlabel("Number of Seasons", color="#aaa")
    ax.set_ylabel("Number of Shows", color="#aaa")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# F. PLOTLY INTERACTIVE CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plotly_content_growth(df: pd.DataFrame) -> go.Figure:
    """
    Interactive Plotly stacked bar: content growth per year.
    Hover to see exact counts; click legend to toggle types.
    """
    yearly = (df[df["year_added"].between(2008, 2021)]
              .groupby(["year_added", "type"])
              .size().reset_index(name="count"))
    yearly["year_added"] = yearly["year_added"].astype(int)

    fig = px.bar(yearly, x="year_added", y="count", color="type",
                 color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#4a9eff"},
                 barmode="stack",
                 title="📈 Content Added to Netflix Per Year (Interactive)",
                 labels={"year_added": "Year", "count": "Titles Added", "type": "Type"})
    fig.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        font_color="#ccc", title_font_size=16,
        legend=dict(bgcolor="#222", bordercolor="#333"),
        hovermode="x unified"
    )
    return fig


def plotly_country_map(df: pd.DataFrame) -> go.Figure:
    """
    Interactive choropleth world map: content count by country.
    Hover over any country to see title count.
    """
    from src.data_cleaner import explode_countries
    country_counts = (explode_countries(df)[lambda x: x["country_single"] != "Unknown"]
                      .groupby("country_single").size().reset_index(name="count"))

    fig = px.choropleth(country_counts,
                        locations="country_single",
                        locationmode="country names",
                        color="count",
                        color_continuous_scale="Reds",
                        title="🌍 Netflix Content by Country",
                        labels={"count": "Titles", "country_single": "Country"})
    fig.update_layout(
        paper_bgcolor="#1a1a1a", font_color="#ccc",
        geo=dict(bgcolor="#1a1a1a", showframe=False,
                 showcoastlines=True, coastlinecolor="#333",
                 showland=True, landcolor="#222",
                 showocean=True, oceancolor="#141414"),
        title_font_size=16
    )
    return fig


def plotly_genre_treemap(df: pd.DataFrame) -> go.Figure:
    """
    Treemap: genre × content type — area proportional to count.
    Great for showing relative size at a glance.
    """
    from src.data_cleaner import explode_genres
    df_g = explode_genres(df)
    counts = (df_g.groupby(["genre", "type"])
              .size().reset_index(name="count"))

    fig = px.treemap(counts, path=["type", "genre"], values="count",
                     color="count", color_continuous_scale="Reds",
                     title="🎭 Genre Treemap (Movies vs TV Shows)")
    fig.update_layout(
        paper_bgcolor="#1a1a1a", font_color="#fff", title_font_size=16
    )
    return fig


def plotly_duration_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Interactive scatter: movie duration vs release year, coloured by rating.
    Reveals whether movies have gotten shorter/longer over time.
    """
    movies = df[(df["type"] == "Movie") &
                df["duration_numeric"].between(30, 240) &
                df["release_year"].between(1990, 2021)]

    fig = px.scatter(movies, x="release_year", y="duration_numeric",
                     color="rating", hover_data=["title", "primary_country"],
                     opacity=0.6,
                     title="🎬 Movie Duration vs Release Year",
                     labels={"release_year": "Release Year",
                             "duration_numeric": "Duration (min)"},
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        font_color="#ccc", title_font_size=16
    )
    return fig
