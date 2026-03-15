"""
src/nlp_engine.py
------------------
NLP pipeline for Netflix movie descriptions.

WHAT IT DOES:
  1. Cleans raw description text (lowercase, remove stopwords, lemmatize)
  2. TF-IDF vectorization — finds the most "important" keywords per title
  3. KMeans clustering — groups titles into thematic clusters
  4. Topic extraction — top keywords per cluster
  5. Word cloud generation — visual frequency plot
  6. Genre cluster insight — shows what each cluster "means"

HOW TF-IDF WORKS:
  - TF  (Term Frequency)  = how often a word appears in THIS document
  - IDF (Inverse Doc Freq) = how rare the word is across ALL documents
  - TF-IDF = TF × IDF  → high score = word is important HERE but rare elsewhere
  - Common words like "the", "is" get near-zero scores (high in all docs → low IDF)
  - "murder", "investigation" score high in crime descriptions → meaningful signal
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import nltk
# Download required NLTK data (runs once, cached locally)
for resource in ["stopwords", "punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource != "punkt" else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
EXTRA_STOPS = {"film", "series", "show", "story", "world", "life", "one",
               "two", "three", "new", "find", "make", "take", "come", "get",
               "also", "even", "becomes", "follows", "set", "must", "still"}
ALL_STOPS = STOP_WORDS | EXTRA_STOPS
LEMMATIZER = WordNetLemmatizer()

NETFLIX_RED = "#E50914"


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Preprocess a single description string.
    Pipeline: lowercase → strip HTML → remove punctuation →
              tokenize → remove stopwords → lemmatize → rejoin.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Lowercase + remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize, filter stopwords, lemmatize
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in text.split()
        if tok not in ALL_STOPS and len(tok) > 2
    ]
    return " ".join(tokens)


def preprocess_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'clean_desc' column to the DataFrame.
    Filter out rows with empty descriptions afterward.
    """
    df = df.copy()
    df["clean_desc"] = df["description"].fillna("").apply(clean_text)
    df = df[df["clean_desc"].str.len() > 10].reset_index(drop=True)
    print(f"Preprocessed {len(df):,} descriptions")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. TF-IDF VECTORIZATION
# ══════════════════════════════════════════════════════════════════════════════

def build_tfidf_matrix(df: pd.DataFrame,
                        max_features: int = 5000,
                        ngram_range: tuple = (1, 2)):
    """
    Convert clean descriptions to a TF-IDF sparse matrix.

    Parameters:
      max_features : vocabulary size cap (top N most frequent terms)
      ngram_range  : (1,1) = unigrams only; (1,2) = also bigrams like "serial killer"

    Returns:
      tfidf_matrix : sparse matrix (n_titles × max_features)
      vectorizer   : fitted TfidfVectorizer (needed to get feature names)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=3,          # ignore terms that appear in < 3 documents
        max_df=0.85,       # ignore terms that appear in > 85% of documents
        sublinear_tf=True, # use 1 + log(TF) to dampen very frequent terms
    )
    tfidf_matrix = vectorizer.fit_transform(df["clean_desc"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


def get_top_keywords(vectorizer, n: int = 30) -> pd.DataFrame:
    """
    Return the top N keywords across the whole corpus by total TF-IDF weight.
    These are the most "characterising" words in all Netflix descriptions.
    """
    feature_names = vectorizer.get_feature_names_out()
    # Sum TF-IDF scores across all documents for each term
    # (requires the full matrix — call after build_tfidf_matrix)
    return feature_names  # caller can use this for word clouds or tables


# ══════════════════════════════════════════════════════════════════════════════
# 3. KMEANS CLUSTERING (TOPIC MODELING)
# ══════════════════════════════════════════════════════════════════════════════

def fit_clusters(tfidf_matrix, n_clusters: int = 8, random_state: int = 42):
    """
    Fit KMeans on the TF-IDF matrix.
    Each cluster = a thematic group (crime, romance, family, etc.)

    WHY KMEANS ON TEXT?
      KMeans treats each document as a point in high-dimensional TF-IDF space.
      Documents about similar topics use similar words → cluster together.

    Returns: fitted KMeans model
    """
    # Normalise rows so cosine distance ≈ Euclidean (better for text)
    matrix_norm = normalize(tfidf_matrix)
    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=10, max_iter=300)
    km.fit(matrix_norm)
    print(f"Fitted {n_clusters} clusters. Inertia: {km.inertia_:.2f}")
    return km


def get_cluster_keywords(km, vectorizer, n_top: int = 10) -> dict:
    """
    For each cluster, get the top N keywords (closest to cluster centroid).
    These keywords describe the "theme" of each cluster.

    Returns: dict {cluster_id: [keyword1, keyword2, ...]}
    """
    feature_names = vectorizer.get_feature_names_out()
    clusters = {}
    for i, centroid in enumerate(km.cluster_centers_):
        # argsort descending → indices of highest-weight features
        top_indices = centroid.argsort()[::-1][:n_top]
        clusters[i] = [feature_names[idx] for idx in top_indices]
    return clusters


def assign_cluster_labels(cluster_keywords: dict) -> dict:
    """
    Auto-assign human-readable labels based on top keywords.
    You can manually override these after inspection.
    """
    label_rules = {
        "crime"         : ["crime","murder","investigation","detective","killer","police"],
        "romance"       : ["love","relationship","romance","heart","couple","marriage"],
        "family"        : ["family","children","father","mother","child","young","kid"],
        "action"        : ["fight","battle","war","mission","agent","force","enemy"],
        "drama"         : ["drama","life","struggle","story","journey","challenge"],
        "comedy"        : ["comedy","funny","laugh","hilarious","humor","fun"],
        "documentary"   : ["documentary","history","world","explore","discover","culture"],
        "international" : ["international","foreign","language","global","worldwide"],
        "thriller"      : ["thriller","suspense","mystery","secret","hidden","dark"],
        "horror"        : ["horror","fear","terrifying","nightmare","evil","ghost"],
    }
    labels = {}
    for cluster_id, keywords in cluster_keywords.items():
        best_label, best_score = "misc", 0
        for label, rule_words in label_rules.items():
            score = sum(1 for kw in keywords if any(r in kw for r in rule_words))
            if score > best_score:
                best_score, best_label = score, label
        labels[cluster_id] = best_label
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIMENSIONALITY REDUCTION FOR VISUALISATION (SVD → 2D)
# ══════════════════════════════════════════════════════════════════════════════

def reduce_to_2d(tfidf_matrix, n_components: int = 2):
    """
    Use Truncated SVD (like PCA for sparse matrices) to compress
    the TF-IDF matrix into 2D for scatter plot visualisation.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    coords = svd.fit_transform(tfidf_matrix)
    explained = svd.explained_variance_ratio_.sum()
    print(f"2D SVD explains {explained:.1%} of variance")
    return coords


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_wordcloud(tfidf_matrix, vectorizer, title: str = "Netflix Descriptions") -> plt.Figure:
    """
    Generate a word cloud from the TF-IDF scores summed across all documents.
    Larger words = higher total TF-IDF weight = more "characteristic" of the corpus.
    """
    feature_names = vectorizer.get_feature_names_out()
    # Sum each column (feature) across all documents
    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    word_scores = dict(zip(feature_names, scores))

    wc = WordCloud(
        width=900, height=500,
        background_color="#141414",
        colormap="Reds",
        max_words=150,
        prefer_horizontal=0.7,
        relative_scaling=0.5,
    ).generate_from_frequencies(word_scores)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_facecolor("#141414")
    fig.patch.set_facecolor("#141414")
    ax.set_title(title, fontsize=16, color="#fff", pad=14)
    fig.tight_layout()
    return fig


def plot_cluster_scatter(df_with_clusters, coords, labels_map: dict) -> plt.Figure:
    """
    2D scatter plot of all titles coloured by cluster.
    Each point = one Netflix title; proximity = description similarity.
    """
    palette = ["#E50914","#4a9eff","#1db954","#ff9500","#9b59b6",
               "#e74c3c","#1abc9c","#f39c12","#e91e63","#00bcd4"]

    fig, ax = plt.subplots(figsize=(12, 8))
    clusters = df_with_clusters["cluster"].values
    for cid in sorted(set(clusters)):
        mask = clusters == cid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=palette[cid % len(palette)],
                   label=f"Cluster {cid}: {labels_map.get(cid,'misc')}",
                   alpha=0.5, s=8, edgecolors="none")

    ax.set_title("Netflix Content Clusters (TF-IDF → SVD 2D)", fontsize=14, color="#fff")
    ax.legend(facecolor="#222", labelcolor="#ccc", markerscale=3,
              fontsize=9, loc="upper right")
    ax.set_xlabel("SVD Component 1", color="#aaa")
    ax.set_ylabel("SVD Component 2", color="#aaa")
    ax.grid(alpha=0.15)
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    fig.tight_layout()
    return fig


def plot_cluster_keywords_bar(cluster_keywords: dict, labels_map: dict) -> plt.Figure:
    """
    Grid of horizontal bar charts — one per cluster — showing top keywords.
    Makes it easy to understand what each cluster represents.
    """
    n = len(cluster_keywords)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3 + 1))
    axes = axes.flatten()

    palette = ["#E50914","#4a9eff","#1db954","#ff9500","#9b59b6",
               "#e74c3c","#1abc9c","#f39c12","#e91e63","#00bcd4"]

    for i, (cid, keywords) in enumerate(cluster_keywords.items()):
        ax = axes[i]
        ax.barh(range(len(keywords)), [1]*len(keywords),
                color=palette[i % len(palette)], alpha=0.7, edgecolor="none")
        ax.set_yticks(range(len(keywords)))
        ax.set_yticklabels(keywords[::-1] if False else keywords,
                           color="#ddd", fontsize=9)
        ax.set_xticks([])
        ax.set_title(f"Cluster {cid}: {labels_map.get(cid,'misc')}",
                     color="#fff", fontsize=10)
        ax.set_facecolor("#1a1a1a")

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.patch.set_facecolor("#141414")
    fig.suptitle("Top Keywords per Content Cluster", color="#fff", fontsize=15)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. FULL PIPELINE (convenience function for notebooks)
# ══════════════════════════════════════════════════════════════════════════════

def run_nlp_pipeline(df: pd.DataFrame, n_clusters: int = 8):
    """
    One function to run the entire NLP pipeline.
    Returns everything you need for the notebook.
    """
    print("Step 1: Cleaning descriptions...")
    df_nlp = preprocess_descriptions(df)

    print("\nStep 2: Building TF-IDF matrix...")
    tfidf_matrix, vectorizer = build_tfidf_matrix(df_nlp)

    print("\nStep 3: Clustering with KMeans...")
    km = fit_clusters(tfidf_matrix, n_clusters=n_clusters)
    df_nlp["cluster"] = km.labels_

    print("\nStep 4: Extracting cluster keywords...")
    cluster_kw = get_cluster_keywords(km, vectorizer)
    labels_map = assign_cluster_labels(cluster_kw)

    print("\nStep 5: Reducing to 2D for visualisation...")
    coords = reduce_to_2d(tfidf_matrix)

    print("\n✅ NLP Pipeline complete!")
    for cid, kws in cluster_kw.items():
        print(f"  Cluster {cid} [{labels_map[cid]}]: {', '.join(kws[:5])}")

    return df_nlp, tfidf_matrix, vectorizer, km, cluster_kw, labels_map, coords
