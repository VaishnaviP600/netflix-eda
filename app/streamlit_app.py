"""
app/streamlit_app.py
---------------------
Netflix Content Intelligence — Interactive Streamlit Web App

HOW TO RUN:
  streamlit run app/streamlit_app.py

WHAT IT DOES:
  - Loads and caches the Netflix dataset
  - Provides 5 interactive pages via sidebar navigation
  - Uses Plotly for all interactive charts
  - Recommendation engine embedded directly in the UI

PAGE MAP:
  🏠 Overview      → KPI cards + quick charts
  📊 EDA Explorer  → Country map, genre treemap, duration charts
  🔤 NLP Analysis  → Word cloud + cluster explorer
  🎬 Recommender   → Type a title → get similar shows/movies
  📈 Time Series   → Growth analysis + Prophet forecast

CACHING STRATEGY (@st.cache_data / @st.cache_resource):
  - @st.cache_data   → for DataFrames (serializable data)
  - @st.cache_resource → for models (non-serializable objects like the recommender)
  - This means the dataset loads ONCE even as users interact with widgets.
"""

import os, sys
# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Netflix Data Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Netflix-themed CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  .main { background-color: #141414; }
  .block-container { padding-top: 2rem; }
  h1 { color: #E50914 !important; }
  h2, h3 { color: #ffffff !important; }
  .stMetric { background: #1c1c1c; border-radius: 10px; padding: 1rem; border: 1px solid #2a2a2a; }
  .metric-label { color: #888 !important; font-size: 12px !important; }
  .stButton > button { background-color: #E50914; color: white; border: none;
                       border-radius: 6px; font-weight: 600; transition: all 0.2s; }
  .stButton > button:hover { background-color: #b5070f; transform: scale(1.02); }
  .rec-card { background: #1c1c1c; border: 1px solid #2a2a2a; border-radius: 10px;
              padding: 1rem 1.25rem; margin: 0.5rem 0; transition: border-color 0.2s; }
  .rec-card:hover { border-color: #E50914; }
  .similarity-badge { background: #E50914; color: white; font-size: 12px;
                      font-weight: 700; padding: 2px 8px; border-radius: 4px; }
  .genre-tag { background: #2a2a2a; color: #ccc; font-size: 11px;
               padding: 2px 8px; border-radius: 12px; margin-right: 4px; }
  div[data-testid="stSidebar"] { background: #0d0d0d; }
  .insight-box { background: #1a1a1a; border-left: 4px solid #E50914;
                 border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Data path ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "netflix_titles.csv")


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS  (run once; reused on every interaction)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading Netflix dataset...")
def load_clean_data():
    """Load + clean data. Cached so it only runs once per session."""
    try:
        from src.data_cleaner import get_clean_df
        return get_clean_df(DATA_PATH)
    except FileNotFoundError:
        st.error("❌ Dataset not found!  \n"
                 "Please download `netflix_titles.csv` from Kaggle and place it in `data/`  \n"
                 "https://www.kaggle.com/datasets/shivamb/netflix-shows")
        st.stop()


@st.cache_resource(show_spinner="Building recommendation engine...")
def get_recommender(df):
    """Fit the recommender once; reuse for all queries."""
    from src.recommender import NetflixRecommender
    rec = NetflixRecommender()
    rec.fit(df)
    return rec


@st.cache_data(show_spinner="Running NLP pipeline...")
def get_nlp_results(_df):
    """Run full NLP pipeline. Underscored arg prevents hashing large DataFrame."""
    from src.nlp_engine import run_nlp_pipeline
    return run_nlp_pipeline(_df, n_clusters=8)


# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor="#1a1a1a",
    paper_bgcolor="#1a1a1a",
    font_color="#ccc",
    title_font_size=16,
    title_font_color="#fff",
    legend=dict(bgcolor="#222", bordercolor="#333"),
    margin=dict(t=50, b=40, l=40, r=20),
    colorway=["#E50914","#4a9eff","#1db954","#ff9500","#9b59b6","#e74c3c"],
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

def sidebar_nav(df):
    with st.sidebar:
        st.markdown("## 🎬 Netflix Analytics")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🏠 Overview", "📊 EDA Explorer", "🔤 NLP Analysis",
             "🎬 Recommender", "📈 Time Series"],
            label_visibility="hidden"
        )
        st.markdown("---")
        st.markdown("### 🔧 Global Filters")
        content_type = st.multiselect(
            "Content Type",
            ["Movie", "TV Show"],
            default=["Movie", "TV Show"]
        )
        year_range = st.slider(
            "Year Added",
            min_value=2008, max_value=2021,
            value=(2008, 2021)
        )
        st.markdown("---")
        st.caption(f"📋 Dataset: {len(df):,} titles  \n"
                   f"🗓️ Range: 2008–2021  \n"
                   f"🌍 Countries: 749")
    return page, content_type, year_range


def apply_filters(df, content_type, year_range):
    mask = (
        df["type"].isin(content_type) &
        df["year_added"].between(*year_range)
    )
    return df[mask]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_overview(df):
    st.title("🎬 Netflix Content Intelligence")
    st.markdown("An end-to-end data analysis portfolio project on the Netflix public dataset.")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Titles", f"{len(df):,}")
    k2.metric("Movies", f"{(df['type']=='Movie').sum():,}")
    k3.metric("TV Shows", f"{(df['type']=='TV Show').sum():,}")
    k4.metric("Countries", f"{df['primary_country'].nunique():,}")
    k5.metric("Genres", f"{df['primary_genre'].nunique():,}")

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    # ── Stacked bar ───────────────────────────────────────────────────────────
    with col_left:
        yearly = (df[df["year_added"].between(2008, 2021)]
                  .groupby(["year_added", "type"])
                  .size().reset_index(name="count"))
        yearly["year_added"] = yearly["year_added"].astype(int)
        fig = px.bar(yearly, x="year_added", y="count", color="type",
                     barmode="stack", title="📈 Content Added Per Year",
                     color_discrete_map={"Movie": "#E50914", "TV Show": "#4a9eff"},
                     labels={"year_added": "Year", "count": "Titles", "type": "Type"})
        fig.update_layout(**PLOTLY_LAYOUT, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # ── Donut ─────────────────────────────────────────────────────────────────
    with col_right:
        type_counts = df["type"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=type_counts.index, values=type_counts.values,
            hole=0.6, marker_colors=["#E50914", "#4a9eff"],
            textfont_size=14, hoverinfo="label+percent+value"
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Content Split",
                           annotations=[{"text": f"{len(df):,}<br>Titles",
                                         "x": 0.5, "y": 0.5,
                                         "font_size": 16, "showarrow": False,
                                         "font_color": "#fff"}])
        st.plotly_chart(fig2, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown("### 💡 Key Insights")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""<div class="insight-box">
            <b>🚀 Explosive Growth</b><br>
            Netflix grew its catalog <b>40×</b> from 2008 to 2019.
            Peak year was 2019 with over 2,000 titles added.
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="insight-box">
            <b>🌍 USA Dominates</b><br>
            41.9% of all content is from the United States.
            India is #2 with 972 titles — Netflix's biggest international bet.
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="insight-box">
            <b>📺 TV Show Surge</b><br>
            TV Shows grew 3× faster than movies after 2016 —
            signalling a deliberate shift to serialised content.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def page_eda(df):
    st.title("📊 EDA Explorer")

    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Countries", "🎭 Genres", "⏱ Duration", "📅 Seasonal"])

    # ── TAB 1: Countries ──────────────────────────────────────────────────────
    with tab1:
        st.subheader("Content by Country")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            n_countries = st.slider("Show top N countries", 5, 30, 15, key="nc")

        top_c = df[df["primary_country"] != "Unknown"]["primary_country"].value_counts().head(n_countries)
        fig = px.bar(top_c.reset_index(), x="count", y="primary_country",
                     orientation="h", title=f"Top {n_countries} Content-Producing Countries",
                     color="count", color_continuous_scale="Reds",
                     labels={"primary_country": "", "count": "Titles"})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # World map
        from src.data_cleaner import explode_countries
        country_counts = (explode_countries(df)[lambda x: x["country_single"] != "Unknown"]
                          .groupby("country_single").size().reset_index(name="count"))
        fig_map = px.choropleth(country_counts,
                                locations="country_single", locationmode="country names",
                                color="count", color_continuous_scale="Reds",
                                title="🌍 Netflix Titles by Country",
                                labels={"count": "Titles", "country_single": "Country"})
        fig_map.update_layout(**PLOTLY_LAYOUT,
                              geo=dict(bgcolor="#1a1a1a", showframe=False,
                                       showland=True, landcolor="#222",
                                       showocean=True, oceancolor="#141414"))
        st.plotly_chart(fig_map, use_container_width=True)

    # ── TAB 2: Genres ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Genre Analysis")
        from src.data_cleaner import explode_genres
        df_g = explode_genres(df)

        col_a, col_b = st.columns(2)
        with col_a:
            genre_counts = df_g["genre"].value_counts().head(15).reset_index()
            fig = px.bar(genre_counts, x="count", y="genre", orientation="h",
                         title="Top 15 Genres", color="count",
                         color_continuous_scale="Reds",
                         labels={"genre": "", "count": "Titles"})
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Treemap
            counts = (df_g.groupby(["genre", "type"]).size().reset_index(name="count")
                      .sort_values("count", ascending=False).head(40))
            fig_tree = px.treemap(counts, path=["type", "genre"], values="count",
                                  color="count", color_continuous_scale="Reds",
                                  title="Genre Treemap")
            fig_tree.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_tree, use_container_width=True)

        # Genre over time
        top6 = df_g["genre"].value_counts().head(6).index.tolist()
        trend = (df_g[df_g["genre"].isin(top6) & df_g["year_added"].between(2013, 2021)]
                 .groupby(["year_added", "genre"]).size().reset_index(name="count"))
        trend["year_added"] = trend["year_added"].astype(int)
        fig_trend = px.line(trend, x="year_added", y="count", color="genre",
                            title="📈 Genre Trends Over Time", markers=True,
                            labels={"year_added": "Year", "count": "Titles", "genre": "Genre"})
        fig_trend.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── TAB 3: Duration ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("Duration Analysis")
        col_a, col_b = st.columns(2)

        movies = df[(df["type"] == "Movie") & df["duration_numeric"].between(30, 300)]
        with col_a:
            fig = px.histogram(movies, x="duration_numeric", nbins=50,
                               title="Movie Runtime Distribution",
                               labels={"duration_numeric": "Duration (minutes)"},
                               color_discrete_sequence=["#E50914"])
            fig.add_vline(x=movies["duration_numeric"].median(), line_dash="dash",
                          line_color="#4a9eff",
                          annotation_text=f"Median: {movies['duration_numeric'].median():.0f} min",
                          annotation_font_color="#4a9eff")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            shows = df[(df["type"] == "TV Show") & df["duration_numeric"].between(1, 15)]
            season_counts = shows["duration_numeric"].value_counts().sort_index().reset_index()
            fig2 = px.bar(season_counts, x="duration_numeric", y="count",
                          title="TV Show Season Count",
                          labels={"duration_numeric": "Seasons", "count": "Shows"},
                          color_discrete_sequence=["#4a9eff"])
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        # Duration by genre box plot
        top_genres = movies["primary_genre"].value_counts().head(10).index
        movies_filtered = movies[movies["primary_genre"].isin(top_genres)]
        fig3 = px.box(movies_filtered, x="primary_genre", y="duration_numeric",
                      title="Movie Duration by Genre",
                      labels={"primary_genre": "", "duration_numeric": "Duration (min)"},
                      color="primary_genre",
                      color_discrete_sequence=px.colors.sequential.Reds)
        fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 4: Seasonal ───────────────────────────────────────────────────────
    with tab4:
        st.subheader("Seasonal Release Patterns")
        monthly = (df[df["year_added"].between(2015, 2021)]
                   .groupby(["year_added", "month_name"])
                   .size().reset_index(name="count"))
        monthly["year_added"] = monthly["year_added"].astype(int)

        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["month_name"] = pd.Categorical(monthly["month_name"], categories=month_order, ordered=True)
        pivot = monthly.pivot(index="year_added", columns="month_name", values="count").fillna(0)

        fig = px.imshow(pivot, color_continuous_scale="Reds",
                        title="🗓️ Monthly Release Heatmap (2015–2021)",
                        labels={"color": "Titles Added"}, aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Day of week
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df["day_of_week"].value_counts().reindex(dow_order).fillna(0).reset_index()
        fig2 = px.bar(dow, x="day_of_week", y="count",
                      title="Content Added by Day of Week",
                      labels={"day_of_week": "", "count": "Titles"},
                      color="count", color_continuous_scale="Reds")
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: NLP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_nlp(df):
    st.title("🔤 NLP Analysis")
    st.markdown("Applying TF-IDF vectorization and KMeans clustering to Netflix descriptions.")

    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
        **TF-IDF** (Term Frequency–Inverse Document Frequency) assigns a weight to each word
        in each description. Words that appear often in ONE description but rarely across ALL
        descriptions get high scores — these are the "signal" words.

        **KMeans Clustering** then groups similar descriptions into thematic clusters.
        Each cluster represents a content theme like "Crime Thriller" or "Family Comedy".

        **Word Cloud**: bigger words have higher total TF-IDF weight across all content.
        """)

    if st.button("🚀 Run NLP Pipeline (takes ~30 seconds first time)", type="primary"):
        with st.spinner("Running NLP pipeline..."):
            df_nlp, tfidf_matrix, vectorizer, km, cluster_kw, labels_map, coords = get_nlp_results(df)
        st.success(f"✅ Clustered {len(df_nlp):,} titles into {len(cluster_kw)} themes!")

        # Word cloud (using plotly-compatible frequency chart)
        import numpy as np
        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        top_words = pd.DataFrame({"word": feature_names, "score": scores})\
                      .sort_values("score", ascending=False).head(40)

        fig = px.bar(top_words, x="score", y="word", orientation="h",
                     title="🔤 Top Keywords by TF-IDF Score",
                     color="score", color_continuous_scale="Reds",
                     labels={"score": "TF-IDF Score", "word": ""})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster overview
        st.subheader("📦 Content Clusters")
        df_nlp["cluster_label"] = df_nlp["cluster"].map(labels_map)
        cluster_sizes = df_nlp.groupby(["cluster", "cluster_label"]).size().reset_index(name="count")
        cluster_sizes["label"] = cluster_sizes.apply(lambda r: f"Cluster {r['cluster']}: {r['cluster_label']}", axis=1)

        fig2 = px.pie(cluster_sizes, values="count", names="label",
                      title="Cluster Size Distribution", hole=0.4,
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

        # Cluster keyword tables
        st.subheader("🔑 Top Keywords per Cluster")
        n_cols = 4
        cluster_items = list(cluster_kw.items())
        for row_start in range(0, len(cluster_items), n_cols):
            cols = st.columns(n_cols)
            for j, (cid, kws) in enumerate(cluster_items[row_start:row_start+n_cols]):
                with cols[j]:
                    label = labels_map.get(cid, "misc")
                    st.markdown(f"**Cluster {cid}: {label}**")
                    for kw in kws[:8]:
                        st.markdown(f"- {kw}")

        # Scatter plot
        df_nlp["x"] = coords[:, 0]
        df_nlp["y"] = coords[:, 1]
        fig3 = px.scatter(df_nlp.sample(min(3000, len(df_nlp))),
                          x="x", y="y", color="cluster_label",
                          hover_data=["title"], opacity=0.6,
                          title="📍 Content Cluster Map (TF-IDF → 2D SVD)",
                          labels={"x": "SVD Component 1", "y": "SVD Component 2", "cluster_label": "Cluster"})
        fig3.update_traces(marker_size=4)
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("👆 Click the button above to run the NLP analysis pipeline.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════

def page_recommender(df):
    st.title("🎬 Content Recommendation Engine")
    st.markdown("**Content-based filtering** using cosine similarity on TF-IDF + genre + cast features.")

    with st.expander("ℹ️ How does cosine similarity work?"):
        st.markdown("""
        Each Netflix title is represented as a **high-dimensional vector** combining:
        - TF-IDF of description (10,000 dims)
        - Genre encoding (500 dims)
        - Cast encoding (3,000 dims)
        - Director encoding (1,000 dims)

        **Cosine similarity** measures the angle between two vectors:
        - Score = 1.0 → identical content profile
        - Score = 0.0 → completely different
        
        We find the N titles with the highest cosine similarity to your chosen title.
        """)

    rec = get_recommender(df)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        title_input = st.text_input("🔍 Enter a movie or TV show title",
                                    placeholder='e.g. "Stranger Things", "The Dark Knight"')
    with col2:
        n_recs = st.selectbox("Number of recommendations", [5, 10, 15, 20], index=1)
    with col3:
        same_type = st.checkbox("Same type only", value=False)

    if title_input:
        try:
            # Show the queried title's info first
            info = rec.get_title_info(title_input)
            st.markdown("---")
            st.markdown(f"### 📌 You searched for: **{info['title']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Type", info["type"])
            c2.metric("Year", str(info["release_year"]))
            c3.metric("Rating", info["rating"])
            c4.metric("Country", info["primary_country"])
            st.markdown(f"*{info['description']}*")
            st.markdown(f"**Genres:** {info['listed_in']}")
            st.markdown("---")

            # Get recommendations
            results = rec.recommend(title_input, n=n_recs, same_type=same_type)
            st.markdown(f"### 🎯 Top {n_recs} Recommendations")

            for i, row in results.iterrows():
                score_pct = int(row["similarity_score"] * 100)
                score_color = "#1db954" if score_pct > 50 else "#ff9500" if score_pct > 25 else "#888"
                with st.container():
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display:flex; justify-content:space-between; align-items:start">
                            <div>
                                <strong style="font-size:15px; color:#fff">{i+1}. {row['title']}</strong>
                                <span style="margin-left:12px; color:#888; font-size:12px">{row['type']} · {row['release_year']} · {row['primary_country']}</span>
                            </div>
                            <span style="background:{score_color}; color:#fff; font-size:12px;
                                         font-weight:700; padding:3px 10px; border-radius:20px; white-space:nowrap">
                                {score_pct}% match
                            </span>
                        </div>
                        <div style="margin-top:6px; font-size:12px; color:#aaa">{row['listed_in']}</div>
                        <div style="margin-top:6px; font-size:13px; color:#ccc">{str(row['description'])[:200]}{'...' if len(str(row['description']))>200 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

        except ValueError as e:
            st.error(str(e))

    # ── Popular titles to try ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎲 Try These Popular Titles")
    sample_titles = ["Stranger Things", "Ozark", "The Crown", "Breaking Bad",
                     "Bird Box", "The Witcher", "Money Heist", "Narcos"]
    cols = st.columns(4)
    for i, title in enumerate(sample_titles):
        with cols[i % 4]:
            if st.button(title, key=f"sample_{i}"):
                st.session_state["sample_title"] = title
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

def page_timeseries(df):
    st.title("📈 Time Series Analysis")
    st.markdown("Growth analysis of Netflix's content library over time.")

    # ── Full stacked area (Plotly) ────────────────────────────────────────────
    yearly = (df[df["year_added"].between(2008, 2021)]
              .groupby(["year_added", "type"])
              .size().unstack(fill_value=0).reset_index())
    yearly.columns = [str(c) for c in yearly.columns]
    yearly["year_added"] = yearly["year_added"].astype(int)
    yearly["Total"] = yearly.get("Movie", 0) + yearly.get("TV Show", 0)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Content Added Per Year",
                                        "YoY Growth Rate (%)",
                                        "Cumulative Library Size",
                                        "Movie vs TV: Growth Index (2012=100)"),
                        vertical_spacing=0.15)

    # ── Sub-plot 1: stacked bar ───────────────────────────────────────────────
    fig.add_trace(go.Bar(x=yearly["year_added"], y=yearly.get("Movie", pd.Series(0)),
                         name="Movies", marker_color="#E50914", showlegend=True), row=1, col=1)
    fig.add_trace(go.Bar(x=yearly["year_added"], y=yearly.get("TV Show", pd.Series(0)),
                         name="TV Shows", marker_color="#4a9eff", showlegend=True), row=1, col=1)

    # ── Sub-plot 2: YoY growth ────────────────────────────────────────────────
    growth_vals = yearly["Total"].pct_change().fillna(0) * 100
    bar_colors = ["#1db954" if v >= 0 else "#E50914" for v in growth_vals]
    fig.add_trace(go.Bar(x=yearly["year_added"], y=growth_vals.round(1),
                         marker_color=bar_colors, showlegend=False,
                         hovertemplate="%{y:.1f}%"), row=1, col=2)

    # ── Sub-plot 3: cumulative ────────────────────────────────────────────────
    cumulative = df[df["year_added"].between(2008, 2021)].groupby("year_added").size().cumsum()
    fig.add_trace(go.Scatter(x=cumulative.index.astype(int), y=cumulative.values,
                             fill="tozeroy", line_color="#ff9500",
                             fillcolor="rgba(255,149,0,0.2)", showlegend=False), row=2, col=1)

    # ── Sub-plot 4: growth index (2012 = 100) ────────────────────────────────
    yr_2012 = yearly[yearly["year_added"] >= 2012]
    for col_name, color, label in [("Movie", "#E50914", "Movies"),
                                    ("TV Show", "#4a9eff", "TV Shows")]:
        if col_name in yr_2012.columns:
            base = yr_2012[col_name].iloc[0]
            indexed = (yr_2012[col_name] / max(base, 1) * 100).values
            fig.add_trace(go.Scatter(x=yr_2012["year_added"], y=indexed,
                                     name=label, line_color=color,
                                     mode="lines+markers", showlegend=False), row=2, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=620, title_text="Netflix Growth Analysis",
                      barmode="stack", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ── Prophet forecast ─────────────────────────────────────────────────────
    st.subheader("🔮 Forecast: Future Content Growth")
    with st.expander("About Prophet Forecasting"):
        st.markdown("""
        **Facebook Prophet** is a time-series forecasting library that:
        - Detects trend changes automatically
        - Handles seasonal patterns (yearly, monthly)
        - Provides uncertainty intervals (shaded area)

        We feed it the yearly title counts and ask it to predict 3 years ahead.
        """)

    if st.button("📊 Run Prophet Forecast"):
        try:
            from prophet import Prophet
            yearly_total = (df[df["year_added"].between(2008, 2021)]
                            .groupby("year_added").size().reset_index())
            yearly_total.columns = ["ds", "y"]
            yearly_total["ds"] = pd.to_datetime(yearly_total["ds"].astype(str) + "-01-01")

            m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                        daily_seasonality=False, changepoint_prior_scale=0.3)
            m.fit(yearly_total)

            future = m.make_future_dataframe(periods=3, freq="YE")
            forecast = m.predict(future)

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=yearly_total["ds"], y=yearly_total["y"],
                                       mode="markers+lines", name="Actual",
                                       line_color="#E50914", marker_size=8))
            fig_f.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                                       mode="lines", name="Forecast",
                                       line={"color": "#4a9eff", "dash": "dash"}))
            fig_f.add_trace(go.Scatter(
                x=list(forecast["ds"]) + list(forecast["ds"][::-1]),
                y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
                fill="toself", fillcolor="rgba(74,158,255,0.15)",
                line_color="rgba(255,255,255,0)", name="Uncertainty",
            ))
            fig_f.update_layout(**PLOTLY_LAYOUT, title="Prophet Forecast: Netflix Content Growth")
            st.plotly_chart(fig_f, use_container_width=True)

            # Show forecast table
            future_rows = forecast[forecast["ds"].dt.year > 2021][
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ].copy()
            future_rows.columns = ["Year", "Predicted", "Lower Bound", "Upper Bound"]
            future_rows["Year"] = future_rows["Year"].dt.year
            future_rows = future_rows.set_index("Year").round(0).astype(int)
            st.dataframe(future_rows, use_container_width=True)

        except ImportError:
            st.error("Prophet not installed. Run: `pip install prophet`")
        except Exception as e:
            st.error(f"Forecast error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df_raw = load_clean_data()
    page, content_type, year_range = sidebar_nav(df_raw)
    df = apply_filters(df_raw, content_type, year_range)

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📊 EDA Explorer":
        page_eda(df)
    elif page == "🔤 NLP Analysis":
        page_nlp(df)
    elif page == "🎬 Recommender":
        page_recommender(df_raw)  # recommender uses full df (pre-filter)
    elif page == "📈 Time Series":
        page_timeseries(df)


if __name__ == "__main__":
    main()
