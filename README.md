# 🎬 Netflix Advanced Data Analysis Portfolio

> A full end-to-end data analyst portfolio project: EDA, NLP, Recommendation Engine, Time-Series, and an interactive Streamlit web app — all on the Netflix public dataset.

---

## 📁 Project Structure

```
netflix-eda/
├── data/
│   └── download_data.py          # Script to download dataset from Kaggle
├── notebooks/
│   ├── 01_data_cleaning.ipynb    # Data wrangling & preprocessing
│   ├── 02_eda.ipynb              # Advanced EDA (Matplotlib, Seaborn, Plotly)
│   ├── 03_nlp_descriptions.ipynb # NLP: TF-IDF, topic modeling, word clouds
│   ├── 04_recommendation.ipynb   # Content-based recommendation engine
│   ├── 05_timeseries.ipynb       # Time-series growth analysis + Prophet forecast
│   └── 06_network_analysis.ipynb # Actor/Director network graph (NetworkX)
├── src/
│   ├── data_cleaner.py           # Reusable data cleaning functions
│   ├── eda_plots.py              # All EDA visualization functions
│   ├── nlp_engine.py             # NLP pipeline (TF-IDF, clusters, word cloud)
│   ├── recommender.py            # Recommendation system class
│   └── network_graph.py          # NetworkX graph builder
├── app/
│   └── streamlit_app.py          #  Interactive Streamlit web app
├── requirements.txt              # All Python dependencies
├── .gitignore
└── README.md
```

---

##  Quick Start

### 1. Clone & install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/netflix-eda.git
cd netflix-eda
pip install -r requirements.txt
```

### 2. Download the dataset
```bash
# Option A: From Kaggle (requires Kaggle API key)
python data/download_data.py

# Option B: Manual download
# Go to: https://www.kaggle.com/datasets/shivamb/netflix-shows
# Download netflix_titles.csv and place it in the data/ folder
```

### 3. Run Jupyter notebooks
```bash
jupyter notebook notebooks/
```

### 4. Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 What's Inside

| Module | Techniques | Libraries |
|--------|-----------|-----------|
| EDA | Trend analysis, heatmaps, distributions | Matplotlib, Seaborn, Plotly |
| NLP | TF-IDF, topic modeling, word clouds | NLTK, scikit-learn, WordCloud |
| Recommendation | Cosine similarity, content-based filtering | scikit-learn, Pandas |
| Time Series | Growth rate, seasonality, Prophet forecast | Prophet, Plotly |
| Network Analysis | Actor/director co-collaboration graphs | NetworkX, Matplotlib |
| Web App | Full interactive dashboard + recommender | Streamlit |

---

## 💡 Key Insights Discovered
- 🇺🇸 USA dominates Netflix catalog (41.9% of all content)
- 📈 TV Shows grew **3× faster** than movies after 2016
- 🎭 Drama + International content = 60%+ of catalog
- 🔍 NLP reveals "crime", "investigation", "murder" cluster as crime-thriller genre
- 🤝 Actors like Adam Sandler appear in 20+ Netflix originals

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **pandas, numpy** — data manipulation
- **matplotlib, seaborn** — static visualization
- **plotly** — interactive charts
- **scikit-learn** — ML (TF-IDF, KMeans, cosine similarity)
- **nltk, wordcloud** — NLP
- **networkx** — graph analysis
- **prophet** — time series forecasting
- **streamlit** — web app deployment

---


