"""
data/download_data.py
---------------------
Downloads the Netflix dataset from Kaggle using the Kaggle API.

SETUP (one-time):
  1. Go to https://www.kaggle.com/account → API → Create New Token
  2. This downloads a kaggle.json file
  3. Place it at: ~/.kaggle/kaggle.json   (Mac/Linux)
                  C:\\Users\\YOU\\.kaggle\\kaggle.json  (Windows)
  4. Run: python data/download_data.py

OR manually:
  Download from: https://www.kaggle.com/datasets/shivamb/netflix-shows
  Place netflix_titles.csv inside the data/ folder.
"""

import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "netflix_titles.csv")

def download_via_kaggle():
    try:
        import kaggle  # pip install kaggle
    except ImportError:
        print("kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    print("Downloading Netflix dataset from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "shivamb/netflix-shows",
        path=DATA_DIR,
        unzip=True
    )
    print(f"✅  Dataset saved to: {CSV_PATH}")

def check_exists():
    if os.path.exists(CSV_PATH):
        print(f"✅  Dataset already exists at: {CSV_PATH}")
        return True
    return False

if __name__ == "__main__":
    if not check_exists():
        download_via_kaggle()
