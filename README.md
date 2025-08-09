# Glioblastoma Detection using Machine Learning (Real UCI dataset)

This repository recreates the Glioma Grading Clinical and Mutation Features project using the **real UCI dataset**.
The code will download the dataset from the UCI repository, preprocess it (preprocessed option: drop rows with missing
values), run EDA, perform feature selection, and train baseline models.

## How this repo is structured
- `data/` — target folder for dataset and processed CSV.
- `scripts/download_and_preprocess.py` — downloads the UCI zip, extracts the CSV, preprocesses and saves `data/glioma_grading.csv`.
- `scripts/preprocess.py` — helper functions for cleaning/encoding/scaling.
- `scripts/feature_selection.py` — runs filter/wrapper/intrinsic selection on the preprocessed data.
- `scripts/train_model.py` — trains Logistic Regression, Random Forest, SVM, Naive Bayes and prints evaluation metrics.
- `notebooks/` — three notebooks: EDA, Feature Selection, Modeling (they call the scripts or load preprocessed CSV).
- `requirements.txt` — required Python packages.

## One-step usage (recommended)
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Run the download + preprocess script (this will fetch from UCI automatically):
   ```bash
   python scripts/download_and_preprocess.py
   ```
   This will create `data/glioma_grading.csv` (preprocessed: rows with missing values dropped, `Grade` encoded to 0/1).
3. Run the feature selection and training scripts:
   ```bash
   python scripts/feature_selection.py
   python scripts/train_model.py
   ```

## Notes
- The dataset is downloaded from UCI: `https://archive.ics.uci.edu/static/public/759/glioma+grading+clinical+and+mutation+features+dataset.zip`
- I chose **preprocessed = drop rows with missing values** for clarity and reproducibility. You can modify imputation strategy in `scripts/preprocess.py`.
- The notebooks are included for exploration; the scripts run headless CLI-friendly processes suitable for CI or automated runs.

## Acknowledgement
Dataset: Tasci et al., UCI Machine Learning Repository (2022).
