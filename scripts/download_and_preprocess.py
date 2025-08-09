import os
import zipfile
import requests
import pandas as pd

UCI_ZIP_URL = 'https://archive.ics.uci.edu/static/public/759/glioma+grading+clinical+and+mutation+features+dataset.zip'
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
ZIP_PATH = os.path.join(DATA_DIR, 'glioma_uci.zip')
CSV_TARGET = os.path.join(DATA_DIR, 'glioma_grading.csv')

def download_zip(url, dest_path):
    print('Downloading dataset from UCI...')
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print('Downloaded zip to', dest_path)

def extract_csv_from_zip(zip_path, dest_dir):
    print('Extracting CSV files from zip...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
        print('Zip contains:', names)
        csv_names = [n for n in names if n.lower().endswith('.csv')]
        if not csv_names:
            raise RuntimeError('No CSV found in zip.')
        chosen = csv_names[0]
        print('Extracting', chosen)
        z.extract(chosen, path=dest_dir)
        extracted_path = os.path.join(dest_dir, chosen)
        return extracted_path

def preprocess(input_csv, output_csv):
    print('Preprocessing CSV:', input_csv)
    df = pd.read_csv(input_csv)
    # Replace placeholder missing values like '--' with NA and drop rows with any NA
    df = df.replace('--', pd.NA).dropna().reset_index(drop=True)
    # Ensure Grade column exists and standardize
    if 'Grade' not in df.columns:
        for c in df.columns:
            if c.strip().lower() == 'grade':
                df = df.rename(columns={c:'Grade'})
                break
    if 'Grade' in df.columns and df['Grade'].dtype == object:
        df['Grade'] = df['Grade'].str.strip().map({'LGG':0, 'GBM':1})
    # Save preprocessed CSV
    df.to_csv(output_csv, index=False)
    print('Saved preprocessed data to', output_csv)
    return output_csv

if __name__ == '__main__':
    download_zip(UCI_ZIP_URL, ZIP_PATH)
    extracted = extract_csv_from_zip(ZIP_PATH, os.path.join(os.path.dirname(__file__), '..', 'data'))
    processed = preprocess(extracted, CSV_TARGET)
    print('Done. Preprocessed file at:', processed)
