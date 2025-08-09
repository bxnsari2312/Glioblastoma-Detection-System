import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'glioma_grading.csv')

def load_preprocessed(path=DATA_CSV):
    df = pd.read_csv(path)
    return df

def clean_and_encode(df, target='Grade'):
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=[target], errors='ignore')
    y = df[target].map({'LGG':0, 'GBM':1}) if df[target].dtype==object else df[target]
    return X, y

def scale_features(X):
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled

if __name__ == '__main__':
    print('This module provides helper functions for preprocessing.')
