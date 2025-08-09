import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression

DATA_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'glioma_grading.csv')

def load_and_prepare():
    df = pd.read_csv(DATA_CSV).dropna().drop_duplicates().reset_index(drop=True)
    if 'Grade' not in df.columns:
        raise RuntimeError('Grade column not found in preprocessed CSV.')
    X = df.drop(columns=['Grade'], errors='ignore')
    # Drop non-numeric columns for feature selection (e.g., Case_ID, Gender) or encode them
    X_num = X.select_dtypes(include=[float,int]).copy()
    y = df['Grade'].map({'LGG':0,'GBM':1}) if df['Grade'].dtype==object else df['Grade']
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns)
    return X_num, X_scaled, y

def filter_method(X_scaled, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=min(k, X_scaled.shape[1]))
    selector.fit(X_scaled, y)
    return list(X_scaled.columns[selector.get_support()])

def wrapper_method(X_scaled, y, k=10):
    estimator = LogisticRegression(max_iter=1000)
    selector = RFE(estimator, n_features_to_select=min(k, X_scaled.shape[1]))
    selector.fit(X_scaled, y)
    return list(X_scaled.columns[selector.get_support()])

def intrinsic_method(X_scaled, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_scaled)
    return list(X_scaled.columns[selector.get_support()])

if __name__ == '__main__':
    X_num, X_scaled, y = load_and_prepare()
    print('Numeric features detected:', list(X_num.columns))
    print('\nFilter-based selected:', filter_method(X_scaled,y, k=10))
    print('Wrapper-based selected:', wrapper_method(X_scaled,y, k=10))
    print('Intrinsic-based selected:', intrinsic_method(X_scaled, threshold=0.01))
