import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

DATA_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'glioma_grading.csv')

def load_and_prepare():
    df = pd.read_csv(DATA_CSV).dropna().drop_duplicates().reset_index(drop=True)
    if 'Grade' not in df.columns:
        raise RuntimeError('Grade column not found in preprocessed CSV.')
    X = df.drop(columns=['Grade'], errors='ignore')
    y = df['Grade'].map({'LGG':0,'GBM':1}) if df['Grade'].dtype==object else df['Grade']
    X_num = X.select_dtypes(include=[float,int]).copy()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_num)
    return X_scaled, y.values, list(X_num.columns)

if __name__ == '__main__':
    X, y, feats = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True),
        'Naive Bayes': GaussianNB()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print('===', name, '===')
        print('Accuracy:', round(accuracy_score(y_test, preds), 4))
        print('F1 (weighted):', round(f1_score(y_test, preds, average='weighted'), 4))
        try:
            prob = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, prob)
            print('ROC AUC:', round(auc,4))
        except Exception:
            pass
        print(classification_report(y_test, preds))
