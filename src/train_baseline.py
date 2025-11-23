import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# Config
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data/processed'
COHORT_PATH = DATA_DIR / 'cohort_split.pkl'
RAW_HOSP = PROJECT_ROOT / 'data/mimic_iv/hosp'

def load_data():
    print("🚀 Loading Data for Baseline...")
    with open(COHORT_PATH, 'rb') as f:
        cohort = pickle.load(f)
        
    df_diag = pd.read_csv(RAW_HOSP / 'diagnoses_icd.csv.gz')
    train_ids = cohort['train_hadm_ids']
    test_ids = cohort['test_hadm_ids']
    labels_map = cohort['mortality_labels']
    
    def get_corpus_labels(hadm_ids):
        subset = df_diag[df_diag['hadm_id'].isin(hadm_ids)]
        # Bag of Codes: "Code1 Code2 Code3"
        corpus_df = subset.groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index()
        
        corpus, labels = [], []
        for hid in hadm_ids:
            row = corpus_df[corpus_df['hadm_id'] == hid]
            if not row.empty:
                corpus.append(row['icd_code'].values[0])
                labels.append(labels_map.get(hid, 0))
        return corpus, labels

    X_train, y_train = get_corpus_labels(train_ids)
    X_test, y_test = get_corpus_labels(test_ids)
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = load_data()
    
    print("   Vectorizing (Bag of Words)...")
    vectorizer = CountVectorizer(binary=True, max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("   Training Logistic Regression...")
    clf = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
    clf.fit(X_train_vec, y_train)
    
    preds = clf.predict_proba(X_test_vec)[:, 1]
    auc = roc_auc_score(y_test, preds)
    ap = average_precision_score(y_test, preds)
    
    print(f"\n✅ BASELINE RESULTS")
    print(f"   ROC-AUC: {auc:.4f} | AUPRC: {ap:.4f}")
    
    with open(PROJECT_ROOT / 'baseline_metrics.pkl', 'wb') as f:
        pickle.dump({'Model': 'Logistic Regression', 'AUC': auc, 'AUPRC': ap}, f)

if __name__ == "__main__":
    main()