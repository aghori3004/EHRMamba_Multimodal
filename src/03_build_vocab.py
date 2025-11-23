import pandas as pd
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_HOSP = PROJECT_ROOT / 'data/mimic_iv/hosp'
PROCESSED = PROJECT_ROOT / 'data/processed'
COHORT_PATH = PROCESSED / 'cohort_split.pkl'

def main():
    print("🚀 Building Vocabulary...")
    with open(COHORT_PATH, 'rb') as f: cohort = pickle.load(f)
    train_ids = set(cohort['train_hadm_ids'])
    
    print("   Loading Diagnoses...")
    df_diag = pd.read_csv(RAW_HOSP / 'diagnoses_icd.csv.gz')
    
    # Filter for Training Set ONLY
    df_train = df_diag[df_diag['hadm_id'].isin(train_ids)]
    
    # Get unique codes
    unique_codes = sorted(list(df_train['icd_code'].unique()))
    
    # Create mapping: Code -> Integer
    # 0 is reserved for Padding, so we start at 1
    vocab = {code: i+1 for i, code in enumerate(unique_codes)}
    vocab['<UNK>'] = len(vocab) + 1  # Handle unknown codes in test set
    
    print(f"   Vocab Size: {len(vocab)} unique codes.")
    
    with open(PROCESSED / 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("✅ Vocab saved.")

if __name__ == "__main__":
    main()