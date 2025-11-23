import pandas as pd
import pickle
import torch
from pathlib import Path
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
RAW_HOSP = PROJECT_ROOT / 'data/mimic_iv/hosp'
PROCESSED = PROJECT_ROOT / 'data/processed'
COHORT_PATH = PROCESSED / 'cohort_split.pkl'
VOCAB_PATH = PROCESSED / 'vocab.pkl'

def process_split(hadm_ids, df_diag, vocab, split_name):
    data_list = []
    subset = df_diag[df_diag['hadm_id'].isin(hadm_ids)]
    grouped = subset.groupby('hadm_id')
    
    unk_id = vocab['<UNK>']
    
    for hadm_id, group in tqdm(grouped, desc=f"Processing {split_name}"):
        # Sort by sequence number to preserve order of events
        group = group.sort_values('seq_num')
        codes = group['icd_code'].tolist()
        
        # Map to IDs
        input_ids = [vocab.get(c, unk_id) for c in codes]
        
        if len(input_ids) < 2: continue # Need at least 2 codes for next-token prediction
            
        # Task: Predict next code. Input: [A, B], Label: [B, C]
        seq_ids = input_ids[:-1]
        seq_labels = input_ids[1:]
        
        data_list.append({
            'hadm_id': hadm_id,
            'input_ids': seq_ids,
            'labels': seq_labels
        })
    return data_list

def main():
    print("🚀 Creating Sequences...")
    with open(COHORT_PATH, 'rb') as f: cohort = pickle.load(f)
    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    
    df_diag = pd.read_csv(RAW_HOSP / 'diagnoses_icd.csv.gz')
    
    train_data = process_split(cohort['train_hadm_ids'], df_diag, vocab, "Train")
    torch.save(train_data, PROCESSED / 'train_data.pt')
    
    test_data = process_split(cohort['test_hadm_ids'], df_diag, vocab, "Test")
    torch.save(test_data, PROCESSED / 'test_data.pt')
    
    print(f"✅ Sequences saved. Train: {len(train_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    main()