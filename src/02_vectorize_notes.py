import torch
import pandas as pd
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent
RAW_NOTE = PROJECT_ROOT / 'data/mimic_iv_note'
PROCESSED = PROJECT_ROOT / 'data/processed'
COHORT_PATH = PROCESSED / 'cohort_split.pkl'

BATCH_SIZE = 32 # Safe for 8GB VRAM
DEVICE = torch.device('cuda') 

def extract_embeddings(hadm_list, all_notes_df, tokenizer, model, desc):
    embeddings = {}
    
    # Filter dataframe to only relevant admissions
    subset = all_notes_df[all_notes_df['hadm_id'].isin(hadm_list)].copy()
    # Sort to match the order (optional but good for debugging)
    subset = subset.set_index('hadm_id').loc[hadm_list].reset_index()
    
    texts = subset['text'].tolist()
    ids = subset['hadm_id'].tolist()
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch_text = texts[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(
            batch_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(DEVICE)
        
        # Forward pass (No Grad to save memory)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Take [CLS] token (first token) as sentence embedding
        cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        for h_id, vec in zip(batch_ids, cls_tokens):
            embeddings[h_id] = vec
            
    return embeddings

def main():
    print("🚀 Starting Note Vectorization...")
    
    # 1. Load Cohort
    with open(COHORT_PATH, 'rb') as f:
        cohort = pickle.load(f)
    
    # 2. Load Model
    print("   Loading Bio_ClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
    model.eval()
    
    # 3. Load Raw Notes (Huge file, might take a moment)
    print("   Reading CSV...")
    df_notes = pd.read_csv(RAW_NOTE / 'discharge.csv.gz')
    
    # 4. Process Train
    train_vecs = extract_embeddings(cohort['train_hadm_ids'], df_notes, tokenizer, model, "Vectorizing Train")
    with open(PROCESSED / 'train_note_embeddings.pkl', 'wb') as f:
        pickle.dump(train_vecs, f)
        
    # 5. Process Test
    test_vecs = extract_embeddings(cohort['test_hadm_ids'], df_notes, tokenizer, model, "Vectorizing Test")
    with open(PROCESSED / 'test_note_embeddings.pkl', 'wb') as f:
        pickle.dump(test_vecs, f)
        
    print(f"✅ Saved {len(train_vecs)} training vectors and {len(test_vecs)} testing vectors.")

if __name__ == "__main__":
    main()