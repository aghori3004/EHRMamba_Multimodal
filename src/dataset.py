import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from pathlib import Path

class EHRDataset(Dataset):
    def __init__(self, data_path, embedding_path, cohort_path, max_len=1024):
        print(f"⚡ Loading data from {data_path}...")
        raw_data = torch.load(data_path)
        
        print(f"⚡ Loading embeddings from {embedding_path}...")
        with open(embedding_path, 'rb') as f:
            self.note_lookup = pickle.load(f)

        with open(cohort_path, 'rb') as f:
            cohort = pickle.load(f)
        self.labels_map = cohort['mortality_labels']
            
        # --- CRITICAL FIX: FILTERING ---
        # Only keep patients who strictly have BOTH sequences and notes
        self.data = []
        missing_count = 0
        
        for item in raw_data:
            hadm_id = item['hadm_id']
            if hadm_id in self.note_lookup:
                self.data.append(item)
            else:
                missing_count += 1
                
        print(f"✅ Dataset loaded. Kept {len(self.data)} samples. Dropped {missing_count} (Missing Notes).")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        hadm_id = item['hadm_id']
        
        # Inputs and Labels for Next Token Prediction
        input_ids = item['input_ids']
        labels = item['labels']
        
        # Fetch the pre-computed BERT embedding (Shape: 768)
        note_vec = self.note_lookup[hadm_id]

        mort_label = self.labels_map.get(hadm_id, 0)
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'note_tensor': torch.tensor(note_vec, dtype=torch.float32),
            'mortality_label': torch.tensor(mort_label, dtype=torch.float32)
        }