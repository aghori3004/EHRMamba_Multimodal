import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from config import *
from dataset import EHRDataset
from model import MambaEHR

DTYPE = torch.float32 

def collate_fn(batch):
    input_ids = pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-100)
    notes = torch.stack([x['note_tensor'] for x in batch])
    mort_labels = torch.stack([x['mortality_label'] for x in batch])
    return {
        'input_ids': input_ids, 'labels': labels, 
        'note_tensor': notes, 'mortality_label': mort_labels
    }

def evaluate_forecasting(model, loader, mode='Multimodal'):
    model.eval()
    correct_1, correct_5, correct_10 = 0, 0, 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"   Forecasting ({mode})"):
            ids = batch['input_ids'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            notes = batch['note_tensor'].to(DEVICE)
            model_notes = None if mode == 'Unimodal' else notes
            
            logits = model(ids, note_tensor=model_notes, task='forecasting')
            
            mask = lbls != -100
            active_logits = logits[mask]
            active_labels = lbls[mask]
            
            if active_labels.numel() == 0: continue
            
            _, top_preds = torch.topk(active_logits, 10, dim=-1)
            expanded_labels = active_labels.unsqueeze(1)
            matches = (top_preds == expanded_labels)
            
            correct_1 += matches[:, :1].any(dim=1).sum().item()
            correct_5 += matches[:, :5].any(dim=1).sum().item()
            correct_10 += matches[:, :10].any(dim=1).sum().item()
            total_tokens += active_labels.numel()
            
    return {
        'Task': 'Forecasting',
        'Model': mode,
        'Metric_1': (correct_1 / total_tokens) * 100,
        'Metric_2': (correct_5 / total_tokens) * 100,
        'Metric_3': (correct_10 / total_tokens) * 100
    }

def evaluate_mortality(model, loader, mode='Multimodal'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"   Mortality ({mode})"):
            ids = batch['input_ids'].to(DEVICE)
            notes = batch['note_tensor'].to(DEVICE)
            targets = batch['mortality_label'].to(DEVICE)
            
            model_notes = None if mode == 'Unimodal' else notes
            
            logits = model(ids, note_tensor=model_notes, task='mortality').squeeze()
            
            if logits.ndim == 0: logits = logits.unsqueeze(0)
            
            # --- FIX: CLAMP LOGITS TO PREVENT NaNs ---
            logits = torch.clamp(logits, min=-10, max=10)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    if np.isnan(all_preds).any():
        print("⚠️ Warning: NaNs detected. Filling with 0.5.")
        all_preds = np.nan_to_num(all_preds, nan=0.5)

    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    
    return {
        'Task': 'Mortality',
        'Model': mode,
        'Metric_1': auc, 
        'Metric_2': ap, 
        'Metric_3': 0.0 
    }

def main():
    print(f"🚀 Generating Final Metrics...")
    
    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    vocab_size = len(vocab) + 1 
    
    dataset = EHRDataset(TEST_DATA, TEST_NOTES, Path(DATA_DIR / '../processed/cohort_split.pkl'))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    results = []
    
    # 1. Forecasting
    print("\n--- Forecasting Evaluation ---")
    model_uni = MambaEHR(vocab_size, D_MODEL, N_LAYERS, DROPOUT).to(DEVICE)
    model_uni.load_state_dict(torch.load(CHECKPOINT_DIR / 'model_unimodal.pth', map_location=DEVICE), strict=False)
    results.append(evaluate_forecasting(model_uni, loader, mode='Unimodal'))
    
    model_multi = MambaEHR(vocab_size, D_MODEL, N_LAYERS, DROPOUT).to(DEVICE)
    model_multi.load_state_dict(torch.load(CHECKPOINT_DIR / 'model_multimodal.pth', map_location=DEVICE), strict=False)
    results.append(evaluate_forecasting(model_multi, loader, mode='Multimodal'))
    
    # 2. Mortality
    print("\n--- Mortality Evaluation ---")
    # Multimodal Mortality
    model_mort_multi = MambaEHR(vocab_size, D_MODEL, N_LAYERS, DROPOUT).to(DEVICE)
    model_mort_multi.load_state_dict(torch.load(CHECKPOINT_DIR / 'model_mortality.pth', map_location=DEVICE))
    results.append(evaluate_mortality(model_mort_multi, loader, mode='Multimodal'))
    
    # Unimodal Mortality (New!)
    model_mort_uni = MambaEHR(vocab_size, D_MODEL, N_LAYERS, DROPOUT).to(DEVICE)
    model_mort_uni.load_state_dict(torch.load(CHECKPOINT_DIR / 'model_mortality_unimodal.pth', map_location=DEVICE))
    results.append(evaluate_mortality(model_mort_uni, loader, mode='Unimodal'))
    
    df = pd.DataFrame(results)
    df.columns = ['Task', 'Model', 'Top-1 / AUC', 'Top-5 / AUPRC', 'Top-10']
    
    print("\n" + "="*60)
    print("📄 FINAL REPORT DATA")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    df.to_csv(PROJECT_ROOT / 'final_metrics.csv', index=False)

if __name__ == "__main__":
    main()