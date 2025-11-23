import torch
import pickle
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import pandas as pd

# Import modules
from config import *
from dataset import EHRDataset
from model import MambaEHR

# --- CONFIG ---
# Ensure this matches training config
USE_BFLOAT16 = True 
DTYPE = torch.bfloat16 if USE_BFLOAT16 else torch.float16

def collate_fn(batch):
    input_ids = pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-100)
    notes = torch.stack([x['note_tensor'] for x in batch])
    return {'input_ids': input_ids, 'labels': labels, 'note_tensor': notes}

def calculate_metrics(model, loader, mode='Multimodal'):
    model.eval()
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    total_tokens = 0
    
    desc = f"Evaluating {mode}"
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch['input_ids'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            notes = batch['note_tensor'].to(DEVICE)
            
            # --- CRITICAL SWITCH ---
            # If evaluating Unimodal, we must pass note_tensor=None
            # to ensure the model ignores the notes, just like in training.
            if mode == 'Unimodal':
                model_notes = None
            else:
                model_notes = notes
                
            with torch.cuda.amp.autocast(dtype=DTYPE):
                logits = model(ids, note_tensor=model_notes)
            
            # Mask padding (label -100)
            mask = lbls != -100
            active_logits = logits[mask]
            active_labels = lbls[mask]
            
            if active_labels.numel() == 0: continue
            
            # Top-k Metrics
            # Get top 10 predictions for each token
            _, top_preds = torch.topk(active_logits, 10, dim=-1)
            
            # Expand labels for comparison
            expanded_labels = active_labels.unsqueeze(1)
            
            # Check matches
            matches = (top_preds == expanded_labels)
            
            # Sum up correct predictions
            correct_1 += matches[:, :1].any(dim=1).sum().item()
            correct_5 += matches[:, :5].any(dim=1).sum().item()
            correct_10 += matches[:, :10].any(dim=1).sum().item()
            total_tokens += active_labels.numel()
            
    return {
        'Model': mode,
        'Top-1 Acc': (correct_1 / total_tokens) * 100,
        'Top-5 Acc': (correct_5 / total_tokens) * 100,
        'Top-10 Acc': (correct_10 / total_tokens) * 100
    }

def main():
    print(f"🚀 Starting Evaluation on {DEVICE}")
    
    # 1. Load Data
    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    vocab_size = len(vocab) + 1 
    
    # Load TEST set
    dataset = EHRDataset(TEST_DATA, TEST_NOTES)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    results = []
    
    # 2. Evaluate Multimodal Model
    print("\n--- Testing Multimodal Model ---")
    model_multi = MambaEHR(vocab_size, D_MODEL, N_LAYERS, DROPOUT).to(DEVICE)
    # Load weights
    model_multi.load_state_dict(torch.load(CHECKPOINT_DIR / 'model_multimodal.pth', map_location=DEVICE))
    res_multi = calculate_metrics(model_multi, loader, mode='Multimodal')
    results.append(res_multi)
    
    # 3. Evaluate Unimodal Model
    print("\n--- Testing Unimodal Baseline ---")
    model_uni = MambaEHR(vocab_size, D_MODEL, N_LAYERS, DROPOUT).to(DEVICE)
    # Load weights
    model_uni.load_state_dict(torch.load(CHECKPOINT_DIR / 'model_unimodal.pth', map_location=DEVICE))
    res_uni = calculate_metrics(model_uni, loader, mode='Unimodal')
    results.append(res_uni)
    
    # 4. Print Final Report
    print("\n" + "="*50)
    print("📊 FINAL RESULTS (Test Set N=5,000)")
    print("="*50)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*50)
    
    # Save to CSV for your report
    df.to_csv(PROJECT_ROOT / 'final_results.csv', index=False)
    print("Results saved to final_results.csv")

if __name__ == "__main__":
    main()