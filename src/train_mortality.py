import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import pickle
from sklearn.metrics import roc_auc_score

from config import *
from dataset import EHRDataset
from model import MambaEHR

# --- CONFIG ---
FINE_TUNE_EPOCHS = 5
USE_BFLOAT16 = True 
GRAD_CLIP = 1.0 

def collate_fn(batch):
    input_ids = pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=0)
    notes = torch.stack([x['note_tensor'] for x in batch])
    mort_labels = torch.stack([x['mortality_label'] for x in batch])
    return {'input_ids': input_ids, 'note_tensor': notes, 'mortality_label': mort_labels}

def main():
    print(f"🚀 Starting STABILIZED Mortality Fine-Tuning on {DEVICE}")
    
    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    vocab_size = len(vocab) + 1 
    
    # --- Use COHORT_PATH from config ---
    dataset = EHRDataset(TRAIN_DATA, TRAIN_NOTES, COHORT_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = MambaEHR(vocab_size, D_MODEL, N_LAYERS, DROPOUT).to(DEVICE)
    print("   Loading Pre-trained Weights...")
    # Load Multimodal Weights
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'model_multimodal.pth'), strict=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float16

    model.train()
    for epoch in range(FINE_TUNE_EPOCHS):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{FINE_TUNE_EPOCHS}")
        
        for batch in pbar:
            ids = batch['input_ids'].to(DEVICE)
            notes = batch['note_tensor'].to(DEVICE)
            targets = batch['mortality_label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=dtype):
                logits = model(ids, note_tensor=notes, task='mortality').squeeze()
                loss = criterion(logits, targets)
            
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            total_loss += loss.item()
            
            if logits.ndim == 0: preds = [torch.sigmoid(logits).detach().float().cpu().item()]; targs = [targets.float().cpu().item()]
            else: preds = torch.sigmoid(logits).detach().float().cpu().numpy().tolist(); targs = targets.float().cpu().numpy().tolist()
                
            all_preds.extend(preds)
            all_labels.extend(targs)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        try:
            auc = roc_auc_score(all_labels, all_preds)
            print(f"✅ Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f} | Train AUC: {auc:.4f}")
        except:
            print(f"✅ Epoch {epoch+1}")
            
        torch.save(model.state_dict(), CHECKPOINT_DIR / 'model_mortality.pth')

if __name__ == "__main__":
    main()