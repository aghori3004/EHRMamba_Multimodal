import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import pickle

from config import *
from dataset import EHRDataset
from model import MambaEHR

# --- STABILITY CONFIG ---
USE_BFLOAT16 = True 
STABLE_LR = 1e-4

def collate_fn(batch):
    input_ids = pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-100)
    notes = torch.stack([x['note_tensor'] for x in batch])
    # Note: We don't need mortality labels for forecasting, but the collate must handle the dict
    return {'input_ids': input_ids, 'labels': labels, 'note_tensor': notes}

def main():
    print(f"🚀 Starting STABILIZED Training on {DEVICE}")
    
    print(f"   Loading Vocab from {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    vocab_size = len(vocab) + 1 
    
    # --- FIX: Passed COHORT_PATH here ---
    dataset = EHRDataset(TRAIN_DATA, TRAIN_NOTES, COHORT_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = MambaEHR(vocab_size, D_MODEL, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=STABLE_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(not USE_BFLOAT16))
    dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float16

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        valid_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, batch in enumerate(pbar):
            ids = batch['input_ids'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            notes = batch['note_tensor'].to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=dtype):
                logits = model(ids, note_tensor=notes)
                loss = criterion(logits.view(-1, vocab_size), lbls.view(-1))
                loss = loss / ACCUMULATE_GRAD
            
            if torch.isnan(loss): continue

            if USE_BFLOAT16:
                loss.backward()
                if (i + 1) % ACCUMULATE_GRAD == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                scaler.scale(loss).backward()
                if (i + 1) % ACCUMULATE_GRAD == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            
            current_loss = loss.item() * ACCUMULATE_GRAD
            total_loss += current_loss
            valid_batches += 1
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})
            
        avg = total_loss / max(1, valid_batches)
        print(f"✅ Epoch {epoch+1} Average Loss: {avg:.4f}")
        
        torch.save(model.state_dict(), CHECKPOINT_DIR / f'mamba_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()