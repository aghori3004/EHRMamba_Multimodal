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
    return {'input_ids': input_ids, 'labels': labels, 'note_tensor': notes}

def main():
    print(f"🚀 Starting UNIMODAL Training on {DEVICE}")
    
    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    vocab_size = len(vocab) + 1 
    
    # --- FIX: Passed COHORT_PATH ---
    dataset = EHRDataset(TRAIN_DATA, TRAIN_NOTES, COHORT_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = MambaEHR(vocab_size, D_MODEL, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=STABLE_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float16

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        valid_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, batch in enumerate(pbar):
            ids = batch['input_ids'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            # We ignore notes for Unimodal
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=dtype):
                # --- UNIMODAL FORCE: note_tensor=None ---
                logits = model(ids, note_tensor=None)
                loss = criterion(logits.view(-1, vocab_size), lbls.view(-1))
                loss = loss / ACCUMULATE_GRAD
            
            if torch.isnan(loss): continue

            loss.backward()
            
            if (i + 1) % ACCUMULATE_GRAD == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUMULATE_GRAD
            valid_batches += 1
            pbar.set_postfix({'loss': f"{loss.item() * ACCUMULATE_GRAD:.4f}"})
            
        avg = total_loss / max(1, valid_batches)
        print(f"✅ Epoch {epoch+1} Average Loss: {avg:.4f}")
        
        torch.save(model.state_dict(), CHECKPOINT_DIR / f'unimodal_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()