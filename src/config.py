from pathlib import Path
import torch

# Project Structure
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data/processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Paths to Artifacts
TRAIN_DATA = DATA_DIR / 'train_data.pt'
TEST_DATA = DATA_DIR / 'test_data.pt'
TRAIN_NOTES = DATA_DIR / 'train_note_embeddings.pkl'
TEST_NOTES = DATA_DIR / 'test_note_embeddings.pkl'
VOCAB_PATH = DATA_DIR / 'vocab.pkl'
COHORT_PATH = DATA_DIR / 'cohort_split.pkl'

# Hyperparameters
CLASS_WEIGHTS = None # Optional for imbalanced loss
BATCH_SIZE = 16      # Optimized for 8GB VRAM
LEARNING_RATE = 1e-4
EPOCHS = 10
GRAD_CLIP = 1.0      # Crucial for Mamba stability
ACCUMULATE_GRAD = 2  # Effective Batch Size = 32

# Model Config (Simplified Mamba)
D_MODEL = 256
N_LAYERS = 4
DROPOUT = 0.1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')