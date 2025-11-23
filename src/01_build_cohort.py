import pandas as pd
import pickle
from pathlib import Path

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent
RAW_HOSP = PROJECT_ROOT / 'data/mimic_iv/hosp'
RAW_NOTE = PROJECT_ROOT / 'data/mimic_iv_note'
PROCESSED = PROJECT_ROOT / 'data/processed'
PROCESSED.mkdir(exist_ok=True, parents=True)

TRAIN_SIZE = 50000
TEST_SIZE = 5000

def main():
    print(f"🚀 Starting Cohort Selection...")
    
    # 1. Load IDs from Notes (Discharge Summaries only)
    print("   Loading Discharge Summary IDs...")
    # We only need the ID columns to find intersection
    df_notes = pd.read_csv(RAW_NOTE / 'discharge.csv.gz', usecols=['subject_id', 'hadm_id'])
    unique_note_hadms = set(df_notes['hadm_id'].unique())
    print(f"   Found {len(unique_note_hadms)} admissions with notes.")

    # 2. Load IDs from Admissions
    print("   Loading Admission IDs & Labels...")
    df_adms = pd.read_csv(RAW_HOSP / 'admissions.csv.gz', usecols=['subject_id', 'hadm_id', 'hospital_expire_flag'])

    mortality_map = dict(zip(df_adms['hadm_id'], df_adms['hospital_expire_flag']))
    unique_adm_hadms = set(df_adms['hadm_id'].unique())
    
    # 3. Find Intersection
    # These are admissions that DEFINITELY have both structured data and text
    valid_hadms = sorted(list(unique_note_hadms.intersection(unique_adm_hadms)))
    print(f"   Found {len(valid_hadms)} valid overlapping admissions.")

    # 4. Strict Split (The Fix for your Bug)
    # We take the first chunk for training, and the NEXT chunk for testing.
    # This guarantees zero overlap.
    train_hadms = valid_hadms[:TRAIN_SIZE]
    test_hadms = valid_hadms[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]
    
    print(f"   Selected {len(train_hadms)} for TRAINING.")
    print(f"   Selected {len(test_hadms)} for TESTING.")

    # 5. Save the lists
    cohort_data = {
        'train_hadm_ids': train_hadms,
        'test_hadm_ids': test_hadms,
        'mortality_labels': mortality_map
    }
    
    output_path = PROCESSED / 'cohort_split.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(cohort_data, f)
        
    print(f"✅ Cohort split & labels saved to {output_path}")

if __name__ == "__main__":
    main()