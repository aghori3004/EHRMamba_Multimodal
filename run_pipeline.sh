#!/bin/bash
echo "🚀 STARTING EHR-MAMBA REPRODUCTION PIPELINE"

# Phase 1: Data
python src/01_build_cohort.py
python src/02_vectorize_notes.py
python src/03_build_vocab.py
python src/04_create_sequences.py

# Phase 2: Deep Learning Training
python src/train.py | tee logs_multimodal.txt
mv checkpoints/mamba_epoch_10.pth checkpoints/model_multimodal.pth

python src/train_unimodal.py | tee logs_unimodal.txt
mv checkpoints/unimodal_epoch_10.pth checkpoints/model_unimodal.pth

python src/train_mortality.py | tee logs_mortality.txt
python src/train_mortality_unimodal.py | tee logs_mortality_unimodal.txt

# Phase 3: Baselines & Eval
python src/train_baseline.py
python src/evaluate_final.py

echo "✅ PIPELINE COMPLETE. OPEN NOTEBOOK TO VIEW RESULTS."
