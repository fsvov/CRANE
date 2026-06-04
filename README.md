# CRANE-Net: Conformal Augmented Gated Encoder Network for Reliable Multimodal Sentiment Analysis

## 1. Code Structure

```
/CRANE/
    utils/
        crane_architecture.py  **CRANE block (cross-modal encoder)
        crane_model.py         **CRANE-Net model with UBG + dual head
        crane_train.py         **Training + conformal evaluation pipeline
        conformal.py          **Conformal predictors (split, adaptive, Mondrian, classification)
        data_loader.py        **Data loading with cal/ES/train split
        metricsTop.py         **Evaluation metrics + conformal metrics
    run.py                    **Main entry point
    run_crane.sh               **SLURM submission script
    run_ensemble.sh           **4-seed ensemble training
    convert_for_crane.py       **Data format conversion
    extract_audio.py          **Audio extraction from raw video
    research_plan.md          **Research roadmap and results
    requirements.txt          **Python dependencies
```

## 2. Data Directory

```
/data/
    mosi/
        raw/
        label.csv
```

## 3. Training

```bash
# Single run with full conformal evaluation
sbatch run_crane.sh

# 4-seed ensemble (parallel)
sbatch run_ensemble.sh
```

```
python run.py

options:
  --seed SEED               random seed (default: 42)
  --batch_size N             batch size (default: 8)
  --lr LR                    learning rate (default: 5e-6)
  --fusion_method METHOD     v1: concatenation, v2: bidirectional gating (default), v3: weighted, v4: transformer
  --dataset NAME             dataset name (default: mosi)
  --num_hidden_layers N      cross-modal encoder layers (default: 1)
  --model_save_path PATH     checkpoint save directory (default: ./saved/)
```
