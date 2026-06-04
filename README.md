# CRANE: Conformal Reliable Augmented Neural Framework

**Reliable Multimodal Sentiment Analysis with Guaranteed Uncertainty Quantification**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Architecture

```mermaid
flowchart TB
    subgraph Input[" Multimodal Input "]
        direction LR
        T["📝 Text<br/><i>'This movie is great'</i>"]
        A["🎵 Audio<br/><i>16kHz waveform</i>"]
    end

    T --> RB["RoBERTa-base<br/><i>frozen · 124M params</i>"]
    A --> D2V["Data2Vec-Audio<br/><i>frozen</i>"]

    RB --> CLS1["CLS Token"]
    D2V --> CLS2["CLS Token"]

    CLS1 --> SB["CRANE Block"]
    CLS2 --> SB

    subgraph SB[" CRANE Block "]
        direction TB
        SA["Self-Attention"]
        FFN1["Feed-Forward"]
        CA["Cross-Attention<br/><i>text ↔ audio</i>"]
        FFN2["Feed-Forward"]
        SA --> FFN1 --> CA --> FFN2
    end

    SB --> FUSE

    subgraph FUSE[" UBG: Uncertainty Bidirectional Gate "]
        direction LR
        BG["Bi-Gating"]
        CPROJ["Confidence Proj.<br/><i>learned: text=0.98, audio=0.32</i>"]
        BG ~~~ CPROJ
    end

    FUSE --> HEADS

    subgraph HEADS[" Dual Head Output "]
        MEAN["Mean Head<br/>→ ŷ ∈ [-3, 3]"]
        VAR["Var Head<br/><i>frozen during std training</i>"]
    end

    MEAN --> CP
    VAR --> MVECP

    subgraph CP[" Conformal Prediction Layer "]
        direction LR
        SPLIT["Split CP<br/><i>constant width</i>"]
        ADAPT["Adaptive CP<br/><i>MC Dropout</i>"]
        MOND["Mondrian CP<br/><i>per-sentiment</i>"]
        CLS["Classification<br/><i>prediction sets</i>"]
    end

    subgraph MVECP[" MVE "]
        MVEADAPT["MVE + Adaptive CP<br/><i>learned variance</i>"]
    end

    SPLIT --> OUT
    ADAPT --> OUT
    MOND --> OUT
    CLS --> OUT
    MVEADAPT --> OUT

    subgraph OUT[" Guaranteed Output "]
        direction TB
        INTERVAL["📏 Prediction Interval<br/>ŷ ± q·σ  →  coverage ≥ 1−α"]
        SET["🏷️ Prediction Set<br/>{−1, 0, 1}  →  class coverage ≥ 1−α"]
    end

    style Input fill:#1a1a2e,stroke:#e94560,color:#eee
    style SB fill:#16213e,stroke:#0f3460,color:#eee
    style FUSE fill:#0f3460,stroke:#e94560,color:#eee
    style HEADS fill:#16213e,stroke:#0f3460,color:#eee
    style CP fill:#1a1a2e,stroke:#533483,color:#eee
    style OUT fill:#533483,stroke:#e94560,color:#eee
```

### Key Innovations vs Baseline

| Component | Baseline | CRANE |
|:---|:---|:---|
| Fusion | Bi-Gating (fixed weights) | **UBG** — learnable per-sample modality confidence |
| Output | Single scalar ŷ | **Dual head** — ŷ + variance σ² |
| Reliability | None | **6 conformal methods** with coverage guarantee |
| Modality gating | Static | **Learned**: conf_text=0.98, conf_audio=0.32 |

---

## Key Results (CMU-MOSI, α=0.10)

```mermaid
gantt
    title Coverage vs Width: Method Comparison (α=0.10, target=90%)
    dateFormat X
    axisFormat %s

    section ❌ Failed
    MC Dropout RAW (35%)     :crit, 0, 35
    MVE+Adaptive (92.3%)     :active, 0, 92
    Mondrian CP (91.1%)      :active, 0, 91
    Adaptive MC (92.7%)      :milestone, 0, 93
    Split CP (90.8%)         :active, 0, 91

    section ✅ Conformal
    Coverage Target (90%)    :milestone, 90, 1
```

| Method | Coverage | Med Width | Training |
|:---|:---:|:---:|:---:|
| MC Dropout RAW | 35.0% ✗ | 0.70 | 0 |
| **Adaptive (MC Dropout)** | **92.7%** ✓ | **2.97** | 0 |
| Split Conformal | 90.8% ✓ | 2.96 | 0 |
| Mondrian Conformal | 91.1% ✓ | 3.39 | 0 |
| MVE + Adaptive | 92.3% ✓ | 3.52 | fine-tune |
| Classification Set | 89.7% | 2.96 avg size | 0 |

> **MC Dropout RAW proves the necessity of conformal**: without calibration, coverage is 35% — a 55pp gap from the 90% target.

---

## UBG: Learned Modality Confidence

UBG learns per-sample modality weights through end-to-end training:

```
Confidence Distribution (test set, 686 samples)

  conf_text  ────────────────────────────────────█─  0.9814 avg
  conf_audio ─────────────█───────────────────────  0.3218 avg

  Corr(conf_text, conf_audio) = −0.415  ←  significant complementarity!

  Negative samples: text=0.980  audio=0.315
  Positive samples: text=0.983  audio=0.336
```

The model **independently discovers** that text dominates sentiment in MOSI, and learns to suppress audio when text is confident — all without manual rules.

---

## Calibration Sensitivity

```mermaid
xychart-beta
    title "Coverage Stability vs Calibration Set Size"
    x-axis "n_cal" [20, 40, 60, 80, 100, 140, 180, 220]
    y-axis "Coverage (%)" 85 --> 95
    line [87.7, 90.8, 90.8, 89.4, 90.8, 90.8, 89.4, 90.8]
    line [85.1, 85.1, 89.2, 88.3, 89.2, 92.1, 90.1, 90.1]
```

> **Only 40 calibration samples** are needed for stable coverage — crucial for data-scarce domains.

---

## Multimodal Uncertainty Decomposition

| Modality | Coverage | Med Width |
|:---|:---:|:---:|
| Text-only | 93.0% | 3.00 |
| Audio-only | 91.1% | **5.40** |
| **Multimodal (UBG)** | **92.7%** | **2.97** |

Multimodal CRANE is the **only configuration where width is narrower than text-only** — UBG's complementary fusion eliminates redundant uncertainty.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Convert CMU-MOSI raw data to CRANE format
python convert_for_crane.py

# Or extract audio from raw videos first
python extract_audio.py --dataset mosi
```

Expected structure:
```
data/MOSI/
├── label.csv          # video_id, clip_id, text, label, mode
└── wav/
    └── <video_id>/
        └── <clip_id>.wav
```

### 3. Train & Evaluate

```bash
# Single run (training + full conformal evaluation)
python run.py --seed 42 --dataset mosi

# SLURM cluster
sbatch run_crane.sh

# 4-seed ensemble (parallel)
sbatch run_ensemble.sh
```

### 4. Conformal Evaluation Output

After training, the pipeline automatically runs all 6 conformal methods:

```
======================================================================
 CONFORMAL PREDICTION EVALUATION
======================================================================
 1. Split Conformal (Constant Width)
 2. Adaptive Conformal (MC Dropout, K=20)
 3. MC Dropout RAW (Gaussian, no calibration)
 4. Mondrian Conformal (Per-Sentiment Conditional)
 5. MVE + Adaptive Conformal
 6. Comparison Table (α=0.10)
 7. Calibration Size Sensitivity
 8. Multimodal Uncertainty Decomposition
 9. Deep Ensemble (if checkpoints exist)
10. Classification Conformal: 7-Class Prediction Sets
======================================================================
```

---

## Code Structure

```
CRANE/
├── run.py                     # Main entry point
├── config.py                  # Model & training configuration
├── run_crane.sh               # SLURM single-run script
├── run_ensemble.sh            # SLURM 4-seed ensemble script
├── convert_for_crane.py       # Data format conversion
├── extract_audio.py           # Audio extraction from video
├── requirements.txt
└── utils/
    ├── en_model.py            # CRANE model (UBG + dual head)
    ├── en_train.py            # Training + conformal evaluation pipeline
    ├── crane_architecture.py  # CRANE cross-attention block
    ├── conformal.py           # 6 conformal predictors + metrics
    ├── data_loader.py         # Data loading (calibration-aware split)
    └── metricsTop.py          # Evaluation metrics + formatting
```

---

## Citation

```bibtex
@misc{crane2026,
  title={CRANE: Conformal Reliable Augmented Neural Framework for
         Multimodal Sentiment Analysis},
  year={2026},
  note={Work in progress}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
