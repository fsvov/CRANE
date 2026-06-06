# CRANE Architecture

Design decisions and rationale for the Conformal Reliable Augmented Neural Framework.

## Data Split: Plan C

```
Full MOSI (2199 samples, pre-defined MOSI train/valid/test splits)
├── Train: ～53% (～1174, from MOSI train, augmentation ON, after ES removal)
├── Cal:   ～10% (～229,  = MOSI valid,   augmentation OFF)
├── ES:    ～5%  (～110, ⊂ MOSI train,    augmentation OFF)
└── Test:  ～31% (～686,  = MOSI test,    augmentation OFF)
```

**Why Plan C (not Plan A or B):**
- Cal from val (not train): calibration residuals must be exchangeable with test. Train residuals are biased small because the model was optimized on them → coverage collapse (42.5%).
- Entire val as cal (not 5% slice): 110 samples is too few for stable quantile estimation → coverage instability (52%).
- ES split from train (not val): early stopping needs an unbiased signal from the training distribution.

## Frozen Encoders

RoBERTa-base and Data2Vec-Audio are frozen during all training.

**Why:** CMU-MOSI has only ~2,200 samples. Fine-tuning 124M+ parameters on this scale would overfit catastrophically. Frozen encoders act as strong regularizers.

## UBG vs Bi-Gating (Fixed Weights)

The baseline uses Bi-Gating with fixed (learned but static) weights. UBG replaces this with per-sample confidence projections:

```
conf_text  = σ(W_text · h_text)    # (768→1) Linear
conf_audio = σ(W_audio · h_audio)  # (768→1) Linear
h_fused    = conf_text * h_text + conf_audio * h_audio
```

**Why:** Fixed weights assume all samples benefit from the same text/audio ratio. UBG learns to suppress audio when text is confident (and vice versa), as shown by the negative correlation (r = -0.20) between per-sample confidences.

## Dual Head Output

```
Mean Head:  h_fused → Linear(768→512) → ReLU → Linear(512→1) → ŷ    (trained with MSE)
Var Head:   h_fused → Linear(768→512) → ReLU → Linear(512→1) → log σ²  (frozen, then fine-tuned with Gaussian NLL)
```

**Why:** Var head is only activated during MVE fine-tuning (freeze backbone → train var head → unfreeze all). This avoids the well-known instability of training mean and variance jointly from scratch.

## Adaptive Conformal as Practical Default

Split CP (constant width): coverage 90.7%, median width 2.93.
Adaptive CP (variable width): coverage 91.0%, median width 2.92.

The aggregate metrics are nearly identical. Why Adaptive?

1. Adaptive's width distribution is right-skewed — narrow for confident predictions, wide for uncertain ones. This enables automatic "trust/no-trust" routing in deployment.
2. Zero additional training cost — only K=20 MC dropout forward passes at inference.
3. If the model encounters distribution shift, MC dropout variance increases → intervals widen automatically — a built-in safety mechanism.

## Mondrian as Oracle Analysis

Mondrian CP groups calibration samples by true sentiment polarity. However, true polarity is unavailable at inference time. The current implementation is therefore **oracle subgroup analysis** — it validates the grouping strategy but cannot be deployed as-is.

For a deployable Mondrian variant, replace true polarity with inferred polarity (ŷ → predicted polarity group).

## Classification Conformal Under-Coverage

Classification CP (88.3%) misses the 90% target because `|ŷ - c_k| ≤ q` discretizes a continuous nonconformity score designed for regression intervals. The coverage loss (~1.7pp) is the price of using a single score function for two output paradigms.

## UBG vs Conformal: Separation of Concerns

Key finding: UBG confidences (text r≈0.13, audio r≈-0.04) are nearly uncorrelated with conformal interval widths.

This supports the three-layer separation:
- **UBG** → modality reliability (which modality to trust)
- **MC Dropout** → predictive uncertainty (how uncertain is this prediction)
- **Conformal** → coverage guarantee (mathematical calibration)

The evidence supports "complementary roles" rather than "proven mutual exclusion" — the correlation analysis is insufficient to rule out that UBG captures some uncertainty-related signal.

## File Responsibilities

| File | Scope |
|------|-------|
| `run.py` | CLI entry point |
| `config.py` | Paths and hyperparameters |
| `utils/crane_architecture.py` | CRANE Block (self-attn, cross-attn, FFN) |
| `utils/en_model.py` | CRANE model (UBG, dual head), MVE variant |
| `utils/en_train.py` | Training loop + 10-section conformal evaluation |
| `utils/conformal.py` | 6 conformal predictors, scoring functions, metrics |
| `utils/data_loader.py` | Plan C data split with calibration set |
| `utils/metricsTop.py` | Regression/classification metrics, formatting |
| `utils/visualization.py` | 9 publication-quality figures |
