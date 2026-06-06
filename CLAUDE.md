# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CRANE (Conformal Reliable Augmented Neural Framework) — multimodal sentiment analysis on CMU-MOSI with conformal prediction for guaranteed uncertainty quantification. Text + audio input → RoBERTa + Data2Vec (frozen) → cross-attention CRANE block → UBG learnable gate → dual-head output (ŷ, σ²) → 6 conformal prediction methods.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Data preparation (one-time)
python convert_for_crane.py
python extract_audio.py --dataset mosi

# Train + full conformal evaluation (single run)
python run.py --seed 42 --dataset mosi

# SLURM cluster
sbatch run_crane.sh        # single run
sbatch run_ensemble.sh     # 4-seed Deep Ensemble

# Run tests
python -m pytest tests/ -v
python -m pytest tests/test_conformal.py -v -k "Split"  # single test class
```

## Data split (Plan C)

```
MOSI (2199) → Train 55% (aug=ON) | Cal 10% (from val, aug=OFF) | ES 5% (from train, aug=OFF) | Test 25% (aug=OFF)
```

**Critical constraint:** Cal and Test both have `augment=False` to maintain exchangeability. Cal must come from the val partition, not train — train residuals are biased small and cause coverage collapse. The data split logic lives in `utils/data_loader.py`.

## Architecture flow

```
Text ──RoBERTa(frozen)──▶ h_text  ──┐
                                     ├── CRANE Block (Self-Attn→FFN→Cross-Attn→FFN) ──▶ UBG Gate ──▶ h_fused ──▶ Mean Head → ŷ
Audio──Data2Vec(frozen)──▶ h_audio──┘                                                                    └── Var Head  → σ² (MVE only)
```

**UBG (core innovation):** Per-sample modality confidence via `conf_text = σ(W_text · h_text)`, `conf_audio = σ(W_audio · h_audio)`. Learns `conf_text≈0.98, conf_audio≈0.37, corr≈-0.20` — discovers text dominance without manual rules.

**Dual head:** Mean head trained with MSE (standard training). Var head frozen during standard training, then fine-tuned with Gaussian NLL for MVE.

## Conformal pipeline

All conformal logic is in `utils/conformal.py`. The pipeline runs 10 evaluation sections in `utils/en_train.py:EnRun()`:

1. Split Conformal → 2. Adaptive (MC Dropout K=20) → 3. MC RAW baseline → 4. Mondrian → 5. MVE → 6. Comparison table → 7. Calibration sensitivity → 8. Multimodal decomposition → 9. Deep Ensemble → 10. Classification Conformal

**Key implementation details:**
- `SplitConformalPredictor`: `nonconformity = |y - ŷ|`, constant-width intervals
- `MCAdaptiveConformalPredictor`: `nonconformity = |y - ŷ| / σ_mc`, adaptive-width intervals
- `MondrianConformalPredictor`: per-group quantiles (currently oracle — uses true polarity, not deployable)
- `ClassificationConformalPredictor`: `|ŷ - c_k| ≤ q` → discrete prediction sets
- `mc_dropout_interval()`: Gaussian baseline (no calibration) — produces ~34% coverage, proving necessity of conformal

**Known caveats:**
- Mondrian with true sentiment labels is oracle analysis, not a deployable method. For deployable Mondrian, use predicted polarity as the grouping variable.
- Classification Conformal (88.3%) misses the 90% target due to discretization loss — `|ŷ - c_k| ≤ q` was designed for regression intervals.
- Do not claim "all methods achieve ≥90% coverage" — Classification CP is ~88.3%.
- Coverage guarantees are finite-sample marginal, not conditional. They require exchangeability between cal and test sets.

## Key files

| File | What it does |
|------|-------------|
| `utils/en_model.py` | `CRANEModel` (UBG + dual head), `CRANEModelMVE` subclass, `UncertaintyBidirectionalGate`, `gaussian_nll_loss()` |
| `utils/en_train.py` | `EnTrainer` — training loop, MC inference (`do_mc_inference`, `do_mc_inference_modality`), MVE fine-tuning (`do_train_mve`). `EnRun()` — master function running all 10 conformal evaluation sections. Contains `_ubg_width_data` deferred assignment to avoid viz dict ordering issues. |
| `utils/conformal.py` | 6 conformal predictors, scoring functions (`compute_coverage`, `compute_interval_width`, `compute_interval_score`), conditional coverage by sentiment/bucket |
| `utils/crane_architecture.py` | `CRANEBlock` — Self-Attn → FFN → Cross-Attn → FFN |
| `utils/data_loader.py` | Plan C split with `cal_loader` (augment=False), `es_loader` (augment=False) |
| `utils/metricsTop.py` | `MetricsTop` (regression/classification), `ConformalMetrics` (static format methods, all flatten inputs to avoid (N,1)→(N,N) broadcast) |
| `utils/visualization.py` | `save_all_figures(viz_data)` — 9 figures, Agg backend |

## Important implementation notes

- **torch.load**: Always use `weights_only=True` (5+ calls in en_train.py).
- **Coverage broadcast bug**: `y_true` (N,1) with `lower` (N,) produces (N,N) matrix — always `.flatten()` inputs in `compute_coverage` and `ConformalMetrics.format_results`.
- **MC std clamping**: `np.maximum(std, 1e-8)` to avoid division by zero in adaptive nonconformity scores.
- **viz data deferred assignment**: In `EnRun()`, UBG-width correlation data is stored in `_ubg_width_data` early (before `viz` dict exists), then populated into `viz` later. Do not reorder.
- **Figure numbering**: Figures are numbered by appearance order (Fig1=UBG, Fig2=Coverage-Width, Fig3=Residual, Fig4=Calibration, Fig5-9 unchanged). `visualization.py` function names and save paths match this ordering.
- **Math rendering**: Inline `$...$` in `CRANE_SUMMARY.md` uses `\mathrm` not `\text` — MathJax 3.x on GitHub/Typora requires `textmacros` extension for `\text` in inline mode.
