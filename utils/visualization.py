"""CRANE visualization module — 8 publication-quality figures.

All figures are saved to figures/ as PNG files (300 DPI).
Uses matplotlib + seaborn. Non-interactive Agg backend for HPC compatibility.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

FIGSIZE = (7, 5)
FIGSIZE_WIDE = (10, 5)
COLORS = {
    'split': '#3498db', 'adaptive': '#e74c3c', 'mondrian': '#9b59b6',
    'mve': '#2ecc71', 'raw': '#e95a2b', 'ensemble': '#e67e22',
    'text': '#1abc9c', 'audio': '#f39c12', 'multi': '#e74c3c',
    'neg': '#e74c3c', 'neutral': '#f39c12', 'pos': '#2ecc71',
    'covered': '#2ecc71', 'missed': '#e74c3c',
}
SET_SIZE_COLORS = {1: '#3498db', 2: '#2ecc71', 3: '#f39c12', 4: '#e74c3c',
                   5: '#9b59b6', 6: '#e67e22', 7: '#95a5a6'}
SENTIMENT_NAMES = ['Negative', 'Neutral', 'Positive']


def _ensure_dir():
    os.makedirs('figures', exist_ok=True)


def _save(name):
    path = os.path.join('figures', name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 1: Coverage-Width Trade-off
# ============================================================

def fig1_coverage_width_tradeoff(data):
    """Pareto-style plot: coverage vs med_width for all methods across α levels.

    data keys:
      methods: list of method names
      alphas: list of α values
      results: dict[method_name] → {α: (coverage, med_width)}
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    markers = {'Split': 's', 'Adaptive': 'o', 'Mondrian': 'D',
               'MVE': '^', 'MC RAW': 'x', 'Ensemble': 'P'}
    method_colors = {'Split': COLORS['split'], 'Adaptive': COLORS['adaptive'],
                     'Mondrian': COLORS['mondrian'], 'MVE': COLORS['mve'],
                     'MC RAW': COLORS['raw'], 'Ensemble': COLORS['ensemble']}

    for method in data['methods']:
        if method not in data['results']:
            continue
        pts = data['results'][method]
        alphas_sorted = sorted(pts.keys())
        covs = [pts[a][0] for a in alphas_sorted]
        widths = [pts[a][1] for a in alphas_sorted]
        mk = markers.get(method, 'o')
        clr = method_colors.get(method, '#333')
        ms = 10 if method == 'MC RAW' else 8
        lw = 2.0 if method == 'MC RAW' else 1.5
        ax.plot(widths, covs, marker=mk, color=clr, label=method, markersize=ms,
                linewidth=lw, markeredgewidth=0.5, markeredgecolor='white')

        if method == 'Adaptive':
            for a, c, w in zip(alphas_sorted, covs, widths):
                offset = (8, 2) if a < 0.15 else (8, -12)
                ax.annotate(f'α={a:.2f}', (w, c), textcoords='offset points',
                            xytext=offset, fontsize=7, color=clr, alpha=0.8)

        if method == 'MC RAW':
            ax.annotate('MC Dropout RAW\n(coverage ≈ 35%, no calibration)',
                        xy=(widths[1], covs[1]), xytext=(widths[1] + 1.5, covs[1] + 0.15),
                        fontsize=8, color=clr, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=clr, lw=1.2))

    for a in sorted(data['alphas']):
        ax.axhline(y=1 - a, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)

    ax.set_xlabel('Median Interval Width (↓ better)', fontsize=11)
    ax.set_ylabel('Coverage (→ target)', fontsize=11)
    ax.set_title('Figure 1: Coverage–Width Trade-off Across Methods and α Levels', fontsize=12)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(0.25, 1.02)
    ax.grid(True, alpha=0.3)
    _save('fig1_coverage_width_tradeoff.png')


# ============================================================
# Figure 2: Calibration Size Sensitivity
# ============================================================

def fig2_calibration_sensitivity(data):
    """Dual-axis plot: coverage and median width vs n_cal.

    data keys:
      n_cal: list of calibration sizes
      split_cov, adapt_cov: coverage for each size
      split_w, adapt_mw: width for each size
    """
    _ensure_dir()
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()

    nc = data['n_cal']
    ax1.plot(nc, data['split_cov'], 's-', color=COLORS['split'], label='Split Coverage', markersize=6)
    ax1.plot(nc, data['adapt_cov'], 'o-', color=COLORS['adaptive'], label='Adaptive Coverage', markersize=6)
    ax1.axhline(y=0.90, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax1.annotate('target 90%', xy=(nc[-1], 0.90), xytext=(10, 5), textcoords='offset points',
                 fontsize=8, color='gray')

    ax2.plot(nc, data['split_w'], 's:', color=COLORS['split'], label='Split Width', markersize=5, alpha=0.6, linewidth=2)
    ax2.plot(nc, data['adapt_mw'], 'o-.', color=COLORS['adaptive'], label='Adaptive Med Width', markersize=5, alpha=0.6, linewidth=2)

    ax1.set_xlabel('Calibration Set Size (n_cal)', fontsize=11)
    ax1.set_ylabel('Coverage', fontsize=11, color='#333')
    ax2.set_ylabel('Median Width', fontsize=11, color='#666')
    ax1.set_title('Figure 2: Calibration Size Sensitivity (α=0.10)', fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=7, framealpha=0.9)

    ax1.set_ylim(0.80, 1.0)
    ax1.grid(True, alpha=0.3)
    _save('fig2_calibration_sensitivity.png')


# ============================================================
# Figure 3: UBG Learned Confidence Distribution
# ============================================================

def fig3_ubg_confidence(conf_text, conf_audio, sentiment_labels):
    """Scatter plot: conf_text vs conf_audio, colored by sentiment polarity.

    Args:
      conf_text, conf_audio: arrays of per-sample confidences
      sentiment_labels: array of 'negative', 'neutral', 'positive' strings
    """
    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: scatter plot
    ax = axes[0]
    sent_colors = {'negative': COLORS['neg'], 'neutral': COLORS['neutral'], 'positive': COLORS['pos']}
    for sent in ['negative', 'neutral', 'positive']:
        mask = np.array(sentiment_labels) == sent
        if mask.sum() == 0:
            continue
        ax.scatter(conf_text[mask], conf_audio[mask], c=sent_colors[sent],
                   label=sent, alpha=0.5, s=12, edgecolors='none')

    ax.set_xlabel('Text Confidence', fontsize=10)
    ax.set_ylabel('Audio Confidence', fontsize=10)
    ax.set_title('UBG Confidence: Text vs Audio', fontsize=11)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    corr = np.corrcoef(conf_text, conf_audio)[0, 1]
    ax.annotate(f'Corr = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=9, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right: histogram
    ax = axes[1]
    bins = np.linspace(0, 1, 31)
    ax.hist(conf_text, bins=bins, alpha=0.6, color=COLORS['text'], label=f'Text (μ={np.mean(conf_text):.3f})')
    ax.hist(conf_audio, bins=bins, alpha=0.6, color=COLORS['audio'], label=f'Audio (μ={np.mean(conf_audio):.3f})')
    ax.set_xlabel('Confidence', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('UBG Confidence Distribution', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Figure 3: UBG Learned Modality Confidence', fontsize=12, y=1.01)
    _save('fig3_ubg_confidence.png')


# ============================================================
# Figure 4: Residual Distribution (Cal vs Test)
# ============================================================

def fig4_residual_distribution(cal_residuals, test_residuals, q_value, alpha=0.10):
    """Overlaid histogram of calibration and test residuals with quantile marker.

    Args:
      cal_residuals: |y_cal - ŷ_cal|
      test_residuals: |y_test - ŷ_test|
      q_value: conformal quantile (split)
      alpha: significance level
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, max(cal_residuals.max(), test_residuals.max()) * 1.1, 50)

    ax.hist(cal_residuals, bins=bins, alpha=0.5, color=COLORS['split'], label=f'Calibration (n={len(cal_residuals)})', density=True)
    ax.hist(test_residuals, bins=bins, alpha=0.5, color=COLORS['adaptive'], label=f'Test (n={len(test_residuals)})', density=True)
    ax.axvline(x=q_value, color='red', linestyle='--', linewidth=1.5, label=f'q={q_value:.3f} (α={alpha:.2f})')

    cal_covered = np.mean(cal_residuals <= q_value) * 100
    test_covered = np.mean(test_residuals <= q_value) * 100
    ax.annotate(f'Cal ≤ q: {cal_covered:.1f}%', xy=(q_value, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] else 0.85),
                xytext=(10, 0), textcoords='offset points', fontsize=8, color=COLORS['split'])
    ax.annotate(f'Test ≤ q: {test_covered:.1f}%', xy=(q_value, ax.get_ylim()[1] * 0.75 if ax.get_ylim()[1] else 0.75),
                xytext=(10, 0), textcoords='offset points', fontsize=8, color=COLORS['adaptive'])

    ax.set_xlabel('Absolute Residual |y − ŷ|', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Figure 4: Residual Distribution — Calibration vs Test', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    _save('fig4_residual_distribution.png')


# ============================================================
# Figure 5: Conditional Coverage Heatmap
# ============================================================

def fig5_conditional_coverage_heatmap(sentiment_cond, bucket_cond):
    """Heatmap of coverage by sentiment polarity and prediction bucket.

    Args:
      sentiment_cond: dict like {'Negative': {'count': N, 'coverage': X, ...}, ...}
      bucket_cond: dict like {'[-2.0,-1.0)': {'count': N, 'coverage': X, ...}, ...}
    """
    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: sentiment bar chart
    ax = axes[0]
    sentiments = ['Negative', 'Neutral', 'Positive']
    covs = [sentiment_cond.get(s, {}).get('coverage', 0) for s in sentiments]
    counts = [sentiment_cond.get(s, {}).get('count', 0) for s in sentiments]
    bars = ax.bar(sentiments, covs, color=[COLORS['neg'], COLORS['neutral'], COLORS['pos']], edgecolor='white')
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.4)
    for bar, c, cnt in zip(bars, covs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{c:.3f}\nn={cnt}', ha='center', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Coverage', fontsize=10)
    ax.set_title('By Sentiment Polarity', fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    # Right: bucket bar chart
    ax = axes[1]
    buckets = list(bucket_cond.keys())
    bucket_covs = [bucket_cond.get(b, {}).get('coverage', 0) for b in buckets]
    bucket_counts = [bucket_cond.get(b, {}).get('count', 0) for b in buckets]
    colors_b = [COLORS['neg'] if float(b.split(',')[0][1:]) < 0 else
                COLORS['neutral'] if '0.0' in b else COLORS['pos']
                for b in buckets]
    bars = ax.bar(range(len(buckets)), bucket_covs, color=colors_b, edgecolor='white')
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.4)
    for i, (bar, c, cnt) in enumerate(zip(bars, bucket_covs, bucket_counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{c:.3f}\nn={cnt}', ha='center', fontsize=8)
    ax.set_xticks(range(len(buckets)))
    _labels = []
    for k, b in enumerate(buckets):
        lb = b.replace('[', '')
        if k == len(buckets) - 1:
            lb = lb.replace(')', ']')
        else:
            lb = lb.replace(')', '')
        _labels.append(lb)
    ax.set_xticklabels(_labels, fontsize=8, rotation=30)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Coverage', fontsize=10)
    ax.set_title('By Prediction Bucket', fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Figure 5: Conditional Coverage (Adaptive CP, α=0.10)', fontsize=12, y=1.01)
    _save('fig5_conditional_coverage.png')


# ============================================================
# Figure 6: Interval Width vs |Prediction|
# ============================================================

def fig6_width_vs_magnitude(y_pred, interval_widths, covered, alpha=0.10):
    """Scatter: interval width vs absolute prediction value.

    Args:
      y_pred: predicted values
      interval_widths: adaptive interval widths per sample
      covered: bool array — true if interval contains the true value
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    yp = np.asarray(y_pred).flatten()
    w = np.asarray(interval_widths).flatten()
    cv = np.asarray(covered).flatten()

    ax.scatter(np.abs(yp[cv]), w[cv], c=COLORS['covered'], alpha=0.4, s=10,
               label=f'Covered ({cv.sum()}/{len(cv)})', edgecolors='none')
    ax.scatter(np.abs(yp[~cv]), w[~cv], c=COLORS['missed'], alpha=0.6, s=15,
               label=f'Missed ({(~cv).sum()}/{len(cv)})', edgecolors='none', marker='x')

    # Trend line
    z = np.polyfit(np.abs(yp), w, 2)
    x_line = np.linspace(0, np.abs(yp).max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k-', linewidth=1.5, alpha=0.5, label='Quadratic fit')

    ax.set_xlabel('|Predicted Sentiment| (0 = neutral, 3 = extreme)', fontsize=10)
    ax.set_ylabel('Adaptive Interval Width', fontsize=10)
    ax.set_title(f'Figure 6: Interval Width vs Prediction Magnitude (α={alpha:.2f})', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    _save('fig6_width_vs_magnitude.png')


# ============================================================
# Figure 7: Classification Prediction Set Sizes
# ============================================================

def fig7_prediction_set_sizes(size_distributions, alpha=0.10):
    """Bar chart of prediction set size distribution.

    Args:
      size_distributions: dict {α: {size: count, ...}, ...}
    """
    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Build consistent color map across both panels
    all_sizes = sorted(set(s for d in size_distributions.values() for s in d.keys()))
    size_color_map = {sz: SET_SIZE_COLORS.get(sz, '#888888') for sz in all_sizes}

    # Left: single α bar chart
    ax = axes[0]
    dist = size_distributions.get(alpha, {})
    sizes_l = sorted(dist.keys())
    counts = [dist[s] for s in sizes_l]
    colors_l = [size_color_map[s] for s in sizes_l]
    bars = ax.bar([str(s) for s in sizes_l], counts, color=colors_l, edgecolor='white')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(cnt), ha='center', fontsize=9)
    ax.set_xlabel('Prediction Set Size', fontsize=10)
    ax.set_ylabel('Number of Samples', fontsize=10)
    ax.set_title(f'Set Size Distribution (α={alpha:.2f})', fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    # Right: stacked bar across α
    ax = axes[1]
    all_alphas = sorted(size_distributions.keys())
    bottom = np.zeros(len(all_alphas))
    alpha_labels = [f'{a:.2f}' for a in all_alphas]
    for sz in all_sizes:
        vals = [size_distributions[a].get(sz, 0) for a in all_alphas]
        ax.bar(alpha_labels, vals, bottom=bottom, label=f'Size={sz}',
               color=size_color_map[sz], alpha=0.85, edgecolor='white')
        bottom += np.array(vals)
    ax.set_xlabel('Significance Level α', fontsize=10)
    ax.set_ylabel('Number of Samples', fontsize=10)
    ax.set_title('Set Size by α Level', fontsize=11)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle('Figure 7: Classification Conformal — Prediction Set Sizes', fontsize=12, y=1.01)
    _save('fig7_prediction_set_sizes.png')


# ============================================================
# Figure 8: Reliability Diagram
# ============================================================

def fig8_reliability_diagram(y_pred, y_true, interval_widths, mc_std, n_bins=10, alpha=0.10):
    """Reliability diagram: binned observed MAE vs predicted uncertainty.

    Left: Observed MAE vs binned MC σ with linear fit.
    Right: Coverage vs bin.

    Args:
      y_pred: predicted values
      y_true: true values
      interval_widths: adaptive interval widths
      mc_std: MC dropout standard deviations
      n_bins: number of bins
    """
    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    yp = np.asarray(y_pred).flatten()
    yt = np.asarray(y_true).flatten()
    w = np.asarray(interval_widths).flatten()
    std = np.asarray(mc_std).flatten()

    # Bin by MC std
    bin_edges = np.percentile(std, np.linspace(0, 100, n_bins + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mae_binned, cov_binned, bin_n = [], [], []
    for i in range(n_bins):
        mask = (std >= bin_edges[i]) & (std < bin_edges[i + 1])
        n_b = mask.sum()
        bin_n.append(n_b)
        if n_b == 0:
            mae_binned.append(0)
            cov_binned.append(0)
        else:
            mae_binned.append(np.mean(np.abs(yt[mask] - yp[mask])))
            cov_binned.append(np.mean((yt[mask] >= yp[mask] - w[mask] / 2) &
                                      (yt[mask] <= yp[mask] + w[mask] / 2)))

    # Left: MAE vs uncertainty (no ideal line — different scales)
    ax = axes[0]
    ax.plot(bin_centers, mae_binned, 'o-', color=COLORS['adaptive'], markersize=6,
            linewidth=2, label='Observed MAE')
    # Linear fit to show monotonic trend
    mask_nonzero = np.array(bin_n) > 0
    if mask_nonzero.sum() >= 3:
        fit = np.polyfit(bin_centers[mask_nonzero], np.array(mae_binned)[mask_nonzero], 1)
        x_fit = np.linspace(bin_centers[0], bin_centers[-1], 50)
        ax.plot(x_fit, np.polyval(fit, x_fit), '--', color='gray', alpha=0.5,
                linewidth=1, label=f'Linear fit (slope={fit[0]:.2f})')
    ax.set_xlabel('Predicted Uncertainty (MC σ)', fontsize=10)
    ax.set_ylabel('Observed MAE', fontsize=10)
    ax.set_title(f'MAE vs Uncertainty Level ({n_bins} bins)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Right: Coverage vs bin
    ax = axes[1]
    bars = ax.bar(range(n_bins), cov_binned, color=COLORS['adaptive'], alpha=0.7, edgecolor='white')
    ax.axhline(y=1 - alpha, color='gray', linestyle='--', alpha=0.4, linewidth=1,
               label=f'Target: {1-alpha:.0%}')
    for i, (bar, sz) in enumerate(zip(bars, bin_n)):
        if sz > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'n={sz}', ha='center', fontsize=6, color='#666')
    ax.set_xlabel('Uncertainty Bin (low → high)', fontsize=10)
    ax.set_ylabel(f'Coverage', fontsize=10)
    ax.set_title(f'Coverage by Uncertainty Level (target: {1-alpha:.0%})', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Figure 8: Reliability Diagram', fontsize=12, y=1.01)
    _save('fig8_reliability_diagram.png')


# ============================================================
# Figure 9: UBG Confidence vs Conformal Interval Width
# ============================================================

def fig9_ubg_confidence_vs_width(conf_text, conf_audio, interval_widths, sentiment_labels):
    """Dual scatter: UBG confidence vs adaptive conformal width.

    Left: conf_text vs width. Right: conf_audio vs width.
    Colored by sentiment polarity. Annotated with correlation coefficients.
    """
    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    w = np.asarray(interval_widths).flatten()
    ct = np.asarray(conf_text).flatten()
    ca = np.asarray(conf_audio).flatten()
    sent_colors = {'negative': COLORS['neg'], 'neutral': COLORS['neutral'], 'positive': COLORS['pos']}

    for ax, conf, name, corr_val in [
        (axes[0], ct, 'Text', np.corrcoef(ct, w)[0, 1]),
        (axes[1], ca, 'Audio', np.corrcoef(ca, w)[0, 1]),
    ]:
        for sent in ['negative', 'neutral', 'positive']:
            mask = np.array(sentiment_labels) == sent
            if mask.sum() == 0:
                continue
            ax.scatter(conf[mask], w[mask], c=sent_colors[sent],
                       label=sent, alpha=0.4, s=10, edgecolors='none')

        # Linear fit
        z = np.polyfit(conf, w, 1)
        x_line = np.linspace(conf.min(), conf.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'k-', linewidth=1.5, alpha=0.5,
                label=f'Linear fit (r={corr_val:.3f})')

        ax.set_xlabel(f'{name} Confidence', fontsize=10)
        ax.set_ylabel('Adaptive Interval Width', fontsize=10)
        ax.set_title(f'{name} Confidence vs Width', fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Figure 9: UBG Confidence — Conformal Width Correlation', fontsize=12, y=1.01)
    _save('fig9_ubg_confidence_vs_width.png')


# ============================================================
# Master: save all figures
# ============================================================

def save_all_figures(viz_data):
    """Generate all 8 figures from a single data dictionary.

    viz_data keys (populated during conformal evaluation):
      - coverage_width: dict for fig1
      - calibration_sensitivity: dict for fig2
      - ubg_confidence: (conf_text, conf_audio, sentiment_labels)
      - residuals: (cal_residuals, test_residuals, q_value, alpha)
      - conditional_coverage: (sentiment_cond, bucket_cond)
      - width_vs_magnitude: (y_pred, widths, covered, alpha)
      - prediction_sets: (size_distributions, alpha)
      - reliability: (y_pred, y_true, widths, mc_std)
    """
    print("\n--- Generating Figures ---")

    if 'coverage_width' in viz_data:
        fig1_coverage_width_tradeoff(viz_data['coverage_width'])

    if 'calibration_sensitivity' in viz_data:
        fig2_calibration_sensitivity(viz_data['calibration_sensitivity'])

    if 'ubg_confidence' in viz_data:
        conf_t, conf_a, labels = viz_data['ubg_confidence']
        fig3_ubg_confidence(conf_t, conf_a, labels)

    if 'residuals' in viz_data:
        cal_r, test_r, q, a = viz_data['residuals']
        fig4_residual_distribution(cal_r, test_r, q, a)

    if 'conditional_coverage' in viz_data:
        sent_cond, buck_cond = viz_data['conditional_coverage']
        fig5_conditional_coverage_heatmap(sent_cond, buck_cond)

    if 'width_vs_magnitude' in viz_data:
        yp, w, cv, a = viz_data['width_vs_magnitude']
        fig6_width_vs_magnitude(yp, w, cv, a)

    if 'prediction_sets' in viz_data:
        dists, a = viz_data['prediction_sets']
        fig7_prediction_set_sizes(dists, a)

    if 'reliability' in viz_data:
        yp, yt, w, std = viz_data['reliability']
        fig8_reliability_diagram(yp, yt, w, std)

    if 'ubg_width_corr' in viz_data:
        ct, ca, aw, sl = viz_data['ubg_width_corr']
        fig9_ubg_confidence_vs_width(ct, ca, aw, sl)

    print("--- All figures saved to figures/ ---")
