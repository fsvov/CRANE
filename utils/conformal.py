import numpy as np


def compute_coverage(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))


def compute_interval_width(lower, upper):
    widths = upper - lower
    return np.mean(widths), np.median(widths)


def compute_interval_score(y_true, lower, upper, alpha):
    """Gneiting & Raftery interval score (lower is better)."""
    width = upper - lower
    penalty_low = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_high = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return np.mean(width + penalty_low + penalty_high)


def conditional_coverage_by_sentiment(y_true, y_pred, lower, upper):
    """Conditional coverage by negative / neutral / positive."""
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()
    lo = np.asarray(lower).flatten()
    up = np.asarray(upper).flatten()

    negative = yt < 0
    neutral = yt == 0
    positive = yt > 0

    results = {}
    for label, mask in [("Negative", negative), ("Neutral", neutral), ("Positive", positive)]:
        if mask.sum() == 0:
            results[label] = {"count": 0, "coverage": 0.0, "avg_width": 0.0}
            continue
        results[label] = {
            "count": int(mask.sum()),
            "coverage": round(compute_coverage(yt[mask], lo[mask], up[mask]), 4),
            "avg_width": round(np.mean(up[mask] - lo[mask]), 4),
        }
    return results


def conditional_coverage_by_bucket(y_pred, y_true, lower, upper):
    """Conditional coverage by prediction buckets [-3,-2), ..., [2,3]."""
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()
    lo = np.asarray(lower).flatten()
    up = np.asarray(upper).flatten()

    bucket_edges = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    results = {}
    for i in range(len(bucket_edges) - 1):
        low, high = bucket_edges[i], bucket_edges[i + 1]
        mask = (yp >= low) & (yp < high)
        label = f"[{low},{high})"
        if mask.sum() == 0:
            results[label] = {"count": 0, "coverage": 0.0, "avg_width": 0.0}
            continue
        results[label] = {
            "count": int(mask.sum()),
            "coverage": round(compute_coverage(yt[mask], lo[mask], up[mask]), 4),
            "avg_width": round(np.mean(up[mask] - lo[mask]), 4),
        }
    return results


class SplitConformalPredictor:
    """Level 1: Split conformal prediction with constant-width intervals."""

    def __init__(self):
        self._q = None
        self._residuals = None

    def calibrate(self, y_true, y_pred):
        residuals = np.abs(np.asarray(y_true).flatten() - np.asarray(y_pred).flatten())
        self._residuals = residuals

    def predict(self, y_pred, alpha):
        if self._residuals is None:
            raise RuntimeError("Must call calibrate() first")
        yp = np.asarray(y_pred).flatten()
        n = len(self._residuals)
        q_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
        q_idx = min(q_idx, n - 1)
        q = np.sort(self._residuals)[q_idx]
        self._q = q
        lower = yp - q
        upper = yp + q
        return lower, upper


class MCAdaptiveConformalPredictor:
    """Level 1.5: Locally adaptive conformal prediction using MC dropout variance."""

    def __init__(self):
        self._q = None
        self._scores = None

    def calibrate(self, y_true, y_pred, mc_std):
        yt = np.asarray(y_true).flatten()
        yp = np.asarray(y_pred).flatten()
        std = np.asarray(mc_std).flatten()
        std = np.maximum(std, 1e-8)
        self._scores = np.abs(yt - yp) / std

    def predict(self, y_pred, mc_std, alpha):
        if self._scores is None:
            raise RuntimeError("Must call calibrate() first")
        yp = np.asarray(y_pred).flatten()
        std = np.asarray(mc_std).flatten()
        std = np.maximum(std, 1e-8)

        n = len(self._scores)
        q_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
        q_idx = min(q_idx, n - 1)
        q = np.sort(self._scores)[q_idx]
        self._q = q

        half_width = q * std
        lower = yp - half_width
        upper = yp + half_width
        return lower, upper


class MondrianConformalPredictor:
    """Mondrian conformal prediction: per-group quantiles for conditional coverage.

    Guarantees coverage ≥ 1-α within each group (e.g. sentiment polarity),
    not just marginally over all samples.
    """

    def __init__(self):
        self._qs = {}

    def calibrate(self, y_true, y_pred, groups):
        yt = np.asarray(y_true).flatten()
        yp = np.asarray(y_pred).flatten()
        self._groups = np.asarray(groups).flatten()
        residuals = np.abs(yt - yp)
        for g in np.unique(self._groups):
            mask = self._groups == g
            if mask.sum() == 0:
                continue
            self._qs[g] = np.sort(residuals[mask])

    def predict(self, y_pred, groups, alpha):
        if not self._qs:
            raise RuntimeError("Must call calibrate() first")
        yp = np.asarray(y_pred).flatten()
        grps = np.asarray(groups).flatten()
        lower = np.zeros_like(yp)
        upper = np.zeros_like(yp)
        for g in np.unique(grps):
            mask = grps == g
            if g not in self._qs:
                continue
            scores = self._qs[g]
            n = len(scores)
            q_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
            q_idx = min(q_idx, n - 1)
            q = scores[q_idx]
            lower[mask] = yp[mask] - q
            upper[mask] = yp[mask] + q
        return lower, upper


def sentiment_group(y):
    """Map continuous sentiment to polarity group: negative, neutral, positive."""
    y = np.asarray(y).flatten()
    groups = np.full_like(y, "neutral", dtype=object)
    groups[y < 0] = "negative"
    groups[y > 0] = "positive"
    return groups


def mc_dropout_interval(y_pred, mc_std, alpha):
    """Baseline: raw MC Dropout intervals (Gaussian assumption, no conformal)."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    yp = np.asarray(y_pred).flatten()
    std = np.asarray(mc_std).flatten()
    std = np.maximum(std, 1e-8)
    lower = yp - z * std
    upper = yp + z * std
    return lower, upper


# ============================================================
# Classification Conformal Prediction
# ============================================================

CLASS_CENTERS = [-3, -2, -1, 0, 1, 2, 3]


def map_to_7class(y):
    """Map continuous sentiment [-3, 3] to 7-class index {-3, -2, ..., 3}."""
    y = np.asarray(y).flatten()
    return np.clip(np.round(y).astype(int), -3, 3)


class ClassificationConformalPredictor:
    """Conformal prediction sets for 7-class sentiment classification.

    Uses the same nonconformity score as regression (|ŷ − y|), but outputs
    discrete prediction sets instead of continuous intervals.
    """

    def __init__(self, class_centers=None):
        self.class_centers = class_centers or CLASS_CENTERS
        self._residuals = None
        self._q = None

    def calibrate(self, y_true, y_pred):
        residuals = np.abs(np.asarray(y_true).flatten() - np.asarray(y_pred).flatten())
        self._residuals = residuals

    def predict(self, y_pred, alpha):
        if self._residuals is None:
            raise RuntimeError("Must call calibrate() first")
        yp = np.asarray(y_pred).flatten()
        n = len(self._residuals)
        q_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
        q_idx = min(q_idx, n - 1)
        q = np.sort(self._residuals)[q_idx]
        self._q = q
        sets = []
        for pred in yp:
            included = sorted([c for c in self.class_centers if abs(pred - c) <= q])
            if not included:
                # Fallback: closest class
                included = [min(self.class_centers, key=lambda c: abs(pred - c))]
            sets.append(included)
        return sets


def classification_set_metrics(y_true_cont, prediction_sets):
    """Compute coverage, set size distribution, and singleton rate."""
    yt = map_to_7class(np.asarray(y_true_cont).flatten())
    n = len(yt)
    covered = sum(true_c in pset for true_c, pset in zip(yt, prediction_sets))
    sizes = [len(s) for s in prediction_sets]
    singletons = sizes.count(1)
    size_dist = {s: sizes.count(s) for s in sorted(set(sizes))}
    return {
        'coverage': round(covered / n, 4),
        'avg_set_size': round(np.mean(sizes), 3),
        'med_set_size': int(np.median(sizes)),
        'singleton_rate': round(singletons / n, 4),
        'max_set_size': max(sizes),
        'size_distribution': size_dist,
    }


def classification_conditional_by_sentiment(y_true_cont, prediction_sets):
    """Conditional set metrics by true sentiment polarity."""
    yt = np.asarray(y_true_cont).flatten()
    negative = yt < 0
    neutral = yt == 0
    positive = yt > 0
    results = {}
    for label, mask in [("Negative", negative), ("Neutral", neutral), ("Positive", positive)]:
        if mask.sum() == 0:
            results[label] = {"count": 0, "coverage": 0.0, "avg_size": 0.0}
            continue
        subset = [prediction_sets[i] for i in range(len(yt)) if mask[i]]
        yt_sub = [map_to_7class(yt[i]) for i in range(len(yt)) if mask[i]]
        cov = sum(t in p for t, p in zip(yt_sub, subset)) / len(subset)
        avg_sz = np.mean([len(s) for s in subset])
        results[label] = {
            "count": int(mask.sum()),
            "coverage": round(cov, 4),
            "avg_size": round(avg_sz, 3),
        }
    return results
