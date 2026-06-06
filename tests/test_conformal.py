"""Unit tests for conformal prediction — coverage guarantees, edge cases, and correctness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from utils.conformal import (
    compute_coverage,
    compute_interval_width,
    compute_interval_score,
    conditional_coverage_by_sentiment,
    conditional_coverage_by_bucket,
    SplitConformalPredictor,
    MCAdaptiveConformalPredictor,
    MondrianConformalPredictor,
    ClassificationConformalPredictor,
    sentiment_group,
    mc_dropout_interval,
    map_to_7class,
    classification_set_metrics,
)


# ============================================================
# Utility functions
# ============================================================

class TestComputeCoverage:
    def test_perfect_coverage(self):
        assert compute_coverage(np.array([0.0]), np.array([-1.0]), np.array([1.0])) == 1.0

    def test_no_coverage(self):
        assert compute_coverage(np.array([0.0]), np.array([1.0]), np.array([2.0])) == 0.0

    def test_half_coverage(self):
        yt = np.array([0.0, 10.0])
        lo = np.array([-1.0, -1.0])
        up = np.array([1.0, 1.0])
        assert compute_coverage(yt, lo, up) == 0.5

    def test_boundary_inclusive(self):
        assert compute_coverage(np.array([1.0]), np.array([0.0]), np.array([1.0])) == 1.0

    def test_vectorized(self):
        rng = np.random.RandomState(42)
        yt = rng.randn(1000)
        lo = yt - 1.0
        up = yt + 1.0
        assert compute_coverage(yt, lo, up) == 1.0


class TestIntervalWidth:
    def test_constant_width(self):
        lo = np.array([-1.0, -1.0])
        up = np.array([1.0, 1.0])
        avg, med = compute_interval_width(lo, up)
        assert avg == 2.0
        assert med == 2.0

    def test_variable_width(self):
        lo = np.array([-1.0, -1.0])
        up = np.array([1.0, 3.0])
        avg, med = compute_interval_width(lo, up)
        assert avg == 3.0
        assert med == 3.0


class TestIntervalScore:
    def test_perfect_prediction(self):
        score = compute_interval_score(
            np.array([0.0]), np.array([-1.0]), np.array([1.0]), alpha=0.1
        )
        assert score == 2.0

    def test_miss_penalty(self):
        score_miss = compute_interval_score(
            np.array([5.0]), np.array([-1.0]), np.array([1.0]), alpha=0.1
        )
        assert score_miss > 2.0


# ============================================================
# Split Conformal Predictor
# ============================================================

class TestSplitConformal:
    def test_coverage_at_alpha_zero(self, rng):
        n = 200
        y_true = rng.randn(n)
        y_pred = y_true + 0.1 * rng.randn(n)

        cal_true, cal_pred = y_true[:100], y_pred[:100]
        test_true, test_pred = y_true[100:], y_pred[100:]

        cp = SplitConformalPredictor()
        cp.calibrate(cal_true, cal_pred)
        lo, up = cp.predict(test_pred, alpha=0.1)

        cov = compute_coverage(test_true, lo, up)
        assert cov >= 0.85

    def test_exchangeable_data_perfect(self):
        rng = np.random.RandomState(1)
        n = 500
        y = rng.randn(n)
        y_hat = y + 0.05 * rng.randn(n)

        cp = SplitConformalPredictor()
        cp.calibrate(y[:250], y_hat[:250])
        lo, up = cp.predict(y_hat[250:], alpha=0.1)

        cov = compute_coverage(y[250:], lo, up)
        assert 0.87 <= cov <= 0.95

    def test_constant_width(self):
        rng = np.random.RandomState(2)
        y = rng.randn(300)
        h = rng.randn(300)

        cp = SplitConformalPredictor()
        cp.calibrate(y[:150], h[:150])
        lo, up = cp.predict(h[150:], alpha=0.1)

        widths = up - lo
        assert np.allclose(widths, widths[0])

    def test_requires_calibrate(self):
        cp = SplitConformalPredictor()
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict(np.array([0.0]), alpha=0.1)

    def test_small_calibration_set(self):
        rng = np.random.RandomState(3)
        y = rng.randn(30)
        cp = SplitConformalPredictor()
        cp.calibrate(y[:10], y[:10])
        lo, up = cp.predict(y[10:], alpha=0.1)
        assert lo.shape == (20,)
        assert up.shape == (20,)

    def test_reproducibility(self):
        rng = np.random.RandomState(4)
        cal_y, cal_p = rng.randn(100), rng.randn(100)
        test_p = rng.randn(50)

        cp1 = SplitConformalPredictor()
        cp1.calibrate(cal_y, cal_p)
        lo1, up1 = cp1.predict(test_p, alpha=0.1)

        cp2 = SplitConformalPredictor()
        cp2.calibrate(cal_y, cal_p)
        lo2, up2 = cp2.predict(test_p, alpha=0.1)

        assert np.allclose(lo1, lo2)
        assert np.allclose(up1, up2)


# ============================================================
# MC Adaptive Conformal Predictor
# ============================================================

class TestAdaptiveConformal:
    def test_coverage_with_adaptive_width(self, rng):
        n = 300
        y = rng.randn(n)
        y_hat = y + 0.1 * rng.randn(n)
        mc_std = 0.1 + 0.5 * np.abs(y)

        cp = MCAdaptiveConformalPredictor()
        cp.calibrate(y[:150], y_hat[:150], mc_std[:150])
        lo, up = cp.predict(y_hat[150:], mc_std[150:], alpha=0.1)

        cov = compute_coverage(y[150:], lo, up)
        assert cov >= 0.85

    def test_adaptive_width_varies(self):
        rng = np.random.RandomState(6)
        n = 200
        y = rng.randn(n)
        y_hat = y + 0.1 * rng.randn(n)
        mc_std_wide = np.full(n, 10.0)
        mc_std_narrow = np.full(n, 0.1)

        cp = MCAdaptiveConformalPredictor()
        cp.calibrate(y[:100], y_hat[:100], mc_std_wide[:100])

        _, up_wide = cp.predict(y_hat[100:], mc_std_wide[100:], alpha=0.1)
        _, up_narrow = cp.predict(y_hat[100:], mc_std_narrow[100:], alpha=0.1)

        assert np.all(up_wide - y_hat[100:] > up_narrow - y_hat[100:])

    def test_zero_std_clamped(self):
        rng = np.random.RandomState(7)
        y = rng.randn(100)
        p = y + 0.1 * rng.randn(100)
        std = np.zeros(100)

        cp = MCAdaptiveConformalPredictor()
        cp.calibrate(y[:50], p[:50], std[:50])
        lo, up = cp.predict(p[50:], std[50:], alpha=0.1)
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(up))


# ============================================================
# Mondrian Conformal Predictor
# ============================================================

class TestMondrianConformal:
    def test_per_group_quantiles(self, rng):
        """Coverage should hold per-group when cal & test are exchangeable."""
        # Interleave groups so each is evenly represented in cal and test
        n_per = 200
        groups = np.array([["A"], ["B"], ["C"]] * n_per).flatten()
        y = rng.randn(3 * n_per)
        y_hat = y + 0.1 * rng.randn(3 * n_per)

        half = 3 * n_per // 2
        cp = MondrianConformalPredictor()
        cp.calibrate(y[:half], y_hat[:half], groups[:half])
        lo, up = cp.predict(y_hat[half:], groups[half:], alpha=0.1)

        cov = compute_coverage(y[half:], lo, up)
        assert cov >= 0.85, f"Marginal coverage {cov:.3f} too low"
        assert len(cp._qs) == 3, f"Expected 3 groups, got {list(cp._qs.keys())}"

    def test_unknown_group_fallback(self):
        rng = np.random.RandomState(9)
        cp = MondrianConformalPredictor()
        cp.calibrate(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array(["A", "A", "B"]),
        )
        lo, up = cp.predict(
            np.array([1.0]),
            np.array(["C"]),
            alpha=0.1,
        )
        assert lo[0] == 0.0
        assert up[0] == 0.0


# ============================================================
# Classification Conformal
# ============================================================

class TestClassificationConformal:
    def test_sets_produced(self):
        rng = np.random.RandomState(10)
        y = rng.uniform(-3, 3, 200)
        y_hat = y + 0.2 * rng.randn(200)

        cp = ClassificationConformalPredictor()
        cp.calibrate(y[:100], y_hat[:100])
        sets = cp.predict(y_hat[100:], alpha=0.1)

        assert len(sets) == 100
        for s in sets:
            assert len(s) >= 1
            assert all(c in [-3, -2, -1, 0, 1, 2, 3] for c in s)

    def test_metrics(self):
        rng = np.random.RandomState(11)
        y = rng.uniform(-3, 3, 200)
        y_hat = y + 0.2 * rng.randn(200)

        cp = ClassificationConformalPredictor()
        cp.calibrate(y[:100], y_hat[:100])
        sets = cp.predict(y_hat[100:], alpha=0.1)

        metrics = classification_set_metrics(y[100:], sets)
        assert 0.0 <= metrics['coverage'] <= 1.0
        assert metrics['avg_set_size'] >= 1.0
        assert 'size_distribution' in metrics

    def test_singleton_when_q_is_zero(self):
        cp = ClassificationConformalPredictor()
        cp._residuals = np.array([0.0, 0.0, 0.0])
        cp._q = 0.0
        sets = cp.predict(np.array([0.0, 1.0, 1.5]), alpha=0.1)
        for s in sets:
            assert len(s) == 1

    def test_fallback_when_empty_set(self):
        cp = ClassificationConformalPredictor(class_centers=[-3, 3])
        cp._residuals = np.array([1.0, 1.0])
        sets = cp.predict(np.array([0.0]), alpha=0.1)
        assert len(sets[0]) >= 1


# ============================================================
# MC Dropout RAW baseline
# ============================================================

class TestMCDropoutRaw:
    def test_output_shape(self):
        lo, up = mc_dropout_interval(np.array([0.0, 1.0]), np.array([0.5, 0.5]), alpha=0.1)
        assert lo.shape == (2,)
        assert up.shape == (2,)
        assert np.all(lo < up)

    def test_gaussian_undercoverage(self):
        rng = np.random.RandomState(12)
        y = rng.randn(500)
        y_hat = y + 0.3 * rng.randn(500)    # true error ~0.3
        mc_std = np.full(500, 0.05)          # model overconfident: thinks error ~0.05

        lo, up = mc_dropout_interval(y_hat, mc_std, alpha=0.1)
        cov = compute_coverage(y, lo, up)
        # z_0.95 * 0.05 = 0.082, but true error ~0.3 → massive under-coverage
        assert cov < 0.5


# ============================================================
# Helper functions
# ============================================================

class TestSentimentGroup:
    def test_group_assignment(self):
        groups = sentiment_group(np.array([-2.0, 0.0, 1.5]))
        assert list(groups) == ["negative", "neutral", "positive"]

    def test_boundary_values(self):
        groups = sentiment_group(np.array([-0.001, 0.0, 0.001]))
        assert list(groups) == ["negative", "neutral", "positive"]


class TestMapTo7Class:
    def test_exact_values(self):
        out = map_to_7class(np.array([-3.0, -1.0, 0.0, 2.0, 3.0]))
        assert list(out) == [-3, -1, 0, 2, 3]

    def test_rounding(self):
        out = map_to_7class(np.array([0.4, 0.6, 1.4, 2.6]))
        assert list(out) == [0, 1, 1, 3]

    def test_clipping(self):
        out = map_to_7class(np.array([-5.0, 5.0]))
        assert list(out) == [-3, 3]


# ============================================================
# Conditional coverage
# ============================================================

class TestConditionalCoverage:
    def test_by_sentiment_keys(self):
        rng = np.random.RandomState(13)
        yt = rng.uniform(-1, 1, 100)
        yp = yt + 0.1 * rng.randn(100)
        lo, up = yp - 1.0, yp + 1.0
        result = conditional_coverage_by_sentiment(yt, yp, lo, up)
        assert set(result.keys()) == {"Negative", "Neutral", "Positive"}
        for v in result.values():
            assert 0.0 <= v["coverage"] <= 1.0

    def test_by_bucket_keys(self):
        rng = np.random.RandomState(14)
        yt = rng.uniform(-3, 3, 200)
        yp = yt + 0.1 * rng.randn(200)
        lo, up = yp - 1.0, yp + 1.0
        result = conditional_coverage_by_bucket(yp, yt, lo, up)
        assert len(result) == 6
        for v in result.values():
            assert "count" in v
            assert "coverage" in v


# ============================================================
# Fixture
# ============================================================

@pytest.fixture
def rng():
    return np.random.RandomState(42)
