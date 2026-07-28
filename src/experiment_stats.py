"""
Pure statistical helpers for the experiment analysis.

Everything here is deliberately small, explicit and unit-testable. No p-hacking:
the primary metric, the MDE and the alpha are declared up front by the caller;
these functions just compute the standard estimators correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class DiffResult:
    control_mean: float
    treatment_mean: float
    absolute_effect: float          # treatment - control
    relative_effect: float          # (t - c) / c
    ci_low: float
    ci_high: float
    ci_halfwidth: float
    se: float
    p_value: float
    n_control: int
    n_treatment: int

    def as_dict(self):
        out = {}
        for k, v in asdict(self).items():
            if v is None:
                out[k] = None
            elif k in ("n_control", "n_treatment"):
                out[k] = int(v)
            elif isinstance(v, (int, float, np.floating)):
                out[k] = float(v)
            else:
                out[k] = v
        return out


# --------------------------------------------------------------------------- #
# Two-proportion test (binary metrics: activation, retention, guardrails)
# --------------------------------------------------------------------------- #
def two_proportion_diff(control, treatment, alpha=0.05) -> DiffResult:
    """Difference in proportions with a Wald CI and a two-proportion z-test.

    The CI uses the unpooled SE (correct for estimating a difference); the
    p-value uses the pooled SE (correct for testing H0: p_t = p_c).
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    n_c, n_t = len(control), len(treatment)
    p_c, p_t = control.mean(), treatment.mean()
    diff = p_t - p_c

    # unpooled SE for the CI
    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = diff - z * se_unpooled, diff + z * se_unpooled

    # pooled SE for the hypothesis test
    p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z_stat = diff / se_pooled if se_pooled > 0 else 0.0
    # survival function, not 1 - cdf: 1 - cdf underflows to exactly 0.0 for
    # |z| > ~8 and would report a fake p = 0 for a strongly significant result.
    p_value = 2 * stats.norm.sf(abs(z_stat))

    return DiffResult(
        control_mean=p_c, treatment_mean=p_t,
        absolute_effect=diff,
        relative_effect=diff / p_c if p_c else np.nan,
        ci_low=ci_low, ci_high=ci_high, ci_halfwidth=z * se_unpooled,
        se=se_unpooled, p_value=p_value,
        n_control=n_c, n_treatment=n_t,
    )


# --------------------------------------------------------------------------- #
# Difference in means (continuous metrics: deposits) via Welch's t
# --------------------------------------------------------------------------- #
def mean_diff(control, treatment, alpha=0.05) -> DiffResult:
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    n_c, n_t = len(control), len(treatment)
    m_c, m_t = control.mean(), treatment.mean()
    diff = m_t - m_c
    se = np.sqrt(control.var(ddof=1) / n_c + treatment.var(ddof=1) / n_t)
    # Welch-Satterthwaite dof
    dof = (control.var(ddof=1) / n_c + treatment.var(ddof=1) / n_t) ** 2 / (
        (control.var(ddof=1) / n_c) ** 2 / (n_c - 1)
        + (treatment.var(ddof=1) / n_t) ** 2 / (n_t - 1)
    )
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    t_stat = diff / se
    p_value = 2 * stats.t.sf(abs(t_stat), dof)  # sf avoids underflow to 0.0
    return DiffResult(
        control_mean=m_c, treatment_mean=m_t,
        absolute_effect=diff,
        relative_effect=diff / m_c if m_c else np.nan,
        ci_low=diff - tcrit * se, ci_high=diff + tcrit * se,
        ci_halfwidth=tcrit * se, se=se, p_value=p_value,
        n_control=n_c, n_treatment=n_t,
    )


# --------------------------------------------------------------------------- #
# Sample Ratio Mismatch (SRM): chi-square goodness of fit vs expected split
# --------------------------------------------------------------------------- #
def srm_test(n_control, n_treatment, expected_ratio=0.5):
    total = n_control + n_treatment
    expected = np.array([total * (1 - expected_ratio), total * expected_ratio])
    observed = np.array([n_control, n_treatment])
    chi2, p = stats.chisquare(observed, expected)
    return {
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "observed_treat_share": float(n_treatment / total),
        "expected_treat_share": float(expected_ratio),
        "chi2": float(chi2),
        "p_value": float(p),
        "srm_flag": bool(p < 0.001),  # standard SRM alarm threshold
    }


# --------------------------------------------------------------------------- #
# Standardized mean difference (covariate balance)
# --------------------------------------------------------------------------- #
def standardized_mean_diff(x_control, x_treatment):
    """SMD for a numeric covariate. |SMD| < 0.1 is the usual 'balanced' rule."""
    x_c = np.asarray(x_control, dtype=float)
    x_t = np.asarray(x_treatment, dtype=float)
    pooled_sd = np.sqrt((x_c.var(ddof=1) + x_t.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((x_t.mean() - x_c.mean()) / pooled_sd)


# --------------------------------------------------------------------------- #
# Power / MDE for a two-proportion test (fixed horizon)
# --------------------------------------------------------------------------- #
def mde_two_proportion(baseline_rate, n_per_arm, alpha=0.05, power=0.8):
    """Minimum detectable ABSOLUTE effect (pp) for the achieved sample size."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p = baseline_rate
    # solve using the pooled-variance approximation
    mde = (z_a + z_b) * np.sqrt(2 * p * (1 - p) / n_per_arm)
    return float(mde)


def power_two_proportion(p_control, p_treatment, n_per_arm, alpha=0.05):
    """Power to detect a given true effect at the achieved sample size."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    diff = abs(p_treatment - p_control)
    se = np.sqrt(p_control * (1 - p_control) / n_per_arm
                 + p_treatment * (1 - p_treatment) / n_per_arm)
    z = diff / se - z_a
    return float(stats.norm.cdf(z))


def required_n_two_proportion(baseline_rate, mde, alpha=0.05, power=0.8):
    """Sample size PER ARM needed to detect an absolute effect `mde`."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p = baseline_rate
    n = (z_a + z_b) ** 2 * 2 * p * (1 - p) / (mde ** 2)
    return int(np.ceil(n))


# --------------------------------------------------------------------------- #
# CUPED variance reduction
# --------------------------------------------------------------------------- #
def cuped_adjust(y, x):
    """Return CUPED-adjusted outcome using pre-experiment covariate x.

    theta = cov(y, x) / var(x); y_cuped = y - theta * (x - mean(x)).
    theta is estimated on the pooled data (independent of assignment), which
    keeps the adjusted estimator unbiased for the ATE while cutting variance.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_cuped = y - theta * (x - x.mean())
    return y_cuped, float(theta)


def variance_and_ci_reduction(se_before, halfwidth_before, se_after, halfwidth_after):
    """Report CUPED's gain on BOTH scales, because they are not the same number.

    Variance reduction is 1 - (se_after/se_before)^2; the confidence interval is
    proportional to the SE, so it narrows by only 1 - se_after/se_before -- about
    half as much in relative terms. Quoting the variance figure as "the CI shrank
    by X%" overstates the precision gain, so both are returned explicitly.
    """
    var_red = 1.0 - (se_after ** 2) / (se_before ** 2)
    ci_red = 1.0 - halfwidth_after / halfwidth_before
    return float(var_red), float(ci_red)


# --------------------------------------------------------------------------- #
# Multiplicity control for exploratory subgroup analyses
# --------------------------------------------------------------------------- #
def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR adjustment.

    The primary/co-primary metrics are pre-registered, but segment-level CATEs
    are exploratory: testing 8 subgroups at alpha = 0.05 gives a ~34% chance of
    at least one false positive under the null. BH-adjusted q-values keep the
    expected false-discovery rate at alpha instead.

    Returns (q_values, rejected) in the caller's original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p downwards
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q = np.empty(m)
    q[order] = q_ranked
    return q, q <= alpha


# --------------------------------------------------------------------------- #
# Bootstrap CI for an arbitrary statistic (used as a robustness check)
# --------------------------------------------------------------------------- #
def bootstrap_diff_ci(control, treatment, stat=np.mean, n_boot=2000,
                      alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        c = rng.choice(control, size=len(control), replace=True)
        t = rng.choice(treatment, size=len(treatment), replace=True)
        diffs[b] = stat(t) - stat(c)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(diffs.mean())
