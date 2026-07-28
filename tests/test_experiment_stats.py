"""
Unit tests for the statistical helpers + a ground-truth recovery test.

Run:  python3 -m pytest tests -q      (or)   python3 tests/test_experiment_stats.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ai_summary                       # noqa: E402
import experiment_stats as es          # noqa: E402
import generate_data                    # noqa: E402


def test_two_proportion_diff_matches_known_effect():
    rng = np.random.default_rng(0)
    control = rng.binomial(1, 0.30, 50_000)
    treatment = rng.binomial(1, 0.35, 50_000)
    r = es.two_proportion_diff(control, treatment)
    assert abs(r.absolute_effect - 0.05) < 0.01
    assert r.ci_low < r.absolute_effect < r.ci_high
    assert r.p_value < 1e-6


def test_srm_passes_for_balanced_split():
    r = es.srm_test(20_000, 20_050)
    assert r["srm_flag"] is False
    assert r["p_value"] > 0.001


def test_srm_flags_imbalance():
    r = es.srm_test(20_000, 22_000)  # clear 48/52 split
    assert r["srm_flag"] is True


def test_cuped_preserves_point_estimate_and_cuts_variance():
    rng = np.random.default_rng(1)
    n = 40_000
    x = rng.normal(0, 1, n)                    # pre-period covariate
    treat = rng.binomial(1, 0.5, n)
    # outcome correlated with x, with a +0.20 treatment effect
    y = 0.5 + 0.4 * x + 0.20 * treat + rng.normal(0, 0.5, n)
    unadj = es.mean_diff(y[treat == 0], y[treat == 1])
    y_cuped, theta = es.cuped_adjust(y, x)
    adj = es.mean_diff(y_cuped[treat == 0], y_cuped[treat == 1])
    # point estimate roughly preserved; variance strictly reduced
    assert abs(adj.absolute_effect - unadj.absolute_effect) < 0.02
    assert adj.se < unadj.se


def test_power_and_mde_are_consistent():
    # if we size for an MDE, power to detect exactly that MDE should be ~0.8
    n = es.required_n_two_proportion(0.30, 0.02, alpha=0.05, power=0.8)
    p = es.power_two_proportion(0.30, 0.32, n, alpha=0.05)
    assert 0.78 < p < 0.82


def test_estimator_recovers_true_ate():
    """The A/B estimator recovers the known synthetic ground-truth ATE."""
    true_ate, activated, treat = generate_data.simulate_activation(seed=42, n=60_000)
    r = es.two_proportion_diff(activated[treat == 0], activated[treat == 1])
    assert r.ci_low <= true_ate <= r.ci_high


def test_p_value_does_not_underflow_to_zero():
    """A strongly significant result must report a tiny p, not a fake exact 0.

    `1 - norm.cdf(|z|)` loses all precision once the true tail probability drops
    below machine epsilon (|z| ~ 8) and returns exactly 0.0 -- a p-value that
    cannot be true. `norm.sf` computes the tail directly. This effect size sits
    in that gap: the naive form returns 0, the survival function does not.
    """
    from scipy import stats

    rng = np.random.default_rng(3)
    control = rng.binomial(1, 0.30, 40_000)
    treatment = rng.binomial(1, 0.35, 40_000)
    r = es.two_proportion_diff(control, treatment)

    z = stats.norm.isf(r.p_value / 2)          # recover |z| from the reported p
    assert 9 < z < 37, f"test no longer probes the underflow gap (z={z:.1f})"
    assert 2 * (1 - stats.norm.cdf(z)) == 0.0  # the bug this guards against
    assert 0.0 < r.p_value < 1e-15             # ...and the fixed behaviour


def test_cuped_ci_reduction_is_smaller_than_variance_reduction():
    """The two CUPED figures must not be conflated.

    The CI is proportional to the SE, so it narrows by roughly half the variance
    reduction. Quoting the variance figure as a CI figure overstates precision --
    this test pins the distinction.
    """
    rng = np.random.default_rng(11)
    n = 40_000
    x = rng.normal(0, 1, n)
    treat = rng.binomial(1, 0.5, n)
    y = 0.5 + 0.4 * x + 0.20 * treat + rng.normal(0, 0.5, n)
    unadj = es.mean_diff(y[treat == 0], y[treat == 1])
    y_cuped, _ = es.cuped_adjust(y, x)
    adj = es.mean_diff(y_cuped[treat == 0], y_cuped[treat == 1])

    var_red, ci_red = es.variance_and_ci_reduction(
        unadj.se, unadj.ci_halfwidth, adj.se, adj.ci_halfwidth)
    assert var_red > 0
    assert 0 < ci_red < var_red
    # ci_red = 1 - sqrt(1 - var_red), up to the two Welch dof giving very
    # slightly different t critical values (a ~1e-9 effect at this n).
    assert abs(ci_red - (1 - np.sqrt(1 - var_red))) < 1e-6


def test_trapezoid_matches_known_integrals():
    """Locks the hand-written integral used by the Qini curve.

    It replaced `np.trapz`, which was deprecated in NumPy 2.0 and removed in 2.1 --
    so the same code passed locally and crashed in CI on a newer NumPy. This test
    pins the replacement against integrals with closed-form answers.
    """
    import analysis

    x = np.linspace(0, 1, 2001)
    assert abs(analysis._trapezoid(x, x) - 0.5) < 1e-6          # ∫x dx = 1/2
    assert abs(analysis._trapezoid(np.ones_like(x), x) - 1.0) < 1e-12
    assert analysis._trapezoid(np.zeros_like(x), x) == 0.0
    # non-uniform spacing must still be handled correctly
    xu = np.array([0.0, 0.1, 0.5, 1.0])
    assert abs(analysis._trapezoid(xu, xu) - 0.5) < 1e-9


def test_benjamini_hochberg_controls_and_orders():
    # A clear signal survives; pure noise does not.
    q, rejected = es.benjamini_hochberg([1e-8, 0.001, 0.20, 0.40, 0.80])
    assert rejected[0] and rejected[1]
    assert not rejected[2:].any()
    # q-values are monotone in p and never below the raw p
    assert all(q[i] <= q[i + 1] for i in range(len(q) - 1))
    assert all(q[i] >= p for i, p in enumerate([1e-8, 0.001, 0.20, 0.40, 0.80]))
    # all-null case: nothing should be declared significant
    _, none_rejected = es.benjamini_hochberg([0.30, 0.45, 0.60, 0.90])
    assert not none_rejected.any()


def test_bh_is_more_conservative_than_uncorrected():
    """BH must never reject more than the uncorrected test at the same alpha."""
    rng = np.random.default_rng(5)
    p_values = rng.uniform(0, 1, 40)
    _, rejected = es.benjamini_hochberg(p_values, alpha=0.05)
    assert rejected.sum() <= (p_values < 0.05).sum()


# --------------------------------------------------------------------------- #
# AI summary numeric guardrail
# --------------------------------------------------------------------------- #
def test_guardrail_accepts_traceable_and_rounded_numbers():
    metrics = {"primary": {"activation": {"absolute_effect": 0.0776,
                                          "control_mean": 0.3408}}}
    allowed = ai_summary.collect_allowed_numbers(metrics)
    # exact, and honestly rounded to fewer decimals
    assert ai_summary.audit_numbers("Activation rose 7.76pp from 34.08%.", allowed) == []
    assert ai_summary.audit_numbers("Activation rose 7.8pp.", allowed) == []


def test_guardrail_rejects_a_fabricated_number():
    metrics = {"primary": {"activation": {"absolute_effect": 0.0776}}}
    allowed = ai_summary.collect_allowed_numbers(metrics)
    bad = ai_summary.audit_numbers("Activation rose 12.4pp, worth $1.3M.", allowed)
    tokens = {u["token"] for u in bad}
    assert "12.4" in tokens and "1.3" in tokens


def test_guardrail_allows_structural_numbers():
    """7-day windows, 95% CIs and 30/90-day follow-ups are not measurements."""
    allowed = ai_summary.collect_allowed_numbers({"x": 0.0776})
    text = "7-day activation, 95% CI, tracked at 30 and 90 days across 3 segments."
    assert ai_summary.audit_numbers(text, allowed) == []


def test_deterministic_summary_passes_its_own_guardrail():
    """The offline fallback must satisfy the same check the LLM draft faces."""
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "outputs",
                                "metrics.json")
    if not os.path.exists(metrics_path):
        return  # pipeline has not been run yet; nothing to check
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    facts = ai_summary.build_facts(metrics)
    allowed = ai_summary.collect_allowed_numbers(facts)
    assert ai_summary.audit_numbers(ai_summary.deterministic_summary(facts),
                                    allowed) == []


def test_guardrail_rejects_an_invented_business_figure():
    """The failure mode that matters: fluent, plausible, and not in the data.

    Scoping the whitelist to the fact sheet (rather than all of metrics.json) is
    what makes this catch the dollar figure -- against the wider set, "2.4"
    collided with an unrelated deposits confidence bound and slipped through.
    """
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "outputs",
                                "metrics.json")
    if not os.path.exists(metrics_path):
        return
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    allowed = ai_summary.collect_allowed_numbers(ai_summary.build_facts(metrics))
    draft = ("Activation rose 7.76pp to 41.84%, adding roughly $2.4M in annual "
             "revenue across 18,000 clients.")
    flagged = {u["token"] for u in ai_summary.audit_numbers(draft, allowed)}
    assert "2.4" in flagged and "18,000" in flagged
    # ...while the two real numbers are left alone
    assert "7.76" not in flagged and "41.84" not in flagged


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
