"""
The committed README and decision memo must match what the code generates.

This guards a real bug that already happened once in this repo: the README's
hand-maintained results table drifted away from the analysis, quoting stale
retention rates and describing CUPED's variance reduction as a CI reduction. A
report that disagrees with its own pipeline is worse than no report.

The check is a pure regeneration from the committed `outputs/metrics.json` --
string formatting only, no re-estimation -- so it is deterministic across
platforms and library versions, unlike diffing recomputed floats would be.

Run:  python3 tests/test_reports_in_sync.py   (or via pytest)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import run  # noqa: E402


def _load():
    with open(os.path.join(ROOT, "outputs", "metrics.json")) as f:
        metrics = json.load(f)
    return metrics, metrics["ground_truth"]


def _read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


def _first_difference(expected, actual, label):
    exp_lines, act_lines = expected.splitlines(), actual.splitlines()
    for i, (e, a) in enumerate(zip(exp_lines, act_lines), start=1):
        if e != a:
            return (f"{label} line {i} is stale:\n"
                    f"  committed: {a!r}\n  generated: {e!r}")
    if len(exp_lines) != len(act_lines):
        return (f"{label}: committed has {len(act_lines)} lines, "
                f"generated has {len(exp_lines)}")
    return None


def test_readme_results_block_matches_metrics():
    metrics, gt = _load()
    readme = _read("README.md")
    assert run.README_BEGIN in readme and run.README_END in readme, \
        "README.md is missing the generated-block markers"

    committed = readme.split(run.README_BEGIN, 1)[1].split(run.README_END, 1)[0]
    expected = run.build_readme_results_block(metrics, gt)
    expected_body = expected.split(run.README_BEGIN, 1)[1].split(run.README_END, 1)[0]

    diff = _first_difference(expected_body, committed, "README key-results block")
    assert diff is None, diff + "\n\nRun: python3 src/run.py"


def test_decision_memo_matches_metrics():
    metrics, gt = _load()
    expected = run.build_memo(metrics, gt, metrics["warehouse_engine"])
    diff = _first_difference(expected, _read("DECISION_MEMO.md"), "DECISION_MEMO.md")
    assert diff is None, diff + "\n\nRun: python3 src/run.py"


def test_readme_real_data_numbers_match_the_json():
    """The real-data section is hand-written prose, so its numbers can drift.

    They already did once: two figures in that section were wrong within minutes
    of being typed. The generated results block is protected by construction; this
    section is not, so each headline number is checked against
    outputs/real_data_validation.json explicitly.
    """
    path = os.path.join(ROOT, "outputs", "real_data_validation.json")
    if not os.path.exists(path):
        print("SKIP  real-data validation has not been run yet")
        return
    with open(path) as f:
        rd = json.load(f)

    ipw = rd["ipw_benchmark"]
    rob = ipw["robustness_across_30_confounding_draws"]
    integ = rd["integrity"]
    lm = rd["uplift_model"]["learned_model"]
    ref = rd["uplift_model"]["single_covariate_reference"]
    cuped = rd["cuped"]

    expected = {
        "experimental benchmark": f"{ipw['experimental_benchmark']*100:+.3f}pp",
        "naive on confounded": f"{ipw['naive_on_confounded']*100:+.3f}pp",
        "ipw estimate": f"{ipw['ipw_estimate']*100:+.3f}pp",
        "regression adjusted": f"{ipw['regression_adjusted_ame']*100:+.3f}pp",
        "mean bias removed": f"{rob['mean_bias_removed_pct']:.1f}%",
        "worst bias removed": f"{rob['worst_bias_removed_pct']:.1f}%",
        "srm p-value": f"{integ['srm_primary_contrast']['p_value']:.3f}",
        "max smd": f"{integ['max_abs_smd']:.3f}",
        "uplift separation": f"{lm['separation_pp']*100:+.2f}pp",
        # 3 decimals on purpose: this p-value is 0.3648, which sits close enough to
        # the 2-decimal boundary that it was written as "0.37" once. Quoting 3
        # decimals removes the ambiguity instead of relying on rounding luck.
        "uplift p-value": f"p = {lm['separation_p_value']:.3f}",
        "single-covariate separation": f"{ref['separation_pp']*100:+.2f}pp",
        "single-covariate p-value": f"p = {ref['separation_p_value']:.3f}",
        "cuped variance reduction": f"{cuped['variance_reduction_pct']:.2f}%",
    }
    readme = _read("README.md")
    missing = {k: v for k, v in expected.items()
               if v.lstrip("+") not in readme and v not in readme}
    assert not missing, (
        "README's real-data section disagrees with real_data_validation.json:\n"
        + "\n".join(f"  {k}: expected to find {v!r}" for k, v in missing.items()))

    # the covariate count is quoted in prose too
    assert f"{len(integ['covariate_balance_smd'])} covariates" in readme, (
        f"README should say '{len(integ['covariate_balance_smd'])} covariates'")


def test_memo_decision_follows_the_preregistered_rule():
    """The memo's verdict must be the declared rule applied to the numbers,
    not a conclusion written independently of them.

    The rule is EXPERIMENT_PLAN §5: activation significant at the stated alpha, its
    CI lower bound past the MDE, and the support-contact guardrail intact. Retention
    and deposits are reported but deliberately do not gate -- see §11.1 and §11.2.
    """
    metrics, _ = _load()
    a = metrics["primary"]["activation"]
    sup = metrics["primary"]["guardrail_support_contact"]
    alpha = metrics["primary"]["alpha_policy"]["alpha_per_metric"]

    import analysis
    should_ship = (a["p_value"] < alpha
                   and a["ci_low"] >= analysis.MDE_DECLARED
                   and sup["ci_high"] <= 0.01)
    memo = _read("DECISION_MEMO.md")
    assert ("**SHIP**" in memo) == should_ship, \
        "memo verdict disagrees with the pre-registered decision rule"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} report-sync tests passed.")
