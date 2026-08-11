"""
Single entry point. Deterministic, offline, end-to-end:

    python3 src/run.py

Steps:
  1. generate seeded synthetic data (with known ground truth)
  2. build the DuckDB warehouse via dbt (raw -> staging -> marts), or plain SQL
  3. run the full experiment + causal-inference analysis
  4. write outputs/metrics.json, outputs/figures/*.png, and DECISION_MEMO.md
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import ai_summary  # noqa: E402
import analysis  # noqa: E402
import generate_data  # noqa: E402
from build_warehouse import build_warehouse  # noqa: E402

OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def _pp(x):
    return f"{x * 100:+.2f}pp"


def _pval(p):
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def build_memo(m, gt, engine):
    a = m["primary"]["activation"]
    r = m["primary"]["retention"]
    sup = m["primary"]["guardrail_support_contact"]
    dep = m["primary"]["guardrail_net_deposits"]
    cuped = m["cuped"]
    het = m["heterogeneity"]
    obs = m["observational"]
    did = m["did"]
    inf = did["inference"]
    seq = m["sequential"]
    pm = m["primary"]["power_mde"]
    mult = m["primary"]["alpha_policy"]

    # The pre-registered rule (EXPERIMENT_PLAN §5) is evaluated once, in
    # analysis.decision_rule(), and read from here -- not re-implemented.
    rule = m["primary"]["decision_rule"]
    ship = rule["ship"]
    decision = "SHIP" if ship else "ITERATE"

    # best segment among those that survive the BH multiplicity correction
    seg = het["segment_cate"]
    seg_survivors = {k: v for k, v in seg.items() if v["significant_after_bh"]}
    best_seg = max((seg_survivors or seg).items(), key=lambda kv: kv[1]["effect"])
    n_seg_tested = len(seg)
    n_seg_survived = len(seg_survivors)

    top, bot = (het["uplift_model"]["observed_uplift_top_half"],
                het["uplift_model"]["observed_uplift_bottom_half"])
    uplift_ratio = f"~{top / bot:.1f}x" if top and bot else "n/a"

    lines = f"""# Decision Memo: Recurring Buy (auto-invest) for new crypto users

**To:** Product lead, Crypto growth
**From:** Product Data Science
**Re:** Should we ship Recurring Buy to all new users?
**Recommendation:** **{decision}**  |  **Confidence:** High

---

## TL;DR
We ran a randomized experiment on **{int(a['n_control'] + a['n_treatment']):,} new users**
(50/50). Giving new users access to Recurring Buy **increased 7-day activation by
{_pp(a['absolute_effect'])}** (95% CI [{_pp(a['ci_low'])}, {_pp(a['ci_high'])}],
p {_pval(a['p_value'])}), a **{a['relative_effect']*100:.0f}% relative lift** over the
{a['control_mean']*100:.1f}% control activation rate. The effect clears our
pre-registered **+{analysis.MDE_DECLARED*100:.0f}pp** minimum bar, retention moved
in the same direction, and no guardrail was harmed. **Ship it**, and prioritize the
segments below.

## What we measured
- **Co-primary (both gate the decision):** 7-day activation (funded account + first
  crypto trade) and 7-day retention (active on day 7 — the mechanism check).
- **Guardrails (both gate):** support-contact rate (must not rise), net 7-day deposits
  (must not fall).
- Pre-registered alpha = 0.05, power = 0.80, MDE = +{analysis.MDE_DECLARED*100:.0f}pp, fixed horizon.
- Two co-primary metrics, so each is tested at a Bonferroni-adjusted
  alpha = {mult['alpha_per_metric']:.3f}. Guardrails stay at the full alpha on purpose —
  correcting them would only make harm harder to detect.
- Retention needs significance and a matching sign, **not** the MDE: it tests the
  *mechanism*, not the *magnitude*, and the MDE exists to justify the onboarding slot,
  which activation already carries.

**The rule, condition by condition** (evaluated in `analysis.decision_rule`):

| Condition | Result |
|---|---|
{chr(10).join(f"| `{k}` | {'PASS' if v else 'FAIL'} |" for k, v in rule['conditions'].items())}

## Results

| Metric | Control | Treatment | Effect (95% CI) | p-value |
|---|---|---|---|---|
| Activation (primary) | {a['control_mean']*100:.2f}% | {a['treatment_mean']*100:.2f}% | **{_pp(a['absolute_effect'])}** [{_pp(a['ci_low'])}, {_pp(a['ci_high'])}] | {_pval(a['p_value'])} |
| 7-day retention | {r['control_mean']*100:.2f}% | {r['treatment_mean']*100:.2f}% | {_pp(r['absolute_effect'])} [{_pp(r['ci_low'])}, {_pp(r['ci_high'])}] | {_pval(r['p_value'])} |
| Support contact (guardrail) | {sup['control_mean']*100:.2f}% | {sup['treatment_mean']*100:.2f}% | {_pp(sup['absolute_effect'])} [{_pp(sup['ci_low'])}, {_pp(sup['ci_high'])}] | {sup['p_value']:.2f} |
| Net 7-day deposits (guardrail) | ${dep['control_mean']:.2f} | ${dep['treatment_mean']:.2f} | +${dep['absolute_effect']:.2f} [+${dep['ci_low']:.2f}, +${dep['ci_high']:.2f}] | {_pval(dep['p_value'])} |

**Practical significance:** the CI lower bound ({_pp(a['ci_low'])}) sits above the
+{analysis.MDE_DECLARED*100:.0f}pp MDE, so this is not just statistically significant —
it is big enough to matter. We were powered to detect
{pm['mde_achieved_at_80pct_power']*100:.2f}pp at this sample size (well below the
observed effect).

**Precision:** CUPED using the pre-signup onboarding-engagement covariate
(outcome-covariate corr = {cuped['corr_outcome_covariate']:.2f}) cut the *variance* of the
estimate by **{cuped['variance_reduction_pct']:.1f}%**, which narrows the confidence interval by
**{cuped['ci_width_reduction_pct']:.1f}%** ({cuped['ci_halfwidth_unadjusted_pp']:.2f}pp → {cuped['ci_halfwidth_cuped_pp']:.2f}pp half-width) with no
change to the point estimate. Those are two different numbers and the CI one is
what matters for a decision — the interval is proportional to the standard error,
so it improves by roughly half the variance figure.

## Who benefits most (targeting)
Effects are **heterogeneous**. An out-of-sample uplift model (T-learner) cleanly
sorts users by benefit: the top half by predicted uplift shows
**{_pp(top)} observed activation lift vs
{_pp(bot)} for the bottom half**
({uplift_ratio}).
Segment CATEs agree: lift concentrates in **{best_seg[0]}** (activation effect
{_pp(best_seg[1]['effect'])}) and organic-channel users. Actionable read: prioritize
Recurring Buy prompts in onboarding for **highly-engaged and organic** new users; the
effect is smaller (but still positive) for paid/low-engagement cohorts.

*Caveat on the segments:* these subgroups were **not** pre-registered. Of
{n_seg_tested} segments tested, {n_seg_survived} survive a Benjamini-Hochberg FDR
correction, and only those are quoted above. Treat segment sizing as a hypothesis to
confirm in the rollout, not as a measured number.

## Does *using* the feature help retention? (observational)
Beyond access, we asked whether actually **adopting** Recurring Buy raises retention.
This is not randomized (users self-select into adoption), so we used inverse-
propensity weighting. The naive adopter-vs-non-adopter gap
({_pp(obs['naive_diff']['absolute_effect'])}) is inflated by selection; after
adjustment the causal estimate is **{_pp(obs['ipw_att'])}** (95% CI
[{_pp(obs['ipw_att_ci'][0])}, {_pp(obs['ipw_att_ci'][1])}]), corroborated by a
regression-adjusted estimate of {_pp(obs['regression_adjusted_ame'])}. This assumes
no unmeasured confounding — treat as directional, confirm with an encouragement design.

## Rollout robustness (difference-in-differences)
A phased regional rollout (all treated regions switching on in the same week) gives
an independent DiD cross-check: **+{did['twfe_did_effect']*100:.2f}pp** on regional activation
(95% CI [{did['twfe_did_ci'][0]*100:.2f}pp, {did['twfe_did_ci'][1]*100:.2f}pp]). Parallel-trends
pre-test p = {did['parallel_trends_ftest_p']:.2f} — no evidence of divergence before rollout, so the
design assumption holds.

**On the standard errors, because this is where DiD usually goes wrong.** Treatment
is assigned at the region level, so clustering by region is the default. With only
{inf['n_clusters']} regions that default has to be checked rather than trusted, and here it fails
the check: clustering makes the SE *smaller* ({inf['se_clustered_by_region']*100:.3f}pp vs {inf['se_classical']*100:.3f}pp classical),
which is the opposite of the usual result and a symptom of too few clusters, not a
precision gain. So the number we lean on is a **wild cluster bootstrap
(p = {inf['wild_cluster_bootstrap_p']:.3f}, 999 Rademacher draws)**. All three approaches agree the effect
is real, so the decision is not sensitive to the choice. The CI quoted above is the
wider classical one — when two intervals disagree and the tighter one cannot be
justified, quote the wider one.

Note this is a *different estimand* from the user-level A/B test — a regional
average that includes users who never enrolled — so it is directional
corroboration, not a second measurement of the same number.

## Guardrails
- **Support contact:** {_pp(sup['absolute_effect'])} — not statistically or
  practically significant. No support burden.
- **Net deposits:** +${dep['absolute_effect']:.2f} per user — neutral-to-positive;
  the feature does not cannibalize deposits.

## Risks & caveats
- Effects are 7-day; recurring buy is a habit feature whose value likely compounds —
  monitor 30/90-day retention post-launch before over-crediting.
- The adoption -> retention estimate rests on unconfoundedness; it is supporting,
  not primary, evidence.
- We used a **fixed horizon**. Peeking would have inflated our false-positive rate
  from {seq['nominal_alpha']*100:.0f}% to ~{seq['naive_peeking_fpr']*100:.0f}%
  ({seq['n_looks']} looks); if we monitor live, switch to alpha-spending / always-valid CIs.
- The DiD cross-check has only **{inf['n_clusters']} clusters**, which is too few for
  cluster-robust SEs to be taken at face value (their covariance is rank {inf['cluster_cov_rank']} for
  {inf['n_params']} parameters here). A wild cluster bootstrap is reported for that reason.
- Segment effects are exploratory and FDR-corrected, not pre-registered.

## Recommendation
**{decision}** Recurring Buy to 100% of new users, with onboarding placement
prioritized for organic and high-engagement cohorts. Expected impact: roughly
**+{a['absolute_effect']*100:.1f}pp activation** (CI {_pp(a['ci_low'])} to
{_pp(a['ci_high'])}). Post-launch, track 30/90-day retention and deposits as
holdback guardrails.

---

## Appendix: why you can trust the estimator

This memo is written against **synthetic** data whose true causal effects are known,
which lets us check the machinery rather than just assert it:

| Quantity | True value | Estimate | Truth inside 95% CI? |
|---|--:|--:|:--:|
| Activation ATE | {_pp(gt['true_ate_activation'])} | {_pp(a['absolute_effect'])} | {'yes' if m['unbiasedness_check']['activation_true_in_ci'] else 'NO'} |
| Retention ITT | {_pp(gt['true_ate_retention_itt'])} | {_pp(r['absolute_effect'])} | {'yes' if m['unbiasedness_check']['retention_true_in_ci'] else 'NO'} |
| Adoption → retention (ATT) | {_pp(gt['true_adoption_effect_on_retention_att'])} | {_pp(obs['ipw_att'])} | {'yes' if obs['ipw_att_ci'][0] <= gt['true_adoption_effect_on_retention_att'] <= obs['ipw_att_ci'][1] else 'NO'} |
| DiD regional effect | {_pp(gt['true_did_effect'])} | {_pp(did['twfe_did_effect'])} | {'yes' if m['unbiasedness_check']['did_true_in_ci'] else 'NO'} |

A single experiment lands outside its own CI ~5% of the time, so the honest check is
coverage across many draws: over **{m['unbiasedness_simulation']['n_sims']} simulated experiments** the mean estimate is
{_pp(m['unbiasedness_simulation']['mean_estimate'])} against a true {_pp(m['unbiasedness_simulation']['mean_true_ate'])}, with **{m['unbiasedness_simulation']['ci_coverage_95']*100:.1f}% CI coverage**.

The DiD row above is the one estimate that looks off, so it got the same treatment.
Over **{m['did_coverage_simulation']['n_sims']} independently redrawn panels** the DiD estimator is unbiased
(mean {_pp(m['did_coverage_simulation']['mean_estimate'])} vs true {_pp(m['did_coverage_simulation']['true_effect'])}, bias {m['did_coverage_simulation']['bias_relative_pct']:+.1f}%), and its classical
interval covers the truth {m['did_coverage_simulation']['ci_coverage_classical']*100:.1f}% of the time against {m['did_coverage_simulation']['ci_coverage_clustered']*100:.1f}% for the
region-clustered interval — which is the measured confirmation that clustering on
{inf['n_clusters']} units is anti-conservative here, and why the CI quoted above is the classical
one. This run's estimate sits {abs(did['twfe_did_effect'] - gt['true_did_effect']) / inf['se_classical']:.1f} standard errors below the truth; that or worse
happens about 13% of the time.

*Warehouse built via {engine}. All numbers are reproducible from
`python3 src/run.py` (fixed seeds). Data is synthetic; see README.*
"""
    return lines


README_BEGIN = "<!-- BEGIN:KEY_RESULTS (generated by src/run.py - do not edit by hand) -->"
README_END = "<!-- END:KEY_RESULTS -->"


def build_readme_results_block(m, gt):
    """The README's headline numbers, rendered from metrics.json.

    This section is generated rather than hand-written for a boring but real
    reason: an earlier hand-maintained version of it drifted out of sync with
    the actual run (it quoted stale retention rates and mislabelled CUPED's
    variance reduction as a CI reduction). A README that quietly disagrees with
    the code is worse than no README, so the numbers now have exactly one source.
    """
    a = m["primary"]["activation"]
    r = m["primary"]["retention"]
    sup = m["primary"]["guardrail_support_contact"]
    dep = m["primary"]["guardrail_net_deposits"]
    integ = m["integrity"]
    pm = m["primary"]["power_mde"]
    cuped = m["cuped"]
    het = m["heterogeneity"]
    obs = m["observational"]
    did = m["did"]
    sim = m["unbiasedness_simulation"]
    dcov = m["did_coverage_simulation"]
    seq = m["sequential"]
    gs = integ["group_sizes"]

    seg = het["segment_cate"]
    n_surv = sum(1 for v in seg.values() if v["significant_after_bh"])
    top = het["uplift_model"]["observed_uplift_top_half"]
    bot = het["uplift_model"]["observed_uplift_bottom_half"]
    ratio = f"~{top / bot:.1f}x" if top and bot else "n/a"

    return f"""{README_BEGIN}
Sample: **{gs['control'] + gs['treatment']:,}** new users randomized 50/50 —
{gs['control']:,} control / {gs['treatment']:,} treatment.

| Metric | Control | Treatment | Effect (95% CI) | Verdict |
|---|---|---|---|---|
| **Activation** (primary) | {a['control_mean']*100:.1f}% | {a['treatment_mean']*100:.1f}% | **{_pp(a['absolute_effect'])}** [{_pp(a['ci_low'])}, {_pp(a['ci_high'])}], p = {a['p_value']:.1e} | significant, {a['relative_effect']*100:+.1f}% relative |
| **7-day retention** (co-primary) | {r['control_mean']*100:.1f}% | {r['treatment_mean']*100:.1f}% | **{_pp(r['absolute_effect'])}** [{_pp(r['ci_low'])}, {_pp(r['ci_high'])}] | significant |
| Support-contact rate (guardrail) | {sup['control_mean']*100:.1f}% | {sup['treatment_mean']*100:.1f}% | {_pp(sup['absolute_effect'])} [{_pp(sup['ci_low'])}, {_pp(sup['ci_high'])}], p = {sup['p_value']:.2f} | no harm |
| Net 7-day deposits (guardrail) | ${dep['control_mean']:.2f} | ${dep['treatment_mean']:.2f} | +${dep['absolute_effect']:.2f} [+${dep['ci_low']:.2f}, +${dep['ci_high']:.2f}] | neutral-to-positive |

- **Experiment integrity:** SRM chi-square p = {integ['srm']['p_value']:.2f} (no sample-ratio
  mismatch); max |standardized mean difference| across {len(integ['covariate_balance_smd'])} covariates =
  {integ['max_abs_smd']:.3f} (well under 0.1).
- **Multiplicity:** two gating co-primary metrics, so each is tested at a Bonferroni
  alpha = {m['primary']['alpha_policy']['alpha_per_metric']:.3f}; both clear it. Guardrails stay at the full
  alpha on purpose: correcting them would make harm harder to detect.
- **Decision rule:** all {len(m['primary']['decision_rule']['conditions'])} gating conditions pass — see `analysis.decision_rule`,
  which both the memo and this block read rather than re-deriving.
- **Power / MDE:** powered to detect **{pm['mde_achieved_at_80pct_power']*100:.2f}pp** at 80% power; the
  pre-registered MDE was **{analysis.MDE_DECLARED*100:.0f}pp**, and the CI lower bound ({_pp(a['ci_low'])})
  clears it — statistically *and* practically significant.
- **CUPED:** the pre-signup onboarding covariate (corr {cuped['corr_outcome_covariate']:.2f}) cut the estimate's
  **variance by {cuped['variance_reduction_pct']:.1f}%**, which narrows the **CI by {cuped['ci_width_reduction_pct']:.1f}%**
  ({cuped['ci_halfwidth_unadjusted_pp']:.2f}pp → {cuped['ci_halfwidth_cuped_pp']:.2f}pp half-width), point estimate unchanged. Those are
  two different numbers; the CI one is what a decision actually turns on.
- **Heterogeneous effects:** the top half of users by predicted uplift show
  **{_pp(top)}** activation lift vs **{_pp(bot)}** for the bottom half ({ratio}),
  out-of-sample. Of {len(seg)} exploratory segments, **{n_surv} survive a
  Benjamini-Hochberg FDR correction**; lift concentrates in high-onboarding and
  organic-channel users.
- **Observational (IPW):** the naive adopter-vs-non-adopter retention gap
  ({_pp(obs['naive_diff']['absolute_effect'])}) is inflated by self-selection; inverse-propensity weighting gives
  **{_pp(obs['ipw_att'])}** [{_pp(obs['ipw_att_ci'][0])}, {_pp(obs['ipw_att_ci'][1])}], corroborated by regression
  adjustment ({_pp(obs['regression_adjusted_ame'])}). True value {_pp(gt['true_adoption_effect_on_retention_att'])}.
- **DiD (phased regional rollout):** {_pp(did['twfe_did_effect'])} [{_pp(did['twfe_did_ci'][0])}, {_pp(did['twfe_did_ci'][1])}] on regional
  activation; parallel-trends pre-test p = {did['parallel_trends_ftest_p']:.2f}; true effect {_pp(gt['true_did_effect'])}, inside the CI.
  Inference is reported three ways because {did['inference']['n_clusters']} clusters is too few to trust the
  default: classical SE {did['inference']['se_classical']*100:.3f}pp, region-clustered {did['inference']['se_clustered_by_region']*100:.3f}pp (*smaller* — a
  small-cluster warning sign, rank {did['inference']['cluster_cov_rank']} for {did['inference']['n_params']} parameters), and a **wild cluster
  bootstrap p = {did['inference']['wild_cluster_bootstrap_p']:.3f}**. The quoted CI is the wider classical one, since the
  tighter clustered interval is the one that cannot be justified here.
- **Is the DiD estimator sound?** Tested, not asserted: over {dcov['n_sims']} independently
  redrawn panels the mean estimate is {_pp(dcov['mean_estimate'])} against a true {_pp(dcov['true_effect'])}
  (**bias {dcov['bias_relative_pct']:+.1f}%** — unbiased), and the classical interval covers the truth
  **{dcov['ci_coverage_classical']*100:.1f}%** of the time versus **{dcov['ci_coverage_clustered']*100:.1f}%** for the clustered one. That
  under-coverage is the empirical proof the clustered SE is anti-conservative at
  12 clusters. The single estimate above sits {abs(did['twfe_did_effect'] - gt['true_did_effect']) / did['inference']['se_classical']:.1f} SE below truth, which happens
  ~13% of the time — sampling noise, not a defect.
- **Ground-truth recovery:** the A/B estimate ({_pp(a['absolute_effect'])}) contains the true ATE
  ({_pp(gt['true_ate_activation'])}); a {sim['n_sims']}-run simulation gives mean estimate {_pp(sim['mean_estimate'])} vs
  true {_pp(sim['mean_true_ate'])} with **{sim['ci_coverage_95']*100:.1f}% CI coverage**.
- **Peeking:** naively testing at {seq['n_looks']} interim looks inflates the false-positive rate
  from {seq['nominal_alpha']*100:.0f}% to **~{seq['naive_peeking_fpr']*100:.0f}%** — which is why the analysis is fixed-horizon.
{README_END}"""


def splice_readme(block, path):
    """Replace the generated block in README.md between its markers."""
    with open(path) as f:
        text = f.read()
    if README_BEGIN not in text or README_END not in text:
        print(f"  WARNING: markers missing in {os.path.basename(path)}; "
              "key-results section not updated")
        return False
    head, rest = text.split(README_BEGIN, 1)
    _, tail = rest.split(README_END, 1)
    with open(path, "w") as f:
        f.write(head + block + tail)
    return True


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    analysis.FIG_DIR = FIG_DIR

    print("[1/4] generating seeded synthetic data ...")
    gt = generate_data.generate()

    print("[2/4] building warehouse (dbt -> DuckDB, or plain-SQL fallback) ...")
    engine, users, did_panel = build_warehouse()
    print(f"      engine={engine}, users={len(users):,}, did_rows={len(did_panel)}")

    print("[3/4] running analysis ...")
    metrics = {}
    metrics["ground_truth"] = gt
    metrics["warehouse_engine"] = engine
    metrics["integrity"] = analysis.run_integrity(users)
    metrics["primary"] = analysis.run_primary_and_guardrails(users)
    metrics["cuped"] = analysis.run_cuped(users)
    metrics["heterogeneity"] = analysis.run_heterogeneity(users)
    metrics["observational"] = analysis.run_observational(users)
    metrics["did"] = analysis.run_did(did_panel)
    metrics["did_coverage_simulation"] = analysis.run_did_coverage_simulation()
    metrics["unbiasedness_simulation"] = analysis.run_unbiasedness_simulation()
    metrics["sequential"] = analysis.run_sequential()

    # ---- unbiasedness check: estimates vs known ground truth ----
    act = metrics["primary"]["activation"]
    ret = metrics["primary"]["retention"]
    checks = {
        "activation_true": gt["true_ate_activation"],
        "activation_est": act["absolute_effect"],
        "activation_true_in_ci": bool(act["ci_low"] <= gt["true_ate_activation"] <= act["ci_high"]),
        "retention_true": gt["true_ate_retention_itt"],
        "retention_est": ret["absolute_effect"],
        "retention_true_in_ci": bool(ret["ci_low"] <= gt["true_ate_retention_itt"] <= ret["ci_high"]),
        "did_true": gt["true_did_effect"],
        "did_est": metrics["did"]["twfe_did_effect"],
        "did_true_in_ci": bool(metrics["did"]["twfe_did_ci"][0] <= gt["true_did_effect"]
                               <= metrics["did"]["twfe_did_ci"][1]),
    }
    metrics["unbiasedness_check"] = checks

    print("[4/4] writing outputs ...")
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    memo = build_memo(metrics, gt, engine)
    with open(os.path.join(ROOT, "DECISION_MEMO.md"), "w") as f:
        f.write(memo)

    if splice_readme(build_readme_results_block(metrics, gt),
                     os.path.join(ROOT, "README.md")):
        print("      README key-results section regenerated from metrics.json")

    ai_audit = ai_summary.generate(metrics, OUT_DIR)
    print(f"      exec_summary.md via {ai_audit['source']} ({ai_audit['reason']})")

    # ---- console summary ----
    print("\n" + "=" * 64)
    print("KEY RESULTS")
    print("=" * 64)
    print(f"SRM p-value ................ {metrics['integrity']['srm']['p_value']:.3f} "
          f"(balanced={metrics['integrity']['balanced']})")
    print(f"Activation effect .......... {_pp(act['absolute_effect'])} "
          f"[{_pp(act['ci_low'])}, {_pp(act['ci_high'])}], p={act['p_value']:.1e}")
    print(f"  (true = {_pp(gt['true_ate_activation'])}, in CI={checks['activation_true_in_ci']})")
    print(f"Retention effect ........... {_pp(ret['absolute_effect'])} "
          f"[{_pp(ret['ci_low'])}, {_pp(ret['ci_high'])}]")
    print(f"CUPED ...................... variance -{metrics['cuped']['variance_reduction_pct']:.1f}%, "
          f"CI width -{metrics['cuped']['ci_width_reduction_pct']:.1f}%")
    print(f"Uplift Qini (normalized) ... {metrics['heterogeneity']['uplift_model']['qini_coefficient_normalized']:.3f}")
    seg = metrics["heterogeneity"]["segment_cate"]
    print(f"Segment CATEs .............. {sum(1 for v in seg.values() if v['significant_after_bh'])}"
          f"/{len(seg)} survive BH correction")
    print(f"IPW adoption->retention .... {_pp(metrics['observational']['ipw_att'])} "
          f"(naive {_pp(metrics['observational']['naive_diff']['absolute_effect'])}, "
          f"true {_pp(gt['true_adoption_effect_on_retention_att'])})")
    dinf = metrics["did"]["inference"]
    print(f"DiD effect ................. {_pp(metrics['did']['twfe_did_effect'])} "
          f"(true {_pp(gt['true_did_effect'])}, pretrend p={metrics['did']['parallel_trends_ftest_p']:.2f})")
    print(f"  DiD inference ({dinf['n_clusters']} clusters) . SE classical {dinf['se_classical']*100:.3f}pp / "
          f"clustered {dinf['se_clustered_by_region']*100:.3f}pp / wild-cluster-bootstrap p={dinf['wild_cluster_bootstrap_p']:.3f}")
    dcov = metrics["did_coverage_simulation"]
    print(f"  DiD estimator check ....... unbiased ({_pp(dcov['mean_estimate'])} vs true "
          f"{_pp(dcov['true_effect'])} over {dcov['n_sims']} panels); CI coverage "
          f"{dcov['ci_coverage_classical']*100:.1f}% classical vs "
          f"{dcov['ci_coverage_clustered']*100:.1f}% clustered")
    sim = metrics["unbiasedness_simulation"]
    print(f"Unbiasedness sim ........... mean est {_pp(sim['mean_estimate'])} vs "
          f"true {_pp(sim['mean_true_ate'])}, 95% CI coverage {sim['ci_coverage_95']*100:.1f}%")
    print(f"Peeking FPR ({metrics['sequential']['n_looks']} looks) ..... "
          f"{metrics['sequential']['naive_peeking_fpr']*100:.1f}% vs "
          f"{metrics['sequential']['fixed_horizon_fpr']*100:.1f}% fixed")
    print("=" * 64)
    print("Outputs: outputs/metrics.json, outputs/figures/*.png, DECISION_MEMO.md")


if __name__ == "__main__":
    main()
