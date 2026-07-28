# Decision Memo: Recurring Buy (auto-invest) for new crypto users

**To:** Product lead, Crypto growth
**From:** Product Data Science
**Re:** Should we ship Recurring Buy to all new users?
**Recommendation:** **SHIP**  |  **Confidence:** High

---

## TL;DR
We ran a randomized experiment on **40,000 new users**
(50/50). Giving new users access to Recurring Buy **increased 7-day activation by
+7.76pp** (95% CI [+6.82pp, +8.71pp],
p < 0.001), a **23% relative lift** over the
34.1% control activation rate. The effect clears our
pre-registered **+2pp** minimum bar, retention moved
in the same direction, and no guardrail was harmed. **Ship it**, and prioritize the
segments below.

## What we measured
- **Primary:** 7-day activation (funded account + first crypto trade).
- **Co-primary:** 7-day retention.
- **Guardrails:** support-contact rate (must not rise), net 7-day deposits (must not fall).
- Pre-registered alpha = 0.05, power = 0.80, MDE = +2pp, fixed horizon.
- Two co-primary metrics, so each is tested at a Bonferroni-adjusted
  alpha = 0.025. Guardrails stay at the full alpha on purpose —
  correcting them would only make harm harder to detect.

## Results

| Metric | Control | Treatment | Effect (95% CI) | p-value |
|---|---|---|---|---|
| Activation (primary) | 34.08% | 41.84% | **+7.76pp** [+6.82pp, +8.71pp] | < 0.001 |
| 7-day retention | 36.96% | 46.21% | +9.26pp [+8.29pp, +10.22pp] | < 0.001 |
| Support contact (guardrail) | 10.19% | 10.31% | +0.12pp [-0.47pp, +0.72pp] | 0.69 |
| Net 7-day deposits (guardrail) | $68.55 | $71.98 | +$3.44 [+$2.38, +$4.49] | < 0.001 |

**Practical significance:** the CI lower bound (+6.82pp) sits above the
+2pp MDE, so this is not just statistically significant —
it is big enough to matter. We were powered to detect
1.33pp at this sample size (well below the
observed effect).

**Precision:** CUPED using the pre-signup onboarding-engagement covariate
(outcome-covariate corr = 0.29) cut the *variance* of the
estimate by **8.2%**, which narrows the confidence interval by
**4.2%** (0.95pp → 0.91pp half-width) with no
change to the point estimate. Those are two different numbers and the CI one is
what matters for a decision — the interval is proportional to the standard error,
so it improves by roughly half the variance figure.

## Who benefits most (targeting)
Effects are **heterogeneous**. An out-of-sample uplift model (T-learner) cleanly
sorts users by benefit: the top half by predicted uplift shows
**+11.17pp observed activation lift vs
+3.21pp for the bottom half**
(~3.5x).
Segment CATEs agree: lift concentrates in **onboarding_tier=high** (activation effect
+14.92pp) and organic-channel users. Actionable read: prioritize
Recurring Buy prompts in onboarding for **highly-engaged and organic** new users; the
effect is smaller (but still positive) for paid/low-engagement cohorts.

*Caveat on the segments:* these subgroups were **not** pre-registered. Of
8 segments tested, 8 survive a Benjamini-Hochberg FDR
correction, and only those are quoted above. Treat segment sizing as a hypothesis to
confirm in the rollout, not as a measured number.

## Does *using* the feature help retention? (observational)
Beyond access, we asked whether actually **adopting** Recurring Buy raises retention.
This is not randomized (users self-select into adoption), so we used inverse-
propensity weighting. The naive adopter-vs-non-adopter gap
(+22.81pp) is inflated by selection; after
adjustment the causal estimate is **+13.19pp** (95% CI
[+11.48pp, +14.95pp]), corroborated by a
regression-adjusted estimate of +13.77pp. This assumes
no unmeasured confounding — treat as directional, confirm with an encouragement design.

## Rollout robustness (difference-in-differences)
A phased regional rollout (all treated regions switching on in the same week) gives
an independent DiD cross-check: **+2.25pp** on regional activation
(95% CI [1.28pp, 3.23pp]). Parallel-trends
pre-test p = 0.97 — no evidence of divergence before rollout, so the
design assumption holds.

**On the standard errors, because this is where DiD usually goes wrong.** Treatment
is assigned at the region level, so clustering by region is the default. With only
12 regions that default has to be checked rather than trusted, and here it fails
the check: clustering makes the SE *smaller* (0.396pp vs 0.493pp classical),
which is the opposite of the usual result and a symptom of too few clusters, not a
precision gain. So the number we lean on is a **wild cluster bootstrap
(p = 0.001, 999 Rademacher draws)**. All three approaches agree the effect
is real, so the decision is not sensitive to the choice. The CI quoted above is the
wider classical one — when two intervals disagree and the tighter one cannot be
justified, quote the wider one.

Note this is a *different estimand* from the user-level A/B test — a regional
average that includes users who never enrolled — so it is directional
corroboration, not a second measurement of the same number.

## Guardrails
- **Support contact:** +0.12pp — not statistically or
  practically significant. No support burden.
- **Net deposits:** +$3.44 per user — neutral-to-positive;
  the feature does not cannibalize deposits.

## Risks & caveats
- Effects are 7-day; recurring buy is a habit feature whose value likely compounds —
  monitor 30/90-day retention post-launch before over-crediting.
- The adoption -> retention estimate rests on unconfoundedness; it is supporting,
  not primary, evidence.
- We used a **fixed horizon**. Peeking would have inflated our false-positive rate
  from 5% to ~20%
  (10 looks); if we monitor live, switch to alpha-spending / always-valid CIs.
- The DiD cross-check has only **12 clusters**, which is too few for
  cluster-robust SEs to be taken at face value (their covariance is rank 11 for
  28 parameters here). A wild cluster bootstrap is reported for that reason.
- Segment effects are exploratory and FDR-corrected, not pre-registered.

## Recommendation
**SHIP** Recurring Buy to 100% of new users, with onboarding placement
prioritized for organic and high-engagement cohorts. Expected impact: roughly
**+7.8pp activation** (CI +6.82pp to
+8.71pp). Post-launch, track 30/90-day retention and deposits as
holdback guardrails.

---

## Appendix: why you can trust the estimator

This memo is written against **synthetic** data whose true causal effects are known,
which lets us check the machinery rather than just assert it:

| Quantity | True value | Estimate | Truth inside 95% CI? |
|---|--:|--:|:--:|
| Activation ATE | +7.69pp | +7.76pp | yes |
| Retention ITT | +9.31pp | +9.26pp | yes |
| Adoption → retention (ATT) | +13.08pp | +13.19pp | yes |
| DiD regional effect | +3.00pp | +2.25pp | yes |

A single experiment lands outside its own CI ~5% of the time, so the honest check is
coverage across many draws: over **500 simulated experiments** the mean estimate is
+7.64pp against a true +7.69pp, with **95.4% CI coverage**.

*Warehouse built via dbt. All numbers are reproducible from
`python3 src/run.py` (fixed seeds). Data is synthetic; see README.*
