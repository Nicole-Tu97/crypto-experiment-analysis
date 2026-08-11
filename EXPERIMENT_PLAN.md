# Experiment plan — Recurring Buy for new crypto users

**Status:** pre-registered. Written and agreed *before* outcome data was looked at.
**Owner:** Product Data Science · **Partners:** Crypto Growth PM, Eng, Design, Support

This document exists so that the analysis cannot be reverse-engineered into a
favourable story after the fact. Everything that determines the decision — the
primary metric, the alpha, the minimum effect worth shipping, the guardrails, the
stopping rule — is fixed here. [`DECISION_MEMO.md`](DECISION_MEMO.md) is scored
against this plan, not against whatever the data turned out to look like.

---

## 1. The decision this experiment serves

Recurring Buy lets a new client schedule an automatic recurring crypto purchase.
The hypothesis is a behavioural one, not a feature-preference one:

> New crypto clients stall at the gap between *opening* an account and *making a
> first trade*, because a first trade requires an active decision about timing —
> which is exactly the decision a nervous first-time buyer is worst at making.
> Removing the timing decision should convert more of them, and should build a
> habit that survives the first week.

The decision on the table: **do we put Recurring Buy in the new-client onboarding
flow for everyone, or not?** Not "is it a nice feature" — whether it earns the
onboarding real estate.

If it works, the mechanism predicts two things we can check: the lift should be
larger for clients who showed more engagement before signing up (they have intent
but lack conviction on timing), and it should show up in retention, not just in a
one-off trade.

## 2. Population and unit

- **Unit of randomization:** client (`user_id`). Not session, not device — the
  outcome is a 7-day behaviour, so the unit has to persist across sessions.
- **Population:** new clients completing signup during the 4-week enrolment window.
- **Exclusions:** none. Excluding low-intent clients would inflate the effect and
  answer a question we are not asking.
- **Assignment:** 50/50 Bernoulli. Even split is optimal for power when both arms
  cost the same to serve, and they do — this is a UI surface, not a paid incentive.

## 3. Metrics

| Role | Metric | Definition | Direction |
|---|---|---|---|
| **Co-primary** | 7-day activation | funded account **and** ≥1 crypto trade within 7 days of signup | ↑ |
| **Co-primary** | 7-day retention | active in the app on day 7 | ↑ |
| **Guardrail** | Support-contact rate | ≥1 support contact within 7 days | must not ↑ |
| **Guardrail** | Net 7-day deposits | net CAD deposited within 7 days | must not ↓ |

**All four gate the decision.** Every metric listed here appears as a condition in §5
and is evaluated in one place, `analysis.decision_rule()`. Nothing is measured for
decoration: if a metric does not carry a decision, it does not belong in this table.

Why activation and not "recurring buys created": adoption of the feature is not
the point. A feature that 40% of clients set up and that moves no one's behaviour
is a failure that an adoption metric would score as a triumph. Activation is the
thing the business actually needs to move.

Why retention is **co-primary rather than primary**: activation is the near-term
decision driver and is measurable in the window; retention is what tells you whether
the activation was real or just a nudged one-off. It is the **mechanism check** — if
activation rises and retention does not, the feature bought a one-off trade rather
than a habit, and that does not earn the slot. Both have to move in the same
direction for the story to hold. Because there are two co-primary metrics, each is
tested at a Bonferroni-adjusted **α = 0.025** — two chances to declare a win is two
chances to be wrong.

Retention is held to a **deliberately looser bar than activation**: significance and a
matching sign, but *not* the +2pp MDE. The MDE exists to justify the onboarding slot
and activation already carries that test; charging retention a second time would be
double-counting. Retention answers *did a habit form*, not *is it big enough*.

Guardrails are deliberately tested at the **full α = 0.05**, uncorrected. A
multiplicity correction makes it *harder* to reject the null, which for a metric
we are trying not to move means making harm harder to detect. Correcting
guardrails is a way of not finding problems.

**Support contact is the guardrail that matters most.** An auto-invest feature
that confuses people generates "why did you take my money" tickets, and that cost
lands on a team that is not in this room. Deposits are the cannibalisation check:
money moved into a recurring schedule is money not deposited some other way.

## 4. Design parameters

| Parameter | Value | Reasoning |
|---|---|---|
| α | 0.05 two-sided (**0.025 per co-primary**, full 0.05 on guardrails) | two-sided because a well-intentioned onboarding change can plausibly hurt; Bonferroni across the two co-primaries; guardrails uncorrected on purpose (see §3) |
| Power | 0.80 | standard |
| **MDE** | **+2.0pp absolute** on activation | below this the effect does not justify the onboarding slot it displaces |
| Horizon | **Fixed.** Analyse once, at the pre-set sample size | see §6 |
| Variance reduction | CUPED on pre-signup onboarding engagement | declared in advance so it cannot be a post-hoc rescue |

The MDE is a **product judgment, not a statistical one**: onboarding space is
scarce and something else gets displaced, so a +0.5pp true effect is a loss even
if it is real and significant. Deciding this before seeing data is the entire
point — afterwards, every observed effect looks like it clears the bar.

Sample size follows from the MDE and an assumed ~34% baseline activation rate:
≈8,800 clients per arm. The enrolment window supplies ~20,000 per arm, so the
design is over-powered — powered to ~1.3pp. That is intentional: the surplus
buys credible **subgroup** and guardrail reads, which are the parts of this
analysis that actually inform *how* to ship rather than *whether* to.

## 5. Decision rule (committed in advance)

**SHIP** to 100% of new clients if **all five** hold:

1. **Activation** significant at α = 0.025; **and**
2. the **lower bound** of activation's 95% CI clears the +2pp MDE; **and**
3. **Retention** significant at α = 0.025 **and moving in the same direction** as
   activation — significance and sign only, no MDE (see §3); **and**
4. **Support-contact** guardrail not harmed: its CI **upper** bound stays below +1pp; **and**
5. **Net deposits** guardrail not harmed: no significant fall at the full α = 0.05.

All five are evaluated in one place — `analysis.decision_rule()`. The memo, the README
results block and `tests/test_reports_in_sync.py` read or independently recompute that
single definition rather than restating it, so no copy of the rule can drift from
another.

Condition 2 is the one that does real work on *magnitude*. "Significant" only rules
out zero; the CI lower bound is what tells us the effect is *large enough to be worth
it*, and a significant +0.4pp would fail this rule on purpose. Condition 3 does the
same job for *mechanism*.

Note the asymmetry in conditions 4 and 5: guardrails are judged by the side of the
interval that would reveal harm, not by the point estimate. "No measured change" is
not the same as "no change".

Otherwise **ITERATE** — and if activation clears the bar but a guardrail is
harmed, that is a hold pending a fix, not a ship with a footnote.

## 6. Why a fixed horizon

Continuous monitoring with a naive fixed-α test at ten interim looks inflates the
false-positive rate from 5% to roughly 20% — simulated explicitly in
`src/analysis.py::run_sequential`. Every look is another chance for noise to cross
the line, and the first crossing is the one that gets screenshotted into Slack.

So: **no interim decisions.** If the team needs live monitoring (a legitimate
need — we want to catch a broken flow on day 2), it runs against an
alpha-spending boundary or always-valid confidence sequences, and stopping *early
for harm* is allowed while stopping early for success is not. Asymmetric by
design, because the costs are asymmetric.

## 7. Known threats, and what we do about them

| Threat | Why it matters here | Mitigation |
|---|---|---|
| **Sample-ratio mismatch** | silent assignment or logging bugs invalidate everything downstream | SRM χ² before any outcome is read; p < 0.001 halts the analysis |
| **Novelty effect** | a new onboarding card gets clicked because it is new | 7-day window is short enough to be vulnerable — flagged as a limitation; 30/90-day post-launch monitoring is part of the ship plan |
| **Short window** | habit features compound; 7 days undercounts the true effect | treat as a **lower** bound, do not extrapolate |
| **Peeking** | inflates Type I error to ~20% | fixed horizon (§6) |
| **Subgroup fishing** | 8 segments at α=0.05 ≈ 34% chance of a false positive | segments are exploratory and Benjamini-Hochberg corrected; they inform targeting, never the ship decision |
| **Adoption ≠ assignment** | adopters self-select on the same traits that drive activation | ITT is the primary estimand; the adoption question is answered separately by IPW and labelled non-randomized |

**The one we cannot fix by analysis:** novelty. A 7-day window cannot distinguish
a durable habit from a new-thing effect, and no amount of statistics inside this
experiment will separate them. That is a monitoring commitment after launch, not a
claim we can make now — which is why the memo's recommendation includes a holdback
rather than declaring victory.

## 8. Estimands, stated precisely

- **Primary (ITT):** effect of *access* to Recurring Buy on activation. This is
  what a ship decision needs, because shipping grants access, not usage.
- **Secondary (ATT, non-randomized):** effect of *adopting* Recurring Buy on
  retention among adopters. Identified only under conditional ignorability,
  positivity and SUTVA. Supporting evidence, never the basis for the decision —
  the honest next step is an encouragement design, which randomizes a *nudge* and
  gives a defensible instrument.
- **Cross-check (DiD):** effect of a phased regional rollout on regional
  activation. A **different estimand** — a regional average including clients who
  never enrolled — so agreement is corroboration, not replication.

## 9. Analysis, fixed in advance

1. Integrity: group sizes, SRM χ², covariate balance (standardized mean differences).
2. Primary and guardrails: two-proportion z-tests, Wald CIs, bootstrap robustness check.
3. CUPED on the declared pre-period covariate. **Variance reduction and CI-width
   reduction reported separately** — the CI improves by roughly half the variance
   figure, and conflating them overstates precision.
4. Heterogeneity: segment CATEs (BH-corrected) plus an out-of-sample T-learner
   uplift model, validated by top-vs-bottom-half observed lift and a Qini curve.
5. Observational: IPW for adoption → retention, with an overlap check and a
   regression-adjusted cross-check.
6. DiD: TWFE with region and week fixed effects, a parallel-trends pre-test and an
   event study. With only 12 clusters, cluster-robust SEs are reported *and
   stress-tested with a wild cluster bootstrap* rather than trusted.

Any deviation from this list gets recorded in the memo as a deviation.

## 10. What would change our mind

Stated up front so it cannot be rationalised later:

- Activation up but **retention flat or down** → the feature nudges a one-off
  trade without building a habit. Do not ship to onboarding; the trade is not
  worth the slot.
- Activation up but **support contacts up materially** → likely a comprehension
  problem in the flow. Fix the flow, re-test.
- Activation up but **deposits down** → cannibalisation, not growth. Investigate
  before shipping.
- Effect concentrated **entirely** in one small segment → ship to that segment,
  not to everyone, and do not quote the pooled average as the expected impact.

---

## 11. Known gaps in this plan (and what I would change)

Written down for the same reason as §10 — a limitation you name is worth more than one
a reader finds.

**1. Retention is defined too loosely for a crypto feature.** "Active in the app on
day 7" counts a client who opened the app to check a managed portfolio and never
touched crypto. The mechanism claim is about a *crypto* habit, so the definition
should be **active in the crypto surface** — a crypto view, an order, or a scheduled
buy executing. The trade-off is baseline: the more crypto-specific the definition, the
lower the base rate and the more sample the same MDE needs. "Any crypto activity" is
the version I would take; "a second trade within 7 days" is too sparse to power.

**2. Model logic is not unit-tested.** The 15 dbt data tests assert things about the
*data*; `tests/test_warehouse_parity.py` catches logic errors only insofar as the dbt
and plain-SQL paths disagree. dbt 1.8+ unit tests would assert the transformation
logic directly against fixture rows.

**3. The 7-day window cannot separate a habit from novelty.** No amount of analysis
inside this experiment fixes that — it is a post-launch monitoring commitment (30/90-day
retention against a holdback), which is why the memo recommends one rather than
declaring the habit proven.

**4. Reject inference is absent from the adoption analysis.** The IPW estimate for
adoption → retention conditions on the covariates we observe. An encouragement design —
randomising a *nudge* to adopt rather than adoption itself — would give a defensible
instrument and replace an untestable assumption with a testable one.
