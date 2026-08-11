"""
Seeded synthetic data generator for the Recurring Buy experiment.

No proprietary data is used anywhere in this project. This script generates a
fully synthetic dataset with a KNOWN ground-truth causal structure, so every
estimator downstream can be checked against the truth (an unbiasedness check).

What it produces (written to ./seeds as CSV, consumed by the dbt/SQL layer):

  raw_users.csv          one row per randomised new user (the A/B experiment)
  raw_activation.csv     7-day activation + guardrail events per user
  raw_did_panel.csv      region x week panel for the staggered-rollout DiD

Ground-truth quantities (the numbers estimators should recover) are written to
./outputs/ground_truth.json.

Design notes
------------
* Assignment is a seeded 50/50 Bernoulli draw -> SRM and covariate balance
  should pass by construction (this is what a healthy experiment looks like).
* Potential outcomes are generated with SHARED uniform draws (Y0 and Y1 use the
  same latent uniform), which is the standard way to simulate a well-defined
  individual treatment effect. Observed outcome = treated ? Y1 : Y0.
* Treatment effect on activation is HETEROGENEOUS (larger for organic-channel
  and high-onboarding users) so the uplift model has real signal to find.
* Adoption of Recurring Buy (setting up an auto-invest) is only possible in the
  treatment arm and is SELF-SELECTED (endogenous). Its effect on retention is
  the non-randomised sub-question handled by IPW.
"""

import json
import os

import numpy as np
import pandas as pd

SEED = 20240700
N_USERS = 40_000

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEEDS_DIR = os.path.join(ROOT, "seeds")
OUT_DIR = os.path.join(ROOT, "outputs")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate():
    os.makedirs(SEEDS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    n = N_USERS
    user_id = np.arange(1, n + 1)

    # ---- pre-treatment covariates (measured before assignment) --------------
    # Onboarding engagement score: minutes active in the app on signup day,
    # BEFORE the feature is assigned. This is our pre-period covariate for CUPED
    # and the main driver of heterogeneity.
    onboarding_raw = rng.gamma(shape=2.0, scale=6.0, size=n)  # skewed, >0
    onboarding_z = (onboarding_raw - onboarding_raw.mean()) / onboarding_raw.std()

    channel = rng.choice(
        ["organic", "paid", "referral"], size=n, p=[0.45, 0.40, 0.15]
    )
    country_tier = rng.choice([1, 2], size=n, p=[0.7, 0.3])
    device = rng.choice(["ios", "android"], size=n, p=[0.55, 0.45])
    age_bucket = rng.choice(
        ["18-24", "25-34", "35-49", "50+"], size=n, p=[0.22, 0.40, 0.28, 0.10]
    )

    # ---- randomised assignment (seeded 50/50) -------------------------------
    treat = rng.binomial(1, 0.5, size=n)

    # helper: map categorical -> numeric contribution
    ch_organic = (channel == "organic").astype(float)
    ch_referral = (channel == "referral").astype(float)
    c_tier2 = (country_tier == 2).astype(float)
    d_android = (device == "android").astype(float)

    # ===================== ACTIVATION (primary metric) =======================
    # Baseline log-odds of 7-day activation (funded + first crypto trade).
    base_a = (
        -0.75
        + 0.55 * onboarding_z
        + 0.30 * ch_organic
        + 0.15 * ch_referral
        - 0.20 * c_tier2
        - 0.05 * d_android
    )
    # Heterogeneous treatment effect on the log-odds scale:
    # bigger for organic + high-onboarding users.
    tau_a = 0.25 + 0.18 * onboarding_z + 0.22 * ch_organic
    p0_a = _sigmoid(base_a)
    p1_a = _sigmoid(base_a + tau_a)

    u_a = rng.uniform(size=n)  # shared latent -> individual treatment effect
    y0_a = (u_a < p0_a).astype(int)
    y1_a = (u_a < p1_a).astype(int)
    activated = np.where(treat == 1, y1_a, y0_a)

    true_ate_activation = float(np.mean(p1_a - p0_a))

    # ===================== ADOPTION (endogenous, treatment only) =============
    # Probability a treated user actually sets up a recurring buy. Driven by the
    # same covariates that drive activation -> confounding for the observational
    # adoption -> retention question.
    logit_adopt = -0.40 + 0.80 * onboarding_z + 0.35 * ch_organic - 0.15 * c_tier2
    p_adopt = _sigmoid(logit_adopt)
    u_adopt = rng.uniform(size=n)
    adopted = ((treat == 1) & (u_adopt < p_adopt)).astype(int)

    # ============ RETENTION (secondary / mechanism check + DiD-style) ========
    # 7-day retention depends on covariates, whether the user activated, and
    # (causally) whether they adopted recurring buy. Treatment also has a small
    # direct effect (access to the feature nudges habit formation).
    R_ADOPT = 0.60   # true structural log-odds effect of adoption on retention
    r_lin = -0.90 + 0.45 * onboarding_z + 0.20 * ch_organic - 0.15 * c_tier2
    r_act = 0.85          # activation strongly predicts retention
    r_direct_treat = 0.10  # small direct ITT effect of access
    u_r = rng.uniform(size=n)

    # world T=0: no adoption possible, activation = y0_a
    logit_r0 = r_lin + r_act * y0_a + R_ADOPT * 0.0 + r_direct_treat * 0.0
    p_r0 = _sigmoid(logit_r0)
    ret0 = (u_r < p_r0).astype(int)

    # world T=1: adoption per propensity (use the realised u_adopt draw), act = y1_a
    adopt_if_treated = (u_adopt < p_adopt).astype(int)
    logit_r1 = r_lin + r_act * y1_a + R_ADOPT * adopt_if_treated + r_direct_treat * 1.0
    p_r1 = _sigmoid(logit_r1)
    ret1 = (u_r < p_r1).astype(int)

    retained = np.where(treat == 1, ret1, ret0)
    true_ate_retention = float(np.mean(p_r1 - p_r0))  # true ITT effect of access

    # True effect of ADOPTION on retention, among adopters (the ATT the IPW
    # analysis targets): compare retention with adoption on vs off, holding the
    # user's own (treated-world) activation and covariates fixed.
    is_adopter = adopted == 1
    logit_adopt_on = r_lin + r_act * y1_a + R_ADOPT * 1.0 + r_direct_treat * 1.0
    logit_adopt_off = r_lin + r_act * y1_a + R_ADOPT * 0.0 + r_direct_treat * 1.0
    true_adoption_att = float(
        np.mean(_sigmoid(logit_adopt_on[is_adopter]) - _sigmoid(logit_adopt_off[is_adopter]))
    )

    # ===================== GUARDRAILS ========================================
    # Support-contact rate: designed to be ~neutral (a real guardrail we must
    # not harm). Small, non-material true effect.
    logit_sup0 = -2.2 + 0.10 * onboarding_z + 0.05 * c_tier2
    logit_sup1 = logit_sup0 + 0.03  # tiny, immaterial
    u_s = rng.uniform(size=n)
    sup0 = (u_s < _sigmoid(logit_sup0)).astype(int)
    sup1 = (u_s < _sigmoid(logit_sup1)).astype(int)
    support_contact = np.where(treat == 1, sup1, sup0)
    true_ate_support = float(np.mean(_sigmoid(logit_sup1) - _sigmoid(logit_sup0)))

    # 7-day net deposits ($). Guardrail: recurring buy should not cannibalise
    # deposits. True effect: small positive (auto-invest nudges deposits).
    dep_mu0 = 60 + 25 * onboarding_z + 10 * ch_organic
    dep_treat_effect = 4.0  # small positive true effect ($)
    dep_noise = rng.normal(0, 55, size=n)
    deposits0 = np.clip(dep_mu0 + dep_noise, 0, None)
    deposits1 = np.clip(dep_mu0 + dep_treat_effect + dep_noise, 0, None)
    deposits = np.where(treat == 1, deposits1, deposits0)
    true_ate_deposits = float(np.mean(deposits1 - deposits0))

    # ---- assemble raw tables ------------------------------------------------
    signup_day = rng.integers(0, 28, size=n)  # day within a 4-week enrolment
    users = pd.DataFrame(
        {
            "user_id": user_id,
            "signup_day": signup_day,
            "assignment": np.where(treat == 1, "treatment", "control"),
            "channel": channel,
            "country_tier": country_tier,
            "device": device,
            "age_bucket": age_bucket,
            "onboarding_minutes": np.round(onboarding_raw, 2),
        }
    )
    activation = pd.DataFrame(
        {
            "user_id": user_id,
            "activated_7d": activated,
            "adopted_recurring_buy": adopted,
            "retained_7d": retained,
            "support_contact_7d": support_contact,
            "net_deposits_7d": np.round(deposits, 2),
        }
    )

    users.to_csv(os.path.join(SEEDS_DIR, "raw_users.csv"), index=False)
    activation.to_csv(os.path.join(SEEDS_DIR, "raw_activation.csv"), index=False)

    # ===================== DiD staggered-rollout panel =======================
    did = _generate_did_panel(rng)
    did.to_csv(os.path.join(SEEDS_DIR, "raw_did_panel.csv"), index=False)

    ground_truth = {
        "seed": SEED,
        "n_users": int(n),
        "true_ate_activation": true_ate_activation,
        "true_ate_retention_itt": true_ate_retention,
        "true_ate_support_contact": true_ate_support,
        "true_ate_net_deposits": true_ate_deposits,
        "true_adoption_effect_on_retention_att": true_adoption_att,
        "true_did_effect": did.attrs.get("true_effect"),
        "notes": (
            "Synthetic data with known structure. ATE = mean(p1 - p0) over the "
            "population using generative probabilities. Estimators should recover "
            "these within confidence intervals."
        ),
    }
    with open(os.path.join(OUT_DIR, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


def simulate_activation(seed, n=8000):
    """Compact re-draw of the ACTIVATION data-generating process only.

    Used by the Monte-Carlo coverage/unbiasedness check: it reuses the exact same
    structural coefficients as generate(), so repeated draws let us verify the
    estimator is unbiased and its 95% CI has ~95% coverage. Returns
    (true_ate, activated, treat).
    """
    rng = np.random.default_rng(seed)
    onboarding_raw = rng.gamma(shape=2.0, scale=6.0, size=n)
    onboarding_z = (onboarding_raw - onboarding_raw.mean()) / onboarding_raw.std()
    channel = rng.choice(["organic", "paid", "referral"], size=n, p=[0.45, 0.40, 0.15])
    country_tier = rng.choice([1, 2], size=n, p=[0.7, 0.3])
    device = rng.choice(["ios", "android"], size=n, p=[0.55, 0.45])
    treat = rng.binomial(1, 0.5, size=n)
    ch_organic = (channel == "organic").astype(float)
    ch_referral = (channel == "referral").astype(float)
    c_tier2 = (country_tier == 2).astype(float)
    d_android = (device == "android").astype(float)
    base_a = (-0.75 + 0.55 * onboarding_z + 0.30 * ch_organic + 0.15 * ch_referral
              - 0.20 * c_tier2 - 0.05 * d_android)
    tau_a = 0.25 + 0.18 * onboarding_z + 0.22 * ch_organic
    p0_a = _sigmoid(base_a)
    p1_a = _sigmoid(base_a + tau_a)
    u_a = rng.uniform(size=n)
    y0_a = (u_a < p0_a).astype(int)
    y1_a = (u_a < p1_a).astype(int)
    activated = np.where(treat == 1, y1_a, y0_a)
    true_ate = float(np.mean(p1_a - p0_a))
    return true_ate, activated, treat


def _generate_did_panel(rng):
    """Region x week panel for a staggered (here: common-timing) rollout DiD.

    Treated regions switch the feature on at ROLLOUT_WEEK. Pre-period trends are
    parallel by construction (a shared week trend + region fixed effects); the
    only divergence is the post-rollout treatment effect, so a parallel-trends
    check should pass.
    """
    n_regions = 12
    n_weeks = 16
    rollout_week = 9
    true_effect = 0.030  # +3.0 pp on regional weekly activation rate

    regions = np.arange(1, n_regions + 1)
    treated_regions = set(regions[: n_regions // 2])  # first half are treated

    region_fe = rng.normal(0, 0.03, size=n_regions)  # baseline level per region
    week_trend = np.linspace(0, 0.04, n_weeks)  # common secular uptrend

    rows = []
    for ri, r in enumerate(regions):
        treated = 1 if r in treated_regions else 0
        for w in range(1, n_weeks + 1):
            post = 1 if w >= rollout_week else 0
            # true weekly activation probability for this region-week
            p = 0.30 + region_fe[ri] + week_trend[w - 1] + true_effect * treated * post
            p = float(np.clip(p, 0.01, 0.99))
            # realistic sampling: rate = successes / cohort size (Binomial noise),
            # so the parallel-trends test has honest, non-degenerate SEs.
            n_users = int(rng.integers(600, 900))
            successes = rng.binomial(n_users, p)
            rate = successes / n_users
            rows.append(
                {
                    "region_id": int(r),
                    "week": int(w),
                    "treated_region": treated,
                    "post": post,
                    "rollout_week": rollout_week,
                    "n_users": n_users,
                    "activation_rate": round(float(rate), 5),
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["true_effect"] = true_effect
    return df


if __name__ == "__main__":
    gt = generate()
    print("Generated synthetic data. Ground truth:")
    print(json.dumps(gt, indent=2))
