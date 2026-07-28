"""
The full experiment + causal-inference analysis.

Each function takes the analysis-ready marts (built by dbt/DuckDB), computes one
block of the analysis, writes its figure(s), and returns a results dict. run.py
assembles everything into metrics.json and the decision memo.

Pre-registration (declared BEFORE looking at outcomes):
  * PRIMARY metric ........ 7-day activation (funded + first crypto trade)
  * CO-PRIMARY ............ 7-day retention
  * GUARDRAILS ............ support-contact rate (must not rise),
                            net 7-day deposits (must not fall)
  * alpha ................. 0.05 (two-sided), fixed horizon
  * power ................. 0.80
  * MDE (declared) ........ 2.0 pp absolute on activation
  * Decision rule ......... SHIP if activation effect is significant AND its CI
                            lower bound clears the +2pp MDE AND no guardrail is
                            significantly harmed.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import patsy
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

import experiment_stats as es
import generate_data
import viz
from viz import plt

ALPHA = 0.05
POWER = 0.80
MDE_DECLARED = 0.02  # 2 pp absolute on activation
SEED = 7

FIG_DIR = None  # set by run.py


def _feature_matrix(df):
    """Pre-treatment covariates -> numeric design matrix (for models)."""
    X = pd.DataFrame(index=df.index)
    X["onboarding_z"] = (df["onboarding_minutes"] - df["onboarding_minutes"].mean()) \
        / df["onboarding_minutes"].std()
    X["organic"] = (df["channel"] == "organic").astype(int)
    X["referral"] = (df["channel"] == "referral").astype(int)
    X["country_tier2"] = (df["country_tier"] == 2).astype(int)
    X["android"] = (df["device"] == "android").astype(int)
    for a in ["25-34", "35-49", "50+"]:
        X[f"age_{a}"] = (df["age_bucket"] == a).astype(int)
    return X


# ======================================================================== #
# 1. Experiment integrity: group sizes, SRM, covariate balance
# ======================================================================== #
def run_integrity(users):
    ctrl = users[users.is_treatment == 0]
    treat = users[users.is_treatment == 1]

    srm = es.srm_test(len(ctrl), len(treat), expected_ratio=0.5)

    # covariate balance via standardized mean differences
    balance = {}
    balance["onboarding_minutes"] = es.standardized_mean_diff(
        ctrl.onboarding_minutes, treat.onboarding_minutes)
    balance["country_tier2"] = es.standardized_mean_diff(
        (ctrl.country_tier == 2), (treat.country_tier == 2))
    balance["android"] = es.standardized_mean_diff(
        (ctrl.device == "android"), (treat.device == "android"))
    for ch in ["organic", "paid", "referral"]:
        balance[f"channel_{ch}"] = es.standardized_mean_diff(
            (ctrl.channel == ch), (treat.channel == ch))
    for a in ["18-24", "25-34", "35-49", "50+"]:
        balance[f"age_{a}"] = es.standardized_mean_diff(
            (ctrl.age_bucket == a), (treat.age_bucket == a))

    max_abs_smd = max(abs(v) for v in balance.values())

    # ---- figure: love plot of |SMD| ----
    viz.set_style()
    names = list(balance.keys())
    vals = [balance[k] for k in names]
    order = np.argsort([abs(v) for v in vals])
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = [viz.RED if abs(v) >= 0.1 else viz.BLUE for v in vals]
    ax.scatter(vals, range(len(vals)), color=colors, s=55, zorder=3)
    ax.axvline(0, color=viz.INK2, lw=0.8)
    ax.axvline(0.1, color=viz.RED, lw=0.9, ls="--")
    ax.axvline(-0.1, color=viz.RED, lw=0.9, ls="--")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Standardized mean difference (treatment - control)")
    ax.set_title("Covariate balance: all |SMD| < 0.1")
    ax.set_xlim(-0.15, 0.15)
    ax.grid(axis="y", visible=False)
    viz.savefig(fig, os.path.join(FIG_DIR, "fig_covariate_balance.png"))

    return {
        "group_sizes": {"control": len(ctrl), "treatment": len(treat)},
        "srm": srm,
        "covariate_balance_smd": {k: float(v) for k, v in balance.items()},
        "max_abs_smd": float(max_abs_smd),
        "balanced": bool(max_abs_smd < 0.1),
    }


# ======================================================================== #
# 2. Primary + guardrail metrics, ATE with CI, power/MDE
# ======================================================================== #
def run_primary_and_guardrails(users):
    ctrl = users[users.is_treatment == 0]
    treat = users[users.is_treatment == 1]
    n_per_arm = min(len(ctrl), len(treat))

    activation = es.two_proportion_diff(ctrl.activated_7d, treat.activated_7d, ALPHA)
    retention = es.two_proportion_diff(ctrl.retained_7d, treat.retained_7d, ALPHA)
    support = es.two_proportion_diff(ctrl.support_contact_7d, treat.support_contact_7d, ALPHA)
    deposits = es.mean_diff(ctrl.net_deposits_7d, treat.net_deposits_7d, ALPHA)

    # power / MDE for the primary metric, using the observed control rate
    base = activation.control_mean
    mde_achieved = es.mde_two_proportion(base, n_per_arm, ALPHA, POWER)
    power_declared = es.power_two_proportion(
        base, base + MDE_DECLARED, n_per_arm, ALPHA)
    req_n = es.required_n_two_proportion(base, MDE_DECLARED, ALPHA, POWER)

    # bootstrap CI as a robustness check on the primary metric
    b_lo, b_hi, b_mean = es.bootstrap_diff_ci(
        ctrl.activated_7d.values, treat.activated_7d.values, seed=SEED)

    # practical significance: does the CI lower bound clear the declared MDE?
    practically_significant = bool(activation.ci_low >= MDE_DECLARED)

    # Two co-primary metrics means two chances to declare a win, so the family-
    # wise error rate is controlled with Bonferroni (alpha/2 each). Guardrails
    # are deliberately excluded: for a metric we are trying NOT to move, a
    # multiplicity correction only makes harm harder to detect.
    alpha_coprimary = ALPHA / 2
    coprimary = {
        "correction": "Bonferroni over 2 co-primary metrics",
        "alpha_per_metric": alpha_coprimary,
        "activation_significant": bool(activation.p_value < alpha_coprimary),
        "retention_significant": bool(retention.p_value < alpha_coprimary),
        "note": ("Guardrails are tested at the full alpha on purpose: "
                 "correcting them would raise the bar for detecting harm."),
    }

    # ---- figure: forest plot (absolute effects with 95% CI) ----
    viz.set_style()
    rows = [
        ("Activation (primary)", activation, "pp", viz.BLUE),
        ("7-day retention", retention, "pp", viz.BLUE),
        ("Support contact (guardrail)", support, "pp", viz.GRAY),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ys = range(len(rows))
    for i, (label, r, _, color) in enumerate(rows):
        ax.errorbar(r.absolute_effect * 100, i,
                    xerr=[[(r.absolute_effect - r.ci_low) * 100],
                          [(r.ci_high - r.absolute_effect) * 100]],
                    fmt="o", color=color, ecolor=color, elinewidth=2,
                    capsize=4, markersize=8, zorder=3)
    ax.axvline(0, color=viz.INK2, lw=0.9)
    ax.axvline(MDE_DECLARED * 100, color=viz.AQUA, lw=1.1, ls="--",
               label=f"declared MDE (+{MDE_DECLARED*100:.0f}pp)")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Absolute effect (percentage points), 95% CI")
    ax.set_title("Treatment effects: primary, co-primary, guardrail")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    viz.savefig(fig, os.path.join(FIG_DIR, "fig_primary_and_guardrails.png"))

    return {
        "config": {"alpha": ALPHA, "power": POWER, "mde_declared": MDE_DECLARED,
                   "n_per_arm": int(n_per_arm)},
        "activation": activation.as_dict(),
        "retention": retention.as_dict(),
        "guardrail_support_contact": support.as_dict(),
        "guardrail_net_deposits": deposits.as_dict(),
        "power_mde": {
            "control_rate": float(base),
            "mde_achieved_at_80pct_power": float(mde_achieved),
            "power_to_detect_declared_mde": float(power_declared),
            "required_n_per_arm_for_declared_mde": int(req_n),
        },
        "activation_bootstrap_ci": {"lo": b_lo, "hi": b_hi, "mean": b_mean},
        "practically_significant_vs_mde": practically_significant,
        "coprimary_multiplicity": coprimary,
    }


# ======================================================================== #
# 3. CUPED variance reduction on the primary metric
# ======================================================================== #
def run_cuped(users):
    ctrl = users[users.is_treatment == 0]
    treat = users[users.is_treatment == 1]

    unadj = es.two_proportion_diff(ctrl.activated_7d, treat.activated_7d, ALPHA)

    # CUPED adjustment using the pre-experiment covariate (onboarding minutes)
    y = users.activated_7d.values.astype(float)
    x = users.onboarding_minutes.values.astype(float)
    y_cuped, theta = es.cuped_adjust(y, x)
    users = users.assign(_y_cuped=y_cuped)
    c_adj = users[users.is_treatment == 0]._y_cuped.values
    t_adj = users[users.is_treatment == 1]._y_cuped.values
    adj = es.mean_diff(c_adj, t_adj, ALPHA)

    rho = np.corrcoef(y, x)[0, 1]
    # Two DIFFERENT numbers that are easy to conflate: variance falls by
    # 1 - (se_new/se_old)^2, but the CI is proportional to the SE so it narrows
    # by only 1 - se_new/se_old. Report both, and never quote one as the other.
    var_reduction, ci_reduction = es.variance_and_ci_reduction(
        unadj.se, unadj.ci_halfwidth, adj.se, adj.ci_halfwidth)

    # ---- figure: CI before vs after CUPED ----
    viz.set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    labels = ["Unadjusted", "CUPED-adjusted"]
    ests = [unadj.absolute_effect * 100, adj.absolute_effect * 100]
    los = [unadj.ci_low * 100, adj.ci_low * 100]
    his = [unadj.ci_high * 100, adj.ci_high * 100]
    colors = [viz.GRAY, viz.ORANGE]
    for i, lab in enumerate(labels):
        ax.errorbar(ests[i], i,
                    xerr=[[ests[i] - los[i]], [his[i] - ests[i]]],
                    fmt="o", color=colors[i], ecolor=colors[i], elinewidth=2.5,
                    capsize=5, markersize=9, zorder=3)
    ax.axvline(0, color=viz.INK2, lw=0.9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Activation effect (percentage points), 95% CI")
    ax.set_title(f"CUPED: variance -{var_reduction*100:.1f}%, "
                 f"CI width -{ci_reduction*100:.1f}% (rho={rho:.2f})")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    viz.savefig(fig, os.path.join(FIG_DIR, "fig_cuped.png"))

    return {
        "theta": theta,
        "corr_outcome_covariate": float(rho),
        "unadjusted": unadj.as_dict(),
        "cuped_adjusted": adj.as_dict(),
        "se_unadjusted": float(unadj.se),
        "se_cuped": float(adj.se),
        "variance_reduction_pct": float(var_reduction * 100),
        "ci_width_reduction_pct": float(ci_reduction * 100),
        "ci_halfwidth_unadjusted_pp": float(unadj.ci_halfwidth * 100),
        "ci_halfwidth_cuped_pp": float(adj.ci_halfwidth * 100),
    }


# ======================================================================== #
# 4. Heterogeneous treatment effects: segment CATE + uplift T-learner
# ======================================================================== #
def run_heterogeneity(users):
    # ---- (a) segment-level CATE with CIs ----
    # These subgroups are EXPLORATORY, not pre-registered: 8 tests at alpha=0.05
    # carry a ~34% chance of at least one false positive under the null, so raw
    # p-values are reported alongside BH-adjusted q-values and the decision memo
    # only leans on segments that survive the adjustment.
    segments = {}
    for dim in ["channel", "onboarding_tier", "country_tier"]:
        for level, sub in users.groupby(dim):
            c = sub[sub.is_treatment == 0].activated_7d
            t = sub[sub.is_treatment == 1].activated_7d
            if len(c) < 50 or len(t) < 50:
                continue
            r = es.two_proportion_diff(c, t, ALPHA)
            segments[f"{dim}={level}"] = {
                "effect": r.absolute_effect, "ci_low": r.ci_low,
                "ci_high": r.ci_high, "p_value": r.p_value, "n": int(len(sub)),
            }

    seg_keys = list(segments)
    q_vals, seg_rejected = es.benjamini_hochberg(
        [segments[k]["p_value"] for k in seg_keys], alpha=ALPHA)
    for k, q, rej in zip(seg_keys, q_vals, seg_rejected):
        segments[k]["q_value_bh"] = float(q)
        segments[k]["significant_after_bh"] = bool(rej)

    # ---- (b) uplift T-learner (two models) evaluated out-of-sample ----
    X = _feature_matrix(users).values
    w = users.is_treatment.values
    y = users.activated_7d.values

    X_tr, X_te, w_tr, w_te, y_tr, y_te = train_test_split(
        X, w, y, test_size=0.35, random_state=SEED, stratify=w)

    m1 = GradientBoostingClassifier(random_state=SEED, max_depth=3,
                                    n_estimators=200, learning_rate=0.05)
    m0 = GradientBoostingClassifier(random_state=SEED, max_depth=3,
                                    n_estimators=200, learning_rate=0.05)
    m1.fit(X_tr[w_tr == 1], y_tr[w_tr == 1])
    m0.fit(X_tr[w_tr == 0], y_tr[w_tr == 0])
    uplift_te = m1.predict_proba(X_te)[:, 1] - m0.predict_proba(X_te)[:, 1]

    mean_pred_uplift = float(uplift_te.mean())

    # validate the model actually sorts users: observed uplift, top vs bottom half
    med = np.median(uplift_te)
    top = uplift_te >= med
    bot = ~top

    def _obs_uplift(mask):
        yy, ww = y_te[mask], w_te[mask]
        if (ww == 1).sum() < 20 or (ww == 0).sum() < 20:
            return np.nan
        return yy[ww == 1].mean() - yy[ww == 0].mean()

    obs_top = _obs_uplift(top)
    obs_bot = _obs_uplift(bot)

    # Qini curve on the test set
    qx, qy, qini_area, qini_coef = _qini_curve(y_te, w_te, uplift_te)

    # ---- figures ----
    viz.set_style()
    # segment CATE forest
    seg_items = sorted(segments.items(), key=lambda kv: kv[1]["effect"])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, (label, d) in enumerate(seg_items):
        col = viz.BLUE if d["significant_after_bh"] else viz.GRAY
        ax.errorbar(d["effect"] * 100, i,
                    xerr=[[(d["effect"] - d["ci_low"]) * 100],
                          [(d["ci_high"] - d["effect"]) * 100]],
                    fmt="o", color=col, ecolor=col, elinewidth=2,
                    capsize=4, markersize=7, zorder=3)
    ax.axvline(0, color=viz.INK2, lw=0.9)
    ax.set_yticks(range(len(seg_items)))
    ax.set_yticklabels([s[0] for s in seg_items])
    ax.set_xlabel("Activation effect (pp), 95% CI")
    ax.set_title("Exploratory segment CATEs (blue = BH-significant, q < 0.05)")
    ax.grid(axis="y", visible=False)
    viz.savefig(fig, os.path.join(FIG_DIR, "fig_uplift_segments.png"))

    # Qini curve
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(qx, qy, color=viz.BLUE, lw=2, label="Uplift model (T-learner)")
    ax.plot([0, 1], [0, qy[-1]], color=viz.GRAY, lw=1.6, ls="--",
            label="Random targeting")
    ax.set_xlabel("Fraction of users targeted (by predicted uplift)")
    ax.set_ylabel("Incremental activations (Qini)")
    ax.set_title(f"Uplift model Qini curve (normalized coef = {qini_coef:.3f})")
    ax.legend(loc="lower right")
    ax.grid(axis="both")
    viz.savefig(fig, os.path.join(FIG_DIR, "fig_uplift_qini.png"))

    def _cast(kk, vv):
        if kk == "n":
            return int(vv)
        if isinstance(vv, (bool, np.bool_)):
            return bool(vv)
        return float(vv)

    return {
        "segment_cate": {k: {kk: _cast(kk, vv) for kk, vv in v.items()}
                         for k, v in segments.items()},
        "segment_cate_note": (
            "Exploratory subgroups (not pre-registered). q_value_bh is the "
            "Benjamini-Hochberg FDR-adjusted p-value across all "
            f"{len(segments)} segments tested."
        ),
        "uplift_model": {
            "type": "T-learner (GradientBoosting), evaluated out-of-sample",
            "mean_predicted_uplift": mean_pred_uplift,
            "observed_uplift_top_half": None if np.isnan(obs_top) else float(obs_top),
            "observed_uplift_bottom_half": None if np.isnan(obs_bot) else float(obs_bot),
            "qini_area_incremental_activations": float(qini_area),
            "qini_coefficient_normalized": float(qini_coef),
            "qini_coefficient_note": (
                "Area between the Qini curve and the random-targeting line, "
                "divided by the total incremental activations at 100% targeting. "
                "0 = no better than random; 0.5 = perfect sorting."
            ),
        },
    }


def _qini_curve(y, w, uplift):
    """Qini curve, plus the area above random targeting on two scales.

    Returns (x, qini, area, coef_normalized):
      * `qini[i]`  incremental activations from targeting the top (i+1) users by
                   predicted uplift, control outcomes rescaled to the treated count.
      * `area`     area between the Qini curve and the random-targeting line,
                   in units of incremental activations.
      * `coef_normalized`  that area divided by total incremental activations at
                   100% targeting -> scale-free and comparable across sample
                   sizes. 0 means no better than random; 0.5 is perfect sorting.

    The normalization matters: dividing the area by n instead (a common shortcut)
    produces a number that shrinks as the test set grows, so it cannot be
    compared between runs or against another model.
    """
    order = np.argsort(-uplift)
    y, w = y[order], w[order]
    cum_t = np.cumsum(w)
    cum_c = np.cumsum(1 - w)
    cum_yt = np.cumsum(y * w)
    cum_yc = np.cumsum(y * (1 - w))
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(cum_c > 0, cum_yc * (cum_t / cum_c), 0.0)
    qini = np.nan_to_num(cum_yt - scaled)
    n = len(y)
    x = np.arange(1, n + 1) / n
    rand = np.linspace(0, qini[-1], n)
    area = _trapezoid(qini - rand, x)
    coef = float(area / qini[-1]) if qini[-1] != 0 else 0.0
    return x, qini, area, coef


def _trapezoid(f, x):
    """Trapezoidal integral of f over x.

    Written out rather than calling numpy: `np.trapz` was deprecated in NumPy 2.0
    and removed in 2.1, so it works on some installs and raises AttributeError on
    others. Four lines of arithmetic is cheaper than a version guard and cannot
    break again.
    """
    f = np.asarray(f, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum((f[1:] + f[:-1]) / 2.0 * np.diff(x)))


# ======================================================================== #
# 5. Observational causal inference: adoption -> retention via IPW
# ======================================================================== #
def run_observational(users):
    """Non-randomised sub-question: among users WITH access to Recurring Buy
    (the treatment arm), does actually ADOPTING it (setting up an auto-invest)
    increase 7-day retention? Adoption is self-selected, so this needs causal
    adjustment, not a raw comparison.

    Identifying assumptions (stated honestly):
      (1) Conditional ignorability / no unmeasured confounding: given the
          observed pre-treatment covariates, adoption is as-good-as-random.
      (2) Positivity/overlap: every covariate profile has a non-trivial chance
          of adopting and of not adopting (checked via the propensity plot).
      (3) SUTVA: one user's adoption does not affect another's retention.
    Assumption (1) is the strong one; in a real setting we would treat the IPW
    estimate as suggestive and confirm with an encouragement/instrument design.
    """
    arm = users[users.is_treatment == 1].copy()
    X = _feature_matrix(arm).values
    a = arm.adopted_recurring_buy.values          # "treatment" = adoption
    y = arm.retained_7d.values

    # naive (confounded) difference
    naive = es.two_proportion_diff(y[a == 0], y[a == 1], ALPHA)

    # propensity of adoption
    ps_model = LogisticRegression(max_iter=1000, C=1.0)
    ps_model.fit(X, a)
    e = ps_model.predict_proba(X)[:, 1]
    e = np.clip(e, 0.02, 0.98)  # trim to respect positivity

    # IPW estimate of the ATT (effect among adopters)
    # ATT = mean_{adopted} y  -  weighted mean of non-adopters (w = e/(1-e))
    treated_mean = y[a == 1].mean()
    w_ctrl = e[a == 0] / (1 - e[a == 0])
    ctrl_weighted = np.sum(w_ctrl * y[a == 0]) / np.sum(w_ctrl)
    att_ipw = treated_mean - ctrl_weighted

    # bootstrap CI for the IPW ATT
    rng = np.random.default_rng(SEED)
    n = len(arm)
    boots = []
    for _ in range(600):
        idx = rng.integers(0, n, n)
        Xi, ai, yi = X[idx], a[idx], y[idx]
        if ai.sum() < 20 or (1 - ai).sum() < 20:
            continue
        m = LogisticRegression(max_iter=500, C=1.0).fit(Xi, ai)
        ei = np.clip(m.predict_proba(Xi)[:, 1], 0.02, 0.98)
        wc = ei[ai == 0] / (1 - ei[ai == 0])
        boots.append(yi[ai == 1].mean()
                     - np.sum(wc * yi[ai == 0]) / np.sum(wc))
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    # regression-adjustment cross-check (outcome model)
    reg = smf.logit("retained_7d ~ adopted_recurring_buy + onboarding_minutes"
                    " + C(channel) + C(country_tier) + C(device)", data=arm).fit(disp=0)
    # average marginal effect of adoption
    d1 = arm.copy(); d1["adopted_recurring_buy"] = 1
    d0 = arm.copy(); d0["adopted_recurring_buy"] = 0
    ame = float((reg.predict(d1) - reg.predict(d0)).mean())

    # ---- figure: propensity overlap + estimate comparison ----
    viz.set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bins = np.linspace(0, 1, 26)
    ax1.hist(e[a == 1], bins=bins, alpha=0.7, color=viz.BLUE,
             label="Adopters", density=True)
    ax1.hist(e[a == 0], bins=bins, alpha=0.6, color=viz.ORANGE,
             label="Non-adopters", density=True)
    ax1.set_xlabel("Estimated propensity to adopt")
    ax1.set_ylabel("Density")
    ax1.set_title("Overlap / positivity check")
    ax1.legend()
    ax1.grid(axis="both")

    labels = ["Naive\n(confounded)", "IPW ATT", "Regression\nadjust"]
    vals = [naive.absolute_effect * 100, att_ipw * 100, ame * 100]
    errs = [[(naive.absolute_effect - naive.ci_low) * 100,
             (att_ipw - ci_low) * 100, 0],
            [(naive.ci_high - naive.absolute_effect) * 100,
             (ci_high - att_ipw) * 100, 0]]
    colors = [viz.GRAY, viz.AQUA, viz.ORANGE]
    ax2.bar(labels, vals, color=colors, width=0.6, zorder=3)
    ax2.errorbar(labels, vals, yerr=errs, fmt="none", ecolor=viz.INK2,
                 elinewidth=1.6, capsize=5, zorder=4)
    ax2.axhline(0, color=viz.INK2, lw=0.9)
    ax2.set_ylabel("Effect of adoption on retention (pp)")
    ax2.set_title("Adjustment shrinks the confounded gap")
    ax2.grid(axis="x", visible=False)
    viz.savefig(fig, os.path.join(FIG_DIR, "fig_observational_ipw.png"))

    return {
        "question": "effect of adopting Recurring Buy on 7-day retention (treatment arm)",
        "naive_diff": naive.as_dict(),
        "ipw_att": float(att_ipw),
        "ipw_att_ci": [float(ci_low), float(ci_high)],
        "regression_adjusted_ame": ame,
        "n_adopters": int((a == 1).sum()),
        "n_non_adopters": int((a == 0).sum()),
        "assumptions": ["conditional ignorability", "positivity/overlap", "SUTVA"],
    }


# ======================================================================== #
# 6. Difference-in-differences (phased regional rollout)
# ======================================================================== #
def _wild_cluster_bootstrap_p(panel, spec, param, cluster_col,
                              n_boot=999, seed=SEED):
    """Null-imposed wild cluster bootstrap p-value (Cameron-Gelbach-Miller 2008).

    The textbook remedy when the number of clusters is small. Procedure:
      1. fit the model WITHOUT `param` (imposing H0: beta = 0) and keep its
         fitted values and residuals;
      2. resample by flipping the sign of every residual in a cluster together
         (Rademacher weights) -- this preserves whatever within-cluster
         dependence exists without assuming G is large;
      3. refit the full model on each pseudo-sample and collect the
         cluster-robust t-statistic;
      4. compare the observed t against that bootstrap distribution.

    Returns (p_value, t_observed).
    """
    y, X = patsy.dmatrices(spec, data=panel, return_type="dataframe")
    yv = np.asarray(y).ravel()
    groups = np.asarray(panel[cluster_col])
    cl = {"groups": groups}

    full = sm.OLS(yv, X).fit(cov_type="cluster", cov_kwds=cl)
    t_obs = float(full.params[param] / full.bse[param])

    restricted = sm.OLS(yv, X.drop(columns=[param])).fit()
    fitted = np.asarray(restricted.fittedvalues)
    resid = np.asarray(restricted.resid)

    uniq, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_boot):
        w = rng.choice([-1.0, 1.0], size=len(uniq))[inverse]
        boot = sm.OLS(fitted + w * resid, X).fit(cov_type="cluster", cov_kwds=cl)
        t_b = boot.params[param] / boot.bse[param]
        if abs(t_b) >= abs(t_obs):
            extreme += 1
    # (extreme + 1) / (B + 1) keeps the p-value valid in finite samples
    return float((extreme + 1) / (n_boot + 1)), t_obs


def run_did(panel):
    """DiD on a phased regional rollout, with honest small-cluster inference.

    Treatment is assigned at the REGION level, so the reflex is to cluster
    standard errors by region. With only 12 regions that reflex needs checking
    rather than trusting, and this function does the checking explicitly:

      * the cluster-robust covariance has rank <= G-1 = 11, so it is rank
        deficient for a 28-parameter spec (and badly so for the 43-parameter
        event study, where it produces NEGATIVE variances);
      * clustering here makes the SE *smaller*, not larger. That is the opposite
        of the usual Bertrand-Duflo-Mullainathan story and is a warning sign:
        with few clusters the cluster-robust estimator is downward biased, and
        there is little within-region serial correlation left for it to pick up
        once region fixed effects are in.

    So all three are reported -- classical, cluster-robust, and a wild cluster
    bootstrap -- and the bootstrap is the one the conclusion rests on.
    """
    panel = panel.copy()

    spec = "activation_rate ~ treated_post + C(region_id) + C(week)"
    n_clusters = int(panel["region_id"].nunique())
    twfe_classical = smf.ols(spec, data=panel).fit()
    twfe = smf.ols(spec, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["region_id"]})

    did_effect = float(twfe.params["treated_post"])
    ci_clustered = twfe.conf_int().loc["treated_post"].tolist()
    ci_classical = twfe_classical.conf_int().loc["treated_post"].tolist()
    did_p = float(twfe.pvalues["treated_post"])

    # HEADLINE CI = the classical (wider) one. Reporting the clustered interval
    # as the headline would contradict the diagnostic below, which shows the
    # cluster-robust SE is anti-conservative at G=12. When two intervals
    # disagree and you cannot justify the tighter one, quote the wider one.
    ci = ci_classical

    wcb_p, wcb_t = _wild_cluster_bootstrap_p(
        panel, spec, "treated_post", "region_id")

    cluster_cov_rank = int(np.linalg.matrix_rank(np.asarray(twfe.cov_params())))

    # event study: interact event_time with treated (baseline = week before rollout)
    ev = panel.copy()
    ev["event_time"] = ev["event_time"].astype(int)
    # drop t = -1 as reference.
    #
    # This model gets HC1 (heteroskedasticity-robust), NOT cluster-robust, SEs --
    # deliberately. It has 43 parameters and there are only 12 clusters, so the
    # cluster-robust covariance collapses to rank 10 and returns negative
    # variances (i.e. NaN standard errors) for several coefficients. Clustering
    # a model this wide on this few units is not conservative, it is undefined.
    es_model = smf.ols(
        "activation_rate ~ C(event_time, Treatment(reference=-1)):treated_region"
        " + C(region_id) + C(week)", data=ev).fit(cov_type="HC1")

    # collect event-time coefficients
    coefs, lows, highs, times = [], [], [], []
    for t in sorted(ev.event_time.unique()):
        if t == -1:
            coefs.append(0.0); lows.append(0.0); highs.append(0.0); times.append(t)
            continue
        name = f"C(event_time, Treatment(reference=-1))[{t}]:treated_region"
        if name in es_model.params.index:
            coefs.append(float(es_model.params[name]))
            lo, hi = es_model.conf_int().loc[name].tolist()
            lows.append(lo); highs.append(hi); times.append(t)

    # parallel-trends test: joint F-test that all PRE-period coefs = 0
    pre_terms = [f"C(event_time, Treatment(reference=-1))[{t}]:treated_region"
                 for t in sorted(ev.event_time.unique()) if t < -1
                 and f"C(event_time, Treatment(reference=-1))[{t}]:treated_region"
                 in es_model.params.index]
    if pre_terms:
        ftest = es_model.f_test(" , ".join(f"{p} = 0" for p in pre_terms))
        pretrend_p = float(np.ravel(ftest.pvalue))
    else:
        pretrend_p = None

    # ---- figures ----
    viz.set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    # (a) raw trends
    for treated, color, lab in [(1, viz.BLUE, "Treated regions"),
                                (0, viz.GRAY, "Control regions")]:
        g = panel[panel.treated_region == treated].groupby("week").activation_rate.mean()
        ax1.plot(g.index, g.values * 100, color=color, lw=2, marker="o",
                 markersize=4, label=lab)
    rollout = int(panel.rollout_week.iloc[0])
    ax1.axvline(rollout, color=viz.RED, lw=1.1, ls="--", label="Rollout")
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Activation rate (%)")
    ax1.set_title("Parallel pre-trends, divergence after rollout")
    ax1.legend(loc="upper left")
    ax1.grid(axis="both")

    # (b) event study
    ax2.errorbar(times, np.array(coefs) * 100,
                 yerr=[(np.array(coefs) - np.array(lows)) * 100,
                       (np.array(highs) - np.array(coefs)) * 100],
                 fmt="o", color=viz.BLUE, ecolor=viz.BLUE, elinewidth=1.6,
                 capsize=3, markersize=6)
    ax2.axhline(0, color=viz.INK2, lw=0.9)
    ax2.axvline(-0.5, color=viz.RED, lw=1.1, ls="--")
    ax2.set_xlabel("Weeks relative to rollout")
    ax2.set_ylabel("DiD coefficient (pp)")
    ax2.set_title("Event study, HC1 SEs (pre-period coefs ~ 0)")
    ax2.grid(axis="both")
    viz.savefig(fig, os.path.join(FIG_DIR, "fig_did.png"))

    return {
        "twfe_did_effect": did_effect,
        "twfe_did_ci": [float(ci[0]), float(ci[1])],
        "twfe_did_ci_basis": "classical (wider of the two; see inference.note)",
        "twfe_did_p_value": did_p,
        "inference": {
            "n_clusters": n_clusters,
            "se_classical": float(twfe_classical.bse["treated_post"]),
            "se_clustered_by_region": float(twfe.bse["treated_post"]),
            "ci_classical": [float(ci_classical[0]), float(ci_classical[1])],
            "ci_clustered": [float(ci_clustered[0]), float(ci_clustered[1])],
            "p_classical": float(twfe_classical.pvalues["treated_post"]),
            "p_clustered": did_p,
            "wild_cluster_bootstrap_p": wcb_p,
            "wild_cluster_bootstrap_t_observed": wcb_t,
            "n_params": int(len(twfe.params)),
            "cluster_cov_rank": cluster_cov_rank,
            "note": (
                f"Treatment is assigned at region level, so clustering by region "
                f"is the default choice -- but with only {n_clusters} clusters the "
                f"cluster-robust covariance is rank {cluster_cov_rank} for "
                f"{len(twfe.params)} parameters, and it makes the SE SMALLER than the "
                "classical one rather than larger. That is the opposite of the "
                "usual Bertrand-Duflo-Mullainathan result and signals "
                "downward bias from too few clusters, not a genuine precision "
                "gain. So the headline CI quoted is the CLASSICAL (wider) one, "
                "and the wild cluster bootstrap p-value is the test the "
                "conclusion rests on. The effect is significant under all "
                "three, so the choice does not change the decision."
            ),
        },
        "parallel_trends_ftest_p": pretrend_p,
        "parallel_trends_ok": bool(pretrend_p is None or pretrend_p > 0.05),
        "event_study_cov_type": "HC1",
        "note": ("Event-study SEs are HC1, not clustered: that spec has 43 "
                 f"parameters against {n_clusters} clusters, so its cluster-robust "
                 "covariance is rank 10 and yields negative variances. Rollout "
                 "timing is common across treated regions, which keeps the TWFE "
                 "estimator clean; genuinely staggered adoption would call for "
                 "Callaway-Sant'Anna / Sun-Abraham to avoid negative-weight bias."),
    }


# ======================================================================== #
# 6b. Monte-Carlo unbiasedness + CI coverage (the real proof)
# ======================================================================== #
def run_unbiasedness_simulation(n_sims=500, n=8000):
    """Repeat the activation experiment over many independent draws and check
    (a) the estimator is unbiased (mean estimate ~ mean true ATE), and
    (b) the 95% CI has ~95% coverage. This is the rigorous demonstration that
    the estimator recovers the known ground truth (one single experiment can
    land outside its CI ~5% of the time; coverage is the honest check)."""
    ests, trues, covered = [], [], 0
    for i in range(n_sims):
        true_ate, activated, treat = generate_data.simulate_activation(
            seed=100_000 + i, n=n)
        r = es.two_proportion_diff(activated[treat == 0], activated[treat == 1], ALPHA)
        ests.append(r.absolute_effect)
        trues.append(true_ate)
        if r.ci_low <= true_ate <= r.ci_high:
            covered += 1
    ests = np.array(ests); trues = np.array(trues)
    coverage = covered / n_sims
    bias = float(ests.mean() - trues.mean())

    viz.set_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(ests * 100, bins=30, color=viz.BLUE, alpha=0.85, zorder=3)
    ax.axvline(trues.mean() * 100, color=viz.RED, lw=2,
               label=f"true ATE = {trues.mean()*100:.2f}pp")
    ax.axvline(ests.mean() * 100, color=viz.ORANGE, lw=2, ls="--",
               label=f"mean estimate = {ests.mean()*100:.2f}pp")
    ax.set_xlabel("Estimated activation effect (pp) across simulated experiments")
    ax.set_ylabel("Count")
    ax.set_title(f"Unbiasedness: {n_sims} sims, 95% CI coverage = {coverage*100:.1f}%")
    ax.legend()
    ax.grid(axis="x", visible=False)
    viz.savefig(fig, os.path.join(FIG_DIR, "fig_unbiasedness.png"))

    return {
        "n_sims": n_sims,
        "n_per_sim": n,
        "mean_true_ate": float(trues.mean()),
        "mean_estimate": float(ests.mean()),
        "bias": bias,
        "ci_coverage_95": float(coverage),
    }


# ======================================================================== #
# 7. Peeking / sequential testing demo
# ======================================================================== #
def run_sequential(control_rate=0.34, n_final=40000, n_looks=10, n_sims=3000):
    """Simulate the false-positive cost of peeking under the NULL (A/A).

    We run many true-null experiments, take `n_looks` equally spaced interim
    looks, and record how often a naive fixed-alpha test EVER crosses 0.05.
    Fixed-horizon testing controls this at 5%; naive peeking inflates it badly.
    """
    rng = np.random.default_rng(SEED)
    look_points = np.linspace(n_final // n_looks, n_final, n_looks).astype(int)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)

    ever_reject = 0
    final_reject = 0
    for _ in range(n_sims):
        # simulate arrival order once, reuse cumulative counts at each look
        c = rng.binomial(1, control_rate, n_final // 2)
        t = rng.binomial(1, control_rate, n_final // 2)  # NULL: same rate
        crossed = False
        for lp in look_points:
            k = lp // 2
            pc, pt = c[:k].mean(), t[:k].mean()
            p = (c[:k].sum() + t[:k].sum()) / (2 * k)
            se = np.sqrt(p * (1 - p) * (2 / k)) if 0 < p < 1 else 1
            z = (pt - pc) / se
            if abs(z) > z_crit:
                crossed = True
                break  # a real team would have stopped and shipped here
        # final-look decision (fixed horizon)
        k = (n_final // 2)
        pc, pt = c.mean(), t.mean()
        p = (c.sum() + t.sum()) / (2 * k)
        se = np.sqrt(p * (1 - p) * (2 / k))
        if abs((pt - pc) / se) > z_crit:
            final_reject += 1
        if crossed:
            ever_reject += 1

    peeking_fpr = ever_reject / n_sims
    fixed_fpr = final_reject / n_sims

    viz.set_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(["Fixed horizon\n(1 test)", f"Naive peeking\n({n_looks} looks)"],
                  [fixed_fpr * 100, peeking_fpr * 100],
                  color=[viz.BLUE, viz.RED], width=0.55, zorder=3)
    ax.axhline(ALPHA * 100, color=viz.INK2, lw=1.1, ls="--",
               label=f"nominal alpha = {ALPHA*100:.0f}%")
    for b, v in zip(bars, [fixed_fpr * 100, peeking_fpr * 100]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%",
                ha="center", color=viz.INK, fontweight="bold")
    ax.set_ylabel("False-positive rate under the null (%)")
    ax.set_title("Why we fix the horizon: peeking inflates Type I error")
    ax.legend()
    ax.grid(axis="x", visible=False)
    viz.savefig(fig, os.path.join(FIG_DIR, "fig_sequential_peeking.png"))

    return {
        "nominal_alpha": ALPHA,
        "fixed_horizon_fpr": float(fixed_fpr),
        "naive_peeking_fpr": float(peeking_fpr),
        "n_looks": n_looks,
        "n_sims": n_sims,
        "remedy": ("Pre-commit to a fixed horizon, or use an alpha-spending "
                   "boundary (O'Brien-Fleming) / always-valid CIs (mSPRT) if "
                   "continuous monitoring is required."),
    }
