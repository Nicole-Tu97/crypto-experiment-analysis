"""
AI-assisted stakeholder summary, with the numbers locked down.

WHY THIS EXISTS
---------------
Writing the same experiment readout for three different audiences (a PM, an ops
lead, an exec) is real work that an LLM is genuinely good at, and it is the part
of a product data scientist's job that AI tooling actually accelerates. What an
LLM is *not* trustworthy for is the numbers: a model that fluently writes
"+7.8pp" when the estimate was +7.76pp, or invents a p-value, has quietly
produced a document nobody should act on.

So the division of labour here is deliberate:

    the model writes the LANGUAGE.   the pipeline owns the NUMBERS.

Every numeric token in the generated draft is extracted and checked against the
values in `outputs/metrics.json`. A number that cannot be traced back to a
computed metric means the draft is **rejected** and the deterministic template is
used instead. The check is mechanical, not another model call — an LLM verifying
an LLM shares the failure mode.

The whitelist is derived automatically from the fact sheet the model was handed,
so it cannot drift out of sync with the analysis. Scoping it to the fact sheet
rather than to all of metrics.json is deliberate and makes the check materially
stricter: with ~30 admissible values instead of ~380, an invented figure is far
less likely to collide with some unrelated metric once rounded. (Whitelisting the
whole file let a fabricated "$2.4M" pass by coinciding with a deposits confidence
bound of 2.38.) A number the model was never shown has no business in the memo
even if it does appear somewhere in the metrics.

OFFLINE BY DEFAULT
------------------
No API key, no network, no `anthropic` package? The deterministic template runs
and the pipeline is unaffected. The LLM path is opt-in:

    ANTHROPIC_API_KEY=... USE_LLM=1 python3 src/run.py

Outputs `outputs/exec_summary.md` and `outputs/ai_summary_audit.json` (the audit
records what was checked, what failed, and which path produced the final text).
"""

from __future__ import annotations

import json
import os
import re

MODEL = "claude-opus-5"

# Numbers a summary may legitimately use that are structural rather than
# measured: the metric windows, percentages-as-round-numbers, small counts used
# in prose ("three segments"), and the alpha/power conventions.
STRUCTURAL_NUMBERS = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    7.0,          # 7-day window
    30.0, 90.0,   # post-launch monitoring windows
    50.0,         # 50/50 split
    95.0,         # confidence level
    100.0,        # percent / full rollout
    80.0,         # power
    0.05, 0.025,  # alpha
}

SYSTEM_PROMPT = """\
You are a senior product data scientist writing for a crypto product team.

You will be given the verified results of a randomized experiment as JSON. Write \
a short executive summary of them.

ABSOLUTE RULE ON NUMBERS: every number you write must be copied from the JSON \
you were given. Do not round differently, do not recompute, do not estimate, and \
do not add a number that is not there. If you want to make a point that would \
require a number you were not given, make the point qualitatively instead. Your \
output is machine-checked against the source data and will be discarded entirely \
if it contains a number that cannot be traced back to it.

Style: plain language for a smart non-specialist. Lead with the decision. No \
jargon without a gloss. Be honest about what is uncertain. Three short sections, \
markdown, no more than 250 words total:

**The decision** — what we should do and why, in two or three sentences.
**What we know** — the effect, its uncertainty, and the guardrails.
**What we do not know** — the genuine limitations, stated plainly.
"""


# --------------------------------------------------------------------------- #
# Numeric guardrail
# --------------------------------------------------------------------------- #
def collect_allowed_numbers(obj, out=None):
    """Every number in `obj`, on both the raw and the pp/percent scale.

    For each numeric leaf, admits the value itself and the value x100 -- because
    a proportion of 0.0776 is legitimately written as 7.76pp in prose. Called on
    the fact sheet handed to the model, so the admissible set is exactly what the
    model was allowed to see.
    """
    if out is None:
        out = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        v = float(obj)
        out.add(v)
        out.add(v * 100.0)
        out.add(abs(v))
        out.add(abs(v) * 100.0)
    elif isinstance(obj, dict):
        for value in obj.values():
            collect_allowed_numbers(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            collect_allowed_numbers(value, out)
    return out


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_numbers(text):
    """Numeric tokens in the draft, as (raw_token, float_value, decimals)."""
    found = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        cleaned = token.replace(",", "").lstrip("+")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        found.append((token, value, decimals))
    return found


def audit_numbers(text, allowed):
    """Check every number in `text` against the allowed set.

    A draft number is accepted if some allowed value agrees with it once both are
    rounded to the draft's own precision -- so "7.8pp" is accepted against a true
    7.76pp, while "8.4pp" is not. Comparing at the quoted precision is the point:
    it permits honest rounding and still catches a fabricated figure.
    """
    allowed = list(allowed)
    unverified = []
    for token, value, decimals in extract_numbers(text):
        if abs(value) in STRUCTURAL_NUMBERS or value in STRUCTURAL_NUMBERS:
            continue
        target = round(value, decimals)
        if any(round(a, decimals) == target for a in allowed):
            continue
        unverified.append({"token": token, "value": value, "decimals": decimals})
    return unverified


# --------------------------------------------------------------------------- #
# The facts handed to the model (a deliberately small, flat slice)
# --------------------------------------------------------------------------- #
def build_facts(m):
    """The subset of metrics.json the model is allowed to see.

    Kept small on purpose: a smaller fact sheet means fewer numbers in play,
    a shorter prompt, and less room for the model to reach for something it was
    not given.
    """
    a = m["primary"]["activation"]
    r = m["primary"]["retention"]
    sup = m["primary"]["guardrail_support_contact"]
    dep = m["primary"]["guardrail_net_deposits"]
    het = m["heterogeneity"]["uplift_model"]
    return {
        "feature": "Recurring Buy (automatic recurring crypto purchase) for new clients",
        "design": {
            "type": "randomized controlled experiment, fixed horizon",
            "n_total": m["integrity"]["group_sizes"]["control"]
            + m["integrity"]["group_sizes"]["treatment"],
            "n_control": m["integrity"]["group_sizes"]["control"],
            "n_treatment": m["integrity"]["group_sizes"]["treatment"],
            "srm_p_value": m["integrity"]["srm"]["p_value"],
            "pre_registered_mde_pp": 2.0,
        },
        "primary_metric": {
            "name": "7-day activation (funded account and first crypto trade)",
            "control_pct": a["control_mean"] * 100,
            "treatment_pct": a["treatment_mean"] * 100,
            "effect_pp": a["absolute_effect"] * 100,
            "ci_low_pp": a["ci_low"] * 100,
            "ci_high_pp": a["ci_high"] * 100,
            "relative_pct": a["relative_effect"] * 100,
        },
        "coprimary_retention": {
            "effect_pp": r["absolute_effect"] * 100,
            "ci_low_pp": r["ci_low"] * 100,
            "ci_high_pp": r["ci_high"] * 100,
        },
        "guardrails": {
            "support_contact_effect_pp": sup["absolute_effect"] * 100,
            "support_contact_p_value": sup["p_value"],
            "net_deposits_effect_dollars": dep["absolute_effect"],
        },
        "targeting": {
            "top_half_lift_pp": (het["observed_uplift_top_half"] or 0) * 100,
            "bottom_half_lift_pp": (het["observed_uplift_bottom_half"] or 0) * 100,
        },
        "limitations": [
            "outcomes are measured over 7 days only; a habit feature likely compounds",
            "a 7-day window cannot separate a durable habit from a novelty effect",
            "segment-level effects are exploratory and FDR-corrected, not pre-registered",
            "the adoption-to-retention estimate is not randomized and assumes no unmeasured confounding",
            "data is synthetic, so effect sizes are illustrative rather than forecasts",
        ],
    }


# --------------------------------------------------------------------------- #
# Deterministic fallback (the default path)
# --------------------------------------------------------------------------- #
def deterministic_summary(facts):
    p = facts["primary_metric"]
    g = facts["guardrails"]
    r = facts["coprimary_retention"]
    return f"""\
# Executive summary — Recurring Buy for new clients

**The decision.** Ship it to all new clients. Giving new clients access to
Recurring Buy raised 7-day activation from {p['control_pct']:.1f}% to
{p['treatment_pct']:.1f}% — a gain of {p['effect_pp']:.2f} percentage points, comfortably above the
{facts['design']['pre_registered_mde_pp']:.0f}-point bar we set before running the test. Nothing we were
watching for harm moved against us.

**What we know.** The experiment randomized {facts['design']['n_total']:,} new clients evenly between
the two arms and the split came out clean. The activation gain is
{p['effect_pp']:.2f} points, and the plausible range runs from {p['ci_low_pp']:.2f} to {p['ci_high_pp']:.2f} points —
the *whole* range clears our bar, which is what makes this a decision rather than
a hint. Seven-day retention moved the same direction, by {r['effect_pp']:.2f} points. Support
contacts were flat ({g['support_contact_effect_pp']:+.2f} points, p = {g['support_contact_p_value']:.2f}), so this is not a
feature that generates confused clients, and deposits rose slightly
(${g['net_deposits_effect_dollars']:+.2f} per client) rather than being cannibalised. The benefit is
uneven: the more engaged half of new clients gained {facts['targeting']['top_half_lift_pp']:.2f} points against
{facts['targeting']['bottom_half_lift_pp']:.2f} for the less engaged half, which is an argument about onboarding
placement, not about whether to ship.

**What we do not know.** Everything here is a 7-day read, and a habit feature is
exactly the kind of thing whose value should compound — so this is a floor, not a
ceiling. The flip side is that seven days also cannot tell us whether we measured
a durable habit or the appeal of a new button; that question needs 30- and 90-day
monitoring after launch, which is why the recommendation includes a holdback. The
segment differences are exploratory and should be treated as a hypothesis to
confirm during rollout.
"""


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #
def llm_summary(facts):
    """Draft the summary with Claude. Returns (text, error)."""
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY not set"

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL,
            # Generous headroom: on this model thinking is on by default and
            # max_tokens caps thinking plus visible text together, so a tight
            # limit truncates the summary rather than the reasoning.
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    "Here are the verified experiment results as JSON. Write the "
                    "executive summary.\n\n"
                    f"```json\n{json.dumps(facts, indent=2)}\n```"
                ),
            }],
        )
    except Exception as exc:  # network, auth, rate limit -- fall back, never fail
        return None, f"{type(exc).__name__}: {exc}"

    if response.stop_reason == "refusal":
        return None, "model declined the request"

    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        return None, "model returned no text"
    return text, None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def generate(metrics, out_dir, use_llm=None):
    """Write outputs/exec_summary.md plus an audit of how it was produced."""
    if use_llm is None:
        use_llm = os.environ.get("USE_LLM") == "1"

    facts = build_facts(metrics)
    # Scoped to the fact sheet, NOT to all of metrics.json -- see module docstring.
    allowed = collect_allowed_numbers(facts)

    audit = {
        "model": MODEL,
        "llm_requested": bool(use_llm),
        "allowed_number_count": len(allowed),
        "guardrail": (
            "Every numeric token in the draft must match a value from the fact "
            "sheet given to the model, at the precision it is quoted to. Any "
            "number that cannot be traced back rejects the whole draft."
        ),
    }

    draft, error = (None, "llm not requested")
    if use_llm:
        draft, error = llm_summary(facts)

    if draft is None:
        audit.update({
            "source": "deterministic_template",
            "reason": error,
            "unverified_numbers": [],
        })
        summary = deterministic_summary(facts)
    else:
        unverified = audit_numbers(draft, allowed)
        audit["numbers_checked"] = len(extract_numbers(draft))
        audit["unverified_numbers"] = unverified
        if unverified:
            # This is the whole point of the layer: a fluent draft with an
            # untraceable number is worse than a plain one, so it does not ship.
            audit.update({
                "source": "deterministic_template",
                "reason": (
                    f"LLM draft rejected: {len(unverified)} number(s) could not be "
                    "traced to metrics.json"
                ),
                "rejected_draft": draft,
            })
            summary = deterministic_summary(facts)
        else:
            audit.update({"source": "llm", "reason": "all numbers verified"})
            summary = draft

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "exec_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary if summary.endswith("\n") else summary + "\n")
    with open(os.path.join(out_dir, "ai_summary_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    return audit


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(os.path.dirname(HERE), "outputs")
    with open(os.path.join(OUT, "metrics.json")) as fh:
        result = generate(json.load(fh), OUT)
    print(json.dumps(result, indent=2)[:2000])
