-- FLAGSHIP ANALYSIS TABLE (one row per user).
--
-- Joins pre-treatment covariates (staging users) to post-assignment outcomes
-- and guardrails. This is the single table every experiment estimator reads:
-- assignment, pre-period covariate, exposure, primary metric, guardrails.

with u as (
    select * from {{ ref('stg_users') }}
),

a as (
    select * from {{ ref('stg_activation') }}
)

select
    u.user_id,

    -- assignment
    u.assignment,
    u.is_treatment,

    -- pre-treatment covariates (safe for balance checks, CUPED, segmentation)
    u.channel,
    u.country_tier,
    u.device,
    u.age_bucket,
    u.onboarding_minutes,
    u.onboarding_tier,
    u.signup_day,

    -- exposure / mechanism (post-treatment; only possible in treatment arm)
    a.adopted_recurring_buy,

    -- primary + co-primary metrics
    a.activated_7d,          -- PRIMARY: funded + first crypto trade within 7d
    a.retained_7d,           -- CO-PRIMARY: active on day 7

    -- guardrail metrics
    a.support_contact_7d,    -- guardrail: must not increase
    a.net_deposits_7d        -- guardrail: must not decrease

from u
join a using (user_id)
