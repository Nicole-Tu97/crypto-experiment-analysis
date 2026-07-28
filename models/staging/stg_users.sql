-- Clean, typed view of the enrolled users and their pre-treatment covariates.
-- Nothing here depends on outcomes: staging is deliberately outcome-free so the
-- covariates remain "pre-treatment" by construction.

with src as (

    select * from {{ ref('raw_users') }}

)

select
    user_id,
    signup_day,
    assignment,
    case when assignment = 'treatment' then 1 else 0 end   as is_treatment,
    channel,
    country_tier,
    device,
    age_bucket,
    onboarding_minutes,
    -- coarse onboarding tier used for segment-level CATE reporting
    case
        when onboarding_minutes >= 18 then 'high'
        when onboarding_minutes >= 8  then 'mid'
        else 'low'
    end as onboarding_tier
from src
