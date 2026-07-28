-- Analysis-ready DiD panel: one row per region x week, with the interaction and
-- event-time columns the difference-in-differences estimator consumes.

select
    region_id,
    week,
    treated_region,
    post,
    treated_post,
    event_time,
    rollout_week,
    n_users,
    activation_rate
from {{ ref('stg_did_panel') }}
