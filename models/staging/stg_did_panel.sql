-- Clean, typed view of the region x week rollout panel.

with src as (

    select * from {{ ref('raw_did_panel') }}

)

select
    region_id,
    week,
    treated_region,
    post,
    rollout_week,
    treated_region * post as treated_post,
    week - rollout_week   as event_time,
    n_users,
    activation_rate
from src
