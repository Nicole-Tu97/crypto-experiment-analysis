-- Clean, typed view of post-assignment outcomes and guardrails.

with src as (

    select * from {{ ref('raw_activation') }}

)

select
    user_id,
    activated_7d,
    adopted_recurring_buy,
    retained_7d,
    support_contact_7d,
    net_deposits_7d
from src
