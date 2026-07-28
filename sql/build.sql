-- Plain-SQL fallback that MIRRORS the dbt project (raw -> staging -> marts).
-- Used only when dbt-duckdb is not installed. Run against DuckDB with the seed
-- CSV directory available. The {SEEDS_DIR} token is substituted by
-- build_warehouse.py before execution.

-- ============================ RAW (load seeds) ============================
create or replace table raw_users as
    select * from read_csv_auto('{SEEDS_DIR}/raw_users.csv', header=true);

create or replace table raw_activation as
    select * from read_csv_auto('{SEEDS_DIR}/raw_activation.csv', header=true);

create or replace table raw_did_panel as
    select * from read_csv_auto('{SEEDS_DIR}/raw_did_panel.csv', header=true);

-- ============================ STAGING =====================================
create or replace view stg_users as
select
    user_id,
    signup_day,
    assignment,
    case when assignment = 'treatment' then 1 else 0 end as is_treatment,
    channel,
    country_tier,
    device,
    age_bucket,
    onboarding_minutes,
    case
        when onboarding_minutes >= 18 then 'high'
        when onboarding_minutes >= 8  then 'mid'
        else 'low'
    end as onboarding_tier
from raw_users;

create or replace view stg_activation as
select
    user_id,
    activated_7d,
    adopted_recurring_buy,
    retained_7d,
    support_contact_7d,
    net_deposits_7d
from raw_activation;

create or replace view stg_did_panel as
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
from raw_did_panel;

-- ============================ MARTS =======================================
create or replace table mart_experiment_users as
select
    u.user_id,
    u.assignment,
    u.is_treatment,
    u.channel,
    u.country_tier,
    u.device,
    u.age_bucket,
    u.onboarding_minutes,
    u.onboarding_tier,
    u.signup_day,
    a.adopted_recurring_buy,
    a.activated_7d,
    a.retained_7d,
    a.support_contact_7d,
    a.net_deposits_7d
from stg_users u
join stg_activation a using (user_id);

create or replace table mart_did_panel as
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
from stg_did_panel;
