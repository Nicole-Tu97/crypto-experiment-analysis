"""
The dbt path and the plain-SQL fallback must produce identical marts.

The README claims the fallback "produces identical marts", and `src/run.py` will
silently take it if dbt is unavailable — so that claim is load-bearing: if the two
DAGs drifted, a machine without dbt would quietly compute different numbers from
the same seeds. This test is what makes the fallback safe to rely on.

Skips (rather than fails) when dbt is not installed, since the fallback is the
supported path in that case.

Run:  python3 tests/test_warehouse_parity.py   (or via pytest)
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import build_warehouse  # noqa: E402
import generate_data    # noqa: E402


def _dbt_available():
    try:
        import dbt.cli.main  # noqa: F401
        import dbt.adapters.duckdb  # noqa: F401
        return True
    except Exception:
        return False


def test_dbt_and_plain_sql_marts_are_identical():
    if not _dbt_available():
        print("SKIP  dbt not installed; plain-SQL fallback is the supported path")
        return

    generate_data.generate()

    engine_sql, users_sql, did_sql = build_warehouse.build_warehouse(engine="plain-sql")
    engine_dbt, users_dbt, did_dbt = build_warehouse.build_warehouse(engine="dbt")
    assert engine_sql == "plain-sql" and engine_dbt == "dbt"

    for name, left, right in [("mart_experiment_users", users_sql, users_dbt),
                              ("mart_did_panel", did_sql, did_dbt)]:
        assert sorted(left.columns) == sorted(right.columns), f"{name}: column mismatch"
        assert len(left) == len(right), f"{name}: row-count mismatch"
        cols = sorted(left.columns)
        pd.testing.assert_frame_equal(
            left[cols].reset_index(drop=True),
            right[cols].reset_index(drop=True),
            check_dtype=False,   # DuckDB CSV inference vs dbt column_types
            check_exact=False,
            rtol=1e-9,
            obj=name,
        )


def test_marts_have_the_shape_the_analysis_assumes():
    generate_data.generate()
    _, users, did = build_warehouse.build_warehouse()

    # one row per user, no duplicate join fan-out
    assert users.user_id.is_unique
    assert len(users) == generate_data.N_USERS

    # binary outcomes really are binary, and assignment is coded consistently
    for col in ["is_treatment", "activated_7d", "retained_7d",
                "support_contact_7d", "adopted_recurring_buy"]:
        assert set(users[col].unique()) <= {0, 1}, col
    assert (users.assignment == "treatment").equals(users.is_treatment == 1)

    # adoption is only possible in the treatment arm -- the analysis relies on
    # this when it restricts the IPW question to that arm
    assert users.loc[users.is_treatment == 0, "adopted_recurring_buy"].sum() == 0

    # the DiD panel is balanced and its derived columns agree with their sources
    assert len(did) == did.region_id.nunique() * did.week.nunique()
    assert (did.treated_post == did.treated_region * did.post).all()
    assert (did.event_time == did.week - did.rollout_week).all()
    assert did.activation_rate.between(0, 1).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} warehouse tests passed.")
