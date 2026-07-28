"""
Build the DuckDB warehouse from the seed CSVs.

Primary path: run the real dbt project (dbt-duckdb) -> raw -> staging -> marts,
including dbt data tests. Fallback path (if dbt is not importable): execute the
equivalent plain SQL in sql/build.sql, which mirrors the same DAG. Either way the
result is a `warehouse.duckdb` file containing `mart_experiment_users` and
`mart_did_panel`, and downstream analysis is identical.
"""

import os
import subprocess
import sys

import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEEDS_DIR = os.path.join(ROOT, "seeds")
DB_PATH = os.path.join(ROOT, "warehouse.duckdb")
SQL_BUILD = os.path.join(ROOT, "sql", "build.sql")


def _try_dbt():
    """Return True if the dbt build succeeds, else False."""
    try:
        import dbt.cli.main  # noqa: F401
    except Exception:
        return False

    env = dict(os.environ)
    env["DBT_PROFILES_DIR"] = ROOT
    # remove any stale db so the run is clean/deterministic
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    # --no-partial-parse is not an optimisation choice: dbt caches ABSOLUTE
    # paths in target/partial_parse.msgpack, so a clone or a moved directory
    # makes every seed fail with "No files found" and the run silently degrades
    # to the plain-SQL fallback. Always re-parse.
    cmd = [sys.executable, "-m", "dbt.cli.main", "build", "--no-partial-parse",
           "--project-dir", ROOT, "--profiles-dir", ROOT]
    print("  running: dbt build --no-partial-parse")
    res = subprocess.run(cmd, env=env, cwd=ROOT,
                         capture_output=True, text=True)
    if res.returncode != 0:
        print("  dbt build FAILED; falling back to plain SQL "
              "(marts are identical, but dbt tests did not run).")
        print(res.stdout[-1500:])
        print(res.stderr[-800:])
        return False
    # surface the dbt PASS/ERROR summary line
    for line in res.stdout.splitlines():
        if "PASS=" in line or "Completed successfully" in line:
            print("  dbt:", line.strip())
    return True


def _run_fallback_sql():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with open(SQL_BUILD) as f:
        script = f.read().replace("{SEEDS_DIR}", SEEDS_DIR)
    con = duckdb.connect(DB_PATH)
    con.execute(script)
    con.close()
    print("  built warehouse via plain SQL (sql/build.sql)")


def build_warehouse(engine="auto"):
    """Build the warehouse and return (engine_used, marts as DataFrames).

    engine="auto" prefers dbt and falls back to plain SQL; "dbt" or "plain-sql"
    force one path (used by the parity test that proves both produce identical
    marts, which is what makes the fallback safe to rely on).
    """
    if engine not in ("auto", "dbt", "plain-sql"):
        raise ValueError(f"unknown engine: {engine}")

    if engine == "plain-sql":
        used = "plain-sql"
        _run_fallback_sql()
    elif engine == "dbt":
        if not _try_dbt():
            raise RuntimeError("dbt build was requested but failed")
        used = "dbt"
    else:
        used = "dbt"
        if not _try_dbt():
            used = "plain-sql"
            _run_fallback_sql()

    con = duckdb.connect(DB_PATH, read_only=True)
    users = con.execute(
        "select * from mart_experiment_users order by user_id").df()
    did = con.execute("select * from mart_did_panel order by region_id, week").df()
    con.close()
    return used, users, did


if __name__ == "__main__":
    engine, users, did = build_warehouse()
    print(f"engine={engine}  users={len(users)}  did_rows={len(did)}")
    print(users.head())
