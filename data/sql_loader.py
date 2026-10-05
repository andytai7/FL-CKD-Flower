"""Runtime landmark-dataframe builder for a real practice's Tomedo→PostgreSQL export (CheCKD SQL).

The deployment counterpart to `fhir_loader.py`: the sandbox rehearses against a Helios FHIR
server (CLAUDE.md §0.9); the real clinic runs Nicolas Conze's two-stage SQL directly against its
own PostgreSQL (source of truth: practice_1/CKD_risk_model_dataframe.sql, mirrored runtime
copies under data/sql/):

- **Stage 1** builds the session-scoped temp tables `t_patient / t_kontakt / t_lab_tag / t_diag`
  (indexed + analysed, once per session, independent of Stichtag and horizon);
- **Stage 2** is a single query whose `params` CTE takes `(praxis, stichtag, horizont_jahre,
  feature_jahre)` as bound parameters and emits the training dataframe
  (`status_g3a/g3b/g4` + `label_g3a/g3b/g4`, NULL = not trainable).

Privacy shape: every statement executes on the practice's own database inside the SuperNode; only
the resulting in-memory frame is used for local training. The temp tables are dropped explicitly
after materialisation and are session-scoped regardless — closing the connection removes them even
on a crash. Nothing is written to disk and no patient row crosses the federation boundary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_SQL_DIR = Path(__file__).resolve().parent / "sql"
STAGE1_SQL = _SQL_DIR / "stage1_temp_tables.sql"
STAGE2_SQL = _SQL_DIR / "stage2_dataframe.sql"

# Stage 1's tables, dropped explicitly after the frame is materialised (the session itself is the
# real guarantee: PostgreSQL TEMP tables vanish with the connection either way).
TEMP_TABLES = ("t_lab_tag", "t_diag", "t_kontakt", "t_patient")

ALLOWED_HORIZONS = (1, 2, 5)

_COMMENT_OR_BLANK = re.compile(r"^\s*(--[^\n]*)?$")


def _stage1_statements(sql_text: str) -> list[str]:
    """Split the Stage-1 script into individual statements for psycopg (one statement per execute).

    The script separates statements with a semicolon at end of line; chunks that reduce to
    comments or whitespace are skipped. Semicolons mid-line (none exist in the script — checked
    at review time) would be the failure mode.
    """
    statements = []
    for chunk in sql_text.split(";\n"):
        if any(line.strip() and not _COMMENT_OR_BLANK.match(line) for line in chunk.splitlines()):
            statements.append(chunk.strip())
    return statements


def load_practice_frame_sql(
    dsn: str,
    *,
    praxis: str,
    stichtag: str,
    horizont_jahre: int = 2,
    feature_jahre: int = 5,
    statement_timeout_ms: int = 600_000,
) -> pd.DataFrame:
    """Build this practice's landmark training dataframe from its local PostgreSQL export.

    Runs Stage 1 (temp tables) and Stage 2 (parameterised dataframe query) in ONE session, then
    drops the temp tables and closes the connection. `stichtag` is an ISO date string
    ("YYYY-MM-DD"); `horizont_jahre` must be one of the audited horizons (1 / 2 / 5).
    """
    if horizont_jahre not in ALLOWED_HORIZONS:
        raise ValueError(f"horizont_jahre must be one of {ALLOWED_HORIZONS}, got {horizont_jahre}")
    if not dsn:
        raise ValueError("Empty DSN — set `sql-dsn` in this SuperNode's node config.")

    import psycopg  # deployment-only dependency; the sandbox paths never touch it

    stage1 = _stage1_statements(STAGE1_SQL.read_text(encoding="utf-8"))
    stage2 = STAGE2_SQL.read_text(encoding="utf-8")

    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
            for stmt in stage1:
                cur.execute(stmt)
            cur.execute(
                stage2,
                {
                    "praxis": praxis,
                    "stichtag": stichtag,
                    "horizont_jahre": horizont_jahre,
                    "feature_jahre": feature_jahre,
                },
            )
            columns = [desc.name for desc in cur.description]
            rows = cur.fetchall()
            cur.execute("DROP TABLE IF EXISTS " + ", ".join(TEMP_TABLES))
        conn.commit()
    finally:
        conn.close()  # connection close alone already removes the TEMP tables

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise RuntimeError(
            f"Landmark query returned 0 rows for praxis={praxis!r}, stichtag={stichtag!r}, "
            f"horizont={horizont_jahre}y — check cohort inclusion and landmark parameters."
        )
    return df
