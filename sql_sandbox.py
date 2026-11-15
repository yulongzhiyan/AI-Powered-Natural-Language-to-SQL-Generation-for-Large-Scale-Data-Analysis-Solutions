"""
Sandbox execution helpers shared by the new pipeline

The original scripts each re-implemented ``execute_sql``.  For the risk
pipeline we need a slightly stricter version:

* a hard timeout so a malformed query cannot block the whole experiment;
* a **read-only** connection (defence in depth for the security requirement
  stated in the thesis, NFR "read-only sandbox");
* result comparison that is order-insensitive but type-aware, mirroring the
  official Spider evaluation protocol (multiset comparison).
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, List, Sequence, Tuple

# SQLite statements that must never reach the sandbox.
_WRITE_KEYWORDS = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
                   "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM")


def is_read_only(sql: str) -> bool:
    """Cheap guard-rail: a generated statement must be a single SELECT."""
    body = sql.strip().rstrip(";").upper()
    if not body.startswith(("SELECT", "WITH")):
        return False
    return not any(f" {kw} " in f" {body} " for kw in _WRITE_KEYWORDS)


def execute_sql(db_path: str, sql: str, timeout_s: float = 5.0,
                read_only: bool = True) -> Tuple[bool, Any]:
    """Execute ``sql`` and return ``(ok, rows | error_message)``.

    The connection is opened on a copy-free read-only URI, so even a
    successful write attempt cannot modify the benchmark database.
    """
    if not sql or not sql.strip():
        return False, "Empty SQL statement."
    if read_only and not is_read_only(sql):
        return False, "Rejected by sandbox: statement is not a read-only SELECT."

    uri = f"file:{db_path}?mode=ro" if read_only else db_path
    conn = None
    try:
        conn = sqlite3.connect(uri, uri=read_only, timeout=timeout_s)
        conn.execute("PRAGMA busy_timeout = %d" % int(timeout_s * 1000))
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return True, rows
    except Exception as exc:  # sqlite3.Error, UnicodeDecodeError, ...
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()


def _canonical(rows: Sequence[Sequence[Any]]) -> List[Tuple[str, ...]]:
    """Multiset of rows, with floats rounded and everything cast to text.

    Rounding avoids false negatives caused by floating point noise, which is
    a known source of noise in execution-accuracy comparisons.
    """
    out: List[Tuple[str, ...]] = []
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, float):
                cells.append(f"{round(cell, 6):.6f}")
            elif cell is None:
                cells.append("<NULL>")
            else:
                cells.append(str(cell).strip())
        out.append(tuple(cells))
    return sorted(out)


def results_match(gold_rows: Sequence[Sequence[Any]],
                  pred_rows: Sequence[Sequence[Any]]) -> bool:
    """Order-insensitive, type-tolerant equality of two result sets."""
    try:
        return _canonical(gold_rows) == _canonical(pred_rows)
    except Exception:
        return False


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars / token) used only for cost reporting.

    Cheap and provider-independent; it must not be presented as an exact
    tokenizer count.
    """
    if not text:
        return 0
    return max(1, int(round(len(text) / 4.0)))


def db_path_for(db_id: str, database_dir: str) -> str:
    return os.path.join(database_dir, db_id, f"{db_id}.sqlite")
