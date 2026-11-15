"""
build a clean, labelled dataset

Two data sources are supported

``--source legacy`` (default, no API cost)
    Re-uses the execution logs already produced by the original pipeline
    (``spider_evaluation_results.csv``).  Every logged query tells us

        * whether the final SQL executed at all  -> ``y_exec_fail``
        * whether it matched the gold result     -> ``y_correct``

    Because the original loop only records an SQL string when the statement
    *did* execute, the materialised sub-population is
    "statements that ran successfully", which is exactly the population in
    which a silent semantic error is possible -- and silent errors are the
    expensive failure mode (a sandbox error is caught for free by the loop).

``--source api``
    Regenerates the *first draft* for each question with the LLM (one call
    per question, temperature 0) and records the true execution state of the
    first attempt.  This gives the textbook P(Error|X) label without any
    ambiguity

Output: ``data/dataset_raw.csv``
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List

from config import (DATABASE_DIR, DATASET_RAW_PATH, MAX_REFLECTION_RETRIES,
                    SPIDER_DEV_PATH, LEGACY_EVAL_CSV)
from sql_sandbox import db_path_for, execute_sql, results_match

# --------------------------------------------------------------------------
# Label helpers
# --------------------------------------------------------------------------
EXEC_FAILED = "EXEC_FAILED"
SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
TYPE_MISMATCH = "TYPE_MISMATCH"
CORRECT = "CORRECT"


def classify_error(error_log: str, is_correct: bool) -> str:
    """Map the free-text log of the original pipeline onto a label."""
    log = (error_log or "").lower()
    if is_correct:
        return CORRECT
    if "failed after" in log or "max retries" in log or "api error" in log:
        return EXEC_FAILED
    if "type/order" in log:
        return TYPE_MISMATCH
    if "mismatch" in log:
        return SEMANTIC_MISMATCH
    # Unknown log but marked wrong: assume the statement ran (the original
    # script logs the SQL string only when execution succeeded).
    return SEMANTIC_MISMATCH


def load_dev_index() -> Dict[tuple, dict]:
    """``(db_id, question) -> dev record``; keys are unique on Spider dev."""
    with open(SPIDER_DEV_PATH, encoding="utf-8") as fh:
        dev = json.load(fh)
    index = {}
    for i, item in enumerate(dev):
        index[(item["db_id"], item["question"])] = {**item, "query_id": i}
    return index


# --------------------------------------------------------------------------
# Dataset from the existing logs
# --------------------------------------------------------------------------
def build_from_legacy(limit: int | None = None) -> List[Dict]:
    index = load_dev_index()
    rows: List[Dict] = []
    seen = set()

    with open(LEGACY_EVAL_CSV, newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            key = (raw["Database"], raw["Question"])
            if key in seen:      # the CSV was appended by two identical runs
                continue
            seen.add(key)
            dev = index.get(key)
            if dev is None:
                continue
            sql = (raw.get("Generated_SQL") or "").strip()
            is_correct = str(raw.get("Is_Correct")).strip().lower() == "true"
            err_type = classify_error(raw.get("Error_Log", ""), is_correct)
            exec_failed = err_type == EXEC_FAILED or sql in ("", "N/A")
            rows.append({
                "query_id": dev["query_id"],
                "db_id": raw["Database"],
                "question": raw["Question"],
                "gold_sql": dev["query"],
                "generated_sql": "" if exec_failed else sql,
                "is_correct": int(is_correct),
                "y_exec_fail": int(exec_failed),
                "y_wrong": int((not is_correct) and not exec_failed),
                "error_type": err_type,
                "error_log": (raw.get("Error_Log") or "")[:300],
                "source": "legacy_csv",
            })
            if limit and len(rows) >= limit:
                break

    rows.sort(key=lambda r: r["query_id"])
    return rows


# --------------------------------------------------------------------------
# Dataset regenerated with the LLM (first draft only)
# --------------------------------------------------------------------------
def build_from_api(limit: int | None = None, sleep_s: float = 0.0) -> List[Dict]:
    import time
    from llm_client import DeepSeekClient
    from rag_store import SchemaIndex

    client = DeepSeekClient()
    index_db = SchemaIndex()
    with open(SPIDER_DEV_PATH, encoding="utf-8") as fh:
        dev = json.load(fh)
    if limit:
        dev = dev[:limit]

    rows: List[Dict] = []
    for i, item in enumerate(dev):
        db_id, question = item["db_id"], item["question"]
        db_path = db_path_for(db_id, DATABASE_DIR)
        schema_text = index_db.retrieve(question, db_id, db_path)
        sql, _usage = client.generate_sql(question, schema_text,
                                          temperature=0.0)
        ok, out = execute_sql(db_path, sql)
        gold_ok, gold_rows = execute_sql(db_path, item["query"])
        correct = bool(ok and gold_ok and results_match(gold_rows, out))
        rows.append({
            "query_id": i,
            "db_id": db_id,
            "question": question,
            "gold_sql": item["query"],
            "generated_sql": sql,
            "is_correct": int(correct),
            "y_exec_fail": int(not ok),
            "y_wrong": int(ok and not correct),
            "error_type": (CORRECT if correct else
                           (EXEC_FAILED if not ok else SEMANTIC_MISMATCH)),
            "error_log": ("" if ok else str(out))[:300],
            "source": "api_first_draft",
        })
        if (i + 1) % 25 == 0:
            print(f"  generated {i + 1}/{len(dev)}")
        if sleep_s:
            time.sleep(sleep_s)
    return rows


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def write_dataset(rows: List[Dict], path: str = DATASET_RAW_PATH) -> None:
    fields = ["query_id", "db_id", "question", "gold_sql", "generated_sql",
              "is_correct", "y_exec_fail", "y_wrong", "error_type",
              "error_log", "source"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarise(rows: List[Dict]) -> None:
    n = len(rows)
    print(f"\nDataset rows            : {n}")
    print("error_type distribution :")
    for k, v in Counter(r["error_type"] for r in rows).most_common():
        print(f"    {k:18s} {v:5d}  ({v / max(n, 1):6.2%})")
    usable = [r for r in rows if not r["y_exec_fail"]]
    pos = sum(r["y_wrong"] for r in usable)
    print(f"Usable for risk model   : {len(usable)} rows, "
          f"{pos} wrong ({pos / max(len(usable), 1):.2%}), "
          f"base rate of silent error")
    print(f"Execution failures      : {sum(r['y_exec_fail'] for r in rows)} "
          f"(handled by the reflective-repair branch, not by the classifier)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the labelled dataset.")
    ap.add_argument("--source", choices=["legacy", "api"], default="legacy")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=DATASET_RAW_PATH)
    args = ap.parse_args()

    if args.source == "legacy":
        data = build_from_legacy(limit=args.limit)
    else:
        data = build_from_api(limit=args.limit)

    if not data:
        raise SystemExit("No rows were produced - check the input paths.")
    write_dataset(data, args.out)
    summarise(data)
    print(f"\nSaved -> {args.out}")
    print(f"(self-correction budget of the original loop was M="
          f"{MAX_REFLECTION_RETRIES})")
