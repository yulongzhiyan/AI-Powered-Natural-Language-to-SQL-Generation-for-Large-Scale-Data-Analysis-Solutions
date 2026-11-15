"""
Ablation study over the six configurations

    LLM_ONLY   zero-shot, whole DDL in the prompt, one attempt
    LLM_RAG    + vector schema retrieval, one attempt (the paper's baseline 2)
    FIXED_3    + error-log driven self-correction, ceiling of 3 attempts
    BAYES_LR   + risk gate with the logistic-regression model
    BAYES_RF   + risk gate with the random-forest model
    BAYES_FULL + risk gate with the multi-sample consistency budget (K=5)

``FIXED_3`` is the original full system ; ``BAYES_*`` insert the
decision layer between generation and execution.

Each configuration appends one row per query to
``ablation_results_<mode>.csv`` so an interrupted run can be resumed and the
raw per-query evidence stays available for the write-up.
"""

from __future__ import annotations

import csv
import json
import os
import time

import chromadb

from config import (DATABASE_DIR, MODEL_DIR, SPIDER_DEV_PATH)
from sql_sandbox import db_path_for, execute_sql, results_match, estimate_tokens

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
# Choose the configuration to run: LLM_ONLY | LLM_RAG | FIXED_3 |
#                                 BAYES_LR | BAYES_RF | BAYES_FULL
ABLATION_MODE = "BAYES_LR"

CSV_OUT_PATH = f"ablation_results_{ABLATION_MODE.lower()}.csv"

# Vector store is only needed by the RAG-based configurations.
chroma_client = chromadb.PersistentClient(path="./chroma_db")
eval_collection = chroma_client.get_or_create_collection(name="spider_eval")

FIELDS = ["Test_ID", "Database", "Question", "Is_Correct", "Error_Log",
          "P_Wrong", "Action", "Calls"]


# --------------------------------------------------------------------------
# 2. Schema sources for the legacy configurations
# --------------------------------------------------------------------------
def get_full_schema_llm_only(db_path: str) -> str:
    """Baseline 1: dump every CREATE TABLE into the prompt (no filtering)."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cur.fetchall()
    conn.close()
    return "\n".join([t[0] for t in tables if t[0]])


def retrieve_schema_rag(query: str, db_id: str, top_k: int = 3) -> str:
    """Baseline 2: top-k tables from the ChromaDB index."""
    results = eval_collection.query(query_texts=[query], n_results=top_k,
                                    where={"db_id": db_id})
    if results["documents"] and results["documents"][0]:
        return "\n".join(results["documents"][0])
    return ""


# --------------------------------------------------------------------------
# 3. One query under the selected configuration
# --------------------------------------------------------------------------
def run_legacy_mode(client, mode: str, question: str, db_path: str,
                    db_id: str) -> dict:
    """One-shot (LLM_ONLY / LLM_RAG) or fixed-3 self-correction."""
    if mode == "LLM_ONLY":
        schema = get_full_schema_llm_only(db_path)
    else:
        schema = retrieve_schema_rag(question, db_id)

    sql, _ = client.generate_sql(question, schema, temperature=0.0)
    calls = 1
    max_attempts = 3 if mode == "FIXED_3" else 1
    ok, out = execute_sql(db_path, sql)
    for _ in range(max_attempts - 1):
        if ok:
            break
        sql, _ = client.repair_sql(question, schema, sql, str(out), temperature=0.0)
        calls += 1
        ok, out = execute_sql(db_path, sql)
    return {"sql": sql, "ok": ok, "rows": out if ok else None,
            "status": "Success" if ok else str(out)[:200],
            "p_wrong": "", "action": "EXECUTE", "calls": calls}


def run_bayes_mode(agent, question: str, db_path: str, db_id: str,
                   gold_rows) -> dict:
    res = agent.run(question, db_id, db_path, gold_rows=gold_rows)
    return {"sql": res.final_sql, "ok": res.executed_ok, "rows": None,
            "status": res.status, "p_wrong": round(res.p_wrong_initial, 4),
            "action": " -> ".join(res.action_path), "calls": res.n_calls,
            "is_correct": res.is_correct}


# --------------------------------------------------------------------------
# 4. Driver
# --------------------------------------------------------------------------
def build_agent(mode: str):
    from llm_client import DeepSeekClient
    from bayesian_agent import BayesianNL2SQLAgent
    from error_classifier import SQLErrorClassifier

    model_key = "RF" if mode == "BAYES_RF" else "LR"
    models = SQLErrorClassifier.load(MODEL_DIR)
    if model_key not in models:
        raise SystemExit(f"No {model_key} model in {MODEL_DIR}; "
                         f"run error_classifier.py first.")
    return BayesianNL2SQLAgent(
        client=DeepSeekClient(), classifier=models, model_key=model_key,
        enable_consistency=(mode == "BAYES_FULL"),
    )


def run_ablation(limit: int | None = None) -> None:
    with open(SPIDER_DEV_PATH, encoding="utf-8") as fh:
        dev_data = json.load(fh)
    if limit:
        dev_data = dev_data[:limit]

    is_bayes = ABLATION_MODE.startswith("BAYES")
    if is_bayes:
        agent = build_agent(ABLATION_MODE)
        client = None
    else:
        from llm_client import DeepSeekClient
        client = DeepSeekClient()
        agent = None

    total = len(dev_data)
    correct = 0
    calls_total = 0
    t0 = time.time()

    file_exists = os.path.isfile(CSV_OUT_PATH)
    with open(CSV_OUT_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()

        print(f"\nAblation run: mode [{ABLATION_MODE}], {total} queries")
        print(f"Results -> {CSV_OUT_PATH}\n")

        for i, item in enumerate(dev_data):
            db_id, question, gold_sql = item["db_id"], item["question"], item["query"]
            db_path = db_path_for(db_id, DATABASE_DIR)
            gold_ok, gold_rows = execute_sql(db_path, gold_sql)

            if is_bayes:
                out = run_bayes_mode(agent, question, db_path, db_id, gold_rows)
                is_correct = bool(out["is_correct"])
            else:
                out = run_legacy_mode(client, ABLATION_MODE, question, db_path, db_id)
                is_correct = bool(gold_ok and out["ok"]
                                  and results_match(gold_rows, out["rows"]))
                out["is_correct"] = is_correct

            calls_total += out["calls"]
            if is_correct:
                correct += 1

            writer.writerow({
                "Test_ID": i + 1, "Database": db_id, "Question": question,
                "Is_Correct": is_correct,
                "Error_Log": out["status"] if not is_correct else "Matched",
                "P_Wrong": out["p_wrong"], "Action": out["action"],
                "Calls": out["calls"],
            })
            csv_file.flush()

            if (i + 1) % 50 == 0 or (i + 1) == total:
                ex = correct / (i + 1) * 100
                print(f"  {i + 1}/{total} | EX so far {ex:.2f}% | "
                      f"calls/query {calls_total / (i + 1):.2f}")

    ex = correct / total * 100
    print("\n" + "=" * 52)
    print(f"Ablation [{ABLATION_MODE}] finished")
    print(f"Execution Accuracy : {ex:.2f}% ({correct}/{total})")
    print(f"Calls per query    : {calls_total / max(total, 1):.3f}")
    print(f"Wall clock         : {time.time() - t0:.0f}s")
    print("=" * 52)


if __name__ == "__main__":
    # Set limit=10 for a smoke test, None for the full 1034 queries.
    run_ablation(limit=None)
