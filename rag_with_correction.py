"""
Interactive entry point: RAG + self-correction + Bayesian risk gate

This file keeps the original demo behaviour (single SQLite file, console
loop) and adds the new decision layer as an *optional* stage.  The legacy
code path is preserved verbatim because the thesis baseline has to stay
reproducible -- set ``NL2SQL_RISK_LAYER=off`` (or pass ``--legacy``) to run
exactly the original pipeline.

New stage in the pipeline
-------------------------
    retrieve schema -> generate SQL -> sandbox execution
      -> (success) risk model P(wrong|x) -> Bayesian decision
           EXECUTE : return the result
           RETRY   : K-sample consistency check, then decide again
           CLARIFY : ask the user to disambiguate instead of guessing
"""

from __future__ import annotations

import argparse
import os
import sys

import chromadb

from config import (DATABASE_DIR, DEEPSEEK_API_KEY, MAX_REFLECTION_RETRIES,
                    MODEL_DIR, TEMPERATURE_GREEDY)
from sql_sandbox import execute_sql, results_match

# --------------------------------------------------------------------------
# Legacy configuration (kept for backward compatibility)
# --------------------------------------------------------------------------
os.environ.setdefault("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="schema_collection")


def retrieve_relevant_schema(query, top_k=2):
    """Retrieve the most relevant table schemas using RAG (legacy helper)."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return "\n".join(results["documents"][0])


def execute_sql_in_sandbox(db_path, sql_query):
    """Legacy wrapper kept so external callers do not break."""
    return execute_sql(db_path, sql_query)


# --------------------------------------------------------------------------
# Legacy pipeline (unchanged semantics)
# --------------------------------------------------------------------------
def generate_and_correct_sql(query, db_path, max_retries=MAX_REFLECTION_RETRIES):
    """Generate SQL -> execute -> reflect on the error log -> retry."""
    from llm_client import DeepSeekClient

    client = DeepSeekClient()
    retrieved_schema = retrieve_relevant_schema(query)
    sql, _ = client.generate_sql(query, retrieved_schema, TEMPERATURE_GREEDY)

    for attempt in range(1, max_retries + 1):
        print(f"\n--- [Attempt {attempt}/{max_retries}] ---")
        print(f"[Generated SQL]: {sql}")
        ok, out = execute_sql(db_path, sql)
        if ok:
            print("[Success]: SQL executed without errors.")
            return sql, out
        print(f"[Error Caught]: {out}")
        if attempt < max_retries:
            print("[Self-Correction]: sending the error log back to the LLM...")
            sql, _ = client.repair_sql(query, retrieved_schema, sql, str(out),
                                       TEMPERATURE_GREEDY)
    print("[Failed]: maximum retries reached.")
    return None, out


# --------------------------------------------------------------------------
# New pipeline: the same loop plus the risk gate
# --------------------------------------------------------------------------
def build_agent(db_id: str | None = None, model_key: str = "LR"):
    """Load the risk stack; return ``None`` when the models are absent."""
    if not os.path.isfile(os.path.join(MODEL_DIR, "lr_risk_model.joblib")):
        print("[risk layer] no model in ./models - run error_classifier.py first.")
        return None
    from bayesian_agent import BayesianNL2SQLAgent
    from error_classifier import SQLErrorClassifier
    from llm_client import DeepSeekClient

    models = SQLErrorClassifier.load(MODEL_DIR)
    return BayesianNL2SQLAgent(client=DeepSeekClient(), classifier=models,
                               model_key=model_key)


def answer(query: str, db_path: str, db_id: str | None = None,
           model_key: str = "LR", legacy: bool = False):
    db_id = db_id or os.path.splitext(os.path.basename(db_path))[0]
    agent = None if legacy else build_agent(db_id, model_key)
    if agent is None:
        sql, data = generate_and_correct_sql(query, db_path)
        return {"status": "answered" if sql else "exec_failed", "sql": sql,
                "data": data, "p_wrong": None, "action_path": ["legacy"]}

    result = agent.run(query, db_id, db_path)
    return {"status": result.status, "sql": result.final_sql,
            "data": None, "p_wrong": result.p_wrong_final,
            "action_path": result.action_path, "calls": result.n_calls,
            "latency_s": result.latency_s}


def main() -> None:
    ap = argparse.ArgumentParser(description="NL2SQL agent (RAG + risk gate)")
    ap.add_argument("--db", default=os.path.join(DATABASE_DIR, "school.sqlite"))
    ap.add_argument("--model", default="LR", choices=["LR", "RF"])
    ap.add_argument("--legacy", action="store_true",
                    help="disable the risk layer (original behaviour)")
    args = ap.parse_args()

    legacy = args.legacy or os.environ.get("NL2SQL_RISK_LAYER", "on").lower() == "off"
    print("NL2SQL agent - RAG + self-correction"
          + ("" if legacy else " + Bayesian risk gate"))
    while True:
        try:
            query = input("\nUser Query: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break

        out = answer(query, args.db, model_key=args.model, legacy=legacy)
        print("\n" + "=" * 56)
        print(f"status   : {out['status']}")
        print(f"trace    : {' -> '.join(out['action_path'])}")
        if out["p_wrong"] is not None:
            print(f"P(wrong) : {out['p_wrong']:.3f}")
        print(f"SQL      : {out['sql']}")
        print("=" * 56)


if __name__ == "__main__":
    main()
