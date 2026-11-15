"""
turn the labelled dataset into a feature table

    data/dataset_raw.csv  --(feature_extractor)-->  data/features_train.csv

The output keeps ``query_id`` and ``y_wrong`` next to the 45 numeric
features, so the classifier can be trained without any further joining.

Rows whose statement never executed are dropped here: they carry no SQL text
to describe, and they are handled by the reflective-repair branch rather
than by the risk model (see build_dataset.py).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd

from config import (DATABASE_DIR, DATASET_RAW_PATH, FEATURE_NAMES_PATH,
                    FEATURES_PATH)
from embeddings import get_embedder
from feature_extractor import (FEATURE_NAMES, SQLFeatureExtractor, load_db_schema)
from sql_sandbox import db_path_for


def build(kind: str = "auto", limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATASET_RAW_PATH).fillna({"generated_sql": ""})
    df["generated_sql"] = df["generated_sql"].astype(str)

    n_total = len(df)
    df = df[df["generated_sql"].str.strip().str.lower() != "n/a"]
    df = df[df["generated_sql"].str.strip() != ""].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    print(f"Rows: {n_total} in dataset_raw -> {len(df)} usable "
          f"(dropped {n_total - len(df)} non-executing statements)")

    # ---- pre-pass: load schemas and build the corpus for the TF-IDF mode
    schemas: Dict[str, object] = {}
    corpus: List[str] = []
    for db_id in df["db_id"].unique():
        schema = load_db_schema(db_path_for(db_id, DATABASE_DIR), db_id)
        schemas[db_id] = schema
        corpus.extend(schema.ddl)
    corpus.extend(df["question"].astype(str).tolist())

    embedder = get_embedder(kind, corpus)
    print(f"Embedder: {embedder.name}")
    extractor = SQLFeatureExtractor(embedder=embedder)

    rows: List[Dict] = []
    for i, rec in df.iterrows():
        f = extractor.extract(str(rec["question"]), str(rec["generated_sql"]),
                             schemas[rec["db_id"]])
        f["query_id"] = int(rec["query_id"])
        f["db_id"] = rec["db_id"]
        f["y_wrong"] = int(rec["y_wrong"])
        f["is_correct"] = int(rec["is_correct"])
        rows.append(f)
        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            print(f"  features {i + 1}/{len(df)}")

    out = pd.DataFrame(rows)[["query_id", "db_id", "y_wrong", "is_correct"]
                             + FEATURE_NAMES]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the feature table.")
    ap.add_argument("--embedder", default="auto",
                    choices=["auto", "chroma", "minilm", "tfidf"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=FEATURES_PATH)
    args = ap.parse_args()

    feats = build(args.embedder, args.limit)
    feats.to_csv(args.out, index=False)
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump(FEATURE_NAMES, fh, indent=2)
    # Provenance: which embedder produced the rag_* features.
    meta = {"embedder": get_embedder(args.embedder).name,
            "n_rows": int(len(feats)), "n_features": len(FEATURE_NAMES),
            "positive_rate": float(feats["y_wrong"].mean())}
    with open(os.path.join(os.path.dirname(args.out), "features_meta.json"),
              "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    pos = int(feats["y_wrong"].sum())
    print(f"\nSaved -> {args.out}")
    print(f"Shape  : {feats.shape[0]} rows x {len(FEATURE_NAMES)} features")
    print(f"Label  : {pos} wrong / {len(feats)} ({pos / len(feats):.2%})")
    print(f"Names  -> {FEATURE_NAMES_PATH}")


if __name__ == "__main__":
    main()
