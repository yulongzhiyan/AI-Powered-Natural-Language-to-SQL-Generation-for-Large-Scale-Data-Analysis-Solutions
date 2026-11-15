"""
Low-cost feature extraction for the NL2SQL risk model

Role in the pipeline
--------------------
    query --RAG--> retrieved DDLs --LLM--> candidate SQL
                                              |
                                    [THIS MODULE]  -> feature vector x
                                              |
                                    error_classifier -> P(Error | x)

Every feature here is computed **without any additional LLM call**, which is
the economic premise of the whole proposal: the risk signal must be cheaper
than the retry it is supposed to save.

Feature families
----------------
1. ``rag_*``  : semantic similarity between the question and the retrieved
                table DDLs.  Low similarity means schema linking was weak.
2. ``q_*``    : surface properties of the natural-language question.
3. ``sql_*``  : structural properties of the generated statement.
4. ``al_*``   : *alignment* features between question and statement, e.g.
                the question asks "how many" but the SQL has no aggregate.
                These carry most of the discriminative power for *silent*
                semantic errors, which a syntax-level check can never see.
5. ``db_*``   : database/schema complexity.
6. ``oov_*``  : schema-grounding violations (identifiers that appear in the
                SQL but exist nowhere in the database).  Direct proxy for
                hallucinated columns/tables.

The extractor never looks at the gold SQL; it only needs the question, the
retrieved schema, the candidate SQL and the full database schema.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from embeddings import BaseEmbedder, get_embedder

# --------------------------------------------------------------------------
# Small vocabularies
# --------------------------------------------------------------------------
SQL_KEYWORDS: Set[str] = {
    "select", "from", "where", "group", "by", "order", "having", "limit",
    "join", "inner", "left", "right", "full", "outer", "cross", "on", "as",
    "and", "or", "not", "in", "is", "null", "like", "between", "distinct",
    "count", "sum", "avg", "max", "min", "union", "intersect", "except",
    "all", "any", "some", "exists", "case", "when", "then", "else", "end",
    "asc", "desc", "offset", "with", "cast", "integer", "real", "text",
    "using", "natural", "values", "into", "create", "table", "primary",
    "key", "foreign", "references", "default", "unique", "check", "date",
    "from", "t", "t1", "t2", "s", "c", "p",
}

STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "is", "are", "was",
    "were", "be", "been", "what", "which", "who", "whom", "whose", "how",
    "many", "much", "all", "and", "or", "that", "this", "these", "those",
    "with", "from", "by", "as", "do", "does", "did", "give", "list", "show",
    "find", "return", "tell", "me", "please", "there", "their", "we", "us",
    "it", "its", "have", "has", "had", "than", "then", "each", "per", "ever",
    "name", "number", "total", "count",
}

AGG_WORDS = ("how many", "number of", "count", "average", "avg", "total",
             "sum", "maximum", "minimum", "highest", "lowest", "largest",
             "smallest", "most", "least", "mean")
SUPERLATIVE_WORDS = ("best", "worst", "top", "highest", "lowest", "most",
                     "least", "largest", "smallest", "maximum", "minimum",
                     "first", "last")
ORDER_WORDS = ("order", "ordered", "sort", "sorted", "ascending",
               "descending", "rank", "ranking")
GROUP_WORDS = ("each", "per", "every", "for every", "grouped", "group by",
               "respectively")
COMPARE_WORDS = ("more than", "less than", "greater", "fewer", "at least",
                 "at most", "between", "exceed", "above", "below", "over",
                 "under")
NEGATION_WORDS = ("not", "no", "never", "without", "except", "excluding",
                  "neither", "nor")
# Words that usually signal an under-specified request (the CLARIFY trigger
# discussed in the proposal: "the best sales" has no ground truth answer).
AMBIGUITY_WORDS = ("best", "good", "nice", "popular", "important",
                   "interesting", "some", "several", "many", "recent",
                   "top", "major", "significant", "reasonable")

_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STR_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_ALIAS_AS_RE = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_TABLE_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Database schema utilities
# --------------------------------------------------------------------------
@dataclass
class DbSchema:
    """Everything we need to know about one SQLite database, cheaply."""

    db_id: str
    ddl: List[str] = field(default_factory=list)          # one string per table
    table_names: List[str] = field(default_factory=list)
    identifiers: Set[str] = field(default_factory=set)    # every table + column

    @property
    def n_tables(self) -> int:
        return len(self.ddl)

    @property
    def n_columns(self) -> int:
        return len(self.identifiers) - len(self.table_names)

    def to_prompt(self, indices: Sequence[int]) -> str:
        return "\n".join(self.ddl[i] for i in indices)


_SCHEMA_CACHE: Dict[str, DbSchema] = {}


def load_db_schema(db_path: str, db_id: str | None = None) -> DbSchema:
    """Read ``sqlite_master`` once per database and cache the result."""
    db_id = db_id or os.path.splitext(os.path.basename(db_path))[0]
    if db_id in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_id]
    schema = DbSchema(db_id=db_id)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        for name, ddl in cur.fetchall():
            if not ddl:
                continue
            schema.ddl.append(ddl)
            schema.table_names.append(name.lower())
            schema.identifiers.add(name.lower())
            # Column names: identifiers inside the CREATE TABLE body.
            body = ddl[ddl.find("(") + 1:] if "(" in ddl else ddl
            for ident in _ID_RE.findall(body):
                low = ident.lower()
                if low not in SQL_KEYWORDS:
                    schema.identifiers.add(low)
    finally:
        conn.close()
    _SCHEMA_CACHE[db_id] = schema
    return schema


# --------------------------------------------------------------------------
# SQL parsing helpers
# --------------------------------------------------------------------------
def strip_literals(sql: str) -> str:
    return _STR_LITERAL_RE.sub("''", sql)


def sql_identifiers(sql: str) -> List[str]:
    """All identifier-like tokens of a statement (lower-cased, literals removed)."""
    return [t.lower() for t in _ID_RE.findall(strip_literals(sql))]


def sql_aliases(sql: str) -> Set[str]:
    """Aliases declared in the statement (``AS x`` or ``FROM t x``)."""
    aliases = {m.group(1).lower() for m in _ALIAS_AS_RE.finditer(sql)}
    for m in _TABLE_ALIAS_RE.finditer(sql):
        if m.group(2):
            cand = m.group(2).lower()
            if cand not in SQL_KEYWORDS:
                aliases.add(cand)
    return aliases


def split_identifier_words(name: str) -> List[str]:
    """``customer_id`` -> ['customer', 'id'] so lexical matching works."""
    parts = re.split(r"[^A-Za-z0-9]+", name)
    out: List[str] = []
    for p in parts:
        # snake_case already split; also split simple camelCase
        out.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p) or [p])
    return [w.lower() for w in out if w]


def _word_pattern(word: str) -> "re.Pattern[str]":
    """Regex with word boundaries so that 'top' does not match 'stop'."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])")


_PATTERN_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _contains_any(text: str, words: Sequence[str]) -> int:
    low = text.lower()
    for w in words:
        pat = _PATTERN_CACHE.get(w)
        if pat is None:
            pat = _PATTERN_CACHE[w] = _word_pattern(w)
        if pat.search(low):
            return 1
    return 0


def _count_any(text: str, words: Sequence[str]) -> int:
    low = text.lower()
    total = 0
    for w in words:
        pat = _PATTERN_CACHE.get(w)
        if pat is None:
            pat = _PATTERN_CACHE[w] = _word_pattern(w)
        total += len(pat.findall(low))
    return total


# --------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # 1. RAG evidence
    "rag_max_sim", "rag_mean_sim", "rag_std_sim", "rag_gap_1_2",
    "n_tables_retrieved",
    # 2. question surface
    "q_n_tokens", "q_n_chars", "q_n_digits",
    "q_agg_word", "q_superlative", "q_order_word", "q_group_word",
    "q_compare_word", "q_negation", "q_ambiguity", "q_ambiguity_count",
    # 3. SQL structure
    "sql_n_tokens", "sql_n_chars", "sql_n_joins", "sql_n_subselect",
    "sql_n_agg", "sql_n_groupby", "sql_n_orderby", "sql_has_having",
    "sql_has_distinct", "sql_has_limit", "sql_has_setop", "sql_n_where",
    "sql_n_select_cols", "sql_n_tables", "sql_has_star", "sql_n_literals",
    # 4. question <-> SQL / schema alignment
    "al_agg_missing", "al_agg_extra", "al_order_missing", "al_group_missing",
    "al_limit_missing", "al_literal_gap", "al_q_schema_coverage",
    "al_q_sql_coverage", "al_q_retrieved_coverage",
    # 5. schema complexity
    "db_n_tables", "db_n_columns", "retrieved_char_len",
    "sql_schema_len_ratio",
    # 6. schema grounding
    "oov_identifier_rate", "n_oov_identifiers",
]


class SQLFeatureExtractor:
    """Turn ``(question, retrieved schema, SQL)`` into a numeric feature dict."""

    def __init__(self, embedder: BaseEmbedder | None = None, top_k: int = 3):
        self.embedder = embedder or get_embedder("auto")
        self.top_k = top_k
        self._ddl_vec_cache: Dict[str, np.ndarray] = {}

    # -- RAG similarity ---------------------------------------------------
    def _retrieval(self, query: str, schema: DbSchema) -> Tuple[List[int], np.ndarray]:
        """Top-k table indices + similarity vector (cosine over DDL embeddings)."""
        if schema.db_id not in self._ddl_vec_cache:
            self._ddl_vec_cache[schema.db_id] = self.embedder.encode(schema.ddl)
        ddl_vecs = self._ddl_vec_cache[schema.db_id]
        q_vec = self.embedder.encode([query])[0]
        sims = ddl_vecs @ q_vec
        order = np.argsort(-sims)[: min(self.top_k, len(schema.ddl))]
        return list(order), sims

    # -- main API ---------------------------------------------------------
    def extract(
        self,
        query: str,
        generated_sql: str,
        schema: DbSchema,
        retrieved_indices: Sequence[int] | None = None,
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {}

        # ---------- 1. RAG evidence ----------
        if retrieved_indices is None:
            retrieved_indices, sims = self._retrieval(query, schema)
        else:
            _, sims = self._retrieval(query, schema)
        top = np.sort(np.asarray([sims[i] for i in retrieved_indices]))[::-1]
        feats["rag_max_sim"] = float(top[0]) if len(top) else 0.0
        feats["rag_mean_sim"] = float(top.mean()) if len(top) else 0.0
        feats["rag_std_sim"] = float(top.std()) if len(top) > 1 else 0.0
        feats["rag_gap_1_2"] = float(top[0] - top[1]) if len(top) > 1 else 0.0
        feats["n_tables_retrieved"] = float(len(retrieved_indices))

        # ---------- 2. Question surface ----------
        q_low = query.lower()
        q_tokens = re.findall(r"[A-Za-z0-9_]+", q_low)
        feats["q_n_tokens"] = float(len(q_tokens))
        feats["q_n_chars"] = float(len(query))
        feats["q_n_digits"] = float(len(_NUM_RE.findall(query)))
        feats["q_agg_word"] = float(_contains_any(q_low, AGG_WORDS))
        feats["q_superlative"] = float(_contains_any(q_low, SUPERLATIVE_WORDS))
        feats["q_order_word"] = float(_contains_any(q_low, ORDER_WORDS))
        feats["q_group_word"] = float(_contains_any(q_low, GROUP_WORDS))
        feats["q_compare_word"] = float(_contains_any(q_low, COMPARE_WORDS))
        feats["q_negation"] = float(_contains_any(q_low, NEGATION_WORDS))
        feats["q_ambiguity"] = float(_contains_any(q_low, AMBIGUITY_WORDS))
        feats["q_ambiguity_count"] = float(_count_any(q_low, AMBIGUITY_WORDS))

        # ---------- 3. SQL structure ----------
        sql = generated_sql or ""
        up = sql.upper()
        sql_tokens = re.findall(r"[A-Za-z0-9_]+", sql)
        feats["sql_n_tokens"] = float(len(sql_tokens))
        feats["sql_n_chars"] = float(len(sql))
        feats["sql_n_joins"] = float(up.count("JOIN"))
        feats["sql_n_subselect"] = float(max(0, up.count("SELECT") - 1))
        feats["sql_n_agg"] = float(sum(up.count(f) for f in
                                       ("COUNT(", "SUM(", "AVG(", "MAX(", "MIN(")))
        feats["sql_n_groupby"] = float(up.count("GROUP BY"))
        feats["sql_n_orderby"] = float(up.count("ORDER BY"))
        feats["sql_has_having"] = float("HAVING" in up)
        feats["sql_has_distinct"] = float("DISTINCT" in up)
        feats["sql_has_limit"] = float("LIMIT" in up)
        feats["sql_has_setop"] = float(any(k in up for k in
                                           ("UNION", "INTERSECT", "EXCEPT")))
        feats["sql_n_where"] = float(up.count("WHERE"))
        feats["sql_has_star"] = float("*" in sql)
        feats["sql_n_literals"] = float(len(_NUM_RE.findall(strip_literals(sql)))
                                        + len(_STR_LITERAL_RE.findall(sql)))
        # number of distinct tables referenced (after FROM / JOIN)
        feats["sql_n_tables"] = float(len({
            m.group(1).lower() for m in _TABLE_ALIAS_RE.finditer(sql)}))
        # select-list width (columns between SELECT and FROM)
        head = sql[: up.find(" FROM ") if " FROM " in up else len(sql)]
        head = re.sub(r"^\s*SELECT\s+(DISTINCT\s+)?", "", head, flags=re.I)
        feats["sql_n_select_cols"] = float(max(1, len(head.split(","))))

        # ---------- 4. Alignment question <-> SQL ----------
        wants_agg = feats["q_agg_word"] > 0
        has_agg = feats["sql_n_agg"] > 0 or feats["sql_n_groupby"] > 0
        feats["al_agg_missing"] = float(wants_agg and not has_agg)
        feats["al_agg_extra"] = float(has_agg and not wants_agg)
        feats["al_order_missing"] = float(feats["q_order_word"] > 0
                                          and feats["sql_n_orderby"] == 0)
        feats["al_group_missing"] = float(feats["q_group_word"] > 0
                                          and feats["sql_n_groupby"] == 0
                                          and feats["sql_n_agg"] > 0)
        # "top 3" without LIMIT is a classic silent error
        top_n = re.search(r"\b(top|first|last)\s+(\d+)\b", q_low)
        feats["al_limit_missing"] = float(bool(top_n) and not feats["sql_has_limit"])
        feats["al_literal_gap"] = float(abs(feats["q_n_digits"]
                                            - feats["sql_n_literals"]))
        # lexical grounding of the question in the database vocabulary
        q_terms = [t for t in q_tokens if t not in STOPWORDS and len(t) > 2]
        schema_words: Set[str] = set()
        for ident in schema.identifiers:
            schema_words.update(split_identifier_words(ident))
        feats["al_q_schema_coverage"] = (
            float(np.mean([t in schema_words for t in q_terms])) if q_terms else 0.0
        )
        sql_words = set(split_identifier_words(strip_literals(sql)))
        feats["al_q_sql_coverage"] = (
            float(np.mean([t in sql_words for t in q_terms])) if q_terms else 0.0
        )
        retr_words: Set[str] = set()
        for i in retrieved_indices:
            retr_words.update(split_identifier_words(schema.ddl[i]))
        feats["al_q_retrieved_coverage"] = (
            float(np.mean([t in retr_words for t in q_terms])) if q_terms else 0.0
        )

        # ---------- 5. Schema complexity ----------
        feats["db_n_tables"] = float(schema.n_tables)
        feats["db_n_columns"] = float(max(0, schema.n_columns))
        retr_len = sum(len(schema.ddl[i]) for i in retrieved_indices)
        feats["retrieved_char_len"] = float(retr_len)
        feats["sql_schema_len_ratio"] = float(len(sql) / max(retr_len, 1))

        # ---------- 6. Schema grounding (hallucination proxy) ----------
        allowed = set(schema.identifiers) | sql_aliases(sql) | SQL_KEYWORDS
        idents = [i for i in sql_identifiers(sql) if not i.isdigit()]
        oov = [i for i in idents if i not in allowed]
        feats["n_oov_identifiers"] = float(len(oov))
        feats["oov_identifier_rate"] = float(len(oov) / max(len(idents), 1))

        return feats

    # -- batch API --------------------------------------------------------
    def extract_batch_dicts(self, samples: Sequence[Dict]) -> List[Dict[str, float]]:
        """``samples``: dicts with keys question / sql / db_id (+ optional schema)."""
        out: List[Dict[str, float]] = []
        for s in samples:
            schema = s.get("schema") or load_db_schema(s["db_path"], s["db_id"])
            out.append(self.extract(s["question"], s["sql"], schema))
        return out

    @staticmethod
    def to_matrix(rows: Sequence[Dict[str, float]]) -> np.ndarray:
        return np.asarray([[r.get(k, 0.0) for k in FEATURE_NAMES] for r in rows],
                          dtype=np.float64)


if __name__ == "__main__":
    # Smoke test on a single well-known Spider question.
    from config import DATABASE_DIR
    db = "concert_singer"
    schema = load_db_schema(os.path.join(DATABASE_DIR, db, f"{db}.sqlite"), db)
    ext = SQLFeatureExtractor()
    f = ext.extract("How many singers do we have?", "SELECT count(*) FROM singer", schema)
    for k in FEATURE_NAMES:
        print(f"{k:26s} {f[k]:.4f}")
