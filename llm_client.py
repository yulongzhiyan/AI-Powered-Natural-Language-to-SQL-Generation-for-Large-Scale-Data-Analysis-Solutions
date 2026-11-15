"""
Thin LLM wrapper with two interchangeable back-ends.

``DeepSeekClient(Original project_wcx)``
    The real thing (same model / base_url as the original project).

``ReplayClient``
    Offline back-end used for (a) smoke-testing the agent without paying for
    API calls and (b) deterministic unit tests.  It replays an SQL statement
    that was recorded earlier for the same question.

Both expose the same two methods so ``bayesian_agent.py`` never needs to
know which one it is talking to:

    generate_sql(question, schema_text, temperature) -> (sql, usage)
    repair_sql(question, schema_text, bad_sql, error, temperature) -> (sql, usage)
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

from config import (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME,
                    TEMPERATURE_GREEDY)
from sql_sandbox import estimate_tokens

GENERATION_PROMPT = """Convert the user's question into a SQL query.
    [Schema]: {schema}
    [Question]: {question}
    Output ONLY raw SQL. No markdown, no explanations."""

REPAIR_PROMPT = """The SQL below failed in the database with this error:
    {error}

    [SQL]: {sql}
    [Schema]: {schema}
    [Question]: {question}
    Output ONLY the corrected raw SQL. No markdown, no explanations."""


def clean_sql(text: str) -> str:
    """Strip markdown fences and stray prose that LLMs like to add."""
    if not text:
        return ""
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    text = text.strip()
    # keep everything up to the last semicolon if a chatty suffix slipped in
    if ";" in text:
        head, _, _ = text.partition(";")
        text = head + ";"
    return text.strip()


class DeepSeekClient:
    """OpenAI-compatible client pointed at DeepSeek."""

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL_NAME):
        from openai import OpenAI

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise RuntimeError(
                "No usable DEEPSEEK_API_KEY. Export a valid key, e.g.\n"
                "    set DEEPSEEK_API_KEY=sk-...   (Windows cmd)\n"
                "    export DEEPSEEK_API_KEY=sk-... (bash)"
            )
        self.model = model
        self._client = OpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- internal ---------------------------------------------------------
    def _chat(self, messages, temperature: float) -> Tuple[str, Dict]:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        self.calls += 1
        usage = getattr(resp, "usage", None)
        p_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        c_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        if not p_tok:   # fall back to a crude estimate
            p_tok = sum(estimate_tokens(m["content"]) for m in messages)
        if not c_tok:
            c_tok = estimate_tokens(resp.choices[0].message.content or "")
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        return clean_sql(resp.choices[0].message.content or ""), {
            "prompt_tokens": p_tok, "completion_tokens": c_tok,
            "total_tokens": p_tok + c_tok, "calls": 1,
        }

    # -- public API -------------------------------------------------------
    def generate_sql(self, question: str, schema_text: str,
                     temperature: float = TEMPERATURE_GREEDY) -> Tuple[str, Dict]:
        messages = [
            {"role": "system", "content": "You are a highly accurate Text-to-SQL assistant."},
            {"role": "user", "content": GENERATION_PROMPT.format(
                schema=schema_text, question=question)},
        ]
        return self._chat(messages, temperature)

    def repair_sql(self, question: str, schema_text: str, bad_sql: str,
                   error: str, temperature: float = TEMPERATURE_GREEDY):
        messages = [
            {"role": "system", "content": "You are a highly accurate Text-to-SQL assistant."},
            {"role": "user", "content": GENERATION_PROMPT.format(
                schema=schema_text, question=question)},
            {"role": "assistant", "content": bad_sql},
            {"role": "user", "content": REPAIR_PROMPT.format(
                error=error, sql=bad_sql, schema=schema_text, question=question)},
        ]
        return self._chat(messages, temperature)

    def stats(self) -> Dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens}


class ReplayClient:
    """Deterministic fake LLM: returns pre-recorded SQL for a question.

    Used by ``--mock`` runs.  It makes the *whole* agent loop (retrieval ->
    generation -> risk scoring -> decision -> sandbox -> logging) executable
    without network access, which is what we need to validate the wiring and
    to reproduce the decision trace offline.
    """

    def __init__(self, table: Dict[str, str]):
        self.table = {k.strip().lower(): v for k, v in table.items()}
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _serve(self, question: str) -> Tuple[str, Dict]:
        self.calls += 1
        sql = self.table.get(question.strip().lower(), "")
        tok = estimate_tokens(sql)
        self.prompt_tokens += 40
        self.completion_tokens += tok
        return sql, {"prompt_tokens": 40, "completion_tokens": tok,
                     "total_tokens": 40 + tok, "calls": 1}

    def generate_sql(self, question: str, schema_text: str,
                     temperature: float = TEMPERATURE_GREEDY):
        return self._serve(question)

    def repair_sql(self, question: str, schema_text: str, bad_sql: str,
                   error: str, temperature: float = TEMPERATURE_GREEDY):
        # Replaying a logged "final" statement: the repair step returns it too.
        return self._serve(question)

    def stats(self) -> Dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens}


def build_client(mode: str = "api", replay_table: Optional[Dict[str, str]] = None):
    """``mode`` in {'api', 'mock'}."""
    if mode == "mock":
        return ReplayClient(replay_table or {})
    return DeepSeekClient()
