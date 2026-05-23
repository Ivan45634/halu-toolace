"""Shared utilities, judges and prompts used by the audit / recover / patch scripts.

Centralizes:
  - JSONL i/o
  - text helpers (truncate, collect_tool_names, derive_type_hint)
  - span localization (find_span, fuzzy match)
  - JSON parsing (JSON_RE, parse_obj)
  - judge backends (LocalJudge, OpenRouterJudge, load_judge)
  - prompt templates used across multiple scripts
"""

from __future__ import annotations

import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------------
# JSONL i/o
# ----------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------


def truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def collect_tool_names(meta: dict) -> str:
    tools = (meta or {}).get("tools") or []
    names: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    if not names:
        return "(unspecified)"
    return ", ".join(names[:12])


def derive_type_hint(reasoning: str) -> str:
    """Heuristic type guess from a judge's free-text reasoning."""
    r = (reasoning or "").lower()
    if "contradict" in r or "conflict" in r:
        return "contradiction"
    if "missing tool" in r or "not available" in r or "no such tool" in r or "tool not in" in r:
        return "missing_tool"
    return "overgeneration"


# ----------------------------------------------------------------------------
# Span localization
# ----------------------------------------------------------------------------


def find_span(output: str, snippet: str) -> tuple[int, int] | None:
    """Locate ``snippet`` in ``output``. Tries exact, case-insensitive, then a fuzzy window."""
    snippet = (snippet or "").strip().strip('"\'')
    snippet = re.sub(r"\s*\.\.\.\s*\[truncated\]\s*$", "", snippet)
    snippet = re.sub(r"\s*\.\.\.\s*$", "", snippet)
    if not snippet or len(snippet) < 3:
        return None
    idx = output.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    idx = output.lower().find(snippet.lower())
    if idx >= 0:
        return idx, idx + len(snippet)
    sn = snippet.lower()
    win = max(20, len(sn))
    out_lower = output.lower()
    step = max(1, win // 4)
    best = (0.0, -1, -1)
    for i in range(0, max(1, len(out_lower) - win + 1), step):
        cand = out_lower[i:i + win]
        ratio = SequenceMatcher(None, cand, sn).ratio()
        if ratio > best[0]:
            best = (ratio, i, i + win)
    if best[0] >= 0.75 and best[1] >= 0:
        match = SequenceMatcher(None, output[best[1]:best[2]], snippet)
        m = match.find_longest_match(0, best[2] - best[1], 0, len(snippet))
        if m.size >= max(8, int(len(snippet) * 0.6)):
            start = best[1] + m.a
            return start, start + m.size
    return None


# ----------------------------------------------------------------------------
# JSON parsing
# ----------------------------------------------------------------------------


JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object found in ``raw``. Returns ``{"_parse_error": ...}`` on failure."""
    raw = (raw or "").strip()
    if not raw:
        return {"_parse_error": "empty"}
    for candidate in [raw] + JSON_RE.findall(raw):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {"_parse_error": "no_json_found"}


# ----------------------------------------------------------------------------
# Judges
# ----------------------------------------------------------------------------


DEFAULT_MAX_NEW_TOKENS = 220


class LocalJudge:
    """Transformers-backed judge on MPS / CPU."""

    def __init__(self, model_id: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device).eval()
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    def __call__(self, system: str, user: str) -> str:
        import torch
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=6144).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tok.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


class OpenRouterJudge:
    """OpenAI-compatible judge proxied through OpenRouter."""

    def __init__(self, model_id: str, env_path: str = "~/.openrouter.env",
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        import os
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(os.path.expanduser(env_path))
        self.client = OpenAI(
            base_url=os.environ["OPENROUTER_BASE_URL"],
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.model_id = model_id
        self.device = "openrouter"
        self.max_new_tokens = max_new_tokens

    def __call__(self, system: str, user: str) -> str:
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=self.max_new_tokens,
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= 3:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"openrouter judge failed after retries: {last_err}")


def load_judge(model_id: str, backend: str = "local"):
    if backend == "openrouter":
        return OpenRouterJudge(model_id)
    return LocalJudge(model_id)


# ----------------------------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------------------------


AUDIT_SYSTEM = (
    "You are an expert annotator for hallucination detection in tool-using AI assistants. "
    "You will receive: USER QUERY, TOOL CONTEXT (the tool's structured response), AVAILABLE TOOLS, "
    "ASSISTANT FINAL ANSWER, and optionally a CANDIDATE HALLUCINATED SPAN already flagged by an "
    "automated pipeline. Decide whether the candidate span is genuinely a hallucination AND "
    "whether the answer contains any OTHER hallucinated phrase that was not labeled.\n\n"
    "DEFINITIONS:\n"
    "  contradiction  - the answer states a fact that directly conflicts with a value in the tool context.\n"
    "  overgeneration - the answer adds substantive claims (recommendations, trends, comparisons, "
    "industry consensus, satisfaction reports, historical stability, etc.) that are NOT supported "
    "by anything in the tool context. Generic praise or unverifiable forecasts COUNT as overgeneration.\n"
    "  missing_tool   - the answer offers to perform an action (book a flight, send an email, "
    "translate, generate an image, reserve, etc.) that requires a tool NOT in the AVAILABLE TOOLS list.\n\n"
    "EXAMPLES OF HALLUCINATIONS:\n"
    "  Tool context says weather is 'sunny'. Answer says 'rainy'. -> contradiction.\n"
    "  Tool context is a Facebook insights dump. Answer adds 'It is highly recommended by most users.' -> overgeneration.\n"
    "  Available tools: [Page Info, Search]. Answer says 'I can also send them an email now.' -> missing_tool.\n\n"
    "If a CANDIDATE SPAN is provided, set label_correct=true when the span really is a "
    "hallucination of the stated type; set false when the span is harmless (paraphrase, "
    "supported by context, or just a code/identifier); set unknown only if the context truly "
    "doesn't let you decide.\n\n"
    "Reply with ONE JSON object on a single line, no markdown, no code fences:\n"
    '{"label_correct": <true|false|"unknown">, '
    '"extra_hallucination": <true|false>, '
    '"extra_text": "<verbatim substring from the answer, or empty>", '
    '"reasoning": "<one short sentence>"}'
)


AUDIT_USER_TEMPLATE = (
    "USER QUERY:\n{query}\n\n"
    "TOOL CONTEXT (truncated):\n{context}\n\n"
    "AVAILABLE TOOLS: {tool_names}\n\n"
    "ASSISTANT FINAL ANSWER (truncated):\n{output}\n\n"
    "CANDIDATE HALLUCINATED SPAN:\n{candidate}\n\n"
    "Respond now with the JSON object only."
)


CONFIRM_SYSTEM = (
    "You are validating a hallucination annotation. You will receive: the user query, "
    "the tool's structured response (context), the available tools, the assistant's "
    "final answer, and a CANDIDATE SPAN that another model thinks is a hallucination. "
    "Decide:\n"
    "  1. Is the candidate span really a hallucination?\n"
    "  2. If yes, classify its type: contradiction / overgeneration / missing_tool.\n"
    "Definitions:\n"
    "  contradiction  : the span contradicts a specific value from the tool context.\n"
    "  overgeneration : the span adds claims that are NOT supported by the tool context.\n"
    "  missing_tool   : the span offers to perform an action requiring a tool not in the "
    "                   available list.\n\n"
    "Reply with ONE JSON object on a single line, no markdown:\n"
    '{"is_hallucination": <true|false>, '
    '"type": <"contradiction"|"overgeneration"|"missing_tool"|"unknown">, '
    '"reasoning": "<one short sentence>"}'
)


CONFIRM_USER_TEMPLATE = (
    "USER QUERY:\n{query}\n\n"
    "TOOL CONTEXT (truncated):\n{context}\n\n"
    "AVAILABLE TOOLS: {tool_names}\n\n"
    "ASSISTANT FINAL ANSWER (truncated):\n{output}\n\n"
    "CANDIDATE HALLUCINATED SPAN:\n  text = {span!r}\n  type previously suspected: {hint!r}\n\n"
    "Respond with the JSON now."
)


RECLASSIFY_SYSTEM = (
    "You are validating a candidate hallucination span in a tool-using assistant's answer. "
    "Previously another judge rejected this span under a strict 3-type taxonomy. You will "
    "decide using an EXTENDED taxonomy that adds an `off_topic` category for cases where the "
    "answer drifts away from the user's request even though the data inside the span may be "
    "technically grounded in the tool context.\n\n"
    "Categories:\n"
    "  contradiction    - the span contradicts a value in the tool context.\n"
    "  overgeneration   - the span adds claims that are not supported by the tool context.\n"
    "  missing_tool     - the span proposes an action requiring a tool not in the available list.\n"
    "  off_topic       - the span (or the answer surrounding it) does not address the user's "
    "                     query, even though it may quote tool data.\n"
    "  not_hallucination - the span is fine; no real issue.\n\n"
    "Reply with ONE JSON object on a single line, no markdown:\n"
    '{"category": "<contradiction|overgeneration|missing_tool|off_topic|not_hallucination>", '
    '"reasoning": "<one short sentence>"}'
)


RECLASSIFY_USER_TEMPLATE = (
    "USER QUERY:\n{query}\n\n"
    "TOOL CONTEXT (truncated):\n{context}\n\n"
    "AVAILABLE TOOLS: {tool_names}\n\n"
    "ASSISTANT FINAL ANSWER (truncated):\n{output}\n\n"
    "CANDIDATE SPAN:\n  text = {span!r}\n\n"
    "Respond with the JSON object now."
)
