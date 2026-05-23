"""Span-level hallucination corruptors for ToolACE dialogues.

Three corruption types per the task spec:
  - contradiction : replace a grounded value in the output with a plausible-but-wrong one.
  - overgeneration: insert content that is not present in the tool output.
  - missing_tool  : suggest invoking a tool that is not in the available tools list.

Design choices (v2):
  - Contradiction spans must be >= MIN_CONTRADICTION_LEN chars; 1-3 char spans are skipped
    because they collapse to a single token and produce a trivial detection signal.
  - Overgeneration and missing_tool are inserted at randomized sentence boundaries
    (not always at the end), reducing positional bias.
  - Plausible-wrong contradiction values are sampled from a cross-record pool of grounded
    values so the replacement is the same TYPE as the original (city -> city, date -> date)
    instead of a deterministic +N shift that any tokenizer can spot.
  - A clean variant with empty hallucination_labels is emitted alongside the corrupted
    ones so models cannot exploit a "every example has a span" prior.
"""

import copy
import json
import random
import re
from typing import Any, Iterable

MIN_CONTRADICTION_LEN = 4
MIN_OUTPUT_LEN = 80
MAX_INSERTION_DOC_CHARS = 4000

STATUS_SWAPS = {
    "sunny": "rainy", "rainy": "sunny", "clear": "stormy", "cloudy": "clear",
    "available": "unavailable", "unavailable": "available",
    "open": "closed", "closed": "open", "active": "inactive", "inactive": "active",
    "enabled": "disabled", "disabled": "enabled",
    "approved": "rejected", "rejected": "approved",
    "success": "failure", "successful": "failed",
    "valid": "invalid", "invalid": "valid",
    "positive": "negative", "negative": "positive",
    "yes": "no", "no": "yes",
    "increasing": "decreasing", "decreasing": "increasing",
    "high": "low", "low": "high",
}

# Multi-word entities preferred over single-word for higher information content.
ENTITY_SWAPS = {
    "Moscow": "London", "London": "Moscow",
    "Paris": "Berlin", "Berlin": "Paris",
    "Tokyo": "Seoul", "Seoul": "Tokyo",
    "Beijing": "Shanghai", "Shanghai": "Beijing",
    "New York": "Los Angeles", "Los Angeles": "New York",
    "San Francisco": "Chicago", "Chicago": "San Francisco",
    "USD": "EUR", "EUR": "USD", "GBP": "JPY", "JPY": "GBP",
    "AAPL": "MSFT", "MSFT": "AAPL", "ETH": "BTC", "BTC": "ETH",
    "India": "Canada", "Canada": "India",
}

# Plausible-sounding overgeneration sentences that aren't tied to numbers
# present in the tool output — picked so a recall-only model can't trivially
# match them via missing-from-context heuristic alone.
OVERGENERATION_TEMPLATES = [
    "Based on similar requests, this tends to remain stable across the next reporting cycle.",
    "It is highly recommended by most users in this category.",
    "Historical trends suggest the values rarely change by more than a few percent week over week.",
    "This option is widely regarded as the most reliable choice in the market.",
    "Additionally, this aligns with the broader industry consensus on the topic.",
    "Customers in the same segment generally report higher satisfaction with this configuration.",
    "Long-term observations show this pattern has held for several years.",
    "Comparable services in the region have shown similar behavior recently.",
]

# Tool actions that imply a tool not present in the available toolset. The
# names are deliberately distinct from common ToolACE function names so the
# "missing_tool" suggestion never accidentally matches an available tool.
MISSING_TOOL_ACTIONS = [
    ("FlightBooking_API", "Would you like me to book a flight for you?"),
    ("Payment_API", "I can also process the payment on your behalf."),
    ("Email_API", "I can draft and send them an email right now."),
    ("Calendar_API", "I can add a calendar event for this directly."),
    ("Reservation_API", "I can reserve a table at the restaurant for you."),
    ("Translation_API", "I can translate this into another language if you need."),
    ("ImageGeneration_API", "I can generate an illustration to go with this."),
    ("SMS_API", "I can send a text-message reminder shortly before the event."),
    ("Maps_API", "I can pull up driving directions for you as well."),
]


def make_corruptions(
    clean_example: dict[str, Any],
    include_clean: bool = True,
    grounded_pool: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return up to four records per base example: clean + 3 corruption variants.

    ``grounded_pool`` is a flat list of grounded values aggregated across the whole
    corpus; it is used to source same-type replacement values for contradictions
    (e.g. a city is replaced by another city seen elsewhere in ToolACE).
    """
    records: list[dict[str, Any]] = []
    if len(clean_example.get("clean_output", "")) < MIN_OUTPUT_LEN:
        return records

    if include_clean:
        records.append(_clean_record(clean_example))

    for corruptor in (contradiction, overgeneration, missing_tool):
        rec = corruptor(clean_example, grounded_pool=grounded_pool)
        if rec is not None:
            records.append(rec)
    return records


def _clean_record(clean_example: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{clean_example['id']}__clean",
        "query": clean_example["query"],
        "context": clean_example["context"],
        "output": clean_example["clean_output"],
        "hallucination_labels": [],
        "meta": _base_meta(clean_example, "clean"),
    }


# Contradiction ----------------------------------------------------------------


def contradiction(
    clean_example: dict[str, Any],
    grounded_pool: list[str] | None = None,
) -> dict[str, Any] | None:
    output = clean_example["clean_output"]
    rng = _rng(clean_example["id"], "contradiction")
    candidates = _contradiction_candidates(
        output=output,
        context=clean_example["context"],
        rng=rng,
        grounded_pool=grounded_pool,
    )
    candidates = [c for c in candidates if len(c[1]) >= MIN_CONTRADICTION_LEN]
    if not candidates:
        return None
    original, replacement, start, end = rng.choice(candidates)
    corrupted = output[:start] + replacement + output[end:]
    return _record(
        clean_example=clean_example,
        corruption_type="contradiction",
        output=corrupted,
        start=start,
        bad_span=replacement,
        extra_meta={"original": original},
    )


def _contradiction_candidates(
    output: str,
    context: str,
    rng: random.Random,
    grounded_pool: list[str] | None,
) -> list[tuple[str, str, int, int]]:
    grounded_values = _grounded_values(context)
    candidates: list[tuple[str, str, int, int]] = []
    seen: set[tuple[int, int, str]] = set()

    for original, replacement in _candidate_replacements(output, grounded_values, grounded_pool, rng):
        if original == replacement or len(original.strip()) < MIN_CONTRADICTION_LEN:
            continue
        for match in _find_token(output, original):
            key = (match.start(), match.end(), replacement)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((original, replacement, match.start(), match.end()))
    return candidates


def _candidate_replacements(
    output: str,
    grounded_values: set[str],
    grounded_pool: list[str] | None,
    rng: random.Random,
) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    pool = grounded_pool or []

    # Prefer multi-word/long grounded values first.
    for value in sorted(grounded_values, key=len, reverse=True):
        replacements.extend(_value_replacements(value, pool, rng))

    for word, replacement in STATUS_SWAPS.items():
        if len(word) >= MIN_CONTRADICTION_LEN:
            replacements.append((word, replacement))

    for entity, replacement in ENTITY_SWAPS.items():
        if len(entity) >= MIN_CONTRADICTION_LEN:
            replacements.append((entity, replacement))

    for match in re.finditer(r"\b\d{1,2}:\d{2}\b", output):
        replacements.append((match.group(0), _shift_time(match.group(0))))

    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", output):
        replacements.append((match.group(0), _shift_date(match.group(0))))

    # Only multi-digit numbers — single digits collapse to 1-char spans.
    for match in re.finditer(r"(?<![\w.])\d{2,}(?:\.\d+)?(?![\w.])", output):
        token = match.group(0)
        replacements.append((token, _plausible_number_swap(token, pool, rng)))

    return replacements


def _value_replacements(
    value: str,
    pool: list[str],
    rng: random.Random,
) -> list[tuple[str, str]]:
    text = str(value).strip()
    lower = text.lower()
    if not text or len(text) > 80:
        return []

    if lower in STATUS_SWAPS:
        return [(text, _preserve_case(text, STATUS_SWAPS[lower]))]

    if text in ENTITY_SWAPS:
        return [(text, ENTITY_SWAPS[text])]

    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        return [(text, _shift_time(text))]

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return [(text, _shift_date(text))]

    if re.fullmatch(r"-?\d{2,}(?:\.\d+)?", text):
        return [(text, _plausible_number_swap(text, pool, rng))]

    # Multi-word proper-noun-ish values: try to swap with another sample of similar length.
    if len(text) >= MIN_CONTRADICTION_LEN and re.search(r"[A-Za-z]", text):
        swap = _pool_swap(text, pool, rng)
        if swap and swap != text:
            return [(text, swap)]

    return []


def _pool_swap(original: str, pool: list[str], rng: random.Random) -> str | None:
    if not pool:
        return None
    target_len = len(original)
    candidates = [
        p for p in pool
        if p != original
        and abs(len(p) - target_len) <= max(4, target_len // 2)
        and len(p) >= MIN_CONTRADICTION_LEN
        and any(c.isalpha() for c in p)
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def _plausible_number_swap(token: str, pool: list[str], rng: random.Random) -> str:
    # Try replacing with another numeric value seen elsewhere in the corpus first,
    # falling back to a +30 shift if the pool has no useful candidate.
    if pool:
        numeric_pool = [
            p for p in pool
            if re.fullmatch(r"-?\d{2,}(?:\.\d+)?", str(p)) and str(p) != token
        ]
        if numeric_pool:
            return str(rng.choice(numeric_pool))
    if "." in token:
        value = float(token)
        return f"{value + 17.0:.2f}".rstrip("0").rstrip(".")
    return str(int(token) + 37)


# Overgeneration ---------------------------------------------------------------


def overgeneration(
    clean_example: dict[str, Any],
    grounded_pool: list[str] | None = None,
) -> dict[str, Any] | None:
    output = clean_example["clean_output"]
    if len(output) < MIN_OUTPUT_LEN:
        return None
    rng = _rng(clean_example["id"], "overgeneration")
    span = rng.choice(OVERGENERATION_TEMPLATES)
    insert_pos = _pick_insertion_pos(output, rng)
    new_output, start = _insert_at(output, insert_pos, span)
    return _record(
        clean_example=clean_example,
        corruption_type="overgeneration",
        output=new_output,
        start=start,
        bad_span=span,
        extra_meta={"insert_pos_ratio": round(insert_pos / max(len(output), 1), 3)},
    )


# Missing tool -----------------------------------------------------------------


def missing_tool(
    clean_example: dict[str, Any],
    grounded_pool: list[str] | None = None,
) -> dict[str, Any] | None:
    output = clean_example["clean_output"]
    if len(output) < MIN_OUTPUT_LEN:
        return None
    available = _available_tool_names(clean_example.get("tools", []))
    absent = [
        (name, phrase) for name, phrase in MISSING_TOOL_ACTIONS
        if name.lower() not in available
    ]
    if not absent:
        return None
    rng = _rng(clean_example["id"], "missing_tool")
    _, phrase = rng.choice(absent)
    insert_pos = _pick_insertion_pos(output, rng)
    new_output, start = _insert_at(output, insert_pos, phrase)
    return _record(
        clean_example=clean_example,
        corruption_type="missing_tool",
        output=new_output,
        start=start,
        bad_span=phrase,
        extra_meta={"insert_pos_ratio": round(insert_pos / max(len(output), 1), 3)},
    )


# Insertion helpers ------------------------------------------------------------


def _pick_insertion_pos(output: str, rng: random.Random) -> int:
    """Choose a sentence boundary in `output`. Uniformly across boundaries -> no end bias."""
    boundaries = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", output)]
    boundaries = [b for b in boundaries if 0 < b <= len(output)]
    if not boundaries:
        return len(output)
    # Skip the final boundary half the time to spread positions.
    if len(boundaries) >= 2 and rng.random() < 0.7:
        boundaries = boundaries[:-1] or boundaries
    return rng.choice(boundaries)


def _insert_at(output: str, pos: int, span: str) -> tuple[str, int]:
    pos = max(0, min(pos, len(output)))
    before = output[:pos].rstrip()
    after = output[pos:].lstrip()
    sep_before = " " if before and not before[-1].isspace() else ""
    sep_after = " " if after else ""
    new_text = before + sep_before + span + sep_after + after
    start = len(before) + len(sep_before)
    return new_text, start


# Record helpers ---------------------------------------------------------------


def _record(
    clean_example: dict[str, Any],
    corruption_type: str,
    output: str,
    start: int,
    bad_span: str,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = _base_meta(clean_example, corruption_type)
    if extra_meta:
        meta.update(extra_meta)
    return {
        "id": f"{clean_example['id']}__{corruption_type}",
        "query": clean_example["query"],
        "context": clean_example["context"],
        "output": output,
        "hallucination_labels": [
            {
                "start": start,
                "end": start + len(bad_span),
                "text": bad_span,
                "label": corruption_type,
            }
        ],
        "meta": meta,
    }


def _base_meta(clean_example: dict[str, Any], corruption_type: str) -> dict[str, Any]:
    return {
        "source": clean_example["source"],
        "base_id": clean_example["id"],
        "corruption_type": corruption_type,
        "tools": clean_example.get("tools", []),
        "tool_call": clean_example.get("tool_call"),
    }


# Grounded values extraction ---------------------------------------------------


def _grounded_values(context: str) -> set[str]:
    values: set[str] = set()
    for obj in _json_objects(context):
        _collect_values(obj, values)
    return values


def _json_objects(text: str) -> list[Any]:
    objects = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        match = re.search(r"[\[{]", text[idx:])
        if match is None:
            break
        start = idx + match.start()
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        objects.append(obj)
        idx = start + end
    return objects


def _collect_values(obj: Any, values: set[str]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            _collect_values(value, values)
    elif isinstance(obj, list):
        for value in obj:
            _collect_values(value, values)
    elif isinstance(obj, (str, int, float, bool)) and obj is not None:
        values.add(str(obj))


def collect_corpus_pool(clean_examples: Iterable[dict[str, Any]]) -> list[str]:
    """Aggregate grounded values across all examples for cross-record contradictions."""
    pool: set[str] = set()
    for ex in clean_examples:
        for v in _grounded_values(ex.get("context", "")):
            if 3 <= len(v) <= 60:
                pool.add(v)
    return sorted(pool)


# Misc helpers -----------------------------------------------------------------


def _find_token(text: str, token: str) -> list[re.Match[str]]:
    if re.fullmatch(r"[A-Za-z]+", token):
        return list(re.finditer(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE))
    return list(re.finditer(re.escape(token), text))


def _shift_time(text: str) -> str:
    hour, minute = text.split(":")
    return f"{(int(hour) + 3) % 24:02d}:{minute}"


def _shift_date(text: str) -> str:
    year, month, day = text.split("-")
    new_month = (int(month) % 12) + 1
    return f"{year}-{new_month:02d}-{day}"


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _available_tool_names(tools: list[Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            names.add(str(name).lower())
    return names


def _rng(base_id: str, corruption_type: str) -> random.Random:
    return random.Random(f"{base_id}:{corruption_type}")


def clone_tool_call(tool_call: Any) -> Any:
    return copy.deepcopy(tool_call)
