"""Build span-level hallucination JSONL splits from minpeter/toolace-parsed.

Produces (under --out-dir):

  data/combined/{train,validation,test}.jsonl   all corruptions + clean negatives
  data/contradiction/{train,validation,test}.jsonl   contradiction-only + clean
  data/missing_tool/{train,validation,test}.jsonl    missing_tool-only + clean
  data/overgeneration/{train,validation,test}.jsonl  overgeneration-only + clean
  data/build_summary.json                        per-split, per-type record counts

Records follow the RAGTruth schema requested in the task:
  query, context, output, hallucination_labels: [{start, end, text, label}]

Filtering applied to source rows (in addition to the original schema checks):
  - drop rows where the final answer is shorter than MIN_OUTPUT_LEN
  - drop rows where final answer is suspiciously long (> MAX_OUTPUT_LEN)
  - drop rows where the lexical overlap between user query and final answer is
    below MIN_QUERY_OUTPUT_OVERLAP (the original ToolACE has a fair number of
    off-topic finals — e.g. caste-system query answered with arrays/linked-list
    explanation — which would inject noise into the corruption pipeline)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

try:
    from .corruptors import collect_corpus_pool, make_corruptions  # package mode
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from corruptors import collect_corpus_pool, make_corruptions  # type: ignore  # script mode


SOURCE_DATASET = "minpeter/toolace-parsed"
SOURCE_CONFIG = "toolace"
MIN_OUTPUT_LEN = 80
MAX_OUTPUT_LEN = 2500
MIN_QUERY_OUTPUT_OVERLAP = 0.10

WORD_RE = re.compile(r"[A-Za-z0-9]+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_DATASET)
    parser.add_argument("--config", default=SOURCE_CONFIG)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--max-clean-examples", type=int, default=5000)
    parser.add_argument(
        "--min-overlap", type=float, default=MIN_QUERY_OUTPUT_OVERLAP,
        help="Minimum query<->output content-word overlap to keep a row.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    clean_examples, source_stats = load_clean_examples(
        source=args.source,
        config=args.config,
        split=args.source_split,
        max_clean_examples=args.max_clean_examples,
        min_overlap=args.min_overlap,
    )
    print(f"loaded {len(clean_examples)} clean source examples")

    pool = collect_corpus_pool(clean_examples)
    print(f"grounded value pool size: {len(pool)}")

    all_records: list[dict[str, Any]] = []
    for clean_example in clean_examples:
        all_records.extend(make_corruptions(clean_example, include_clean=True, grounded_pool=pool))

    by_type: Counter = Counter(r["meta"]["corruption_type"] for r in all_records)
    print(f"records by type: {dict(by_type)}")

    split_by_base = split_base_ids([ex["id"] for ex in clean_examples])

    # Write the combined dataset and three per-type datasets.
    written: dict[str, dict[str, int]] = {}
    combined_dir = out_root / "combined"
    written["combined"] = write_partition(
        combined_dir,
        records=all_records,
        split_by_base=split_by_base,
        accept=lambda r: True,
    )

    for corruption_type in ("contradiction", "missing_tool", "overgeneration"):
        per_type_dir = out_root / corruption_type
        written[corruption_type] = write_partition(
            per_type_dir,
            records=all_records,
            split_by_base=split_by_base,
            accept=lambda r, t=corruption_type: r["meta"]["corruption_type"] in (t, "clean"),
        )

    # Backwards-compat copies in the data root so the existing zero_shot_eval script keeps working.
    for split in ("train", "validation", "test"):
        src = combined_dir / f"{split}.jsonl"
        if src.exists():
            (out_root / f"{split}.jsonl").write_bytes(src.read_bytes())

    summary = {
        "source": args.source,
        "config": args.config,
        "source_split": args.source_split,
        "clean_examples": len(clean_examples),
        "records": len(all_records),
        "source_stats": source_stats,
        "records_by_type": dict(sorted(by_type.items())),
        "records_by_dataset": written,
        "grounded_pool_size": len(pool),
        "filters": {
            "min_output_len": MIN_OUTPUT_LEN,
            "max_output_len": MAX_OUTPUT_LEN,
            "min_query_output_overlap": args.min_overlap,
        },
    }
    (out_root / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def write_partition(
    out_dir: Path,
    records: list[dict[str, Any]],
    split_by_base: dict[str, str],
    accept,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if not accept(rec):
            continue
        split = split_by_base[rec["meta"]["base_id"]]
        splits[split].append(rec)
    for split in ("train", "validation", "test"):
        write_jsonl(out_dir / f"{split}.jsonl", splits[split])
    return {split: len(splits[split]) for split in ("train", "validation", "test")}


def load_clean_examples(
    source: str,
    config: str | None,
    split: str,
    max_clean_examples: int,
    min_overlap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config:
        dataset = load_dataset(source, config, split=split)
    else:
        dataset = load_dataset(source, split=split)

    clean_examples = []
    reject_reasons: Counter = Counter()
    for index, row in enumerate(dataset):
        clean, reject = normalize_row(row=row, index=index, source=source, min_overlap=min_overlap)
        if clean is None:
            reject_reasons[reject or "unknown"] += 1
            continue
        clean_examples.append(clean)
        if len(clean_examples) >= max_clean_examples:
            break

    stats = {
        "rows_seen": index + 1,
        "valid_rows": len(clean_examples),
        "rejected_rows": sum(reject_reasons.values()),
        "reject_reasons": dict(sorted(reject_reasons.items())),
    }
    return clean_examples, stats


def normalize_row(
    row: dict[str, Any], index: int, source: str, min_overlap: float
) -> tuple[dict[str, Any] | None, str | None]:
    messages = parse_maybe_json(row.get("messages"))
    tools = parse_maybe_json(row.get("tools"))
    extra = parse_maybe_json(row.get("extra"))
    if not isinstance(messages, list):
        return None, "missing_messages"
    if not isinstance(tools, list) or not tools:
        return None, "missing_tools"

    user_query = first_user_query(messages)
    tool_calls = assistant_tool_calls(messages)
    tool_responses = tool_response_messages(messages)
    final_answer = final_assistant_answer(messages)

    if not user_query:
        return None, "missing_user_query"
    if not tool_calls:
        return None, "missing_tool_call"
    if not tool_responses:
        return None, "missing_tool_response"
    if not final_answer:
        return None, "missing_final_answer"

    output = final_answer.strip()
    if len(output) < MIN_OUTPUT_LEN:
        return None, "output_too_short"
    if len(output) > MAX_OUTPUT_LEN:
        return None, "output_too_long"

    overlap = query_output_overlap(user_query, output)
    if overlap < min_overlap:
        return None, "off_topic_output"

    context = "\n".join(format_tool_response(message) for message in tool_responses)
    if not context.strip():
        return None, "empty_context"

    base_id = base_id_from_extra(extra) or f"toolace_train_{index:05d}"
    return {
        "id": base_id,
        "source": source,
        "query": user_query.strip(),
        "tools": tools,
        "tool_call": tool_calls[0] if len(tool_calls) == 1 else tool_calls,
        "context": context.strip(),
        "clean_output": output,
        "query_output_overlap": overlap,
    }, None


def query_output_overlap(query: str, output: str) -> float:
    qw = {t.lower() for t in WORD_RE.findall(query) if len(t) > 2}
    ow = {t.lower() for t in WORD_RE.findall(output) if len(t) > 2}
    if not qw:
        return 0.0
    return len(qw & ow) / len(qw)


def parse_maybe_json(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped.lower() == "null":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def first_user_query(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    return None


def assistant_tool_calls(messages: list[dict[str, Any]]) -> list[Any]:
    calls = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        function_call = message.get("function_call")
        if tool_calls:
            calls.append(tool_calls)
        elif function_call:
            calls.append(function_call)
    return calls


def tool_response_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        message
        for message in messages
        if message.get("role") in {"tool", "function"} and str(message.get("content") or "").strip()
    ]


def final_assistant_answer(messages: list[dict[str, Any]]) -> str | None:
    seen_tool_response = False
    final = None
    for message in messages:
        role = message.get("role")
        if role in {"tool", "function"} and str(message.get("content") or "").strip():
            seen_tool_response = True
            continue
        if role != "assistant" or not seen_tool_response:
            continue
        content = str(message.get("content") or "").strip()
        if content and not message.get("tool_calls") and not message.get("function_call"):
            final = content
    return final


def format_tool_response(message: dict[str, Any]) -> str:
    payload = {
        "role": message.get("role"),
        "name": message.get("name"),
        "content": parse_maybe_json(message.get("content")),
    }
    if message.get("tool_call_id"):
        payload["tool_call_id"] = message.get("tool_call_id")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def base_id_from_extra(extra: Any) -> str | None:
    if isinstance(extra, dict):
        for key in ("id", "source_id", "conversation_id", "uuid"):
            value = extra.get(key)
            if value:
                return str(value)
    return None


def split_base_ids(base_ids: list[str]) -> dict[str, str]:
    unique_ids = sorted(set(base_ids), key=stable_sort_key)
    train_end = int(len(unique_ids) * 0.8)
    validation_end = int(len(unique_ids) * 0.9)
    mapping: dict[str, str] = {}
    for idx, base_id in enumerate(unique_ids):
        if idx < train_end:
            split = "train"
        elif idx < validation_end:
            split = "validation"
        else:
            split = "test"
        mapping[base_id] = split
    return mapping


def stable_sort_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
