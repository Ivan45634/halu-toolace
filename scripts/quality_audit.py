"""LLM-as-judge quality audit for the hallucination dataset.

For each record (clean or corrupted) the judge model is asked, in JSON-only mode,
whether the labeled hallucination span is genuinely a hallucination and whether
there is any *other* unlabeled hallucinated phrase in the model output. The
judge's verdict is then cross-checked against the LettuceDetect zero-shot
predictions stored from a previous run of ``scripts/zero_shot_eval.py``.

Outputs (per dataset/split):
  data/quality_audit/<dataset>/<split>/decisions.jsonl
  data/quality_audit/<dataset>/<split>/summary.md
  data/quality_audit/<dataset>/<split>/high_confidence.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 220
MAX_CONTEXT_CHARS = 1500
MAX_OUTPUT_CHARS = 1200

JUDGE_SYSTEM = (
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

JUDGE_USER_TEMPLATE = (
    "USER QUERY:\n{query}\n\n"
    "TOOL CONTEXT (truncated):\n{context}\n\n"
    "AVAILABLE TOOLS: {tool_names}\n\n"
    "ASSISTANT FINAL ANSWER (truncated):\n{output}\n\n"
    "CANDIDATE HALLUCINATED SPAN:\n{candidate}\n\n"
    "Respond now with the JSON object only."
)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def collect_tool_names(meta: dict) -> str:
    tools = meta.get("tools") or []
    names = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    if not names:
        return "(unspecified)"
    return ", ".join(names[:12])


def build_prompt(record: dict) -> tuple[str, str]:
    labels = record.get("hallucination_labels") or []
    meta = record.get("meta", {})
    if labels:
        lab = labels[0]
        parts = [
            f"text = {lab['text']!r}",
            f"label_type = {lab['label']!r}",
            f"location = characters [{lab['start']}, {lab['end']})",
        ]
        if lab["label"] == "contradiction" and meta.get("original"):
            parts.append(f"replaces_original_value = {meta['original']!r}")
        candidate = "\n  ".join(parts)
    else:
        candidate = "(none — record is marked clean. Set label_correct=unknown. Use extra_hallucination=true if you spot any unlabeled hallucination.)"

    user = JUDGE_USER_TEMPLATE.format(
        query=truncate(record["query"], 600),
        context=truncate(record["context"], MAX_CONTEXT_CHARS),
        tool_names=collect_tool_names(meta),
        output=truncate(record["output"], MAX_OUTPUT_CHARS),
        candidate=candidate,
    )
    return JUDGE_SYSTEM, user


JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_judge_reply(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {"_parse_error": "empty"}
    # First try strict JSON of the whole reply.
    for candidate in [raw] + JSON_RE.findall(raw):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {"_parse_error": "no_json_found"}


def normalize_decision(parsed: dict[str, Any]) -> dict[str, Any]:
    if "_parse_error" in parsed:
        return {
            "label_correct": "unknown",
            "extra_hallucination": False,
            "extra_text": "",
            "reasoning": "",
            "parse_error": parsed["_parse_error"],
        }
    lc = parsed.get("label_correct")
    if isinstance(lc, bool):
        label_correct = "true" if lc else "false"
    else:
        s = str(lc).strip().lower()
        if s in ("true", "yes", "y"):
            label_correct = "true"
        elif s in ("false", "no", "n"):
            label_correct = "false"
        else:
            label_correct = "unknown"
    extra = bool(parsed.get("extra_hallucination", False))
    extra_text = str(parsed.get("extra_text", "") or "").strip()
    reasoning = str(parsed.get("reasoning", "") or "").strip()
    return {
        "label_correct": label_correct,
        "extra_hallucination": extra,
        "extra_text": extra_text,
        "reasoning": reasoning,
    }


# Inference -------------------------------------------------------------------


def load_judge(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()
    return tok, model, device


def judge_one(tok, model, device, system: str, user: str) -> str:
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=6144).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


# Cross-check with LettuceDetect ---------------------------------------------


def load_lettuce_predictions(records: list[dict], model_id: str) -> dict[str, list[dict]]:
    try:
        from lettucedetect.models.inference import HallucinationDetector
        detector = HallucinationDetector(method="transformer", model_path=model_id)
    except Exception as e:
        print(f"  lettuce-detect unavailable, skipping cross-check: {e}", file=sys.stderr)
        return {}
    preds: dict[str, list[dict]] = {}
    for r in records:
        try:
            spans = detector.predict(
                context=[r["context"]],
                question=r["query"],
                answer=r["output"],
                output_format="spans",
            )
        except Exception:
            spans = []
        preds[r["id"]] = spans
    return preds


def lettuce_overlap_with_label(label: dict | None, preds: list[dict], output_len: int) -> dict:
    """Return whether any LettuceDetect span overlaps the labeled span, or where they fire if no label."""
    if not label:
        # Clean record: see if lettuce fired anywhere.
        any_fired = bool(preds)
        return {"fired": any_fired, "n_spans": len(preds)}
    a0, a1 = label["start"], label["end"]
    matched = False
    for sp in preds:
        b0, b1 = int(sp.get("start", 0)), int(sp.get("end", 0))
        if max(a0, b0) < min(a1, b1):
            matched = True
            break
    return {"fired": matched, "n_spans": len(preds)}


# Driver ----------------------------------------------------------------------


def audit_split(
    dataset_dir: Path,
    split: str,
    out_dir: Path,
    max_records: int,
    use_lettuce: bool,
    judge_model: str = JUDGE_MODEL,
) -> dict[str, Any]:
    src = dataset_dir / f"{split}.jsonl"
    records = [json.loads(line) for line in src.open()]
    if max_records:
        records = records[:max_records]
    print(f"[{dataset_dir.name}/{split}] {len(records)} records")

    if use_lettuce:
        print(f"  loading lettuce-detect for cross-check...")
        lettuce_preds = load_lettuce_predictions(records, "KRLabsOrg/lettucedect-base-modernbert-en-v1")
    else:
        lettuce_preds = {}

    print(f"  loading judge model {judge_model}...")
    tok, model, device = load_judge(judge_model)
    print(f"  judge running on {device}")

    decisions: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(records):
        system, user = build_prompt(r)
        raw = judge_one(tok, model, device, system, user)
        parsed = parse_judge_reply(raw)
        decision = normalize_decision(parsed)

        label = (r["hallucination_labels"] or [None])[0]
        lettuce = lettuce_overlap_with_label(label, lettuce_preds.get(r["id"], []), len(r["output"]))
        decisions.append({
            "id": r["id"],
            "corruption_type": r["meta"].get("corruption_type"),
            "labeled_span": label,
            "judge": decision,
            "judge_raw": raw.strip()[:400],
            "lettuce": lettuce,
        })
        if (i + 1) % 20 == 0 or i == len(records) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(records) - i - 1) / max(rate, 1e-6)
            print(f"  {i+1}/{len(records)} | {rate:.2f} rec/s | ETA {eta:.0f}s")

    out_split = out_dir / dataset_dir.name / split
    out_split.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_split / "decisions.jsonl", decisions)

    summary, high_conf = summarize(records, decisions)
    (out_split / "summary.md").write_text(summary)
    write_jsonl(out_split / "high_confidence.jsonl", high_conf)

    print(f"  wrote {out_split}/decisions.jsonl ({len(decisions)} rows)")
    print(f"  wrote {out_split}/summary.md")
    print(f"  wrote {out_split}/high_confidence.jsonl ({len(high_conf)} rows)")

    return {"dataset": dataset_dir.name, "split": split, "n_records": len(records)}


def summarize(records: list[dict], decisions: list[dict]) -> tuple[str, list[dict]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        by_type[d["corruption_type"] or "unknown"].append(d)

    lines = ["# Quality audit summary", "", f"Total records: {len(decisions)}", ""]
    lines.append("| corruption_type | n | label_correct=true | label_correct=false | label_correct=unknown | extra_found | parse_errors | lettuce_agrees |")
    lines.append("|---|---|---|---|---|---|---|---|")

    high_conf_ids: set[str] = set()
    for ctype, items in sorted(by_type.items()):
        n = len(items)
        nt = sum(1 for x in items if x["judge"]["label_correct"] == "true")
        nf = sum(1 for x in items if x["judge"]["label_correct"] == "false")
        nu = sum(1 for x in items if x["judge"]["label_correct"] == "unknown")
        nx = sum(1 for x in items if x["judge"]["extra_hallucination"])
        npe = sum(1 for x in items if "parse_error" in x["judge"])
        # Lettuce "agrees" definition:
        #   for hallucinated records: lettuce span overlaps the label
        #   for clean records: lettuce did NOT fire
        nle = 0
        for x in items:
            fired = x["lettuce"].get("fired", False)
            if ctype == "clean":
                if not fired:
                    nle += 1
            else:
                if fired:
                    nle += 1
        lines.append(
            f"| {ctype} | {n} | {nt} | {nf} | {nu} | {nx} | {npe} | {nle}/{n} |"
        )

        for x in items:
            # High confidence:
            #   clean records: judge says label_correct=unknown AND extra=False AND lettuce did NOT fire
            #   corrupted: judge says label_correct=true AND lettuce overlaps OR judge says true AND extra=False
            if ctype == "clean":
                if (
                    x["judge"]["label_correct"] == "unknown"
                    and not x["judge"]["extra_hallucination"]
                    and not x["lettuce"].get("fired", False)
                    and "parse_error" not in x["judge"]
                ):
                    high_conf_ids.add(x["id"])
            else:
                if (
                    x["judge"]["label_correct"] == "true"
                    and "parse_error" not in x["judge"]
                ):
                    high_conf_ids.add(x["id"])

    lines.append("")
    lines.append(f"High-confidence subset: {len(high_conf_ids)} of {len(decisions)} records "
                 f"({100 * len(high_conf_ids) / max(len(decisions), 1):.1f}%).")
    lines.append("")
    lines.append("Definitions:")
    lines.append("- `label_correct=true`  : judge confirms the labeled span is a real hallucination of the right type.")
    lines.append("- `label_correct=false` : judge thinks the labeled span is NOT a hallucination — the label is likely wrong.")
    lines.append("- `label_correct=unknown`: judge couldn't decide (often clean records where no candidate was provided).")
    lines.append("- `extra_found`: judge found a hallucinated phrase that is NOT in `hallucination_labels`.")
    lines.append("- `lettuce_agrees`: LettuceDetect's prediction overlaps the label for corrupted records, or is silent for clean records.")
    lines.append("- `high_confidence`: judge agrees the label is correct (or for clean: judge sees no extra issue and LettuceDetect is silent). Suitable for downstream training.")

    high_conf_records = [r for r, d in zip(records, decisions) if d["id"] in high_conf_ids]
    return "\n".join(lines), high_conf_records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "combined"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "quality_audit"))
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--no-lettuce", action="store_true")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    args = parser.parse_args(argv)

    audit_split(
        dataset_dir=Path(args.dataset_dir),
        split=args.split,
        out_dir=Path(args.out_dir),
        max_records=args.max_records,
        use_lettuce=not args.no_lettuce,
        judge_model=args.judge_model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
