"""LLM-assisted corruption generator.

For each clean source example, use a local instruction LLM to draft a
*semantically plausible* corruption snippet (for contradiction / overgeneration /
missing_tool). The snippet is then inserted with the same exact-offset bookkeeping
as the rule-based corruptors in `corruptors.py`, so the resulting record still
matches the RAGTruth schema with `output[start:end] == text`.

Why this pipeline:
  - Rule-based corruptors are deterministic but produce surface-level swaps
    (city -> city, sunny -> rainy, fixed appended phrases). A detector trained
    on those overfits to lexical shortcuts.
  - Asking an LLM to *write* the corruption gives more varied lexical surface
    while still letting us control: (a) the corruption *type*, (b) the *position*
    inside the answer, and (c) the *exact span text* recorded as the label.

Outputs:
  data/llm_aug/<corruption_type>/{train,validation,test}.jsonl

Each output record has the same schema as the regex-built dataset but with
meta.corruption_source set to "llm_augment".
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


SYSTEM_PROMPT = (
    "You are a synthetic-data generator for hallucination-detection training. "
    "Given a clean tool-using assistant answer and the tool's structured response (context), "
    "produce ONE short hallucinated phrase of the requested type. The phrase MUST be a single "
    "sentence (or a clause that can stand on its own), <= 200 characters, written in the same "
    "voice as the original answer. Reply with ONE JSON object on a single line, no markdown.\n\n"
    "Types:\n"
    "  contradiction  : a claim that directly conflicts with a specific value in the context.\n"
    "  overgeneration : a claim that adds information not present anywhere in the context.\n"
    "  missing_tool   : an offer to perform an action that requires a tool NOT in the available list.\n\n"
    'Schema: {"phrase": "<the hallucinated phrase>", "rationale": "<one short reason>"}\n'
    "Constraints:\n"
    "  - Do NOT mention the words 'hallucination' or 'incorrect'.\n"
    "  - The phrase must sound natural in the answer, not robotic.\n"
    "  - For contradiction, the phrase must contradict the actual context values (numbers / "
    "names / statuses), not be a vague hedge.\n"
)


USER_TEMPLATE = (
    "Tool context (truncated):\n{context}\n\n"
    "Available tools: {tool_names}\n\n"
    "Clean assistant answer (truncated):\n{output}\n\n"
    "Requested corruption type: {ctype}\n\n"
    "Produce the JSON now."
)


def truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "...[truncated]"


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()
    return tok, model, device


def generate(tok, model, device, system: str, user: str, max_new_tokens: int = 160) -> str:
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_phrase(raw: str) -> str | None:
    raw = raw.strip()
    candidates = [raw] + JSON_RE.findall(raw)
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("phrase"), str):
            phrase = obj["phrase"].strip()
            if 5 < len(phrase) <= 220 and "\n" not in phrase:
                return phrase
    return None


def pick_insertion_pos(output: str, rng: random.Random) -> int:
    boundaries = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", output)]
    boundaries = [b for b in boundaries if 0 < b <= len(output)]
    if not boundaries:
        return len(output)
    if len(boundaries) >= 2 and rng.random() < 0.7:
        boundaries = boundaries[:-1] or boundaries
    return rng.choice(boundaries)


def insert_at(output: str, pos: int, phrase: str) -> tuple[str, int, int]:
    pos = max(0, min(pos, len(output)))
    before = output[:pos].rstrip()
    after = output[pos:].lstrip()
    sep_before = " " if before and not before[-1].isspace() else ""
    sep_after = " " if after else ""
    new_text = before + sep_before + phrase + sep_after + after
    start = len(before) + len(sep_before)
    return new_text, start, start + len(phrase)


def tool_names_of(meta: dict) -> str:
    tools = meta.get("tools") or []
    names = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return ", ".join(names[:12]) or "(unspecified)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "data" / "combined"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "llm_aug"))
    parser.add_argument("--n", type=int, default=30, help="Per corruption type")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--types", nargs="+", default=["contradiction", "overgeneration", "missing_tool"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_path = Path(args.source) / f"{args.split}.jsonl"
    records = [json.loads(l) for l in src_path.open()]
    # Use only the clean-base outputs as sources for augmentation.
    clean_sources = [
        r for r in records
        if r["meta"].get("corruption_type") == "clean"
    ]
    rng = random.Random(args.seed)
    rng.shuffle(clean_sources)
    print(f"loaded {len(records)} records, {len(clean_sources)} clean sources")

    print(f"loading model {args.model} ...")
    tok, model, device = load_model(args.model)
    print(f"model loaded on {device}")

    out_root = Path(args.out_dir)
    stats: dict[str, dict[str, int]] = {}
    t0 = time.time()

    for ctype in args.types:
        kept: list[dict] = []
        pool = list(clean_sources)
        rng.shuffle(pool)
        attempts = 0
        for src in pool:
            if len(kept) >= args.n:
                break
            attempts += 1
            user = USER_TEMPLATE.format(
                context=truncate(src["context"], 1400),
                tool_names=tool_names_of(src.get("meta", {})),
                output=truncate(src["output"], 1000),
                ctype=ctype,
            )
            raw = generate(tok, model, device, SYSTEM_PROMPT, user)
            phrase = parse_phrase(raw)
            if not phrase:
                continue

            insertion_rng = random.Random(f"{src['id']}:{ctype}:{args.seed}")
            pos = pick_insertion_pos(src["output"], insertion_rng)
            new_output, start, end = insert_at(src["output"], pos, phrase)
            base_id = src["meta"].get("base_id") or src["id"]
            rec = {
                "id": f"{base_id}__llm_{ctype}",
                "query": src["query"],
                "context": src["context"],
                "output": new_output,
                "hallucination_labels": [
                    {"start": start, "end": end, "text": phrase, "label": ctype}
                ],
                "meta": {
                    **{k: v for k, v in src["meta"].items() if k in ("source", "tools", "tool_call")},
                    "base_id": base_id,
                    "corruption_type": ctype,
                    "corruption_source": "llm_augment",
                    "llm_model": args.model,
                    "llm_rationale_seen": True,
                },
            }
            assert new_output[start:end] == phrase
            kept.append(rec)
            elapsed = time.time() - t0
            if len(kept) % 5 == 0:
                print(f"  [{ctype}] {len(kept)}/{args.n} kept, {attempts} attempts, {elapsed:.0f}s")

        out_dir = out_root / ctype
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.split}.jsonl"
        with path.open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[ctype] = {"kept": len(kept), "attempts": attempts}
        print(f"  wrote {path} ({len(kept)} records)")

    print(json.dumps({"stats": stats, "elapsed_s": time.time() - t0}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
