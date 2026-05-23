"""Validation + zero-shot baseline for the tool-calling hallucination dataset.

Usage:
  python scripts/zero_shot_eval.py [--dataset-dir data/combined] [--split validation]
                                   [--no-model] [--max-records N]
                                   [--model-id KRLabsOrg/lettucedect-base-modernbert-en-v1]

Three checks per dataset split:
  1. Sanity         — uniqueness, label length distribution, position bias, query<->output overlap.
  2. Lexical        — char-level F1 of a "token-not-in-context" baseline.
  3. LettuceDetect  — zero-shot transformer baseline. Requires the `lettucedetect` package
                      and a torch / transformers (>=4.48) install with ModernBERT support.

Output:
  <dataset_dir>/validation_report.md
  <dataset_dir>/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "data" / "combined"

WORD_RE = re.compile(r"[A-Za-z0-9]+")


# IO ---------------------------------------------------------------------------


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text)]


def words_set(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 2}


def char_spans_to_char_mask(text: str, spans: list[dict]) -> list[bool]:
    mask = [False] * len(text)
    for s in spans:
        for i in range(s["start"], min(s["end"], len(text))):
            mask[i] = True
    return mask


def f1_from_masks(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(1 for p, g in zip(pred, gold) if p and g)
    fp = sum(1 for p, g in zip(pred, gold) if p and not g)
    fn = sum(1 for p, g in zip(pred, gold) if not p and g)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def aggregate_metrics(per_record: list[dict]) -> dict:
    tp = sum(r["tp"] for r in per_record)
    fp = sum(r["fp"] for r in per_record)
    fn = sum(r["fn"] for r in per_record)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n": len(per_record)}


# 1. Sanity --------------------------------------------------------------------


def sanity_check(records: list[dict]) -> dict:
    outputs = [r["output"] for r in records]
    queries = [r["query"] for r in records]

    output_counts = Counter(outputs)
    dup_outputs = sum(c for c in output_counts.values() if c > 1)

    label_lengths: list[int] = []
    label_count: list[int] = []
    type_counts: Counter = Counter()
    clean_count = 0

    for r in records:
        labels = r["hallucination_labels"]
        label_count.append(len(labels))
        if not labels:
            clean_count += 1
        for lab in labels:
            label_lengths.append(lab["end"] - lab["start"])
            type_counts[lab["label"]] += 1

    short_labels = sum(1 for l in label_lengths if l <= 3)

    overlaps = []
    for q, o in zip(queries, outputs):
        qw, ow = words_set(q), words_set(o)
        if qw:
            overlaps.append(len(qw & ow) / len(qw))

    pos_from_start = []
    for r in records:
        out_len = max(len(r["output"]), 1)
        for lab in r["hallucination_labels"]:
            pos_from_start.append(lab["start"] / out_len)

    avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0

    return {
        "n_records": len(records),
        "n_clean": clean_count,
        "n_hallucinated": len(records) - clean_count,
        "n_unique_outputs": len(output_counts),
        "n_duplicate_output_records": dup_outputs,
        "avg_labels_per_record": avg(label_count),
        "label_length_min": min(label_lengths) if label_lengths else 0,
        "label_length_max": max(label_lengths) if label_lengths else 0,
        "label_length_mean": avg(label_lengths),
        "short_labels_le_3_chars": short_labels,
        "short_labels_pct": (short_labels / len(label_lengths) * 100) if label_lengths else 0.0,
        "query_output_overlap_mean": avg(overlaps),
        "labels_by_type": dict(type_counts),
        "label_relative_start_pos_mean": avg(pos_from_start),
    }


# 2. Lexical baseline ----------------------------------------------------------


def lexical_predict(context: str, output: str) -> list[bool]:
    ctx_words = words_set(context)
    mask = [False] * len(output)
    for m in WORD_RE.finditer(output):
        tok = m.group(0).lower()
        if len(tok) <= 2:
            continue
        if tok not in ctx_words:
            for i in range(m.start(), m.end()):
                mask[i] = True
    return mask


def lexical_baseline(records: list[dict]) -> dict:
    per_record: list[dict] = []
    per_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        gold = char_spans_to_char_mask(r["output"], r["hallucination_labels"])
        pred = lexical_predict(r["context"], r["output"])
        m = f1_from_masks(pred, gold)
        per_record.append(m)
        record_type = r["meta"].get("corruption_type", "unknown")
        per_type[record_type].append(m)
    return {
        "overall": aggregate_metrics(per_record),
        "by_type": {t: aggregate_metrics(v) for t, v in per_type.items()},
    }


# 3. Zero-shot LettuceDetect ---------------------------------------------------


def model_zero_shot(records: list[dict], model_id: str) -> dict | None:
    try:
        from lettucedetect.models.inference import HallucinationDetector
    except Exception as e:
        return {"error": f"lettucedetect unavailable: {e}"}

    try:
        detector = HallucinationDetector(method="transformer", model_path=model_id)
    except Exception as e:
        return {"error": f"failed to load {model_id}: {e}"}

    per_record: list[dict] = []
    per_type: dict[str, list[dict]] = defaultdict(list)

    for idx, r in enumerate(records):
        out = r["output"]
        try:
            spans = detector.predict(
                context=[r["context"]],
                question=r["query"],
                answer=out,
                output_format="spans",
            )
        except Exception as e:
            spans = []
            if idx < 3:
                print(f"  predict error on record {idx}: {e}", file=sys.stderr)

        pred_mask = [False] * len(out)
        for sp in spans:
            start = int(sp.get("start", 0))
            end = int(sp.get("end", 0))
            for i in range(max(0, start), min(end, len(out))):
                pred_mask[i] = True

        gold = char_spans_to_char_mask(out, r["hallucination_labels"])
        m = f1_from_masks(pred_mask, gold)
        per_record.append(m)
        record_type = r["meta"].get("corruption_type", "unknown")
        per_type[record_type].append(m)

    return {
        "model_id": model_id,
        "overall": aggregate_metrics(per_record),
        "by_type": {t: aggregate_metrics(v) for t, v in per_type.items()},
    }


# Report -----------------------------------------------------------------------


def fmt_metrics(m: dict) -> str:
    return f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['n']})"


def write_report(out_dir: Path, dataset_name: str, split: str, sanity: dict, lex: dict, mdl: dict | None) -> None:
    md = [f"# Validation report — {dataset_name} / {split}", ""]
    md += ["## 1. Sanity", ""]
    md += [f"- Records: {sanity['n_records']} (clean: {sanity['n_clean']}, hallucinated: {sanity['n_hallucinated']})"]
    md += [f"- Unique outputs: {sanity['n_unique_outputs']} ({sanity['n_duplicate_output_records']} duplicates)"]
    md += [f"- Avg labels per record: {sanity['avg_labels_per_record']:.2f}"]
    md += [f"- Label length min/mean/max: {sanity['label_length_min']} / {sanity['label_length_mean']:.1f} / {sanity['label_length_max']}"]
    md += [f"- Short labels (<=3 chars): {sanity['short_labels_le_3_chars']} ({sanity['short_labels_pct']:.1f}%)"]
    md += [f"- Query<->output content-word overlap (mean): {sanity['query_output_overlap_mean']:.3f}"]
    md += [f"- Labels by type: {sanity['labels_by_type']}"]
    md += [f"- Mean label start position (0=start, 1=end): {sanity['label_relative_start_pos_mean']:.3f}"]
    md += [""]

    md += ["## 2. Lexical baseline (token not in context)", ""]
    md += [f"- Overall: {fmt_metrics(lex['overall'])}"]
    for t, m in sorted(lex["by_type"].items()):
        md += [f"  - {t}: {fmt_metrics(m)}"]
    md += [""]

    md += ["## 3. Zero-shot LettuceDetect", ""]
    if mdl is None:
        md += ["- skipped (--no-model)"]
    elif "error" in mdl:
        md += [f"- error: {mdl['error']}"]
    else:
        md += [f"- model: {mdl['model_id']}"]
        md += [f"- Overall: {fmt_metrics(mdl['overall'])}"]
        for t, m in sorted(mdl["by_type"].items()):
            md += [f"  - {t}: {fmt_metrics(m)}"]
    md += [""]

    md += ["## Interpretation guide", ""]
    md += ["- Lexical F1 > 0.5 → task is largely solvable by lexical overlap; dataset is too easy."]
    md += ["- Lexical F1 << model F1 → model uses signal beyond surface lexical mismatch (good)."]
    md += ["- Position bias OK if mean start pos ~ 0.4-0.6 (varied) — close to 0 or 1 = strong bias."]
    md += ["- Short-label % above ~5% inflates contradiction F1 trivially; aim < 5%."]

    out_md = out_dir / f"validation_report_{split}.md"
    out_json = out_dir / f"validation_report_{split}.json"
    out_md.write_text("\n".join(md))
    out_json.write_text(json.dumps({"sanity": sanity, "lexical": lex, "model": mdl}, indent=2))
    print(f"wrote {out_md} and {out_json}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--model-id", default="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    args = parser.parse_args(argv)

    ds_dir = Path(args.dataset_dir)
    path = ds_dir / f"{args.split}.jsonl"
    records = load(path)
    if args.max_records:
        records = records[: args.max_records]
    print(f"loaded {len(records)} records from {path}")

    sanity = sanity_check(records)
    print("sanity:", json.dumps(sanity, indent=2))

    lex = lexical_baseline(records)
    print(f"\nlexical overall: {fmt_metrics(lex['overall'])}")
    for t, m in sorted(lex["by_type"].items()):
        print(f"  {t}: {fmt_metrics(m)}")

    mdl = None
    if not args.no_model:
        print(f"\nrunning zero-shot model: {args.model_id} ...")
        mdl = model_zero_shot(records, args.model_id)
        if mdl and "error" not in mdl:
            print(f"model overall: {fmt_metrics(mdl['overall'])}")
            for t, m in sorted(mdl["by_type"].items()):
                print(f"  {t}: {fmt_metrics(m)}")
        else:
            print(f"model: {mdl}")

    write_report(ds_dir, ds_dir.name, args.split, sanity, lex, mdl)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
