"""Quality audit subcommands for the hallucination dataset.

Subcommands:
  run         Run the LLM judge over a split; writes decisions.jsonl + summary.md +
              high_confidence.jsonl.
  summary     Regenerate summary.md and high_confidence.jsonl from an existing
              decisions.jsonl (no model calls).
  report      Build decisions_with_verdict.jsonl and verdict_summary.md by adding
              an explicit `verdict` + `judge_verified` field to each decision.
  filter      Filter the source dataset down to the judge's high-confidence subset
              (mirrors the rule used inside `summarize`).
  export      Export decisions / recovered / skipped into a multi-sheet Excel
              workbook.

Shared helpers live in common.py. Each subcommand uses the same data layout:

  audit-dir = data/quality_audit_openrouter/combined/<split>/decisions.jsonl
  source    = data/combined/<split>.jsonl
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .common import (
    AUDIT_SYSTEM,
    AUDIT_USER_TEMPLATE,
    collect_tool_names,
    load_jsonl,
    load_judge,
    parse_json_object,
    truncate,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
JUDGE_MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"
MAX_CONTEXT_CHARS = 1500
MAX_OUTPUT_CHARS = 1200
SPLITS = ("train", "validation", "test")
VERDICT_VALUES = ("confirmed", "false_positive", "false_negative", "wrong_type", "uncertain")


# ----------------------------------------------------------------------------
# Shared building blocks
# ----------------------------------------------------------------------------


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
        candidate = ("(none — record is marked clean. Set label_correct=unknown. "
                     "Use extra_hallucination=true if you spot any unlabeled hallucination.)")
    user = AUDIT_USER_TEMPLATE.format(
        query=truncate(record["query"], 600),
        context=truncate(record["context"], MAX_CONTEXT_CHARS),
        tool_names=collect_tool_names(meta),
        output=truncate(record["output"], MAX_OUTPUT_CHARS),
        candidate=candidate,
    )
    return AUDIT_SYSTEM, user


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
    return {
        "label_correct": label_correct,
        "extra_hallucination": bool(parsed.get("extra_hallucination", False)),
        "extra_text": str(parsed.get("extra_text", "") or "").strip(),
        "reasoning": str(parsed.get("reasoning", "") or "").strip(),
    }


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
            spans = detector.predict(context=[r["context"]], question=r["query"],
                                       answer=r["output"], output_format="spans")
        except Exception:
            spans = []
        preds[r["id"]] = spans
    return preds


def lettuce_overlap_with_label(label: dict | None, preds: list[dict], output_len: int) -> dict:
    if not label:
        return {"fired": bool(preds), "n_spans": len(preds)}
    a0, a1 = label["start"], label["end"]
    matched = any(max(a0, int(sp.get("start", 0))) < min(a1, int(sp.get("end", 0))) for sp in preds)
    return {"fired": matched, "n_spans": len(preds)}


def summarize(records: list[dict], decisions: list[dict]) -> tuple[str, list[dict]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        by_type[d["corruption_type"] or "unknown"].append(d)

    lettuce_was_run = any("fired" in d.get("lettuce", {}) for d in decisions) and \
                      any(d.get("lettuce", {}).get("n_spans", 0) > 0 for d in decisions)

    lines = ["# Quality audit summary", "", f"Total records: {len(decisions)}",
             f"LettuceDetect cross-check: {'enabled' if lettuce_was_run else 'skipped'}", ""]
    header = ["corruption_type", "n", "label_correct=true", "label_correct=false",
              "label_correct=unknown", "extra_found", "parse_errors"]
    if lettuce_was_run:
        header.append("lettuce_agrees")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    high_conf_ids: set[str] = set()
    for ctype, items in sorted(by_type.items()):
        n = len(items)
        nt = sum(1 for x in items if x["judge"]["label_correct"] == "true")
        nf = sum(1 for x in items if x["judge"]["label_correct"] == "false")
        nu = sum(1 for x in items if x["judge"]["label_correct"] == "unknown")
        nx = sum(1 for x in items if x["judge"]["extra_hallucination"])
        npe = sum(1 for x in items if "parse_error" in x["judge"])
        row = [ctype, n, nt, nf, nu, nx, npe]
        if lettuce_was_run:
            nle = 0
            for x in items:
                fired = x["lettuce"].get("fired", False)
                if (ctype == "clean" and not fired) or (ctype != "clean" and fired):
                    nle += 1
            row.append(f"{nle}/{n}")
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

        for x in items:
            if ctype == "clean":
                lettuce_silent = (not x["lettuce"].get("fired", False)) if lettuce_was_run else True
                if (x["judge"]["label_correct"] == "unknown"
                        and not x["judge"]["extra_hallucination"]
                        and lettuce_silent
                        and "parse_error" not in x["judge"]):
                    high_conf_ids.add(x["id"])
            else:
                if x["judge"]["label_correct"] == "true" and "parse_error" not in x["judge"]:
                    high_conf_ids.add(x["id"])

    lines += ["",
              f"High-confidence subset: {len(high_conf_ids)} of {len(decisions)} records "
              f"({100 * len(high_conf_ids) / max(len(decisions), 1):.1f}%).",
              "",
              "Definitions:",
              "- `label_correct=true`  : judge confirms the labeled span is a real hallucination of the right type.",
              "- `label_correct=false` : judge thinks the labeled span is NOT a hallucination — the label is likely wrong.",
              "- `label_correct=unknown`: judge couldn't decide (often clean records where no candidate was provided).",
              "- `extra_found`: judge found a hallucinated phrase that is NOT in `hallucination_labels`."]
    if lettuce_was_run:
        lines.append("- `lettuce_agrees`: LettuceDetect's prediction overlaps the label for corrupted records, or is silent for clean records.")
    lines.append("- `high_confidence`: judge agrees the label is correct (or for clean: judge sees no extra issue"
                 + (" and LettuceDetect is silent" if lettuce_was_run else "") + "). Suitable for downstream training.")

    high_conf_records = [r for r, d in zip(records, decisions) if d["id"] in high_conf_ids]
    return "\n".join(lines), high_conf_records


def classify_verdict(decision: dict, corruption_type: str) -> tuple[str, bool]:
    judge = decision.get("judge", {})
    label_correct = judge.get("label_correct")
    extra = judge.get("extra_hallucination", False)
    if "parse_error" in judge or label_correct not in ("true", "false", "unknown"):
        return "uncertain", False
    if corruption_type == "clean":
        return ("false_negative", False) if extra else ("confirmed", True)
    if label_correct == "true":
        return "confirmed", False
    if label_correct == "false":
        return "false_positive", (not extra)
    return "wrong_type", False


# ----------------------------------------------------------------------------
# Subcommand: run
# ----------------------------------------------------------------------------


def cmd_run(args) -> int:
    dataset_dir = Path(args.dataset_dir)
    src = dataset_dir / f"{args.split}.jsonl"
    records = load_jsonl(src)
    if args.max_records:
        records = records[: args.max_records]
    print(f"[{dataset_dir.name}/{args.split}] {len(records)} records | backend={args.backend}", flush=True)

    if args.no_lettuce:
        lettuce_preds: dict[str, list[dict]] = {}
    else:
        print("  loading lettuce-detect for cross-check...", flush=True)
        lettuce_preds = load_lettuce_predictions(records, "KRLabsOrg/lettucedect-base-modernbert-en-v1")

    print(f"  loading judge model {args.judge_model} (backend={args.backend})...", flush=True)
    judge = load_judge(args.judge_model, backend=args.backend)
    print(f"  judge running on {judge.device}", flush=True)

    n_workers = 1 if args.backend == "local" else args.workers

    def _process(r: dict) -> tuple[dict, str]:
        system, user = build_prompt(r)
        try:
            raw = judge(system, user)
        except Exception as e:  # noqa: BLE001
            return ({"label_correct": "unknown", "extra_hallucination": False,
                     "extra_text": "", "reasoning": "", "parse_error": str(e)[:120]}, "")
        return normalize_decision(parse_json_object(raw)), raw

    def _row(r: dict, decision: dict, raw: str) -> dict:
        label = (r["hallucination_labels"] or [None])[0]
        lettuce = lettuce_overlap_with_label(label, lettuce_preds.get(r["id"], []), len(r["output"]))
        return {
            "id": r["id"],
            "corruption_type": r["meta"].get("corruption_type"),
            "labeled_span": label,
            "judge": decision,
            "judge_raw": (raw or "").strip()[:400],
            "lettuce": lettuce,
        }

    decisions: list[dict] = [None] * len(records)  # type: ignore
    done = 0
    t0 = time.time()

    if n_workers == 1:
        for i, r in enumerate(records):
            decision, raw = _process(r)
            decisions[i] = _row(r, decision, raw)
            done += 1
            if done % 10 == 0 or done == len(records):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(records) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(records)} | {rate:.2f} rec/s | ETA {eta:.0f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_process, r): i for i, r in enumerate(records)}
            for fut in as_completed(futs):
                i = futs[fut]
                decision, raw = fut.result()
                decisions[i] = _row(records[i], decision, raw)
                done += 1
                if done % 10 == 0 or done == len(records):
                    rate = done / max(time.time() - t0, 1e-6)
                    eta = (len(records) - done) / max(rate, 1e-6)
                    print(f"  {done}/{len(records)} | {rate:.2f} rec/s | ETA {eta:.0f}s", flush=True)

    out_split = Path(args.out_dir) / dataset_dir.name / args.split
    out_split.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_split / "decisions.jsonl", decisions)

    summary, high_conf = summarize(records, decisions)
    (out_split / "summary.md").write_text(summary)
    write_jsonl(out_split / "high_confidence.jsonl", high_conf)

    print(f"  wrote {out_split}/decisions.jsonl ({len(decisions)} rows)")
    print(f"  wrote {out_split}/summary.md")
    print(f"  wrote {out_split}/high_confidence.jsonl ({len(high_conf)} rows)")
    return 0


# ----------------------------------------------------------------------------
# Subcommand: summary  (regenerate from existing decisions.jsonl)
# ----------------------------------------------------------------------------


def cmd_summary(args) -> int:
    decisions = load_jsonl(Path(args.decisions))
    by_id = {d["id"] for d in decisions}
    records = [r for r in load_jsonl(Path(args.source)) if r["id"] in by_id]
    summary, high_conf = summarize(records, decisions)
    out_summary = Path(args.decisions).with_name("summary.md")
    out_hc = Path(args.decisions).with_name("high_confidence.jsonl")
    out_summary.write_text(summary)
    write_jsonl(out_hc, high_conf)
    print(f"wrote {out_summary}")
    print(f"wrote {out_hc} ({len(high_conf)} rows)")
    return 0


# ----------------------------------------------------------------------------
# Subcommand: report  (decisions_with_verdict.jsonl + verdict_summary.md)
# ----------------------------------------------------------------------------


def cmd_report(args) -> int:
    audit_root = Path(args.audit_dir)
    src_root = Path(args.source_dir)
    decisions_path = audit_root / args.split / "decisions.jsonl"
    source_path = src_root / f"{args.split}.jsonl"
    if not decisions_path.exists():
        raise SystemExit(f"missing: {decisions_path}")
    if not source_path.exists():
        raise SystemExit(f"missing: {source_path}")

    decisions = {d["id"]: d for d in load_jsonl(decisions_path)}
    counts: dict[str, Counter] = {ctype: Counter() for ctype in
                                    ("clean", "contradiction", "missing_tool", "overgeneration")}
    augmented: list[dict] = []
    n_judge_verified = 0
    for rec in load_jsonl(source_path):
        ctype = rec["meta"].get("corruption_type", "unknown")
        d = decisions.get(rec["id"])
        if d is None:
            verdict, judge_verified = "uncertain", False
        else:
            verdict, judge_verified = classify_verdict(d, ctype)
        if judge_verified:
            n_judge_verified += 1
        counts.setdefault(ctype, Counter())[verdict] += 1
        augmented.append({
            "id": rec["id"],
            "corruption_type": ctype,
            "judge_verified": judge_verified,
            "verdict": verdict,
            "labeled_span": (rec["hallucination_labels"] or [None])[0],
            "judge_reasoning": (d or {}).get("judge", {}).get("reasoning", ""),
            "judge_extra_text": (d or {}).get("judge", {}).get("extra_text", ""),
        })

    out_path = audit_root / args.split / "decisions_with_verdict.jsonl"
    summary_path = audit_root / args.split / "verdict_summary.md"
    write_jsonl(out_path, augmented)

    lines = [f"# Verdict summary — {args.split}", "",
             f"Total records: {len(augmented)} | judge_verified (judge sees no hallucination): "
             f"{n_judge_verified} ({100*n_judge_verified/max(len(augmented),1):.1f}%)", ""]
    lines.append("| corruption_type | n | " + " | ".join(VERDICT_VALUES) + " |")
    lines.append("|---|---|" + "---|" * len(VERDICT_VALUES))
    for ctype in ("clean", "contradiction", "missing_tool", "overgeneration"):
        c = counts.get(ctype, Counter())
        n = sum(c.values())
        if not n:
            continue
        cells = [str(c.get(v, 0)) for v in VERDICT_VALUES]
        lines.append(f"| {ctype} | {n} | " + " | ".join(cells) + " |")
    lines += ["",
              "## Verdict meaning",
              "- `confirmed`      — our label matches the judge",
              "- `false_positive` — we labeled a hallucination, judge disagrees (regex put a bad span)",
              "- `false_negative` — we marked clean, judge found a hidden hallucination",
              "- `wrong_type`     — corrupted record but our type / span doesn't match what judge sees",
              "- `uncertain`      — judge parse error or no decision",
              "",
              "## judge_verified",
              "True iff the record contains no hallucination according to the judge."]
    summary_path.write_text("\n".join(lines))
    print(f"wrote {out_path} ({len(augmented)} rows)")
    print(f"wrote {summary_path}")
    return 0


# ----------------------------------------------------------------------------
# Subcommand: filter  (keep only judge-confirmed records, mirror summarize's rule)
# ----------------------------------------------------------------------------


def cmd_filter(args) -> int:
    audit_path = Path(args.audit_dir) / args.split / "decisions.jsonl"
    src_path = Path(args.source_dir) / f"{args.split}.jsonl"
    out_path = Path(args.out_dir) / f"{args.split}.jsonl"
    if not audit_path.exists():
        raise SystemExit(f"audit decisions missing: {audit_path}")
    if not src_path.exists():
        raise SystemExit(f"source split missing: {src_path}")

    decisions = {d["id"]: d for d in load_jsonl(audit_path)}
    kept_by_type: Counter = Counter()
    dropped_by_type: Counter = Counter()
    written: list[dict] = []

    for rec in load_jsonl(src_path):
        d = decisions.get(rec["id"])
        ctype = rec["meta"].get("corruption_type", "unknown")
        if d is None:
            dropped_by_type[ctype] += 1
            continue
        judge = d["judge"]
        ok = False
        if "parse_error" not in judge:
            if ctype == "clean":
                ok = (judge["label_correct"] == "unknown"
                      and not judge.get("extra_hallucination", False))
            else:
                ok = (judge["label_correct"] == "true")
        if ok:
            kept_by_type[ctype] += 1
            written.append(rec)
        else:
            dropped_by_type[ctype] += 1

    write_jsonl(out_path, written)
    print(f"kept   : {sum(kept_by_type.values())}  {dict(kept_by_type)}")
    print(f"dropped: {sum(dropped_by_type.values())}  {dict(dropped_by_type)}")
    print(f"wrote  : {out_path}")
    return 0


# ----------------------------------------------------------------------------
# Subcommand: export  (Excel workbook with audit + recovered + skipped)
# ----------------------------------------------------------------------------


def cmd_export(args) -> int:
    import pandas as pd

    audit_root = Path(args.audit_dir)
    src_root = Path(args.source_dir)
    rec_root = Path(args.recovered_dir) if args.recovered_dir else None
    out_path = Path(args.out)

    decisions_by_split: dict[str, list[dict]] = {}
    source_by_split: dict[str, list[dict]] = {}
    for split in SPLITS:
        decisions_by_split[split] = load_jsonl(audit_root / split / "decisions.jsonl")
        source_by_split[split] = load_jsonl(src_root / f"{split}.jsonl")

    summary_rows: list[dict] = []
    for split in SPLITS:
        decisions = decisions_by_split[split]
        if not decisions:
            continue
        per_type: dict[str, Counter] = {}
        for d in decisions:
            ctype = d["corruption_type"] or "unknown"
            verdict, _ = classify_verdict(d, ctype)
            per_type.setdefault(ctype, Counter())[verdict] += 1
        for ctype in ("clean", "contradiction", "missing_tool", "overgeneration"):
            c = per_type.get(ctype, Counter())
            n = sum(c.values())
            if not n:
                continue
            summary_rows.append({
                "split": split, "corruption_type": ctype, "n": n,
                **{v: c.get(v, 0) for v in VERDICT_VALUES},
                "%confirmed": round(100 * c.get("confirmed", 0) / n, 1),
            })
    summary_df = pd.DataFrame(summary_rows)

    def decisions_df(decisions: list[dict], source: list[dict]) -> pd.DataFrame:
        by_id = {r["id"]: r for r in source}
        rows = []
        for d in decisions:
            rec = by_id.get(d["id"], {})
            ctype = d["corruption_type"] or "unknown"
            verdict, judge_verified = classify_verdict(d, ctype)
            lab = d.get("labeled_span") or {}
            judge = d.get("judge", {})
            rows.append({
                "id": d["id"],
                "corruption_type": ctype,
                "verdict": verdict,
                "judge_verified": judge_verified,
                "label_correct": judge.get("label_correct"),
                "extra_hallucination": judge.get("extra_hallucination", False),
                "labeled_span_text": lab.get("text", ""),
                "labeled_span_type": lab.get("label", ""),
                "labeled_span_start": lab.get("start", ""),
                "labeled_span_end": lab.get("end", ""),
                "judge_extra_text": judge.get("extra_text", ""),
                "judge_reasoning": judge.get("reasoning", ""),
                "parse_error": judge.get("parse_error", ""),
                "query": (rec.get("query") or "")[:300],
                "output_preview": (rec.get("output") or "")[:300],
            })
        return pd.DataFrame(rows)

    def recovered_df(records: list[dict]) -> pd.DataFrame:
        rows = []
        for r in records:
            lab = (r.get("hallucination_labels") or [{}])[0]
            rows.append({
                "id": r["id"],
                "new_label_type": lab.get("label"),
                "span_text": lab.get("text", "")[:300],
                "span_start": lab.get("start"),
                "span_end": lab.get("end"),
                "judge_reasoning": r.get("meta", {}).get("judge_reasoning", "")[:300],
                "query": r.get("query", "")[:200],
                "output_preview": (r.get("output") or "")[:300],
            })
        return pd.DataFrame(rows)

    def skipped_df(records: list[dict]) -> pd.DataFrame:
        rows = []
        for r in records:
            rows.append({
                "id": r.get("id"),
                "reason": r.get("reason"),
                "snippet": r.get("snippet", "")[:300],
                "confirm_is_hall": r.get("confirm", {}).get("is_hallucination", ""),
                "confirm_type": r.get("confirm", {}).get("type", ""),
                "confirm_reasoning": r.get("confirm", {}).get("reasoning", "")[:300],
            })
        return pd.DataFrame(rows)

    sheets: dict[str, pd.DataFrame] = {"Summary": summary_df}
    for split in SPLITS:
        if decisions_by_split[split]:
            sheets[f"Decisions_{split}"] = decisions_df(decisions_by_split[split], source_by_split[split])
    if rec_root is not None:
        for split in SPLITS:
            rec = load_jsonl(rec_root / f"{split}.jsonl")
            skip = load_jsonl(rec_root / f"{split}.skipped.jsonl")
            if rec:
                sheets[f"Recovered_{split}"] = recovered_df(rec)
            if skip:
                sheets[f"Skipped_{split}"] = skipped_df(skip)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            for col_idx, col in enumerate(df.columns, start=1):
                series = df[col].astype(str)
                width = min(60, max(12, series.map(len).max() if len(df) else 12))
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = width

    print(f"wrote {out_path}")
    for name, df in sheets.items():
        print(f"  {name[:31]:30s} {len(df):6d} rows")
    return 0


# ----------------------------------------------------------------------------
# CLI dispatcher
# ----------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the LLM judge over a split.")
    p_run.add_argument("--dataset-dir", default=str(ROOT / "data" / "combined"))
    p_run.add_argument("--split", default="validation")
    p_run.add_argument("--out-dir", default=str(ROOT / "data" / "quality_audit_openrouter"))
    p_run.add_argument("--max-records", type=int, default=0)
    p_run.add_argument("--no-lettuce", action="store_true")
    p_run.add_argument("--backend", choices=["local", "openrouter"], default="openrouter")
    p_run.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    p_run.add_argument("--workers", type=int, default=6)
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summary", help="Regenerate summary.md from existing decisions.jsonl.")
    p_sum.add_argument("--decisions", required=True)
    p_sum.add_argument("--source", required=True)
    p_sum.set_defaults(func=cmd_summary)

    p_rep = sub.add_parser("report", help="Emit decisions_with_verdict.jsonl + verdict_summary.md.")
    p_rep.add_argument("--audit-dir", required=True)
    p_rep.add_argument("--source-dir", required=True)
    p_rep.add_argument("--split", default="train")
    p_rep.set_defaults(func=cmd_report)

    p_filt = sub.add_parser("filter", help="Filter source split to judge-confirmed records.")
    p_filt.add_argument("--audit-dir", required=True)
    p_filt.add_argument("--source-dir", required=True)
    p_filt.add_argument("--split", default="train")
    p_filt.add_argument("--out-dir", required=True)
    p_filt.set_defaults(func=cmd_filter)

    p_exp = sub.add_parser("export", help="Export multi-sheet Excel report.")
    p_exp.add_argument("--audit-dir", required=True)
    p_exp.add_argument("--source-dir", required=True)
    p_exp.add_argument("--recovered-dir", default=None)
    p_exp.add_argument("--out", required=True)
    p_exp.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
