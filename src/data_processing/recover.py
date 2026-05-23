"""Judge-driven label recovery subcommands.

Subcommands:
  cleans       For records the audit marked clean but where the judge found a
               hidden hallucination, locate the extra span and emit a brand new
               labeled record (corruption_source = "judge_recovered").
  extra-spans  For corrupted records where the judge confirmed our label AND
               flagged a *second* hallucination, append an extra label to the
               existing record (extra_label_source = "judge_recovered").
  other        Re-classify confirm-rejected spans under an extended taxonomy that
               adds `off_topic`. Salvaged records get corruption_type = "other".

All three use the same judge (OpenRouter gpt-oss-120b by default), the same
prompts (CONFIRM_SYSTEM / RECLASSIFY_SYSTEM from common.py), and the same span
localization helper (find_span).
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .common import (
    CONFIRM_SYSTEM,
    CONFIRM_USER_TEMPLATE,
    RECLASSIFY_SYSTEM,
    RECLASSIFY_USER_TEMPLATE,
    OpenRouterJudge,
    collect_tool_names,
    derive_type_hint,
    find_span,
    load_jsonl,
    parse_json_object,
    truncate,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
CONFIGS = ("combined", "contradiction", "missing_tool", "overgeneration")
JUDGE_MODEL_DEFAULT = "openai/gpt-oss-120b:free"


# ----------------------------------------------------------------------------
# Shared judge call helpers
# ----------------------------------------------------------------------------


def _confirm_user(rec: dict, start: int, end: int, hint: str) -> str:
    return CONFIRM_USER_TEMPLATE.format(
        query=truncate(rec["query"], 600),
        context=truncate(rec["context"], 1500),
        tool_names=collect_tool_names(rec.get("meta", {})),
        output=truncate(rec["output"], 1200),
        span=rec["output"][start:end],
        hint=hint,
    )


def _reclassify_user(rec: dict, start: int, end: int) -> str:
    return RECLASSIFY_USER_TEMPLATE.format(
        query=truncate(rec["query"], 600),
        context=truncate(rec["context"], 1500),
        tool_names=collect_tool_names(rec.get("meta", {})),
        output=truncate(rec["output"], 1200),
        span=rec["output"][start:end],
    )


def _retry_judge(judge, system: str, user: str, attempts: int = 3) -> dict:
    for attempt in range(attempts):
        try:
            return parse_json_object(judge(system, user))
        except Exception as e:  # noqa: BLE001
            if attempt == attempts - 1:
                return {"_parse_error": str(e)[:120]}
            time.sleep(1.5 * (attempt + 1))
    return {"_parse_error": "exhausted"}


def _run_pool(items: list, work, workers: int, label: str) -> list:
    results: list = [None] * len(items)  # type: ignore
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, item): i for i, item in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(items):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(items) - done) / max(rate, 1e-6)
                print(f"  {label} {done}/{len(items)} | {rate:.2f} rec/s | ETA {eta:.0f}s", flush=True)
    return results


# ----------------------------------------------------------------------------
# Subcommand: cleans
# ----------------------------------------------------------------------------


def cmd_cleans(args) -> int:
    decisions = {d["id"]: d for d in load_jsonl(Path(args.decisions))}
    source = {r["id"]: r for r in load_jsonl(Path(args.source))}
    split_name = args.split or Path(args.source).stem
    out_dir = Path(args.out_dir)

    candidates: list[dict] = []
    for sid, d in decisions.items():
        if d["corruption_type"] != "clean":
            continue
        judge_dec = d.get("judge", {})
        if not judge_dec.get("extra_hallucination"):
            continue
        extra_text = (judge_dec.get("extra_text") or "").strip()
        if not extra_text or sid not in source:
            continue
        candidates.append({"id": sid, "decision": d, "extra_text": extra_text})
    print(f"candidate false-negative cleans: {len(candidates)}")

    recovered: list[dict] = []
    skipped: list[dict] = []
    for c in candidates:
        rec = source[c["id"]]
        loc = find_span(rec["output"], c["extra_text"])
        if loc is None:
            skipped.append({"id": c["id"], "reason": "span_not_found",
                            "snippet": c["extra_text"][:200]})
            continue
        start, end = loc
        recovered.append({
            "id": c["id"], "rec": rec, "start": start, "end": end,
            "type_hint": derive_type_hint(c["decision"]["judge"].get("reasoning", "")),
            "judge_reasoning": c["decision"]["judge"].get("reasoning", ""),
        })
    print(f"located in output: {len(recovered)} ({len(skipped)} couldn't locate)")

    if not args.no_confirm and recovered:
        judge = OpenRouterJudge(args.judge_model)

        def _work(item: dict) -> dict:
            item["confirm"] = _retry_judge(judge, CONFIRM_SYSTEM,
                                            _confirm_user(item["rec"], item["start"], item["end"], item["type_hint"]))
            return item

        recovered = _run_pool(recovered, _work, args.workers, "confirm")

    final: list[dict] = []
    type_counter: Counter = Counter()
    rejected_by_confirm = 0
    for item in recovered:
        confirm = item.get("confirm", {})
        if args.no_confirm:
            ctype = item["type_hint"]
            is_hall = True
        else:
            if "_parse_error" in confirm:
                skipped.append({"id": item["id"], "reason": f"confirm_parse_error:{confirm['_parse_error']}"})
                continue
            is_hall = bool(confirm.get("is_hallucination"))
            ctype = str(confirm.get("type", "")).strip().lower()
            if ctype not in ("contradiction", "overgeneration", "missing_tool"):
                ctype = item["type_hint"]
        if not is_hall:
            rejected_by_confirm += 1
            skipped.append({"id": item["id"], "reason": "confirm_rejected", "confirm": confirm})
            continue
        rec = item["rec"]
        new_meta = dict(rec.get("meta", {}))
        new_meta.update({
            "corruption_type": ctype,
            "corruption_source": "judge_recovered",
            "judge_reasoning": item["judge_reasoning"],
            "previous_corruption_type": "clean",
        })
        final.append({
            "id": rec["id"] + "__recovered",
            "query": rec["query"],
            "context": rec["context"],
            "output": rec["output"],
            "hallucination_labels": [{
                "start": item["start"], "end": item["end"],
                "text": rec["output"][item["start"]:item["end"]], "label": ctype,
            }],
            "meta": new_meta,
        })
        type_counter[ctype] += 1

    write_jsonl(out_dir / f"{split_name}.jsonl", final)
    write_jsonl(out_dir / f"{split_name}.skipped.jsonl", skipped)
    print(f"\nrecovered: {len(final)} {dict(type_counter)}")
    print(f"skipped:   {len(skipped)} (confirm-rejected: {rejected_by_confirm})")
    return 0


# ----------------------------------------------------------------------------
# Subcommand: extra-spans
# ----------------------------------------------------------------------------


def _overlaps(start: int, end: int, labels: list[dict]) -> bool:
    return any(max(lab["start"], start) < min(lab["end"], end) for lab in labels)


def cmd_extra_spans(args) -> int:
    decisions_by_split: dict[str, dict[str, dict]] = {
        split: {d["id"]: d for d in load_jsonl(Path(args.audit_dir) / split / "decisions.jsonl")}
        for split in SPLITS
    }
    judge = OpenRouterJudge(args.judge_model)
    patch_jobs: dict[str, list[dict]] = {split: [] for split in SPLITS}

    for split in SPLITS:
        src_path = Path(args.filtered_dir) / "combined" / f"{split}.jsonl"
        for rec in load_jsonl(src_path):
            if rec["meta"].get("corruption_type") == "clean":
                continue
            d = decisions_by_split[split].get(rec["id"])
            if not d or not d.get("judge", {}).get("extra_hallucination"):
                continue
            extra_text = (d["judge"].get("extra_text") or "").strip()
            if not extra_text:
                continue
            loc = find_span(rec["output"], extra_text)
            if loc is None:
                patch_jobs[split].append({"rec": rec, "decision": d, "loc": None,
                                           "extra_text": extra_text, "skip_reason": "span_not_found"})
                continue
            start, end = loc
            if _overlaps(start, end, rec["hallucination_labels"]):
                patch_jobs[split].append({"rec": rec, "decision": d, "loc": None,
                                           "extra_text": extra_text, "skip_reason": "overlaps_existing"})
                continue
            patch_jobs[split].append({"rec": rec, "decision": d, "loc": (start, end),
                                       "extra_text": extra_text})

    for split in SPLITS:
        located = sum(1 for j in patch_jobs[split] if j["loc"] is not None)
        print(f"[{split}] patch candidates: {len(patch_jobs[split])} (located {located})")

    def _confirm_job(job: dict) -> dict:
        if job["loc"] is None:
            return job
        start, end = job["loc"]
        hint = derive_type_hint(job["decision"]["judge"].get("reasoning", ""))
        job["confirm"] = _retry_judge(judge, CONFIRM_SYSTEM,
                                       _confirm_user(job["rec"], start, end, hint))
        return job

    for split in SPLITS:
        jobs = [j for j in patch_jobs[split] if j["loc"] is not None]
        if jobs:
            print(f"[{split}] confirming {len(jobs)} located patches...")
            confirmed = _run_pool(jobs, _confirm_job, args.workers, label=f"{split}")
            patch_jobs[split] = [j for j in patch_jobs[split] if j["loc"] is None] + confirmed

    patches_by_id: dict[str, dict[str, dict]] = {split: {} for split in SPLITS}
    log_by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for job in patch_jobs[split]:
            rid = job["rec"]["id"]
            if job["loc"] is None:
                log_by_split[split].append({
                    "id": rid, "status": "skip",
                    "reason": job.get("skip_reason", "span_not_found"),
                    "extra_text_preview": job["extra_text"][:200],
                })
                continue
            confirm = job.get("confirm", {})
            if "_parse_error" in confirm:
                log_by_split[split].append({"id": rid, "status": "skip",
                                              "reason": f"confirm_parse_error:{confirm['_parse_error']}"})
                continue
            if not confirm.get("is_hallucination"):
                log_by_split[split].append({"id": rid, "status": "skip",
                                              "reason": "confirm_rejected", "confirm": confirm})
                continue
            ctype = str(confirm.get("type", "")).strip().lower()
            if ctype not in ("contradiction", "overgeneration", "missing_tool"):
                ctype = derive_type_hint(job["decision"]["judge"].get("reasoning", ""))
            start, end = job["loc"]
            patches_by_id[split][rid] = {
                "start": start, "end": end,
                "text": job["rec"]["output"][start:end], "label": ctype,
            }
            log_by_split[split].append({"id": rid, "status": "applied", "label": ctype,
                                          "span_text": job["rec"]["output"][start:end][:200]})

    totals: Counter = Counter()
    for cfg in CONFIGS:
        for split in SPLITS:
            src = Path(args.filtered_dir) / cfg / f"{split}.jsonl"
            if not src.exists():
                continue
            out_records: list[dict] = []
            for rec in load_jsonl(src):
                new_label = patches_by_id[split].get(rec["id"])
                if new_label is not None and not _overlaps(new_label["start"], new_label["end"],
                                                             rec["hallucination_labels"]):
                    rec = json.loads(json.dumps(rec))
                    rec["hallucination_labels"] = list(rec["hallucination_labels"]) + [new_label]
                    rec.setdefault("meta", {})["extra_label_source"] = "judge_recovered"
                    totals[f"{cfg}/{split}"] += 1
                out_records.append(rec)
            write_jsonl(Path(args.out_dir) / cfg / f"{split}.jsonl", out_records)

    for split in SPLITS:
        write_jsonl(Path(args.out_dir) / "combined" / f"{split}.patch_log.jsonl",
                    log_by_split[split])

    print("\napplied:")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {args.out_dir}/<config>/<split>.jsonl and combined/*.patch_log.jsonl")
    return 0


# ----------------------------------------------------------------------------
# Subcommand: other
# ----------------------------------------------------------------------------


def _collect_other_candidates(split: str, audit_dir: Path, source_dir: Path,
                                recovered_dir: Path, patched_dir: Path) -> list[dict]:
    source_by_id = {r["id"]: r for r in load_jsonl(source_dir / f"{split}.jsonl")}
    decisions_by_id = {
        d["id"]: d for d in load_jsonl(audit_dir / split / "decisions.jsonl")
    }
    candidates: list[dict] = []
    for row in load_jsonl(recovered_dir / f"{split}.skipped.jsonl"):
        if not str(row.get("reason", "")).startswith("confirm_rejected"):
            continue
        sid = row["id"]
        if sid not in source_by_id:
            continue
        extra = (decisions_by_id.get(sid, {}).get("judge", {}).get("extra_text") or "").strip()
        if not extra:
            continue
        candidates.append({"id": sid, "rec": source_by_id[sid], "extra_text": extra,
                            "origin": "recover"})
    for row in load_jsonl(patched_dir / "combined" / f"{split}.patch_log.jsonl"):
        if row.get("status") != "skip" or not str(row.get("reason", "")).startswith("confirm_rejected"):
            continue
        sid = row["id"]
        if sid not in source_by_id:
            continue
        extra = (decisions_by_id.get(sid, {}).get("judge", {}).get("extra_text") or "").strip()
        if not extra:
            continue
        candidates.append({"id": sid, "rec": source_by_id[sid], "extra_text": extra,
                            "origin": "patch"})
    return candidates


def cmd_other(args) -> int:
    judge = OpenRouterJudge(args.judge_model)
    out_root = Path(args.out_dir)
    grand_total: Counter = Counter()

    audit_dir = Path(args.audit_dir)
    source_dir = Path(args.source_dir)
    recovered_dir = Path(args.recovered_dir)
    patched_dir = Path(args.patched_dir)

    for split in SPLITS:
        cands = _collect_other_candidates(split, audit_dir, source_dir, recovered_dir, patched_dir)
        print(f"[{split}] candidates: {len(cands)}", flush=True)

        def _work(item: dict) -> dict:
            loc = find_span(item["rec"]["output"], item["extra_text"])
            if loc is None:
                item["loc"] = None
                return item
            item["loc"] = loc
            item["reply"] = _retry_judge(judge, RECLASSIFY_SYSTEM,
                                          _reclassify_user(item["rec"], loc[0], loc[1]))
            return item

        results = _run_pool(cands, _work, args.workers, label=split)

        added: list[dict] = []
        notes: list[dict] = []
        per_category: Counter = Counter()
        for item in results:
            note = {"id": item["id"], "origin": item["origin"],
                    "extra_text_preview": item["extra_text"][:200]}
            if item.get("loc") is None:
                note["status"] = "span_not_found"
                notes.append(note); continue
            reply = item.get("reply", {})
            if "_parse_error" in reply:
                note["status"] = f"parse_error:{reply['_parse_error']}"
                notes.append(note); continue
            cat = str(reply.get("category", "")).strip().lower()
            note["category"] = cat
            note["reasoning"] = reply.get("reasoning", "")
            if cat not in ("contradiction", "overgeneration", "missing_tool", "off_topic"):
                note["status"] = "not_hallucination"
                notes.append(note); continue
            rec = item["rec"]
            start, end = item["loc"]
            new_meta = dict(rec.get("meta", {}))
            new_meta.update({
                "corruption_type": "other",
                "subtype": cat,
                "corruption_source": "judge_reclassified",
                "previous_corruption_type": rec["meta"].get("corruption_type"),
                "reclassify_reasoning": reply.get("reasoning", ""),
            })
            added.append({
                "id": rec["id"] + "__other",
                "query": rec["query"],
                "context": rec["context"],
                "output": rec["output"],
                "hallucination_labels": [{
                    "start": start, "end": end,
                    "text": rec["output"][start:end], "label": "other",
                }],
                "meta": new_meta,
            })
            note["status"] = "added"
            notes.append(note)
            per_category[cat] += 1

        write_jsonl(out_root / f"{split}.jsonl", added)
        write_jsonl(out_root / f"{split}.notes.jsonl", notes)
        print(f"[{split}] added: {len(added)} {dict(per_category)}", flush=True)
        for k, v in per_category.items():
            grand_total[k] += v

    print(f"\ntotal added (all splits): {sum(grand_total.values())} {dict(grand_total)}")
    return 0


# ----------------------------------------------------------------------------
# CLI dispatcher
# ----------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cl = sub.add_parser("cleans", help="Recover false-negative cleans as labeled records.")
    p_cl.add_argument("--decisions", required=True)
    p_cl.add_argument("--source", required=True)
    p_cl.add_argument("--out-dir", required=True)
    p_cl.add_argument("--split", default=None)
    p_cl.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    p_cl.add_argument("--no-confirm", action="store_true")
    p_cl.add_argument("--workers", type=int, default=6)
    p_cl.set_defaults(func=cmd_cleans)

    p_ex = sub.add_parser("extra-spans", help="Add extra labels to confirmed-corrupted records.")
    p_ex.add_argument("--filtered-dir", default=str(ROOT / "data" / "combined_filtered"))
    p_ex.add_argument("--audit-dir",    default=str(ROOT / "data" / "quality_audit_openrouter" / "combined"))
    p_ex.add_argument("--out-dir",      default=str(ROOT / "data" / "combined_patched"))
    p_ex.add_argument("--judge-model",  default=JUDGE_MODEL_DEFAULT)
    p_ex.add_argument("--workers",      type=int, default=6)
    p_ex.set_defaults(func=cmd_extra_spans)

    p_ot = sub.add_parser("other", help="Reclassify confirm-rejected spans into `other`.")
    p_ot.add_argument("--audit-dir",     default=str(ROOT / "data" / "quality_audit_openrouter" / "combined"))
    p_ot.add_argument("--source-dir",    default=str(ROOT / "data" / "combined"))
    p_ot.add_argument("--recovered-dir", default=str(ROOT / "data" / "recovered"))
    p_ot.add_argument("--patched-dir",   default=str(ROOT / "data" / "combined_patched"))
    p_ot.add_argument("--out-dir",       default=str(ROOT / "data" / "other"))
    p_ot.add_argument("--judge-model",   default=JUDGE_MODEL_DEFAULT)
    p_ot.add_argument("--workers",       type=int, default=6)
    p_ot.set_defaults(func=cmd_other)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
