"""Merge patched, recovered and reclassified records into the final dataset.

Sources:
  --patched-dir     <root>/<config>/<split>.jsonl   filtered + extra-label patches
  --recovered-dir   <root>/<split>.jsonl            new corrupted records from
                                                    judge-recovered false-negative cleans
  --other-dir       <root>/<split>.jsonl            judge-reclassified "other" records
                                                    (off-topic etc.)

Output:
  --out-dir/<config>/<split>.jsonl                  final per-config splits

For each split we union the three sources, deduplicate by id, and route records
into the right configurations:

  combined        all records
  contradiction   clean + records whose primary label is contradiction
  missing_tool    clean + records whose primary label is missing_tool
  overgeneration  clean + records whose primary label is overgeneration

Records with `meta.corruption_type == "other"` go ONLY into combined (no per-
type subset).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SPLITS = ("train", "validation", "test")
CONFIGS = ("combined", "contradiction", "missing_tool", "overgeneration")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()] if p.exists() else []


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def primary_label(record: dict) -> str:
    labels = record.get("hallucination_labels") or []
    if not labels:
        return record["meta"].get("corruption_type", "clean")
    # First label is the canonical one (set by build pipeline / recover / other)
    return labels[0]["label"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patched-dir",   default="data/combined_patched")
    p.add_argument("--recovered-dir", default="data/recovered")
    p.add_argument("--other-dir",     default="data/other")
    p.add_argument("--out-dir",       default="data/final")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    summary = {split: Counter() for split in SPLITS}

    for split in SPLITS:
        # 1. Start from patched/combined (full set, possibly with multi-labels).
        records_by_id: dict[str, dict] = {}
        for r in load_jsonl(Path(args.patched_dir) / "combined" / f"{split}.jsonl"):
            records_by_id[r["id"]] = r

        # 2. Add recovered records (former cleans, now labeled corruptions).
        for r in load_jsonl(Path(args.recovered_dir) / f"{split}.jsonl"):
            records_by_id[r["id"]] = r

        # `other` records (corruption_type="other") are excluded from the final
        # dataset: they violate the validator's strict 3-type taxonomy and are
        # kept only under data/other/<split>.jsonl for downstream consumers who
        # want them. Skipped here by not loading from --other-dir.

        # Drop any secondary label whose label.label != meta.corruption_type so
        # each record has labels of a single type matching its corruption_type
        # (the validator enforces this). This loses the patch-added "extra"
        # label when its type differs from the primary; that information is
        # preserved in data/combined_patched/ if needed.
        cleaned: list[dict] = []
        for r in records_by_id.values():
            primary_type = r["meta"].get("corruption_type")
            kept = [lab for lab in r.get("hallucination_labels", []) if lab.get("label") == primary_type]
            if r.get("hallucination_labels") and not kept and primary_type != "clean":
                # All labels were stripped (paranoia branch — shouldn't happen
                # because the primary label always matches by construction).
                continue
            r = dict(r)
            r["hallucination_labels"] = kept
            cleaned.append(r)
        all_records = cleaned

        # Route into per-config slices.
        for cfg in CONFIGS:
            slice_records: list[dict] = []
            for rec in all_records:
                ctype = rec["meta"].get("corruption_type", "clean")
                if cfg == "combined":
                    slice_records.append(rec)
                else:
                    # per-type config: keep clean + records with that primary corruption_type
                    primary = primary_label(rec)
                    if ctype == "clean" or primary == cfg:
                        slice_records.append(rec)
            write_jsonl(out_root / cfg / f"{split}.jsonl", slice_records)
            summary[split][cfg] = len(slice_records)
        # Also count by corruption_type (combined view)
        for rec in all_records:
            summary[split][f"type:{rec['meta'].get('corruption_type','?')}"] += 1
        # Multi-label rows (patched)
        multi = sum(1 for rec in all_records if len(rec.get("hallucination_labels", [])) >= 2)
        summary[split]["multi_label"] = multi

    print("=== final dataset summary ===")
    for split in SPLITS:
        s = summary[split]
        print(f"[{split}]")
        for cfg in CONFIGS:
            print(f"  {cfg:20s} {s[cfg]:5d}")
        print(f"  by type: " + "  ".join(f"{k.split(':')[1]}={v}" for k, v in s.items() if k.startswith("type:")))
        print(f"  multi-label rows: {s['multi_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
