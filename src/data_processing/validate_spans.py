import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_LABELS = {"contradiction", "overgeneration", "missing_tool"}
REQUIRED_TOP_LEVEL = {"id", "query", "context", "output", "hallucination_labels", "meta"}
REQUIRED_LABEL_FIELDS = {"start", "end", "text", "label"}
REQUIRED_META_FIELDS = {"source", "base_id", "corruption_type", "tools", "tool_call"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hallucination span JSONL files.")
    parser.add_argument("files", nargs="*", default=["data/train.jsonl", "data/validation.jsonl", "data/test.jsonl"])
    parser.add_argument("--allow-clean", action="store_true")
    args = parser.parse_args()

    stats = Counter()
    base_to_split: dict[str, str] = {}
    errors = []
    for file_name in args.files:
        path = Path(file_name)
        split = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stats["rows"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                    continue
                errors.extend(validate_record(record, path, line_number, args.allow_clean))

                meta = record.get("meta") if isinstance(record, dict) else {}
                base_id = meta.get("base_id") if isinstance(meta, dict) else None
                if base_id:
                    previous = base_to_split.setdefault(str(base_id), split)
                    if previous != split:
                        errors.append(
                            f"{path}:{line_number}: base_id {base_id!r} appears in both {previous!r} and {split!r}"
                        )
                corruption_type = meta.get("corruption_type") if isinstance(meta, dict) else None
                if corruption_type:
                    stats[f"type:{corruption_type}"] += 1
                    stats[f"split:{split}"] += 1

    if errors:
        print("\n".join(errors[:100]))
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors")
        raise SystemExit(1)

    print(json.dumps(dict(sorted(stats.items())), indent=2))


def validate_record(record: Any, path: Path, line_number: int, allow_clean: bool) -> list[str]:
    errors = []
    prefix = f"{path}:{line_number}"
    if not isinstance(record, dict):
        return [f"{prefix}: record must be an object"]

    missing = REQUIRED_TOP_LEVEL - set(record)
    if missing:
        errors.append(f"{prefix}: missing fields {sorted(missing)}")
        return errors

    for field in ("query", "context", "output"):
        if not isinstance(record[field], str) or not record[field].strip():
            errors.append(f"{prefix}: {field} must be a non-empty string")

    labels = record["hallucination_labels"]
    if not isinstance(labels, list):
        errors.append(f"{prefix}: hallucination_labels must be a list")
        return errors

    meta = record["meta"]
    if not isinstance(meta, dict):
        errors.append(f"{prefix}: meta must be an object")
        return errors
    missing_meta = REQUIRED_META_FIELDS - set(meta)
    if missing_meta:
        errors.append(f"{prefix}: meta missing fields {sorted(missing_meta)}")

    corruption_type = meta.get("corruption_type")
    if corruption_type == "clean":
        if labels and allow_clean:
            errors.append(f"{prefix}: clean sample must have no labels")
        elif not allow_clean:
            errors.append(f"{prefix}: clean sample found but --allow-clean was not set")
        return errors

    if corruption_type not in VALID_LABELS:
        errors.append(f"{prefix}: invalid corruption_type {corruption_type!r}")
    if not labels:
        errors.append(f"{prefix}: corrupted sample must have at least one label")

    output = record.get("output", "")
    for label_index, label in enumerate(labels):
        label_prefix = f"{prefix}:hallucination_labels[{label_index}]"
        errors.extend(validate_label(label, output, corruption_type, label_prefix))
    return errors


def validate_label(label: Any, output: str, corruption_type: str, prefix: str) -> list[str]:
    errors = []
    if not isinstance(label, dict):
        return [f"{prefix}: label must be an object"]
    missing = REQUIRED_LABEL_FIELDS - set(label)
    if missing:
        errors.append(f"{prefix}: missing fields {sorted(missing)}")
        return errors

    start = label.get("start")
    end = label.get("end")
    text = label.get("text")
    label_type = label.get("label")
    if not isinstance(start, int) or not isinstance(end, int):
        errors.append(f"{prefix}: start and end must be integers")
        return errors
    if start < 0 or end <= start or end > len(output):
        errors.append(f"{prefix}: invalid span [{start}, {end}) for output length {len(output)}")
        return errors
    if output[start:end] != text:
        errors.append(f"{prefix}: span text mismatch: expected {text!r}, got {output[start:end]!r}")
    if label_type != corruption_type:
        errors.append(f"{prefix}: label {label_type!r} does not match corruption_type {corruption_type!r}")
    if label_type not in VALID_LABELS:
        errors.append(f"{prefix}: invalid label {label_type!r}")
    return errors


if __name__ == "__main__":
    main()
