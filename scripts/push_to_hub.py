"""Push the generated hallucination dataset to the Hugging Face Hub.

Each per-type subdirectory under ``--data-dir`` (combined, contradiction,
missing_tool, overgeneration) becomes a separate dataset configuration on the
Hub, so consumers can pick the one they want via
``load_dataset(repo_id, "contradiction")`` etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


CONFIGS = ("combined", "contradiction", "missing_tool", "overgeneration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Target dataset repo, e.g. <user>/toolace-hallucination-spans.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--readme", default="DATASET_CARD.md")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS),
        help="Subset of configs to push (default: all).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    pushed: dict[str, dict[str, int]] = {}
    for config in args.configs:
        config_dir = data_root / config
        data_files = {
            "train": str(config_dir / "train.jsonl"),
            "validation": str(config_dir / "validation.jsonl"),
            "test": str(config_dir / "test.jsonl"),
        }
        missing = [split for split, path in data_files.items() if not Path(path).exists()]
        if missing:
            print(f"skip {config}: missing splits {missing}")
            continue
        dataset = load_dataset("json", data_files=data_files)
        dataset.push_to_hub(args.repo_id, config_name=config, private=args.private)
        pushed[config] = {split: dataset[split].num_rows for split in data_files}
        print(f"pushed config={config}: {pushed[config]}")

    readme = Path(args.readme)
    if readme.exists():
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"uploaded {readme} as README.md")

    print(json.dumps({"repo_id": args.repo_id, "pushed": pushed}, indent=2))


if __name__ == "__main__":
    main()
