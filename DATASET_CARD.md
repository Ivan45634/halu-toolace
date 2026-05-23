---
license: apache-2.0
language:
- en
task_categories:
- text-classification
- token-classification
tags:
- hallucination-detection
- tool-calling
- span-labeling
- synthetic-corruption
- toolace
- ragtruth-format
pretty_name: ToolACE Hallucination Spans
size_categories:
- 1K<n<10K
configs:
- config_name: combined
  data_files:
    - split: train
      path: combined/train-*.parquet
    - split: validation
      path: combined/validation-*.parquet
    - split: test
      path: combined/test-*.parquet
- config_name: contradiction
  data_files:
    - split: train
      path: contradiction/train-*.parquet
    - split: validation
      path: contradiction/validation-*.parquet
    - split: test
      path: contradiction/test-*.parquet
- config_name: missing_tool
  data_files:
    - split: train
      path: missing_tool/train-*.parquet
    - split: validation
      path: missing_tool/validation-*.parquet
    - split: test
      path: missing_tool/test-*.parquet
- config_name: overgeneration
  data_files:
    - split: train
      path: overgeneration/train-*.parquet
    - split: validation
      path: overgeneration/validation-*.parquet
    - split: test
      path: overgeneration/test-*.parquet
---

# ToolACE Hallucination Spans

Span-level hallucination dataset for tool-calling dialogues, derived from [`minpeter/toolace-parsed`](https://huggingface.co/datasets/minpeter/toolace-parsed). Stored in the RAGTruth-compatible schema (`query`, `context`, `output`, `hallucination_labels`).

The dataset is intended for the task described in *Hallucination Detection in Tool Calling*: train and evaluate span-level detectors (LettuceDetect-style, LookBackLens, etc.) on tool-grounded model responses, then improve over baselines.

## Configs

Four configurations are published. Each config has `train` / `validation` / `test` splits and the same record schema; the only difference is which hallucination type is present.

| Config            | Purpose                                                                 | Hallucinated examples carry label(s) |
|-------------------|-------------------------------------------------------------------------|--------------------------------------|
| `combined`        | All three corruption types mixed with clean negatives. Use for general training. | `contradiction`, `missing_tool`, `overgeneration` |
| `contradiction`   | Only `contradiction` corruptions + clean negatives. Focused evaluation. | `contradiction` |
| `missing_tool`    | Only `missing_tool` corruptions + clean negatives.                      | `missing_tool` |
| `overgeneration`  | Only `overgeneration` corruptions + clean negatives.                    | `overgeneration` |

Clean negatives (records with `hallucination_labels: []`) are included in every config so detectors learn to abstain instead of always firing.

## Record schema

```json
{
  "id": "toolace_train_00000__contradiction",
  "query": "User request text",
  "context": "Serialized tool/function response(s) (JSON-encoded per turn)",
  "output": "Assistant final answer (clean or corrupted)",
  "hallucination_labels": [
    {
      "start": 32,
      "end": 37,
      "text": "rainy",
      "label": "contradiction"
    }
  ],
  "meta": {
    "source": "minpeter/toolace-parsed",
    "base_id": "toolace_train_00000",
    "corruption_type": "contradiction",
    "tools": [],
    "tool_call": {}
  }
}
```

`output[start:end] == text` for every label. Clean records have `hallucination_labels: []` and `meta.corruption_type == "clean"`.

## Corruption types

`contradiction` — replace a grounded value in the assistant's final answer with a plausible-but-wrong alternative. Replacement values are sourced from a cross-record pool of grounded entities and statuses (e.g. a city is swapped with another city seen elsewhere in ToolACE) so the contradiction cannot be detected by trivial "not in context" lookups. Single-character substitutions (e.g. one digit) are not allowed — every contradiction span is at least 4 characters.

`overgeneration` — insert a sentence that adds claims not supported by the tool response (industry generalizations, temporal stability claims, comparative recommendations). The insertion position is chosen randomly across sentence boundaries within the answer, not just at the end, to remove positional bias.

`missing_tool` — insert a sentence offering an action that requires a tool not present in the available tool schemas (flight booking, payment, calendar, etc.). Insertion position is also randomized across sentence boundaries.

## Source filtering

Only ToolACE rows that pass all of the following are kept:

- non-empty user query
- at least one available tool schema
- at least one assistant tool call
- at least one tool/function response
- a non-empty assistant final text after the tool response
- `80 <= len(output) <= 2500` characters (avoid trivial or overlong answers)
- `len(query_words ∩ output_words) / len(query_words) >= 0.10` — drops off-topic rows that the parsed ToolACE occasionally contains (e.g. caste-system query answered with a linked-list explanation), which would otherwise pollute the corruption signal.

## Pipeline

The dataset is built in four stages:

1. **Build** (`build_from_toolace.py` + `corruptors.py`) — load ToolACE, filter, inject regex-based corruptions.
2. **Audit** (`audit run` with `openai/gpt-oss-120b:free` via OpenRouter) — LLM-as-judge validates every label.
3. **Recover + Patch** (`recover cleans`, `recover extra-spans`) — salvage records the audit found problematic:
   - `recover` turns false-negative cleans (clean records where judge spotted a hallucination) into labeled records.
   - `patch` adds extra labels to confirmed-corrupted records where judge found a *second* hallucination.
4. **Merge** (`merge_final.py`) — assemble `data/final/<config>/<split>.jsonl` from patched + recovered, dropping labels whose type doesn't match the record's primary `corruption_type` to satisfy the strict RAGTruth one-type-per-record schema.

## Build statistics

| Item | Value |
|---|---|
| ToolACE rows scanned | 11,072 |
| Rows accepted by schema + length + overlap filters | 720 |
| Records after regex corruption | 2,646 (720 clean + 486 contradiction + 720 missing_tool + 720 overgeneration) |
| Records confirmed by LLM judge (gpt-oss-120b) | 1,675 / 209 / 205 (train/val/test) |
| Records recovered from false-negative cleans | 280 / 32 / 30 |
| Final dataset (combined config) | **train 1,955 / val 241 / test 235** = 2,431 records |
| Splits | 80 / 10 / 10 by deterministic hash over `base_id`; all variants of one base stay in one split |

Counts per type across all configs (each clean record appears in all four):

| Type | Total |
|---|---:|
| clean | 1,324 |
| contradiction | 726 |
| missing_tool | 1,404 |
| overgeneration | 2,070 |

## Zero-shot baseline (validation split)

> Numbers measured before the LLM-as-judge audit and recovery. They establish the pre-audit floor.

| Config | Lexical baseline F1 | LettuceDetect F1 |
|---|---|---|
| `combined` (n=264) | 0.156 | 0.198 |
| `contradiction` (n=120) | 0.023 | 0.030 |
| `missing_tool` (n=144) | 0.104 | 0.118 |
| `overgeneration` (n=144) | 0.167 | 0.225 |

- `lexical baseline`: char marked as hallucinated iff it belongs to a content word that is absent from the tool context. Recall is high (~0.75), precision is very low (~0.09).
- `LettuceDetect`: zero-shot inference with [`KRLabsOrg/lettucedect-base-modernbert-en-v1`](https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1). Trained on RAGTruth (news/QA domain), not tool-calling, so precision in this domain is also low (~0.11) but consistently improves recall and F1 over lexical.

Reproducible from this repo:

```bash
python -m src.data_processing.zero_shot_eval --dataset-dir data/final/combined --split validation
python -m src.data_processing.zero_shot_eval --dataset-dir data/final/contradiction --split validation
python -m src.data_processing.zero_shot_eval --dataset-dir data/final/missing_tool --split validation
python -m src.data_processing.zero_shot_eval --dataset-dir data/final/overgeneration --split validation
```

## Build

```bash
python -m pip install -r requirements.txt
python -m src.data_processing.build_from_toolace                    # → data/combined/...
python -m src.data_processing.audit run --backend openrouter \
    --judge-model openai/gpt-oss-120b:free \
    --dataset-dir data/combined --split train           # repeat for val/test
python -m src.data_processing.audit filter \
    --audit-dir data/quality_audit_openrouter/combined \
    --source-dir data/combined --split train \
    --out-dir data/combined_filtered/combined
python -m src.data_processing.recover cleans \
    --decisions data/quality_audit_openrouter/combined/train/decisions.jsonl \
    --source data/combined/train.jsonl \
    --out-dir data/recovered --split train
python -m src.data_processing.recover extra-spans
python -m src.data_processing.merge_final                            # → data/final/
python -m src.data_processing.validate_spans --allow-clean \
    data/final/combined/*.jsonl data/final/*/*.jsonl
```

## Push to the Hub

```bash
python -m src.data_processing.push_to_hub <user>/toolace-hallucination-spans \
    --data-dir data/final --readme DATASET_CARD.md
```

`--data-dir data/final` is the default; pass `--public` to make the repo public.

## Known limitations

- **Synthetic & deterministic.** Regex-based corruptions plus LLM-recovered real hallucinations. They don't cover naturally occurring cascading errors across multi-turn dialogue.
- **Strict single-type schema.** Each record is annotated with a single corruption_type, and every label in the record must match it. When the audit found a *second* hallucination of a different type inside an already-corrupted record, the secondary label is dropped at merge — **the hallucinated text remains in the output but is no longer annotated**. This affected 288 records (231 train / 30 val / 27 test). The unannotated spans are preserved in `data/combined_patched/` for downstream consumers who want them.
- **Off-topic answers** (data is grounded but the answer doesn't address the user query) don't fit the 3-type RAGTruth taxonomy. They are collected separately under `data/other/` (47 records) and excluded from the final dataset to keep the validator schema strict.
- **`contradiction` is the hardest type.** Single-character substitutions are not allowed (`MIN_CONTRADICTION_LEN = 4`), and records without any grounded value of length ≥4 in the output are simply skipped, so `contradiction` has fewer records than the other types.
- **`missing_tool` actions** come from a curated list — they don't cover every tool capability that might be implied by an arbitrary user query.
- **Context format**: JSON-serialized tool messages, not re-rendered into natural language, so token positions and length distribution differ from RAGTruth's news-corpus context.
- **Pre-fine-tune baselines only.** Numbers above are zero-shot. Fine-tuning is the natural next step (see the `Improve baselines` section of the task spec).
