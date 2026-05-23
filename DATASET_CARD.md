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

## Build statistics

| Item | Value |
|---|---|
| Rows scanned | 11,072 |
| Rows accepted | 720 |
| Records by type | clean: 720, contradiction: 486, missing_tool: 720, overgeneration: 720 |
| Records (combined config) | train 2,118 / val 264 / test 264 |
| Splits | 80 / 10 / 10 by deterministic hash over `base_id` (all variants of one base stay in one split) |

## Zero-shot baseline (validation split)

| Config | Lexical baseline F1 | LettuceDetect F1 |
|---|---|---|
| `combined` (n=264) | 0.156 | 0.198 |
| `contradiction` (n=120) | 0.023 | 0.030 |
| `missing_tool` (n=144) | 0.104 | 0.118 |
| `overgeneration` (n=144) | 0.167 | 0.225 |

- `lexical baseline`: char marked as hallucinated iff it belongs to a content word that is absent from the tool context. Recall is high (~0.75), precision is very low (~0.09) — i.e. tool answers mention many context-absent words even on clean records.
- `LettuceDetect`: zero-shot inference with [`KRLabsOrg/lettucedect-base-modernbert-en-v1`](https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1). The model was trained on RAGTruth (news/QA domain), not tool-calling, so its precision in this domain is also low (~0.11) but it consistently improves recall and F1 over the lexical baseline.

Reproducible from this repo:

```bash
python scripts/zero_shot_eval.py --dataset-dir data/combined --split validation
python scripts/zero_shot_eval.py --dataset-dir data/contradiction --split validation
python scripts/zero_shot_eval.py --dataset-dir data/missing_tool --split validation
python scripts/zero_shot_eval.py --dataset-dir data/overgeneration --split validation
```

The per-record / per-type breakdown is written to `validation_report_validation.{md,json}` inside each dataset directory.

## Build

```bash
python -m pip install -r requirements.txt
python scripts/build_from_toolace.py
python scripts/validate_spans.py --allow-clean data/combined/*.jsonl
```

## Push to the Hub

```bash
python scripts/push_to_hub.py <user>/toolace-hallucination-spans --private
```

This pushes all four configs (`combined`, `contradiction`, `missing_tool`, `overgeneration`) as separate dataset configurations.

## Known limitations

- Corruptions are synthetic and deterministic: each base example produces the same corruption seed-by-seed across runs. Useful for supervised span localization, but does not cover naturally occurring hallucination patterns (e.g. cascading errors across multi-turn dialogue).
- `missing_tool` actions come from a curated list — they do not cover every tool capability that might be implied by an arbitrary user query.
- `contradiction` quality depends on whether the clean final answer contains values that can be plausibly substituted. Examples without any grounded value of length ≥4 in the output are simply skipped, which is why `contradiction` has fewer records (486) than the other types (720 each).
- The context is serialized JSON from tool/function messages, not re-rendered into natural language, so token positions and length distribution differ from RAGTruth's news-corpus context.
- Baseline numbers above are zero-shot; we have not fine-tuned a detector yet. Fine-tuning is the natural next step (see the `Improve baselines` section of the task spec).
