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


## Configs

| Config            | Records the hallucination_labels carry |
|-------------------|----------------------------------------|
| `combined`        | all three types + clean                |
| `contradiction`   | `contradiction` only + clean           |
| `missing_tool`    | `missing_tool` only + clean            |
| `overgeneration`  | `overgeneration` only + clean          |

Splits per config: `train` / `validation` / `test` (80/10/10 by deterministic
hash over `base_id` and all variants of one base stay in one split). Clean
negatives are included in every config for detectors abstain learning

## Record schema

```json
{
  "id": "toolace_train_00000__contradiction",
  "query": "User request text",
  "context": "Serialized tool/function response(s)",
  "output": "Assistant final answer (clean or corrupted)",
  "hallucination_labels": [
    {"start": 32, "end": 37, "text": "rainy", "label": "contradiction"}
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

`output[start:end] == text` for every label. Clean records have
`hallucination_labels: []` and `meta.corruption_type == "clean"`

## Corruption types

- **`contradiction`** : replace a grounded value in the answer with a same-type
  substitute drawn from a cross-record pool. >= 4 chars to avoid trivial
  single-token detection
- **`overgeneration`** : insert a content-free sentence at a random sentence
  boundary inside the answer
- **`missing_tool`** : insert an offer to use a tool not present in
  `meta.tools`

## Build statistics

| Item | Value |
|---|---|
| ToolACE rows scanned | 11,072 |
| Accepted by filters | 720 |
| Final combined dataset (train / val / test) | 1,955 / 241 / 235 |

Per-type record counts across configs (each clean record appears in all four):

| Type | Count |
|---|---:|
| clean | 1,324 |
| contradiction | 726 |
| missing_tool | 1,404 |
| overgeneration | 2,070 |
