# halu-toolace

Span-level hallucination detection dataset and pipeline for tool-calling dialogues,
built on top of [`minpeter/toolace-parsed`](https://huggingface.co/datasets/minpeter/toolace-parsed).

- **Dataset on Hugging Face:** [`Ivan1008/toolace-hallucination-spans`](https://huggingface.co/datasets/Ivan1008/toolace-hallucination-spans)
- **Schema:** RAGTruth-compatible (`query`, `context`, `output`, `hallucination_labels`)
- **Configs:** `combined`, `contradiction`, `missing_tool`, `overgeneration`

## Quickstart

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 1. Build the dataset locally from ToolACE
.venv/bin/python scripts/build_from_toolace.py

# 2. Validate spans and run zero-shot baselines
.venv/bin/python scripts/validate_spans.py --allow-clean data/combined/*.jsonl
.venv/bin/python scripts/zero_shot_eval.py --dataset-dir data/combined --split validation

# 3. (Optional) LLM-as-judge quality audit
.venv/bin/python scripts/quality_audit.py --dataset-dir data/combined --split validation

# 4. Push to Hugging Face
.venv/bin/python scripts/push_to_hub.py <user>/<repo> --readme DATASET_CARD.md
```

## Layout

```
scripts/
  build_from_toolace.py   ToolACE → JSONL splits (4 configs)
  corruptors.py           contradiction / overgeneration / missing_tool
  validate_spans.py       schema + offset check
  zero_shot_eval.py       sanity + lexical + LettuceDetect baselines
  quality_audit.py        Qwen2.5-3B-Instruct as judge
  llm_augment.py          LLM-generated semantic corruptions
  push_to_hub.py          push 4 configs to HF
notebooks/
  data_pipeline.ipynb     end-to-end walkthrough
PIPELINE.md               pipeline architecture
DATASET_USAGE.md          how to train / validate / test on the dataset
DATASET_CARD.md           README of the published HF dataset
```

## License

Apache 2.0 (matches `minpeter/toolace-parsed`).
