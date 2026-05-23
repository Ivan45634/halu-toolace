# halu-toolace

Span-level hallucination detection dataset and pipeline for tool-calling dialogues,
built on top of [`minpeter/toolace-parsed`](https://huggingface.co/datasets/minpeter/toolace-parsed).

- **Dataset on Hugging Face:** [`Ivan1008/toolace-hallucination-spans`](https://huggingface.co/datasets/Ivan1008/toolace-hallucination-spans)
- **Schema:** RAGTruth-compatible (`query`, `context`, `output`, `hallucination_labels`)
- **Configs:** `combined`, `contradiction`, `missing_tool`, `overgeneration`

## Quickstart

Run commands in package mode from the repository root (`python -m src.data_processing.X`).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 1. Build the dataset locally from ToolACE
python -m src.data_processing.build_from_toolace

# 2. Validate spans and run zero-shot baselines
python -m src.data_processing.validate_spans --allow-clean data/combined/*.jsonl
python -m src.data_processing.zero_shot_eval --dataset-dir data/combined --split validation

# 3. (Optional) LLM-as-judge audit + recovery cycle
python -m src.data_processing.audit run --backend openrouter --split train
python -m src.data_processing.recover cleans \
    --decisions data/quality_audit_openrouter/combined/train/decisions.jsonl \
    --source data/combined/train.jsonl --out-dir data/recovered --split train
python -m src.data_processing.recover extra-spans
python -m src.data_processing.recover other

# 4. Final merge + push
python -m src.data_processing.merge_final
python -m src.data_processing.push_to_hub <user>/<repo> --readme DATASET_CARD.md
```

## Layout

```
src/data_processing/
  common.py             shared utils / judges / prompts
  corruptors.py         contradiction / overgeneration / missing_tool (regex)
  build_from_toolace.py ToolACE → JSONL splits (4 configs)
  validate_spans.py     schema + offset check
  zero_shot_eval.py     sanity + lexical + LettuceDetect baselines
  audit.py              5 subcommands: run / summary / report / filter / export
  recover.py            3 subcommands: cleans / extra-spans / other
  merge_final.py        patched + recovered → data/final/
  llm_augment.py        LLM-generated semantic corruptions (optional)
  push_to_hub.py        push 4 configs to HF
notebooks/
  data_pipeline.ipynb   end-to-end walkthrough
PIPELINE.md             pipeline architecture
DATASET_CARD.md         README of the published HF dataset
```

## License

Apache 2.0 (matches `minpeter/toolace-parsed`).
