# Pipeline

Span-level hallucination dataset for tool-calling dialogues, built on top of
[`minpeter/toolace-parsed`](https://huggingface.co/datasets/minpeter/toolace-parsed)
and published as [`Ivan1008/toolace-hallucination-spans`](https://huggingface.co/datasets/Ivan1008/toolace-hallucination-spans).
RAGTruth-compatible schema: `query`, `context`, `output`, `hallucination_labels`.

## Stages

```
ToolACE  -->  filter  -->  regex corruptors  -->  LLM-as-judge audit  -->  recover + patch  -->  merge  -->  HF
                           (3 types + clean)       (gpt-oss-120b)
```

1. **`build_from_toolace.py`** : drop records that fail any of: 5-part schema,
   `80 ≤ len(output) ≤ 2500`, query - output word overlap ≥ 0.10.
   Dropped 70 of 790 source rows (around 9% noisiest).

2. **`corruptors.py`** : produce three corruption variants + a clean copy per
   surviving record:
   - `contradiction` — replace a grounded value with a same-type substitute from
     a *cross-record* pool (city - city, currency - currency, ...). Minimum 4-char
     span avoids trivial one-token detection
   - `overgeneration` — insert a content-free sentence ("historical trends
     suggest...") at a random sentence boundary
   - `missing_tool` — insert an offer to invoke a tool absent from `meta.tools`.
     Action names disjoint from common ToolACE function names

3. **`audit run`** : for each record, ask `openai/gpt-oss-120b:free` via
   OpenRouter (a) is the labeled span actually a hallucination of the stated
   type? (b) any other hallucinated text missed? Outputs `decisions.jsonl` +
   `summary.md`

4. **`recover {cleans,extra-spans,other}`** : the audit surfaces two problems
   the rule-based corruptor can't catch:
   - *False-negative cleans*: records marked clean but the source ToolACE
     answer already contained a hallucination (mostly overgeneration). Recover
     turns these into properly labeled records (+342 across splits)
   - *Second hallucinations* inside already-corrupted records (multi-label).
     Patch appends the extra label. Affects 577 records
   - `recover other` salvages off-topic answers under an `other/` namespace;
     excluded from `data/final/` because they break the 3-type schema

5. **`merge_final.py`** : combine patched + recovered records per split into
   the four published configs. Secondary labels whose type differs from the
   record primary `corruption_type` are dropped at merge (strict RAGTruth
   schema). Original text is preserved in `data/combined_patched/` for
   consumers who want full multi-label annotation

## Key insights

- **Filter before corrupting.** ToolACE has off-topic finals (caste question
  -> linked-list answer). Without the overlap filter, every contradiction would
  inherit topic-mismatch noise
- **Cross-record value pool over fixed shifts.** Replacing `2025` with `2024+N`
  is detectable by tokenizer rarity; replacing with a date sampled from the
  rest of the corpus is not
- **Insert mid-sentence, not append.** Position bias was the biggest signal
  v1 leaked. Mean label start position went from 0.79 -> 0.49 after fixing
- **`contradiction` is intrinsically hardest.** single-value swaps are short,
  often lexically plausible against context, and have the smallest training
  sample

## Reproduce

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt

# 1. Build + corrupt
python -m src.data_processing.build_from_toolace
python -m src.data_processing.validate_spans --allow-clean data/combined/*.jsonl

# 2. Zero-shot sanity (lexical + LettuceDetect) on validation
for ds in combined contradiction missing_tool overgeneration; do
  python -m src.data_processing.zero_shot_eval --dataset-dir data/$ds --split validation
done

# 3. LLM-as-judge audit
for split in train validation test; do
  python -m src.data_processing.audit run \
      --backend openrouter --judge-model openai/gpt-oss-120b:free \
      --dataset-dir data/combined --split $split --no-lettuce
done

# 4. Recover + merge
for split in train validation test; do
  python -m src.data_processing.recover cleans \
      --decisions data/quality_audit_openrouter/combined/$split/decisions.jsonl \
      --source data/combined/$split.jsonl --out-dir data/recovered --split $split
done
python -m src.data_processing.recover extra-spans
python -m src.data_processing.recover other
python -m src.data_processing.merge_final
python -m src.data_processing.validate_spans --allow-clean data/final/*/*.jsonl

# 5. Publish
python -m src.data_processing.push_to_hub <user>/toolace-hallucination-spans \
    --readme DATASET_CARD.md
```
