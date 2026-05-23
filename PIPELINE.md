# Quality Data Pipeline — Tool-Calling Hallucination Spans

End-to-end pipeline for building and quality-auditing a span-level hallucination
detection dataset on top of `minpeter/toolace-parsed` (ToolACE).

## Goal

Produce a dataset of tool-using assistant dialogues annotated with
character-level hallucination spans (RAGTruth schema) that:

1. Reflects the three corruption types from the task spec: `contradiction`,
   `overgeneration`, `missing_tool`.
2. Doesn't bake in trivial detection shortcuts (positional bias, 1-character
   spans, fixed templates).
3. Mixes corrupted records with clean negatives so detectors learn to abstain.
4. Has a measurable quality floor — every record has a paper trail of *who*
   labeled it (rules vs LLM judge) and *why* it's considered correct.

## Architecture

```
ToolACE
  │  load_dataset("minpeter/toolace-parsed", "toolace", split="train")
  ▼
┌───────────────────────────────────────┐
│ src/data_processing/build_from_toolace.py         │
│  - schema filter (5 mandatory parts)  │
│  - len & query↔output overlap filter  │
│  - per-base deterministic split       │
│  - collect_corpus_pool() for swaps    │
└─────────────────┬─────────────────────┘
                  │ clean source examples (~720)
                  ▼
┌───────────────────────────────────────┐
│ src/data_processing/corruptors.py  (rule-based)   │
│  - contradiction (≥4-char spans,      │
│    cross-record value pool)           │
│  - overgeneration (8 templates,       │
│    sentence-boundary insertion)       │
│  - missing_tool (9 actions,           │
│    sentence-boundary insertion)       │
│  - clean (no labels)                  │
└─────────────────┬─────────────────────┘
                  │ ≈2,646 records, 4 configs
                  ▼
┌───────────────────────────────────────┐
│ src/data_processing/validate_spans.py             │
│  - schema validation                  │
│  - output[start:end] == text          │
│  - same base_id stays in one split    │
└─────────────────┬─────────────────────┘
                  │
   ┌──────────────┼──────────────┐
   ▼              ▼              ▼
┌──────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│ zero_shot_eval.py    │ │ audit.py (run subcommand)       │ │ llm_augment.py         │
│  - sanity stats      │ │  - LLM-as-judge        │ │  - LLM-generated       │
│  - lexical baseline  │ │    (Qwen2.5-3B-Instruct│ │    contradictions /    │
│  - LettuceDetect ZS  │ │     on MPS)            │ │    overgenerations /   │
│                      │ │  - high_confidence/    │ │    missing_tool spans  │
│  → validation_report │ │    flagged subsets     │ │  - position from same  │
└──────────────────────┘ └────────────────────────┘ │    sentence-boundary   │
                                                    │    logic               │
                                                    └────────────────────────┘
                                                                │
                  ┌─────────────────────────────────────────────┘
                  ▼
┌───────────────────────────────────────┐
│ src/data_processing/push_to_hub.py                │
│  - 4 configs as separate dataset      │
│    configurations (combined +         │
│    per-type)                          │
└─────────────────┬─────────────────────┘
                  ▼
       Ivan1008/toolace-hallucination-spans
       (public, 4 configs, RAGTruth schema)
```

## Stage details

### 1. Source filtering (`build_from_toolace.py`)

Beyond the original schema checks (messages / tools / tool_call / tool_response /
final_answer present), the rebuild added:

| Filter | Threshold | Why |
|---|---|---|
| `len(output)` | `80 ≤ x ≤ 2500` chars | very short answers have no room for meaningful corruption; very long ones break the 4k modernbert context window during eval. |
| `query↔output` overlap | `≥ 0.10` (content-word Jaccard / query) | ToolACE has off-topic finals (e.g. caste-system question answered with linked-list explanation); these would inject noise into corruption labels. |
| `MIN_CONTRADICTION_LEN` | `≥ 4` chars | 1-char digit swaps collapse to a single token in any tokenizer — the model trivially learns "rare numeric token = hallucination." |

This dropped the source pool from 790 to 720 but eliminated the noisiest 9% of records.

### 2. Corruptors (`corruptors.py`)

- `contradiction`: a grounded value in the answer is replaced. Replacement
  values come, in order of preference, from
  (a) a status-swap dictionary (sunny↔rainy, open↔closed, ...),
  (b) an entity-swap dictionary (city↔city, currency↔currency, ...),
  (c) a *cross-record* pool of grounded values aggregated from the whole corpus
  (`collect_corpus_pool`). For numbers we sample from the numeric pool rather
  than apply a fixed `+90` shift, so the substituted value still looks like a
  plausible value of the same type.
- `overgeneration`: a sentence is *inserted* (not appended) at a randomly
  chosen sentence boundary. Templates are deliberately content-free
  ("historical trends suggest...", "industry consensus on the topic", ...) so a
  recall-only baseline can't trivially detect them by missing-from-context.
- `missing_tool`: same insertion strategy with a sentence that offers an
  action requiring a tool absent from `meta.tools`. The action map is
  intentionally disjoint from common ToolACE function names so the inserted
  span never accidentally matches an available tool.
- `clean`: same record passed through with empty `hallucination_labels`. About
  one third of every config consists of clean records.

### 3. Validation (`validate_spans.py` + `zero_shot_eval.py`)

`validate_spans.py` is the structural gate — it errors out if any record
violates the RAGTruth schema or if a base_id leaks across splits.

`zero_shot_eval.py` is the *signal* gate — it computes:

| Metric | What it tells us | Target |
|---|---|---|
| Short-label ratio | %(labels ≤ 3 chars). High = trivial. | < 5% |
| Mean label start position | If near 0 or 1, model can overfit to position. | 0.4–0.6 |
| Query↔output overlap | Confirms filter is doing its job. | ≥ 0.1 |
| Lexical baseline F1 | If > 0.5 the dataset is too easy. | < 0.5 |
| LettuceDetect zero-shot F1 | Must exceed lexical baseline by a meaningful margin. | lexical + 0.04 minimum |

Current numbers on `combined/validation`:

| Metric | Value |
|---|---|
| Short-label ratio | **0.0%** (was 13.5% in v1) |
| Mean label position | **0.49** (was 0.79) |
| Lexical baseline F1 | 0.156 |
| LettuceDetect F1 | 0.198 (+0.04) |

### 4. LLM-as-judge audit (`audit run`)

For each record we ask an LLM judge, in JSON-only mode:

1. *Given the user query, the tool context, the available tools, and the
   assistant's answer, is the candidate hallucination span actually a
   hallucination of the stated type?*
2. *Are there any other hallucinated phrases the labelers missed?*

Two backends are supported:
- `--backend local` runs a transformers model on MPS/CPU (Qwen2.5-3B-Instruct
  used in initial trials, too weak in practice).
- `--backend openrouter --judge-model openai/gpt-oss-120b:free` proxies through
  OpenRouter (used in the final audit; much stronger but ~3 sec per call). The
  driver parallelizes 6 calls at a time and writes incremental progress.

Decisions per record are written verbatim plus parsed:

```json
{
  "id": "toolace_train_00020__contradiction",
  "corruption_type": "contradiction",
  "labeled_span": {"start": 32, "end": 37, "text": "rainy", "label": "contradiction"},
  "judge": {
    "label_correct": "true",
    "extra_hallucination": false,
    "extra_text": "",
    "reasoning": "..."
  },
  "judge_raw": "..."
}
```

Two outputs are derived from each split:
- `high_confidence.jsonl` — judge confirms the label (corrupted) or sees no extra
  hallucination (clean).
- `summary.md` — per-type counts of `label_correct = true / false / unknown` and
  `extra_found`. Reveals where the rule-based corruptor is too naive.

### 5. Recover + Patch (`recover cleans`, `recover extra-spans`)

The audit surfaces two distinct quality problems that the basic filter cannot fix
on its own:

- **False-negative cleans.** The judge often flags hidden hallucinations inside
  records we labeled clean (309 / 39 / 41 across train / val / test). These come
  from imperfect ToolACE source generations, not from our pipeline. `recover`
  locates the judge's `extra_text` in the output, asks a second judge call to
  confirm and classify, and emits a new record with that span as a labeled
  hallucination. Result: 280 / 32 / 30 *real* hallucinations added to the
  dataset, mostly overgeneration.
- **Extra hallucinations inside confirmed-corrupted records.** Judge confirms
  our label AND finds a second hallucination of (often) a different type. `patch`
  reuses the same recover/confirm machinery to append an extra label to the
  existing record so the second hallucination is annotated too. 464 / 61 / 52
  records receive a second label.

`recover other` is a follow-up pass on records the confirm-judge rejected
above. With an extended `{contradiction, overgeneration, missing_tool, off_topic,
not_hallucination}` taxonomy, it salvages 47 records as `other` (mostly
`off_topic` — answers that don't address the user query). These records live
under `data/other/` and are intentionally excluded from `data/final` because
they break the RAGTruth 3-type validator schema; downstream consumers who want
them can opt in.

### 6. Merge (`merge_final.py`)

Combines patched + recovered records per split, then routes them into the four
configurations. Because the project keeps the strict RAGTruth schema (one
`corruption_type` per record, all labels of that type), any patched secondary
label whose type differs from the record's primary `corruption_type` is dropped
at merge. The hallucinated text remains in the output but is no longer
annotated — preserved in `data/combined_patched/` for callers who want full
multi-type annotation.

### 7. LLM augmentation (`llm_augment.py`) — optional

For extra lexical diversity beyond rule-based templates: an LLM generates a new
hallucinated phrase for the requested type, then the sentence-boundary insertion
logic from the rule-based pipeline places it inside a clean answer with exact
character offsets. `meta.corruption_source = "llm_augment"` lets downstream
training mix or split this subset cleanly. Not enabled in the current published
dataset.

## How to reproduce

```bash
# (one-time) create venv (Python 3.11 needed for datasets' lzma import)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 1. Build
.venv/bin/python src/data_processing/build_from_toolace.py
.venv/bin/python src/data_processing/validate_spans.py --allow-clean data/combined/*.jsonl

# 2. Signal-level evaluation (sanity + lexical + LettuceDetect)
for ds in combined contradiction missing_tool overgeneration; do
  .venv/bin/python src/data_processing/zero_shot_eval.py --dataset-dir data/$ds --split validation
done

# 3. LLM-as-judge audit (gpt-oss-120b via OpenRouter)
for split in train validation test; do
  .venv/bin/python src/data_processing/audit.py run \
      --backend openrouter --judge-model openai/gpt-oss-120b:free \
      --dataset-dir data/combined --split $split --no-lettuce
done

# 4. Recover + patch + reclassify
for split in train validation test; do
  .venv/bin/python src/data_processing/recover.py cleans \
      --decisions data/quality_audit_openrouter/combined/$split/decisions.jsonl \
      --source data/combined/$split.jsonl \
      --out-dir data/recovered --split $split
done
.venv/bin/python src/data_processing/recover.py extra-spans
.venv/bin/python src/data_processing/recover.py other

# 5. Final merge + validation
.venv/bin/python src/data_processing/merge_final.py
.venv/bin/python src/data_processing/validate_spans.py --allow-clean \
    data/final/combined/*.jsonl data/final/*/*.jsonl

# 6. Publish (data/final is the default --data-dir)
.venv/bin/python src/data_processing/push_to_hub.py <user>/toolace-hallucination-spans \
    --readme DATASET_CARD.md
```
