# Hallucination Detection in Tool Calling

Span-level hallucination detection in tool-calling dialogues 

## Results

![Method comparison](docs/methods_comparison.png)

Sentence-level F1 on the published test split (the metric used for leaderboard ranking):

| method | combined | contradiction | missing_tool | overgeneration |
|---|---:|---:|---:|---:|
| lexical floor | 0.302 | 0.231 | 0.218 | 0.319 |
| LettuceDetect-base (baseline 1.1) | 0.331 | 0.286 | 0.287 | 0.321 |
| LettuceDetect-large (baseline 1.2) | 0.361 | 0.315 | 0.330 | 0.335 |
| LookBackLens zero-shot from RAGTruth (baseline 2) | 0.308 | 0.273 | 0.232 | 0.312 |
| LookBackLens in-domain on ToolACE | 0.489 | 0.377 | 0.406 | 0.508 |
| NLI zero-shot (DeBERTa-v3-large-mnli) | 0.326 | 0.288 | 0.252 | 0.322 |
| Rule-based missing_tool | 0.043 | 0.000 | 0.156 | 0.000 |
| **ModernBERT fine-tune** | **0.798** | 0.763 | **0.966** | **0.697** |
| **Qwen-2.5-7B LoRA** (generative) | 0.771 | **0.800** ⭐ | 0.927 | 0.672 |
| **LightGBM ensemble** | **0.871** | **0.877** | **0.993** | **0.824** |

The strongest single model (ModernBERT fine-tune) improves the best baseline
sentence F1 by **+0.31 absolute** on `combined`. The ensemble adds another
+0.07. The **Qwen-2.5-7B LoRA** generative detector (LLM emits
`<halu type="...">...</halu>` markers, char offsets recovered by regex) reaches
**sentence F1 0.800 on `contradiction`** — the best single-model number on the
hardest type, +4 pp over ModernBERT. 

See `notebooks/solution.ipynb` for full code, training curves and analytics.

## Repository layout

```
notebooks/
  solution.ipynb               main notebook: covers both baselines + dataset construction
                                 + the four improvements + ensemble + analysis
  results/
    lettucedetect_baseline/    raw predictions for both LettuceDetect checkpoints
    lookbacklens_baseline/     Llama-2-7b attention features + 12 LookBackLens variants
    modernbert_ft/             fine-tuned ModernBERT checkpoint + log + metrics
    nli_zeroshot/              NLI predictions + metrics
    missing_tool_rule/         rule-based predictions + metrics
    ensemble/                  4 LightGBM models + predictions + feature importance
    qwen_lora/                 LoRA adapter + training log + Qwen test predictions
    analytics/                 heatmaps + JSON (coverage/calibration/disagreement/type-confusion/entropy)
    baselines_resentence/      baselines re-scored at sentence + span level
src/data_processing/           dataset construction package
docs/methods_comparison.png    bar chart above
PIPELINE.md                    dataset construction
DATASET_CARD.md                README of the published HF dataset
```


* `notebooks/solution.ipynb`:

```
Part 1 Data
Part 2 Methods
  LettuceDetect - baseline 1 (lexical + base + large)
  LookBackLens - baseline 2 (Llama-2-7b + LR)
Part 3 Improvement methods
  ModernBERT fine-tune     
  NLI zero-shot
  Rule-based missing_tool
  LightGBM ensemble
  Qwen-2.5-7B LoRA (generative detector)
  Ablation analytics — confidence/calibration/disagreement/type-confusion/entropy heatmaps
  Final results table + cross-method plot
  Qualitative inspection + Discussion + Reproducibility
```


## Dataset curation

- **Original dataset:** ['minpeter/toolace-parsed`](https://huggingface.co/datasets/minpeter/toolace-parsed)
- **Dataset on Hugging Face:** [`Ivan1008/toolace-hallucination-spans`](https://huggingface.co/datasets/Ivan1008/toolace-hallucination-spans)
- **Schema:** RAGTruth-compatible (`query`, `context`, `output`, `hallucination_labels`)
- **Configs:** `combined`, `contradiction`, `missing_tool`, `overgeneration`

Check `PIPELINE.md` and `README.md`

## Reproducing solution

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
    lightgbm peft accelerate sentence-transformers jupyter ipykernel scikit-learn

.venv/bin/jupyter lab notebooks/solution.ipynb

#  RUN_HEAVY = True for full retraining
```

## Checkpoints

| Checkpoint | Location |
|---|---|
| ModernBERT fine-tune (token classifier, 7 BIO classes) (`AutoModelForTokenClassification`) | HF: [`ArsenyIvanov/toolace-halu-modernbert-large`](https://huggingface.co/ArsenyIvanov/toolace-halu-modernbert-large) |
| **Qwen-2.5-7B LoRA adapter (40 MB)** — load via `PeftModel.from_pretrained(base, adapter)` | HF: [`ArsenyIvanov/toolace-halu-qwen-lora`](https://huggingface.co/ArsenyIvanov/toolace-halu-qwen-lora) |
| LightGBM ensemble (4 per-config models) (`lgb.Booster`) | `notebooks/results/ensemble/lgbm_<cfg>.txt`|
| LookBackLens attention feature cache | `notebooks/results/lookbacklens_baseline/features/*.npz` |

External HF pretrained model:
`KRLabsOrg/lettucedect-{base,large}-modernbert-en-v1`,
`NousResearch/Llama-2-7b-chat-hf`,
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`,
`sentence-transformers/all-MiniLM-L6-v2`.

## Team project

Elizaveta Kamenskaya <br>
Ivan Listopadov <br>
Arseny Ivanov <br>
Maksim Smirnov