# Как пользоваться датасетом `Ivan1008/toolace-hallucination-spans`

Кратко, без инженерии. Если интересует как собирался под капотом — см. PIPELINE.md.

---

## Что внутри

4 конфигурации, в каждой — `train` / `validation` / `test`:

| Config | train | val | test | Что внутри |
|---|---:|---:|---:|---|
| `combined` | 2,118 | 264 | 264 | Всё вместе: clean + 3 типа галлюцинаций |
| `contradiction` | 966 | 120 | 120 | Только clean + contradiction |
| `missing_tool` | 1,152 | 144 | 144 | Только clean + missing_tool |
| `overgeneration` | 1,152 | 144 | 144 | Только clean + overgeneration |

Каждая запись — это RAGTruth-формат:

```python
{
  "query":   "Help me check the weather in Beijing.",
  "context": "{\"role\": \"tool\", \"name\": \"Weather_API\", \"content\": {...}}",
  "output":  "The weather in Beijing today is rainy with temperature 24...",
  "hallucination_labels": [
    {"start": 32, "end": 37, "text": "rainy", "label": "contradiction"}
  ],
  "meta": {"corruption_type": "contradiction", "base_id": "toolace_train_00020", ...}
}
```

`output[start:end] == text` гарантируется. Для clean записей `hallucination_labels: []`.

---

## Как собирался (быстрая версия)

```
ToolACE-parsed (11,072 диалогов)
        │ фильтр: есть user query + tool call + tool response + final answer
        │ фильтр: 80 ≤ len(output) ≤ 2500 символов
        │ фильтр: query↔output content-word overlap ≥ 0.10
        ▼
720 базовых диалогов с чистыми ответами ассистента
        │
        │ для КАЖДОГО диалога создаются варианты:
        │   • clean         — ответ без изменений
        │   • contradiction — заменяем значение из context на другое того же типа
        │   • overgeneration — вставляем фразу с inferred-claim
        │   • missing_tool  — вставляем offer выполнить действие, требующее
        │                     инструмента не из tools list
        ▼
2,646 уникальных записей
        │ split 80/10/10 детерминистично по hash(base_id)
        │ важно: ВСЕ варианты одного base_id попадают в один split
        │        (иначе при обучении на train модель увидела бы те же тексты
        │        в validation через clean-вариант → утечка)
        ▼
combined: train 2118 / val 264 / test 264
```

3 per-type конфигурации — это просто фильтр поверх combined: оставляем
только записи нужного типа + все clean. Сплиты те же.

---

## Зачем нужен `combined`

**Для обучения универсального детектора**, который ловит все 3 типа галлюцинаций одной моделью.

В реальной задаче ты не знаешь *какой* тип галлюцинации тебе сейчас встретится — модель должна уметь решать всё сразу. На `combined` ты получаешь:
- больше данных (2,118 в train против 1,152 в самом крупном per-type)
- сбалансированное обучение по типам (каждый тип ≈ равное представительство)
- единые метрики P/R/F1 поверх всех галлюцинаций

Это **основной датасет для production-обучения**.

---

## Зачем нужны per-type конфиги

**Для контролируемой оценки и ablation**.

Когда обученная на `combined` модель показывает плохой F1 — непонятно, она плохо ловит contradiction, или overgeneration, или missing_tool? Per-type датасеты позволяют:

1. **Раздельная оценка качества**: запустить ту же модель на 3 per-type validation и понять, какой тип она проваливает.
2. **Эксперимент "specialised vs generalist"**: обучить отдельные детекторы на каждом per-type и сравнить их с одним детектором на combined. Иногда specialised лучше — особенно если типы галлюцинаций требуют разных сигналов.
3. **Изоляция проблемного типа** при отладке: если contradiction даёт F1=0.03, можно копать только в contradiction config, не мешая остальное.

Это указано прямо в задании: "you will have **three datasets** with different types of span-based hallucinations".

---

## Правила сплитов (что куда)

| Split | % | Используется для | Когда смотреть |
|---|---|---|---|
| `train` | 80% | Обучение модели | Каждую эпоху |
| `validation` | 10% | Подбор гиперпараметров, early stopping, мониторинг overfit | Каждую эпоху или каждые N шагов |
| `test` | 10% | **Финальная оценка**. Один раз, в конце. | После того как все решения по модели приняты |

**Ключевое правило:** не настраивай ничего на test. Если ты выбираешь между двумя моделями, глядя в test — это **больше не test**, это второй validation. После такого test set "сгорает" и финальные числа уже не достоверны.

---

## Как обучаться

### Сценарий 1: универсальный детектор (рекомендуется)

```python
from datasets import load_dataset

repo = "Ivan1008/toolace-hallucination-spans"
ds = load_dataset(repo, "combined")
train_set = ds["train"]      # 2118 записей
val_set   = ds["validation"] # 264 записей

# для span-detection (token classification):
# - токенизируешь output как последовательность токенов
# - метка на токен: 0 = не галлюцинация, 1 = галлюцинация
#   (или 4 класса: O / B-contradiction / B-overgen / B-missing-tool — если хочешь
#    различать ТИП галлюцинации, а не только её наличие)
# - модель: AutoModelForTokenClassification (DeBERTa-v3-base / ModernBERT-base / etc.)
```

В качестве context-источника для токенизации подавай `(context, output)` парой:
`tok(record["context"], record["output"], return_offsets_mapping=True)` — это
ровно тот формат, на котором обучались LettuceDetect-подобные модели.

### Сценарий 2: специализированный детектор на один тип

```python
# Например, только contradiction:
ds = load_dataset(repo, "contradiction")
# train: 966 записей (часть clean + часть contradiction)
# Обучаешь бинарный детектор: hallucination_labels пустой = клин, иначе = corrupt
```

Это полезно если в твоей prod-ситуации интересует только конкретный тип ошибки.

### Совмещённый сценарий (research)

Обучи 4 модели (`combined` + три per-type) одинаково. Сравни их между собой на одних и тех же per-type validation split. Часто видишь, что:
- `combined`-модель ≈ specialised на самом частом типе
- specialised лучше на своём типе, но хуже на других
- combined даёт лучший средний F1 → её и тащить в прод

---

## Как валидировать (во время обучения)

После каждой эпохи (или каждые N шагов):

1. Считай **char-level F1** по `hallucination_labels`:
   - для каждой записи строишь gold mask: `mask[start:end] = 1`
   - модель предсказывает свою mask
   - TP / FP / FN считаешь посимвольно
   - агрегируешь по всему validation
2. Смотри метрики **раздельно по типам** — даже на `combined` validation у тебя в `meta.corruption_type` есть тип каждой записи.
3. Делай **early stopping** по validation F1, не по loss. Loss может падать, а F1 — застрять или ухудшаться.

Скрипт `scripts/zero_shot_eval.py` уже делает это для zero-shot бейзлайнов
(lexical + LettuceDetect). Адаптируй его под свою модель.

Целевые цифры на validation (текущие baseline):

| Метрика | Lexical baseline | LettuceDetect zero-shot | Куда стремиться |
|---|---|---|---|
| F1 (combined) | 0.156 | 0.198 | ≥ 0.4 после fine-tuning |
| F1 (contradiction) | 0.023 | 0.030 | ≥ 0.3 (трудный тип) |
| F1 (missing_tool) | 0.104 | 0.118 | ≥ 0.5 |
| F1 (overgeneration) | 0.167 | 0.225 | ≥ 0.5 |

---

## Как тестировать (финальный шаг)

**Только один раз**, после того как ты:
- выбрал архитектуру
- зафиксировал гиперпараметры
- провёл все эксперименты на validation

```python
test_set = load_dataset(repo, "combined", split="test")
# или per-type test для разбивки
predictions = model.predict(test_set)
final_f1 = compute_char_f1(predictions, test_set["hallucination_labels"])
```

Если результат на test заметно хуже validation (gap > 0.05 F1) — есть overfit
на validation (вероятно, ты что-то подстраивал по val-метрикам слишком долго).

---

## Что НЕ делать

| Антипаттерн | Почему плохо |
|---|---|
| Брать все данные из `train` + `validation` + `test` для обучения | Тогда нечем оценивать — модель видела всё |
| Смешивать base_id между train/val/test | Утечка: модель видит `clean` версию ответа в train и `contradiction` версию того же ответа в val. Поэтому split по base_id, а не по record id |
| Считать F1 только overall | Маскирует слабый тип. Всегда per-type разбивка |
| Выбирать модель по test F1 | См. выше — test сгорит, финальная оценка станет недостоверной |
| Обучать на combined и тестировать только на combined | Не узнаешь, где модель проседает по типам |

---

## Шпаргалка: что под рукой

```bash
# Загрузить и посмотреть
python -c "
from datasets import load_dataset
ds = load_dataset('Ivan1008/toolace-hallucination-spans', 'combined')
print(ds)
print(ds['train'][0])
"

# Запустить zero-shot бейзлайны на validation
.venv/bin/python scripts/zero_shot_eval.py --dataset-dir data/combined --split validation

# Сгенерировать дополнительные семантические corruptions через LLM
.venv/bin/python scripts/llm_augment.py --source data/combined --split train --n 100

# Проверить качество меток через LLM-судью
.venv/bin/python scripts/quality_audit.py --dataset-dir data/combined --split validation
```
