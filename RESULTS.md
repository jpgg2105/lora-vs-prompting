# Results: when does LoRA fine-tuning beat a good prompt?

Run: 31 October (GPU, Colab T4) and 1 November 2026 (Gemini) · 300 MASSIVE en-US test utterances ·
every prediction is in `results/preds/`, scores in `results/scores.json`, tables in
`results/REPORT.md`. Intervals are 95% bootstrap intervals; differences are paired (same utterances).
"Exact record" means the intent and every slot are right.

## The short version

- **With 100 labelled examples, prompting the large model wins easily.** Gemini 3.5 Flash-Lite with
  10 retrieved examples got 55% of records exactly right; Qwen2.5-1.5B fine-tuned with LoRA on the
  same 100 got 32% (three seeds: 33, 31, 32). The gap is 22 points [16, 28] against the
  seed-13 run (paired), 23 against the three-seed mean.
- **With 2,000 examples, the fine-tuned 1.5B model comes close on the record, not on every field.**
  LoRA 64% vs Gemini 66% exact: −2 points [−7, +3]. That is within noise, but a gap of up to 7 points
  isn't ruled out. Intent accuracy is the same (86%).
  On slot extraction Gemini stays ahead: F1 0.78 vs 0.72, −5 points [−10, −1].
- **On the same small model, fine-tuning beats prompting once there is data.** LoRA vs the same Qwen
  with 10 retrieved examples: a tie at 100 examples (+2 [−4, +7]), then +12 [+6, +17] at 500 and
  +15 [+9, +21] at 2,000.
- **Format was never the problem.** Every method produced valid JSON on 99.7–100% of inputs. What
  went wrong at 100 examples was labels: the fine-tuned model invented intents and slot types
  ("recipe", "currency"), because its prompt carries no label list and 22 of the 60 intents never
  appear in its 100 training examples.
- **The fine-tuned model is cheaper to run.** 31 input tokens per document against about 900 for
  the few-shot prompts (29× fewer), 190 GPU-seconds per 1,000 documents on a T4 against 355 for the
  same model prompted, and 3 minutes of training for the 2,000-example run.

## Hypotheses (written before running)

| | Hypothesis | Result |
|---|---|---|
| H1 | With about 100 examples, few-shot prompting of a frontier model wins | **True.** 55% vs 32% exact record; 85% vs 61% intent accuracy. |
| H2 | With about 2,000 examples, the LoRA-tuned small model matches or beats it on field accuracy and valid-JSON rate | **Partly true; fails on slot F1.** No significant difference on exact record (−2 [−7, +3]), intent (−0.3) or valid JSON (100% vs 99.7%); behind on slot F1 (−5 [−10, −1], significant). The API model is Flash-Lite, not a frontier model, so this is a bar a larger model would raise. |
| H3 | LoRA mainly improves format reliability before it improves field accuracy | **False.** JSON validity was near 100% from the first run. Schema compliance (known labels only) was *worse* after fine-tuning on 100 examples (70% vs 90% for the prompt) and caught up only with more data. |

## Learning curve

Exact record match, 300 test utterances. The pool is the labelled data available: LoRA trains on
it, few-shot prompts retrieve their 10 examples from it.

| Method | 0 | 100 | 500 | 2,000 |
|---|---|---|---|---|
| Gemini 3.5 Flash-Lite, 10 retrieved examples | – | **55%** | **64%** | **66%** |
| Qwen2.5-1.5B + LoRA | – | 32% (±1, 3 seeds) | 51% | 64% |
| Qwen2.5-1.5B, 10 retrieved examples | 9% (no examples) | 31% | 40% | 49% |

| At 2,000 examples | Gemini few-shot | Qwen + LoRA | Difference [95% CI] |
|---|---|---|---|
| Exact record | 66.0% | 63.7% | −2.3 [−7.3, +2.7] |
| Intent accuracy | 86.0% | 85.7% | −0.3 [−4.3, +3.7] |
| Slot F1 | 0.776 | 0.724 | −5.2 [−9.6, −0.8] |
| Schema OK | 98.7% | 97.3% | −1.3 [−3.7, +0.7] |
| Valid JSON | 99.7% | 100% | |

## Why LoRA struggled at 100 examples

The few-shot prompts list all 60 intents and 55 slot types; the fine-tuned model gets a one-line
instruction and has to learn the labels from its examples. With 100 examples it saw 38 of the 60
intents, and 44 of the 300 test utterances need one it never saw. It filled the gaps with plausible
inventions: 58 replies used an intent that doesn't exist (`recipe`, `qa_date`, `email_send`) and 53
replies used an unknown slot type (61 occurrences: `recipe_name`, `currency`, `definition`). These
counts are for the seed-13 run. The 2,000-example pool covers all 60 intents, and schema compliance
rose to 97%.

The obvious next step is to give the fine-tuned model the label list too, or to constrain its
decoding to the schema. Neither was part of the plan, so neither is tested here.

## Cost

| Method | Training | Input tokens per document | Inference per 1,000 documents |
|---|---|---|---|
| Qwen + LoRA | 40 s (100 ex.), 72 s (500), 190 s (2,000) on a T4 | 31 | 190 GPU-seconds (T4) |
| Qwen, 10 retrieved examples | none | ~880 | 355 GPU-seconds (T4) |
| Gemini Flash-Lite, 10 retrieved examples | none | ~900 | API, about 900k input tokens |

The adapter has 18.5 million trainable parameters, 1.2% of the model. Prices aren't compared because
the API ran on the free tier; the token counts are the cost driver.

## Limits

- One task (short commands with short spans), one small model, one API model. MASSIVE intents are
  well-defined labels; open-ended extraction would favour the larger model more.
- One seed for the 500 and 2,000 LoRA runs; the three seeds at 100 varied by ±1 point.
- LoRA settings were fixed before running and not tuned; tuning would likely help LoRA.
- The API model is the free Flash-Lite, not the largest available.
