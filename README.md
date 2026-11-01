# LoRA vs prompting: when does fine-tuning beat a good prompt?

> Status: **run on 31 October and 1 November 2026. Full write-up in [RESULTS.md](RESULTS.md).**
> The hypotheses below were written before the run and are not edited afterwards.

## Headline result

- With 100 labelled examples, few-shot prompting of Gemini 3.5 Flash-Lite won: 55% of records
  exactly right against 32% for Qwen2.5-1.5B fine-tuned with LoRA.
- With 2,000 examples, the fine-tuned 1.5B model caught up on exact records (64% vs 66%, within
  noise) and intent accuracy, while staying 5 points behind on slot F1.
- On the same small model, LoRA beat few-shot prompting by 12 to 15 points from 500 examples up.
- Format was never the bottleneck (valid JSON 99.7–100% everywhere); with 100 examples the
  fine-tuned model invented labels it had never seen.
- The fine-tuned model reads 31 tokens per document against about 900 for the prompts.

Teams often ask whether to fine-tune a small model or prompt a large one. This project answers it for
one concrete task, structured extraction, and finds how many training examples it takes for a
LoRA-tuned small model to catch up with few-shot prompting.

## Questions

1. On a structured-extraction task, how does a small open model fine-tuned with LoRA compare with the
   same model prompted few-shot, and with a frontier model prompted few-shot?
2. How does the fine-tuned model's accuracy change with 100, 500 and 2,000 training examples?
3. What does each option cost to train and to run per 1,000 documents?

## Hypotheses (written before running)

- H1: with about 100 examples, few-shot prompting of a frontier model wins.
- H2: with about 2,000 examples, the LoRA-tuned small model matches or beats it on field accuracy
  and on valid-JSON rate.
- H3: LoRA mainly improves format reliability (valid, schema-conforming JSON) before it improves
  field accuracy.

## Setup

**Task.** Turn a short request to a voice assistant into JSON with its intent and its slots, the
spans that carry the details:

```
"wake me up at five am this week"
-> {"intent": "alarm_set", "slots": {"time": "five am", "date": "this week"}}
```

**Data.** MASSIVE 1.1, English (FitzGerald et al., 2023; CC BY 4.0): 60 intents, 55 slot types.
With seed 13, `lvp.cli prepare` draws pools of 100, 500 and 2,000 training utterances (each pool
contains the smaller ones) and 300 test utterances from MASSIVE's test split.

**Same data for every method.** At each pool size, the fine-tuned model trains on the pool, and the
few-shot prompts get the 10 pool examples most similar to the test utterance (BM25), along with the
full lists of intents and slot types. So the x-axis is "labelled examples you have", for both.

**Methods**

| Name | Model | How |
| --- | --- | --- |
| qwen-fewshot-n0 | Qwen2.5-1.5B-Instruct | zero-shot: instructions and label lists only |
| qwen-fewshot-nN | Qwen2.5-1.5B-Instruct | 10 retrieved examples from the pool of N |
| qwen-lora-nN | Qwen2.5-1.5B-Instruct + LoRA | fine-tuned on the pool of N, short instruction, no examples |
| gemini-fewshot-nN | Gemini 3.5 Flash-Lite (API) | 10 retrieved examples from the pool of N |

The API model is the "large model" baseline. It isn't the largest available, but it is free to run
here and far larger than the 1.5B model.

**LoRA settings, fixed before running** (`src/lvp/hf.py`): r = 16, alpha = 32, dropout 0.05 on all
attention and MLP projections; learning rate 2e-4, cosine schedule, 5% warmup, effective batch 16;
8 / 3 / 2 epochs for 100 / 500 / 2,000 examples (about 50, 94 and 250 steps); loss on the answer
tokens only. The pool of 100 is trained with three seeds to show run-to-run variance. Every
method decodes greedily.

**Metrics** (`src/lvp/scoring.py`), with 95% bootstrap intervals:
- valid JSON: the whole reply parses (a code fence or a sentence around it counts as invalid)
- schema OK: a known intent, known slot types, string values
- intent accuracy; slot precision, recall and F1 over (type, value) pairs; exact record match
- cost: training time, GPU seconds per 1,000 documents, tokens per document

### Changes to the plan, made before running

The first draft left the dataset open ("name, date, amount, category"); MASSIVE was chosen for its
licence, its size and its human-labelled spans. The draft planned at least 300 test documents; the
final set is exactly 300, which keeps the API baseline at 900 free-tier calls. Hypotheses are unchanged.

## Run it

1. **GPU part**, in Google Colab (free T4, about an hour): open
   [`notebooks/lora_vs_prompting_colab.ipynb`](notebooks/lora_vs_prompting_colab.ipynb), choose a T4
   runtime and run all cells. Results are saved to Google Drive as they finish.
2. **Gemini part**, on your computer:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run_gemini.ps1 -Limit 5   # a first look
   powershell -ExecutionPolicy Bypass -File scripts\run_gemini.ps1            # 900 calls, resumable
   ```
3. Put every prediction file in `results/preds/` and run `python -m lvp.cli report`.

Any OS, without the scripts:

```bash
pip install -e ".[gemini,dev]" && pytest -q
python -m lvp.cli prepare
python -m lvp.cli gemini        # needs GEMINI_API_KEY in .env
python -m lvp.cli gpu           # needs a CUDA GPU: pip install -e ".[gpu]"
python -m lvp.cli report
```

## References

- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
- Biderman et al. (2024). LoRA Learns Less and Forgets Less. TMLR.
- FitzGerald et al. (2023). MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages. ACL.
- Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
