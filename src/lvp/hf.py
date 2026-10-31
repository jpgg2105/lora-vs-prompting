"""GPU side (run in Colab): Qwen2.5-1.5B-Instruct zero-shot, few-shot, and LoRA-tuned at each pool size.

Fixed before running, not tuned on the test set:
  LoRA r=16, alpha=32, dropout 0.05 on every attention and MLP projection; lr 2e-4, cosine schedule,
  5% warmup, effective batch 16; epochs 8 / 3 / 2 for pools of 100 / 500 / 2,000 (about 50, 94 and
  250 optimizer steps); loss on the answer tokens only; greedy decoding, up to 96 new tokens.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EPOCHS = {100: 8, 500: 3, 2000: 2}
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def _dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load(model_name: str = MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_dtype(), device_map="cuda")
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def generate(model, tok, conversations: list[list[dict]], batch_size: int = 8, max_new_tokens: int = 96) -> tuple[list[str], dict]:
    import torch

    model.eval()
    replies, n_in, n_out = [], 0, 0
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(0, len(conversations), batch_size):
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations[i : i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
        new = out[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = [x for x in row.tolist() if x != tok.pad_token_id]
            n_out += len(ids)
            replies.append(tok.decode(ids, skip_special_tokens=True).strip())
        n_in += int(enc["attention_mask"].sum())
        print(f"    {min(i + batch_size, len(conversations))}/{len(conversations)}", end="\r")
    torch.cuda.synchronize()
    return replies, {"gpu_seconds": time.perf_counter() - t, "input_tokens": n_in, "output_tokens": n_out}


class _SFT:
    """Chat-formatted examples with the loss on the assistant answer only."""

    def __init__(self, tok, pool):
        from .prompts import sft_messages
        self.items = []
        for ex in pool:
            full = tok.apply_chat_template(sft_messages(ex["text"], ex["target"]), tokenize=False)
            prompt = tok.apply_chat_template(sft_messages(ex["text"]), tokenize=False, add_generation_prompt=True)
            ids = tok(full, add_special_tokens=False)["input_ids"]
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
            self.items.append({"input_ids": ids, "labels": [-100] * n_prompt + ids[n_prompt:]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate(pad_id):
    import torch

    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id); lab = torch.full((len(batch), n), -100); att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"]); ids[i, :L] = torch.tensor(b["input_ids"]); lab[i, :L] = torch.tensor(b["labels"]); att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return fn


def train_lora(pool: list[dict], out_dir: Path, seed: int = 13, model_name: str = MODEL):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    model, tok = load(model_name)
    tok.padding_side = "right"
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bf16 = _dtype() == torch.bfloat16
    args = TrainingArguments(output_dir=str(out_dir), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                             num_train_epochs=EPOCHS[len(pool)], learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
                             logging_steps=10, save_strategy="no", report_to="none", seed=seed, bf16=bf16, fp16=not bf16,
                             remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=_SFT(tok, pool), data_collator=_collate(tok.pad_token_id))
    torch.cuda.synchronize(); t = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    meta = {"train_seconds": time.perf_counter() - t, "trainable_params": trainable, "epochs": EPOCHS[len(pool)],
            "steps": trainer.state.global_step, "final_loss": trainer.state.log_history[-1].get("train_loss")}
    tok.padding_side = "left"
    return model, tok, meta


def environment() -> dict:
    import peft
    import torch
    import transformers
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "dtype": str(_dtype())}


def save_preds(path: Path, ids: list[str], replies: list[str], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for i, r in zip(ids, replies):
            f.write(json.dumps({"id": i, "reply": r}, ensure_ascii=False) + "\n")
