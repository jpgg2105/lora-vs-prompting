"""python -m lvp.cli prepare            build data/ from MASSIVE (needs internet once)
   python -m lvp.cli gpu [--quick]      Colab: Qwen zero-shot, few-shot and LoRA runs (resumable)
   python -m lvp.cli gemini [--limit N] few-shot on Gemini for each pool size (resumable, needs GEMINI_API_KEY)
   python -m lvp.cli report             score every file in results/preds and write results/REPORT.md
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from . import data as D
from .prompts import Retriever, fewshot_messages, sft_messages

RESULTS = Path(os.environ.get("LVP_RESULTS", D.ROOT / "results"))
PREDS = RESULTS / "preds"
LORA_SEEDS = {100: [13, 14, 15], 500: [13], 2000: [13]}


def fewshot_jobs(pool_n: int, test: list[dict], labels: dict, k: int = 10) -> list[tuple[str, list[dict]]]:
    if pool_n == 0:
        return [(ex["id"], fewshot_messages(ex["text"], [], labels)) for ex in test]
    r = Retriever(D.load(f"pool_{pool_n}"))
    return [(ex["id"], fewshot_messages(ex["text"], r.top(ex["text"], k), labels)) for ex in test]


def cmd_gpu(quick: bool) -> None:
    from . import hf

    global PREDS
    test, labels = D.load("test"), D.labels()
    if quick:
        test, PREDS = test[:20], PREDS / "quick"   # kept apart so the full run doesn't skip anything
    ids = [ex["id"] for ex in test]
    env = hf.environment()
    print("GPU:", env)
    sizes = list(D.SIZES)
    pending_prompt = [n for n in [0] + sizes if not (PREDS / f"qwen-fewshot-n{n}.jsonl").exists()]
    if pending_prompt:
        model, tok = hf.load()
        for n in pending_prompt:
            print(f"[qwen {'zero-shot' if n == 0 else f'few-shot, pool {n}'}]")
            jobs = fewshot_jobs(n, test, labels)
            replies, meta = hf.generate(model, tok, [m for _, m in jobs], batch_size=8)
            hf.save_preds(PREDS / f"qwen-fewshot-n{n}.jsonl", ids, replies, {**meta, **env, "method": "qwen-fewshot", "pool": n, "shots": 0 if n == 0 else 10})
        del model
    for n in sizes:
        for seed in (LORA_SEEDS[n][:1] if quick else LORA_SEEDS[n]):
            out = PREDS / f"qwen-lora-n{n}-s{seed}.jsonl"
            if out.exists():
                print(f"[skip] {out.name}")
                continue
            print(f"[qwen LoRA, pool {n}, seed {seed}]")
            pool = D.load(f"pool_{n}")
            if quick:
                pool = pool[:100] if n == 100 else None
            if pool is None:
                continue
            model, tok, tmeta = hf.train_lora(pool, RESULTS / "tmp", seed=seed)
            replies, meta = hf.generate(model, tok, [sft_messages(ex["text"]) for ex in test], batch_size=16)
            hf.save_preds(out, ids, replies, {**meta, **tmeta, **env, "method": "qwen-lora", "pool": n, "seed": seed, "lora": hf.LORA})
            del model, tok
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
    print("\nDone. Download the results folder (the notebook's last cell zips it).")


def cmd_gemini(limit: int, model: str, rpm: float) -> None:
    from .gemini import load_env, run

    load_env(D.ROOT / ".env")
    test, labels = D.load("test"), D.labels()
    if limit:
        test = test[:limit]
    for n in D.SIZES:
        code = asyncio.run(run(fewshot_jobs(n, test, labels), PREDS / f"gemini-fewshot-n{n}.jsonl", model, rpm=rpm))
        if code:
            print("Run the same command again after the quota resets; finished answers are kept.")
            return
    print("Gemini runs complete.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lvp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    g = sub.add_parser("gpu"); g.add_argument("--quick", action="store_true", help="20 test examples, to check the setup")
    m = sub.add_parser("gemini"); m.add_argument("--limit", type=int, default=0); m.add_argument("--model", default="gemini-3.5-flash-lite"); m.add_argument("--rpm", type=float, default=12)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        s = D.build()
        print({k: len(v) for k, v in s.items()}, "->", D.DATA)
    elif a.cmd == "gpu":
        cmd_gpu(a.quick)
    elif a.cmd == "gemini":
        cmd_gemini(a.limit, a.model, a.rpm)
    else:
        from .report import write_report
        print(write_report(PREDS, RESULTS / "REPORT.md"))


if __name__ == "__main__":
    main()
