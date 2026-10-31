"""Few-shot baseline on Gemini (free tier) through the OpenAI-compatible endpoint, resumable."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def run(jobs: list[tuple[str, list[dict]]], out: Path, model: str, rpm: float = 12, concurrency: int = 2,
              reasoning_effort: str | None = "low", client=None) -> int:
    """jobs = [(example id, messages)]. Appends {"id", "reply", "usage"} lines to ``out``; skips ids already there."""
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines()[1:]:
            done.add(json.loads(line)["id"])
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"meta": {"model": model, "reasoning_effort": reasoning_effort}}) + "\n", encoding="utf-8")
    todo = [j for j in jobs if j[0] not in done]
    print(f"  {out.name}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return 0
    if client is None:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=URL, max_retries=6, timeout=90)
    gap, last, lock, sem, stop = 60 / rpm if rpm else 0, [0.0], asyncio.Lock(), asyncio.Semaphore(concurrency), asyncio.Event()
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    async def one(i, msgs):
        async with sem:
            if stop.is_set():
                return
            async with lock:
                wait = last[0] + gap - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                last[0] = time.monotonic()
            t = time.perf_counter()
            try:
                r = await client.chat.completions.create(model=model, messages=msgs, temperature=0, **extra)
            except Exception as e:  # noqa: BLE001 - quota or network: stop cleanly, resume later
                print(f"  stopped: {type(e).__name__}: {str(e)[:160]}")
                stop.set()
                return
            u = getattr(r, "usage", None)
            rec = {"id": i, "reply": r.choices[0].message.content or "", "seconds": time.perf_counter() - t,
                   "usage": {"input_tokens": getattr(u, "prompt_tokens", 0), "output_tokens": getattr(u, "completion_tokens", 0)}}
            async with lock:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(i, m) for i, m in todo))
    return 1 if stop.is_set() else 0
