"""Prompts. Few-shot prompts carry the label lists and 10 examples retrieved from the same pool the
LoRA model is trained on, so at every pool size both methods see the same labelled data. The
fine-tuned model gets a short instruction only: training replaces the instructions and examples."""
from __future__ import annotations

import json
import math
import re
from collections import Counter

K_SHOTS = 10
_W = re.compile(r"\w+")

FEWSHOT_SYSTEM = """You extract structured data from short requests to a voice assistant.
Reply with one JSON object and nothing else: {{"intent": "<intent>", "slots": {{"<slot type>": "<words copied from the request>"}}}}
Use exactly one intent from this list: {intents}
Slot types (use only these, and only for words that appear in the request): {slots}
If a slot type occurs twice, give its values as a list. If there are no slots, use "slots": {{}}."""

SFT_SYSTEM = "Extract the intent and slots of the request as JSON."


def dumps(t: dict) -> str:
    return json.dumps(t, ensure_ascii=False, separators=(", ", ": "))


class Retriever:
    """BM25 over the pool's utterances: the K most similar examples go into the prompt."""

    def __init__(self, pool: list[dict]) -> None:
        self.pool = pool
        self.tf = [Counter(_W.findall(p["text"].lower())) for p in pool]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for t in self.tf for w in t)
        n = len(pool)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def top(self, text: str, k: int = K_SHOTS) -> list[dict]:
        q = [w for w in _W.findall(text.lower()) if w in self.idf]
        sc = [sum(self.idf[w] * tf[w] * 2.5 / (tf[w] + 1.5 * (.25 + .75 * dl / self.avg)) for w in q if tf[w])
              for tf, dl in zip(self.tf, self.len)]
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:k]
        return [self.pool[i] for i in reversed(order)]  # most similar last, next to the question


def fewshot_messages(text: str, shots: list[dict], labels: dict) -> list[dict]:
    msgs = [{"role": "system", "content": FEWSHOT_SYSTEM.format(intents=", ".join(labels["intents"]), slots=", ".join(labels["slots"]))}]
    for s in shots:
        msgs += [{"role": "user", "content": s["text"]}, {"role": "assistant", "content": dumps(s["target"])}]
    msgs.append({"role": "user", "content": text})
    return msgs


def sft_messages(text: str, answer: dict | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SFT_SYSTEM}, {"role": "user", "content": text}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": dumps(answer)})
    return msgs
