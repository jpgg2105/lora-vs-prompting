"""Scores every prediction file and writes the tables (interpretation is written by hand)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from . import data as D
from .scoring import aggregate, score_one

METRICS = [("valid_json", "valid JSON"), ("schema_ok", "schema OK"), ("intent_acc", "intent"), ("slot_f1", "slot F1"), ("exact", "exact record")]


def load_preds(path: Path) -> tuple[dict, dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])["meta"]
    return meta, {json.loads(x)["id"]: json.loads(x) for x in lines[1:] if x.strip()}


def score_file(path: Path, test: list[dict], labels: dict) -> dict:
    meta, preds = load_preds(path)
    rows = [score_one(preds[ex["id"]]["reply"], ex["target"], labels) for ex in test if ex["id"] in preds]
    agg = aggregate(rows) if rows else {"n": 0}
    toks = [p.get("usage", {}) for p in preds.values()]
    agg["cost"] = {k: meta.get(k) for k in ("gpu_seconds", "train_seconds", "input_tokens", "output_tokens", "steps", "trainable_params") if meta.get(k) is not None}
    if toks and toks[0]:
        agg["cost"]["input_tokens"] = sum(t.get("input_tokens", 0) for t in toks)
        agg["cost"]["output_tokens"] = sum(t.get("output_tokens", 0) for t in toks)
    agg["meta"] = {k: v for k, v in meta.items() if k in ("method", "pool", "seed", "gpu", "model", "dtype")}
    return agg


def write_report(preds_dir: Path, out: Path) -> Path:
    test, labels = D.load("test"), D.labels()
    res = {p.stem: score_file(p, test, labels) for p in sorted(preds_dir.glob("*.jsonl"))}  # quick/ is a subfolder, not included
    L = ["# LoRA vs prompting: results", "", f"Test set: {len(test)} MASSIVE en-US utterances. 95% bootstrap intervals in brackets.", "",
         "| Run | n | " + " | ".join(t for _, t in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for name, a in res.items():
        if not a.get("n"):
            continue
        L.append(f"| {name} | {a['n']} | " + " | ".join(f"{a[k]:.3f} [{a['ci'][k][0]:.2f}, {a['ci'][k][1]:.2f}]" for k, _ in METRICS) + " |")
    # learning curve: exact match by pool size, LoRA seeds averaged
    curve = defaultdict(dict)
    for name, a in res.items():
        m = re.match(r"(.+?)-n(\d+)(?:-s\d+)?$", name)
        if m and a.get("n"):
            curve[m.group(1)].setdefault(int(m.group(2)), []).append(a)
    L += ["", "## Learning curve (exact record match; LoRA averaged over seeds where there are several)", "",
          "| Method | " + " | ".join(f"pool {n}" for n in (0,) + D.SIZES) + " |", "|---|" + "---|" * (len(D.SIZES) + 1)]
    for meth, by in sorted(curve.items()):
        cells = []
        for n in (0,) + D.SIZES:
            runs = by.get(n)
            cells.append("–" if not runs else f"{sum(r['exact'] for r in runs) / len(runs):.3f}" + (f" (±{(max(r['exact'] for r in runs) - min(r['exact'] for r in runs)) / 2:.3f})" if len(runs) > 1 else ""))
        L.append(f"| {meth} | " + " | ".join(cells) + " |")
    L += ["", "## Cost", "", "| Run | train s | inference s (300 docs) | per 1,000 docs s | input tokens / doc | output tokens / doc |", "|---|---|---|---|---|---|"]
    for name, a in res.items():
        c, n = a.get("cost", {}), a.get("n") or 1
        inf, tr = c.get("gpu_seconds"), c.get("train_seconds")
        L.append(f"| {name} | {f'{tr:.0f}' if tr else '–'} | {f'{inf:.0f}' if inf else '–'} | {f'{inf / n * 1000:.0f}' if inf else '–'} | "
                 f"{c.get('input_tokens', 0) / n:,.0f} | {c.get('output_tokens', 0) / n:,.0f} |")
    L += ["", "Gemini runs use the API, so they have no GPU time; their cost is the token counts at the API's price."]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_name("scores.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return out
