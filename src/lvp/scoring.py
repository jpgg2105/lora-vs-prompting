"""Scoring, all by rules:
  valid_json     the whole reply parses as one JSON object (stray text or fences count as invalid)
  schema_ok      it has a known intent and a "slots" object whose keys are known slot types and
                 whose values are strings or lists of strings
  intent_acc     intent matches
  slot P/R/F1    over (slot type, value) pairs, values lower-cased with spaces normalized, micro-averaged
  exact          intent and every slot pair match
Replies that aren't valid JSON are still scored for intent and slots through a lenient parse (the
first {...} block), so format failures and extraction failures can be told apart.
"""
from __future__ import annotations

import json
import random
import re


def _norm(v: str) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def pairs(slots) -> set:
    out = set()
    if isinstance(slots, dict):
        for k, v in slots.items():
            for x in (v if isinstance(v, list) else [v]):
                out.add((str(k), _norm(x)))
    return out


def parse(reply: str) -> tuple[dict | None, bool]:
    s = reply.strip()
    try:
        obj = json.loads(s)
        return (obj, True) if isinstance(obj, dict) else (None, False)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj, False
        except json.JSONDecodeError:
            pass
    return None, False


def score_one(reply: str, gold: dict, labels: dict) -> dict:
    obj, strict = parse(reply)
    pred_slots = obj.get("slots", {}) if obj else {}
    schema = bool(obj) and obj.get("intent") in labels["intents"] and isinstance(pred_slots, dict) and all(
        k in labels["slots"] and (isinstance(v, str) or (isinstance(v, list) and all(isinstance(x, str) for x in v)))
        for k, v in pred_slots.items())
    p, g = pairs(pred_slots), pairs(gold["slots"])
    intent_ok = bool(obj) and obj.get("intent") == gold["intent"]
    return {"valid_json": strict, "schema_ok": strict and schema, "intent_ok": intent_ok,
            "tp": len(p & g), "n_pred": len(p), "n_gold": len(g), "exact": intent_ok and p == g}


def aggregate(rows: list[dict], n_boot: int = 2000, seed: int = 13) -> dict:
    def summary(rs):
        tp, npred, ngold = (sum(r[k] for r in rs) for k in ("tp", "n_pred", "n_gold"))
        prec = tp / npred if npred else 0.0
        rec = tp / ngold if ngold else 0.0
        return {"valid_json": sum(r["valid_json"] for r in rs) / len(rs), "schema_ok": sum(r["schema_ok"] for r in rs) / len(rs),
                "intent_acc": sum(r["intent_ok"] for r in rs) / len(rs), "slot_p": prec, "slot_r": rec,
                "slot_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "exact": sum(r["exact"] for r in rs) / len(rs)}
    base = summary(rows)
    rng = random.Random(seed)
    boots = [summary([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(n_boot)]
    ci = {k: [sorted(b[k] for b in boots)[int(.025 * n_boot)], sorted(b[k] for b in boots)[int(.975 * n_boot) - 1]] for k in base}
    return {"n": len(rows), **base, "ci": ci}
