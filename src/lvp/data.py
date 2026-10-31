"""MASSIVE en-US (FitzGerald et al., 2023; CC BY 4.0): utterance -> {"intent", "slots"} as JSON.

    "wake me up at [time : five am] [date : this week]"
    -> {"intent": "alarm_set", "slots": {"time": "five am", "date": "this week"}}

Splits, fixed by seed 13 and written to data/:
  pool_2000.jsonl   2,000 training examples; pool_500 and pool_100 are its first 500 and 100,
                    so every larger pool contains the smaller ones
  test.jsonl        300 examples from MASSIVE's test partition
"""
from __future__ import annotations

import io
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
SIZES = (100, 500, 2000)
TEST_N = 300
SEED = 13
_SLOT = re.compile(r"\[\s*([\w]+)\s*:\s*([^\]]+?)\s*\]")


def parse_annot(annot: str) -> dict:
    """'[time : five am] and [time : six]' -> {"time": ["five am", "six"]}; single values stay strings."""
    out: dict[str, list[str]] = {}
    for k, v in _SLOT.findall(annot):
        out.setdefault(k, []).append(v)
    return {k: (v[0] if len(v) == 1 else v) for k, v in out.items()}


def target(row: dict) -> dict:
    return {"intent": row["intent"], "slots": parse_annot(row["annot_utt"])}


def download(dest: Path = DATA / "en-US.jsonl") -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading MASSIVE 1.1 (about 40 MB, once)...")
    with urllib.request.urlopen(URL, timeout=300) as r:
        tf = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    member = next(m for m in tf.getmembers() if m.name.endswith("/en-US.jsonl"))
    dest.write_bytes(tf.extractfile(member).read())
    return dest


def build(raw: Path | None = None) -> dict[str, list[dict]]:
    raw = raw or download()
    rows = [json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines() if x.strip()]
    by = {p: [r for r in rows if r["partition"] == p] for p in ("train", "test")}
    rng = random.Random(SEED)
    rng.shuffle(by["train"])
    rng.shuffle(by["test"])
    ex = lambda r: {"id": r["id"], "text": r["utt"], "target": target(r)}  # noqa: E731
    splits = {f"pool_{n}": [ex(r) for r in by["train"][:n]] for n in SIZES}
    splits["test"] = [ex(r) for r in by["test"][:TEST_N]]
    labels = {"intents": sorted({r["intent"] for r in rows}),
              "slots": sorted({k for r in rows for k in parse_annot(r["annot_utt"])})}
    for name, items in splits.items():
        write(DATA / f"{name}.jsonl", items)
    (DATA / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return splits


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(name: str) -> list[dict]:
    return [json.loads(x) for x in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def labels() -> dict:
    return json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
