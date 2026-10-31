import asyncio
import json
import random

from lvp import data as D
from lvp.prompts import Retriever, fewshot_messages, sft_messages
from lvp.scoring import aggregate, pairs, parse, score_one

INTENTS = ["alarm_set", "weather_query", "play_music", "calendar_set"]
SLOTS = ["time", "date", "place_name", "artist_name", "event_name"]


def fake_massive(path, n_train=2100, n_test=320, seed=0):
    rng = random.Random(seed)
    rows = []
    for part, n in (("train", n_train), ("test", n_test), ("dev", 10)):
        for i in range(n):
            it = rng.choice(INTENTS)
            annot = {"alarm_set": "wake me up at [time : five am] [date : this week]",
                     "weather_query": f"what's the weather in [place_name : city {i % 7}]",
                     "play_music": f"play something by [artist_name : band {i % 5}]",
                     "calendar_set": "remind me about [event_name : lunch] at [time : noon] and [time : two pm]"}[it]
            rows.append({"id": f"{part}{i}", "locale": "en-US", "partition": part, "intent": it,
                         "utt": annot.replace("[", "").replace("]", "").replace(" : ", " "), "annot_utt": annot})
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_parse_annot():
    assert D.parse_annot("wake me up at [time : five am] [date : this week]") == {"time": "five am", "date": "this week"}
    assert D.parse_annot("at [time : noon] and [time : two pm]") == {"time": ["noon", "two pm"]}
    assert D.parse_annot("no slots here") == {}


def test_build_nested_pools(tmp_path, monkeypatch=None):
    raw = tmp_path / "en-US.jsonl"
    fake_massive(raw)
    old = D.DATA
    D.DATA = tmp_path / "out"
    try:
        s = D.build(raw)
        assert [len(s[f"pool_{n}"]) for n in D.SIZES] == [100, 500, 2000] and len(s["test"]) == 300
        assert s["pool_500"][:100] == s["pool_100"] and s["pool_2000"][:500] == s["pool_500"]
        assert not {x["id"] for x in s["test"]} & {x["id"] for x in s["pool_2000"]}
        again = D.build(raw)
        assert again["test"] == s["test"]          # deterministic
        assert set(D.labels()["intents"]) == set(INTENTS)
    finally:
        D.DATA = old


def test_scoring_strict_vs_lenient():
    labels = {"intents": INTENTS, "slots": SLOTS}
    gold = {"intent": "alarm_set", "slots": {"time": "five am", "date": "this week"}}
    good = score_one('{"intent": "alarm_set", "slots": {"time": "Five  AM", "date": "this week"}}', gold, labels)
    assert good["valid_json"] and good["schema_ok"] and good["exact"] and good["tp"] == 2
    fenced = score_one('```json\n{"intent": "alarm_set", "slots": {"time": "five am"}}\n```', gold, labels)
    assert not fenced["valid_json"] and fenced["intent_ok"] and fenced["tp"] == 1 and not fenced["exact"]
    bad_schema = score_one('{"intent": "alarm_set", "slots": {"hour": "five am"}}', gold, labels)
    assert bad_schema["valid_json"] and not bad_schema["schema_ok"]
    assert score_one("sorry", gold, labels)["n_pred"] == 0
    assert pairs({"time": ["noon", "Two pm"]}) == {("time", "noon"), ("time", "two pm")}
    assert parse("[1, 2]") == (None, False)


def test_aggregate_f1_and_ci():
    rows = [{"valid_json": True, "schema_ok": True, "intent_ok": True, "tp": 1, "n_pred": 2, "n_gold": 1, "exact": False}] * 10
    a = aggregate(rows, n_boot=200)
    assert abs(a["slot_p"] - .5) < 1e-9 and a["slot_r"] == 1.0 and abs(a["slot_f1"] - 2 / 3) < 1e-9
    assert a["ci"]["intent_acc"] == [1.0, 1.0]


def test_prompts_and_retrieval():
    pool = [{"id": str(i), "text": t, "target": {"intent": "play_music", "slots": {}}} for i, t in
            enumerate(["play jazz", "weather in paris", "wake me at five", "play some rock music"])]
    top = Retriever(pool).top("please play rock", k=2)
    assert top[-1]["text"] == "play some rock music"      # most similar sits last, next to the question
    msgs = fewshot_messages("play rock", top, {"intents": INTENTS, "slots": SLOTS})
    assert msgs[0]["role"] == "system" and "alarm_set" in msgs[0]["content"] and len(msgs) == 1 + 2 * 2 + 1
    assert sft_messages("x", {"intent": "a", "slots": {}})[-1]["content"] == '{"intent": "a", "slots": {}}'


def test_gemini_runner_resumes(tmp_path):
    from lvp.gemini import run

    class Msg:
        def __init__(self, c): self.message = type("m", (), {"content": c})
    class Resp:
        def __init__(self, c): self.choices = [Msg(c)]; self.usage = type("u", (), {"prompt_tokens": 10, "completion_tokens": 3})
    class Client:
        calls = 0
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    Client.calls += 1
                    return Resp('{"intent": "alarm_set", "slots": {}}')
    out = tmp_path / "g.jsonl"
    jobs = [(str(i), [{"role": "user", "content": "x"}]) for i in range(5)]
    assert asyncio.run(run(jobs[:3], out, "m", rpm=0, client=Client)) == 0
    asyncio.run(run(jobs, out, "m", rpm=0, client=Client))
    assert Client.calls == 5 and len(out.read_text().splitlines()) == 6
