# Data notice

`pool_*.jsonl`, `test.jsonl` and `labels.json` are derived from MASSIVE 1.1, en-US
(FitzGerald et al., 2023, "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset
with 51 Typologically-Diverse Languages"), © Amazon.com, Inc., licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Changes: a fixed random sample (seed 13), and the
annotated utterances converted to `{"intent", "slots"}` JSON by `src/lvp/data.py`.
