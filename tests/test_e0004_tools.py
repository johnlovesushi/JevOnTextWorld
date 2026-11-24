"""E0004 tooling: --save-reasonings, the peak GPU memory of an evaluation, scripts/sample_reasonings.py."""
import json
import sys
from pathlib import Path

from jevtw import train
from jevtw.testing import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_save_reasonings_and_eval_peak(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"policy": {"prompt_format": "chat_think", "forecast_format": "chat"}}))
    out = tmp_path / "eval"
    train.main(["--smoke", "--out", str(out), "--config", str(cfg), "--batches", "0", "--save-reasonings",
                "--save-forecasts"])
    (row,) = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert row["split"] == "val" and "gpu_peak_gb" in row          # None on the CPU
    records = [json.loads(line) for line in (out / "reasonings" / "batch0000.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (out / "forecasts" / "batch0000.jsonl").read_text().splitlines()]
    assert len(records) == row["decisions"] == sum(ep["length"] for ep in episodes)
    # the same decisions, in the same order, as the forecasts file
    assert [(r["game"], r["sample_seed"], r["t"], r["command"], r["z"]) for r in records] == \
        [(ep["game"], ep["sample_seed"], s["t"], s["command"], ep["z"]) for ep in episodes for s in ep["steps"]]
    tokenizer = ByteTokenizer()
    for r in records:
        assert r["reasoning_end"] in ("answer", "eos", "cap", "cap_in_think")
        assert 1 <= r["reasoning_tokens"] <= 8
        assert len(tokenizer(r["reasoning"], add_special_tokens=False)["input_ids"]) >= 1
    for end in ("answer", "eos", "cap", "cap_in_think"):
        share = sum(r["reasoning_end"] == end for r in records) / len(records)
        assert abs(share - row[f"reasoning_end_{end}"]) < 1e-9

    from summarize_run import markdown, summarize
    summary = summarize(out, "E", "t")
    assert summary["gpu_peak_gb"] is None and "Training" not in markdown(summary, None, None)
    with (out / "metrics.jsonl").open("w") as f:                    # what a GPU evaluation writes
        f.write(json.dumps({**row, "gpu_peak_gb": 7.25}) + "\n")
    summary = summarize(out, "E", "t")
    assert summary["gpu_peak_gb"] == 7.25
    assert "peak GPU memory (evaluation): 7.25 GiB" in markdown(summary, None, None)


def test_sample_reasonings(tmp_path, capsys):
    import sample_reasonings

    records = [{"game": f"g{i}", "sample_seed": 17, "z": i % 2, "length": 3, "t": 0, "command": "wait",
                "reasoning_end": "answer", "reasoning_tokens": 0 if i < 2 else 5, "reasoning": "x" * (100 * i)}
               for i in range(8)]
    chosen = sample_reasonings.pick(records, seed=3)
    assert [r["z"] for r in chosen] == [1, 0] and all(r["reasoning_tokens"] > 0 for r in chosen)
    assert chosen == sample_reasonings.pick(records, seed=3)
    assert sample_reasonings.pick([r for r in records if r["z"] == 0], seed=3)[0]["z"] == 0
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    sample_reasonings.main([str(path), "--seed", "3", "--chars", "50"])
    text = capsys.readouterr().out
    assert "**won episode**" in text and "**lost episode**" in text and "x" * 50 + " […]" in text
    assert "x" * 51 not in text
