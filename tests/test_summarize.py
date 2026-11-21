"""The report generator on a real (smoke) run directory."""
import json
import subprocess
import sys
from pathlib import Path

from jevtw.train import main as train_main

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_run_report(tmp_path):
    run = tmp_path / "run"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"grpo": {"lam": 0.5, "alpha": 0.1}}))
    train_main(["--smoke", "--out", str(run), "--config", str(cfg)])
    meta = json.loads((run / "config.json").read_text())["meta"]
    assert {"commit", "host", "torch", "started"} <= set(meta)

    log = tmp_path / "job.out"
    log.write_text("\n".join(f"line {i}" for i in range(80)))
    notes = tmp_path / "notes.md"
    notes.write_text("Ran as specified.")
    out = tmp_path / "reports"
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_run.py"), str(run), "--id", "T0001",
                    "--title", "smoke", "--log", str(log), "--notes", str(notes), "--out", str(out)], check=True)

    summary = json.loads((out / "T0001.json").read_text())
    assert summary["progress"] == {"batches_done": 3, "batches_planned": 3, "last_batch": 3}
    assert [v["batch"] for v in summary["val"]] == [0, 2, 3]
    assert summary["non_default_config"]["grpo"]["lam"] == 0.5
    assert summary["train_last"]["sec_per_decision"] > 0
    assert summary["anomalies"] == []
    md = (out / "T0001.md").read_text()
    for section in ("## Anomalies", "## Validation", "## Training", "## Log tail", "## Executor notes"):
        assert section in md
    assert "line 79" in md and "line 29" not in md and "Ran as specified." in md


def test_anomalies_detected(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"args": {"batches": 5}, "grpo": {"lam": 0.0}, "policy": {}}))
    row = {"split": "train", "active_groups": 0, "mixed_groups": 0, "sampling_vs_old_max_abs_logp_diff": 0.2,
           "passes": [{"loss": float("nan"), "grad_norm": 1.0, "clip_frac": 0.5}]}
    (run / "metrics.jsonl").write_text("\n".join(json.dumps({"batch": b, **row}) for b in (1, 2)) + '\n{"cut')
    sys.path.insert(0, str(ROOT / "scripts"))
    from summarize_run import summarize

    text = " | ".join(summarize(run, "T", "")["anomalies"])
    for expected in ("loss = nan", "incomplete: 2 of 5", "no learning signal", "max diff 0.2", "clip fraction",
                     "no validation"):
        assert expected in text
