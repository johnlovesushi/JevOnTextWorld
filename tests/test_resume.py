"""--start-batch / --ref-model: a run resumed from a checkpoint draws the same games as the uninterrupted run and
takes its KL reference from --ref-model, not from the checkpoint it starts from (E0015: runs killed by a full
disk at batch ~140 had to be continued)."""
import json

import torch

from jevtw import train


def record_jobs(monkeypatch):
    seen = []
    real = train.draw_jobs

    def spy(rng, games, g):
        jobs = real(rng, games, g)
        seen.append([(name, env_seed, seed) for job in jobs for (name, _, env_seed, seed) in job])
        return jobs

    monkeypatch.setattr(train, "draw_jobs", spy)
    return seen


def lines(run):
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


def test_resume_replays_the_game_draws_and_loads_the_reference(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"grpo": {"beta": 0.04}}))      # beta > 0: a reference model is loaded
    full = tmp_path / "full"
    seen = record_jobs(monkeypatch)
    train.main(["--smoke", "--out", str(full), "--config", str(cfg), "--batches", "3"])
    assert len(seen) == 3
    full_jobs = list(seen)

    references = []
    real_trainer = train.Trainer

    def capture(policy, cfg, reference_model=None, **kw):
        references.append(reference_model)
        return real_trainer(policy, cfg, reference_model=reference_model, **kw)

    monkeypatch.setattr(train, "Trainer", capture)
    seen.clear()
    ref_dir = full / "checkpoints" / "batch0003"          # any directory other than --model
    cont = tmp_path / "cont"
    train.main(["--smoke", "--out", str(cont), "--config", str(cfg), "--batches", "3", "--start-batch", "3",
                "--model", str(full / "checkpoints" / "batch0002"), "--ref-model", str(ref_dir)])

    # the fast-forward redraws batches 1-2, so the one batch played is batch 3 of the uninterrupted run
    assert seen == full_jobs
    rows = lines(cont)
    assert [(r["batch"], r["split"]) for r in rows] == [(3, "train"), (3, "val")]   # no evaluation of batch 0

    from transformers import AutoModelForCausalLM
    want = AutoModelForCausalLM.from_pretrained(ref_dir, dtype=torch.float32).state_dict()
    got = references[0].state_dict()
    assert all(torch.equal(got[k], want[k]) for k in want)


def test_start_batch_is_checked(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        train.main(["--smoke", "--out", str(tmp_path / "x"), "--batches", "3", "--start-batch", "4"])
