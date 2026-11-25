"""E0008: scripts/forecast_calibration.py (log loss, saturation, per-step and reliability tables)."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

A, B = 0.5, -1.0


def sig(x):
    return 1 / (1 + math.exp(-x))


def episode(z, logits, platt=False, game="g"):
    steps = [{"t": t, "command": "c", "s": s, "p": sig(A * s + B) if platt else sig(s), "s_options": [s]}
             for t, s in enumerate(logits)]
    return {"game": game, "env_seed": 0, "sample_seed": 1, "z": z, "length": len(steps), "steps": steps}


def write_run(tmp_path, records_by_batch, platt, train_rows=()):
    run = tmp_path / "E9999x"
    (run / "forecasts").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"policy": {"platt": platt}, "grpo": {"cal_loss": "bce"}}))
    lines = list(train_rows)
    for batch, records in records_by_batch.items():
        (run / "forecasts" / f"batch{batch:04d}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        lines.append({"batch": batch, "split": "val", "win_rate": sum(r["z"] for r in records) / len(records)})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in lines) + "\n")
    return run


def test_log_loss_and_baselines():
    import forecast_calibration as fc

    records = [episode(1, [2.0, 0.0]), episode(0, [-1.0, 3.0, 0.5])]
    e = fc.evaluate(records, platt=False)
    expected = -(math.log(sig(2.0)) + math.log(sig(0.0)) + math.log(1 - sig(-1.0)) + math.log(1 - sig(3.0))
                 + math.log(1 - sig(0.5))) / 5
    assert e["log_loss"] == pytest.approx(expected)
    w = 2 / 5                                              # decision-weighted, not 1 / 2 episodes
    assert e["w"] == pytest.approx(w) and e["win_rate"] == 0.5 and e["wins"] == 1 and e["decisions"] == 5
    assert e["log_loss_const"] == pytest.approx(-(w * math.log(w) + (1 - w) * math.log(1 - w)))
    assert e["brier_const"] == pytest.approx(w * (1 - w))
    assert e["auc_t0"] == 1.0 and e["platt_fit"] is None
    # from s the loss stays finite where sigmoid(s) rounds to 1
    far = fc.evaluate([episode(0, [800.0]), episode(1, [800.0])], platt=False)
    assert far["log_loss"] == pytest.approx(400.0) and far["saturated"] == 1.0


def test_platt_uses_p_counts_exact_values_and_recovers_a_b():
    import forecast_calibration as fc

    records = [episode(1, [4.0, 1.0, 6.0], platt=True), episode(0, [0.0, -2.0], platt=True)]
    e = fc.evaluate(records, platt=True)
    ps = [(sig(A * s + B), z) for z, ss in ((1, [4.0, 1.0, 6.0]), (0, [0.0, -2.0])) for s in ss]
    assert e["log_loss"] == pytest.approx(-sum(math.log(p if z else 1 - p) for p, z in ps) / 5)
    assert e["platt_fit"]["a"] == pytest.approx(A) and e["platt_fit"]["b"] == pytest.approx(B)
    assert e["p_exact_01"] == 0
    records[1]["steps"][0]["p"] = 1.0                      # a lost episode forecast with p exactly 1
    e = fc.evaluate(records, platt=True)
    assert e["p_exact_01"] == 1 and math.isfinite(e["log_loss"]) and e["log_loss"] > -math.log(fc.EPS) / 5


def test_saturation_steps_and_reliability():
    import forecast_calibration as fc

    won = episode(1, [0.0] * 8)
    lost = episode(0, [0.0] * 3)
    for st, p in zip(won["steps"], [0.999, 0.994, 0.993, 0.95, 0.55, 0.05, 0.007, 0.001]):
        st["p"] = p
    for st, p in zip(lost["steps"], [0.006, 0.5, 1.0]):
        st["p"] = p
    e = fc.evaluate([won, lost], platt=True)
    assert e["saturated"] == pytest.approx(5 / 11)         # 0.999, 0.994, 0.001, 0.006, 1.0; the bounds are not
    steps = {r["t"]: r for r in e["by_step"]}
    assert [r["t"] for r in e["by_step"]] == ["0", "1", "2", "3", "4", "5", ">=6"]
    assert steps["0"]["n"] == 2 and steps["3"]["n"] == 1 and steps[">=6"]["n"] == 2 and steps[">=6"]["wins"] == 2
    assert steps["0"]["auc"] == 0.5 and steps["3"]["auc"] is None      # all s equal; one class only
    assert steps["0"]["brier"] == pytest.approx(((0.999 - 1) ** 2 + 0.006 ** 2) / 2)
    rel = e["reliability"]
    assert len(rel) == 10 and sum(r["n"] for r in rel) == 11
    assert rel[9]["n"] == 5 and rel[9]["win_rate"] == pytest.approx(4 / 5)   # p = 1.0 falls into the last bin
    assert rel[0]["n"] == 4 and rel[0]["win_rate"] == pytest.approx(3 / 4)
    assert rel[5]["n"] == 2 and rel[5]["mean_p"] == pytest.approx(0.525) and rel[2]["mean_p"] is None


def test_analyze_and_markdown(tmp_path, capsys):
    import forecast_calibration as fc

    recs = [episode(1, [1.0, 2.0], platt=True), episode(0, [-1.0], platt=True)]
    train = [{"batch": b, "split": "train", "mixed_groups": b, "platt_a": A, "platt_b": B,
              "passes": [{"loss": 0.1, "grad_norm": 1.0, "kl": 0.0, "kl_old": 0.0, "clip_frac": 0.0},
                         {"loss": 0.1, "grad_norm": 1.0, "kl": 0.1 * b, "kl_old": 0.05, "clip_frac": 0.2}]}
             for b in (1, 2)]
    run = write_run(tmp_path, {0: recs, 2: recs}, platt=True, train_rows=train)
    out = tmp_path / "cal.json"
    fc.main([str(run), "--out", str(out), "--markdown"])
    result = json.loads(out.read_text())
    assert result["platt"] and [e["batch"] for e in result["evals"]] == [0, 2]
    assert "kl" not in result["evals"][0] and result["evals"][1]["kl"] == pytest.approx(0.2)
    assert result["evals"][1]["clip_frac"] == 0.2 and result["evals"][1]["logged"]["win_rate"] == 0.5
    t = result["training"]
    assert t["mean_mixed_groups"] == 1.5 and t["max_kl_last_pass"] == pytest.approx(0.2) and t["max_kl_batch"] == 2
    assert t["all_finite"]
    text = capsys.readouterr().out
    assert "Platt a" in text and "Reliability, batch 0" in text and "Reliability, batch 2" in text
    assert "| 2 | 1/2 | 0.5000 |" in text

    run2 = write_run(tmp_path / "np", {0: [episode(1, [1.0]), episode(0, [2.0])]}, platt=False)
    fc.main([str(run2), "--markdown"])
    assert "Platt a" not in capsys.readouterr().out
