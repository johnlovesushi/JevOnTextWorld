"""T0007: SFT on expert states (jevtw.sft, scripts/build_sft_data.py), RL from an SFT checkpoint, the final
evaluation (scripts/final_eval.py) and its aggregation (scripts/aggregate_final.py)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from jevtw import sft, train  # noqa: E402
from jevtw.config import PolicyConfig  # noqa: E402
from jevtw.envs import KeyChestEnv, Observation, TextWorldEnv  # noqa: E402
from jevtw.policy import LMPolicy, build_prompt  # noqa: E402
from jevtw.testing import ByteTokenizer, tiny_model  # noqa: E402


def no_reasoning_policy(seed=0, **overrides):
    cfg = PolicyConfig(max_reasoning_tokens=0, history_window=3, yes_text="Y", no_text="N",
                       confidence_prompt="\nWill you win? Y or N:", **overrides)
    return LMPolicy(tiny_model(seed=seed), ByteTokenizer(), cfg)


def _rows(run):
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


# ---- 1. build_sft_data follows the walkthrough of a real game

def test_build_sft_data_matches_walkthrough(cooking_game, tmp_path, capsys):
    import build_sft_data as bsd
    from diagnose_history import walk, walkthrough

    out = tmp_path / "sft" / "train.jsonl"
    assert bsd.main(["--games", str(cooking_game.parent), "--out", str(out)]) == 0
    rows = sft.read_rows(out)
    env = TextWorldEnv(cooking_game)
    expected = [c for c in walkthrough(cooking_game) if env._keep(c)]
    env.close()
    assert [r["command"] for r in rows] == expected and [r["t"] for r in rows] == list(range(len(expected)))
    assert all(r["game"] == cooking_game.name for r in rows)
    states = [item for item in walk(cooking_game) if not isinstance(item, dict)]
    cfg_full, cfg_feedback = PolicyConfig(max_reasoning_tokens=0), PolicyConfig(max_reasoning_tokens=0,
                                                                                history_mode="feedback")
    for row, (t, objective, history, obs, a_star) in zip(rows, states):
        assert row["obs"]["commands"][row["a_star"]] == a_star == row["command"]
        assert len(row["history"]) == t and [h["command"] for h in row["history"]] == expected[:t]
        assert set(row["obs"]) == {"text", "commands", "done", "won", "feedback"}
        got = sft.row_state(row)
        assert got == (objective, history, obs, obs.commands.index(a_star))
        for cfg in (cfg_full, cfg_feedback):                  # the raw material rebuilds the prompt of either mode
            assert build_prompt(got[0], got[1], got[2], cfg) == build_prompt(objective, history, obs, cfg)
    stats = json.loads(Path(str(out) + ".stats.json").read_text())
    assert stats["games"] == stats["games_used"] == 1 and stats["states"] == len(expected) and stats["dropped"] == []
    assert stats["states_per_game"] == {str(len(expected)): 1}
    assert sum(stats["action_types"].values()) == len(expected) and stats["action_types"]["eat"] == 1
    assert '"states"' in capsys.readouterr().out
    with pytest.raises(SystemExit):                           # an existing file is not overwritten
        bsd.main(["--games", str(cooking_game.parent), "--out", str(out)])


def test_keychest_expert_rows_win():
    rows = sft.keychest_expert_rows(range(6))
    assert len(rows) == 18 and {r["command"] for r in rows} == {"open drawer", "open box", "take key", "open chest"}
    for seed in range(6):
        env, mine = KeyChestEnv(), [r for r in rows if r["game"] == f"keychest-{seed}"]
        obs = env.reset(seed)
        for r in mine:
            assert sft.row_state(r)[2] == obs and obs.commands[r["a_star"]] == r["command"]
            obs = env.step(r["command"])
        assert obs.won


# ---- 2. the loss is the cross-entropy of a* under the softmax of decision_forward's scores

def test_sft_loss_equals_hand_computed_cross_entropy():
    policy = no_reasoning_policy()
    for row in sft.keychest_expert_rows([0, 1]):
        objective, history, obs, a_star = sft.row_state(row)
        d = sft.expert_decision(policy, objective, history, obs, a_star)
        assert d.reasoning_ids == [] and d.action == a_star and d.command == row["command"]
        assert d.prompt_ids == policy.encode(build_prompt(objective, history, obs, policy.cfg))
        assert len(d.option_ids) == len(obs.commands)
        loss, hit = sft.sft_loss(policy, d)
        with torch.no_grad():
            _, scores, _ = policy.decision_forward(policy.model, d, forecast=False)
        scores = [float(v) for v in scores]
        by_hand = math.log(sum(math.exp(v) for v in scores)) - scores[a_star]
        assert float(loss.detach()) == pytest.approx(by_hand, abs=1e-5)
        assert hit == (max(range(len(scores)), key=scores.__getitem__) == a_star)
        # the same number as the action term GRPO trains: -log q(a_t) of LMPolicy.score
        logps, _ = policy.score(d)
        assert float(loss.detach()) == pytest.approx(-float(logps[-1].detach()), abs=1e-6) and logps.numel() == 1


def test_step_accumulates_the_gradient_of_the_mean_loss_and_warms_up():
    policy = no_reasoning_policy()
    decisions = [sft.expert_decision(policy, *sft.row_state(r)) for r in sft.keychest_expert_rows([0])]
    params = [p for p in policy.model.parameters() if p.requires_grad]
    mean_loss = sum(sft.sft_loss(policy, d)[0] for d in decisions) / len(decisions)
    expected = torch.autograd.grad(mean_loss, params)

    assert sft.warmup_steps(10, 0.1) == 1 and sft.warmup_steps(35, 0.1) == 4 and sft.warmup_steps(10, 0.0) == 0
    assert [sft.lr_factor(k, 4) for k in (1, 2, 4, 5)] == [0.25, 0.5, 1.0, 1.0] and sft.lr_factor(1, 0) == 1.0
    trainer = sft.SFTTrainer(policy, lr=1e-3, weight_decay=0.0, grad_clip=1e9, total_steps=40, warmup_frac=0.1)
    before = [p.detach().clone() for p in params]
    stats = trainer.step(decisions)
    for p, g in zip(params, expected):                        # .grad is still that of the step
        assert torch.allclose(p.grad, g, atol=1e-6)
    assert stats["n"] == 3 and stats["loss_sum"] == pytest.approx(float(mean_loss.detach()) * 3, rel=1e-5)
    assert stats["lr"] == pytest.approx(1e-3 / 4) and trainer.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)
    assert any(not torch.equal(p, b) for p, b in zip(params, before))
    lrs = [trainer.step(decisions)["lr"] for _ in range(4)]
    assert lrs == pytest.approx([5e-4, 7.5e-4, 1e-3, 1e-3])
    assert trainer.optimizer.defaults["weight_decay"] == 0.0


# ---- 3. on the KeyChest expert the loss falls and the accuracy rises

def test_keychest_expert_is_learned():
    policy = no_reasoning_policy()
    rows = sft.keychest_expert_rows(range(8))
    decisions = [sft.expert_decision(policy, *sft.row_state(r)) for r in rows]
    later = [i for i, r in enumerate(rows) if r["t"] > 0]

    def measure():
        with torch.no_grad():
            results = [sft.sft_loss(policy, d) for d in decisions]
        return (sum(float(v) for v, _ in results) / len(results), sum(hit for _, hit in results) / len(results),
                sum(results[i][1] for i in later) / len(later))

    loss0, acc0, later0 = measure()
    trainer = sft.SFTTrainer(policy, lr=3e-3, weight_decay=0.0, grad_clip=1.0, total_steps=60, warmup_frac=0.1)
    for _ in range(20):
        for i in range(0, len(decisions), 8):
            trainer.step(decisions[i:i + 8])
    loss1, acc1, later1 = measure()
    assert loss1 < 0.5 * loss0 and acc1 > acc0, (loss0, acc0, loss1, acc1)
    # The first state is the same text in every game while the expert's command depends on where the key is hidden,
    # so only the majority container can be right there; every later state is decided by what the agent has seen.
    first = [r["command"] for r in rows if r["t"] == 0]
    majority = max(first.count("open drawer"), first.count("open box"))
    assert later1 == 1.0 and later1 > later0 and acc1 == pytest.approx((len(later) + majority) / len(rows))


# ---- 4. end to end: SFT (smoke) -> checkpoint -> jevtw.train --model <dir> and scripts/final_eval.py

def test_sft_checkpoint_feeds_rl_and_final_eval(tmp_path, capsys):
    import final_eval

    run = tmp_path / "sft"
    sft.main(["--smoke", "--out", str(run), "--epochs", "3", "--lr", "3e-3", "--batch-decisions", "8"])
    rows = _rows(run)
    train_rows, val_rows = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]
    assert [r["batch"] for r in train_rows] == [1, 2, 3] == [r["batch"] for r in val_rows]
    assert all({"sft_loss", "train_acc", "lr", "grad_norm"} <= set(r) and "passes" not in r for r in train_rows)
    assert train_rows[-1]["sft_loss"] < train_rows[0]["sft_loss"] and train_rows[-1]["steps"] == 9
    assert all(r["episodes"] == 8 and 0.0 <= r["win_rate"] <= 1.0 for r in val_rows)
    config = json.loads((run / "config.json").read_text())
    assert config["sft"] == {"states": 24, "games": 8, "steps_per_epoch": 3, "total_steps": 9, "warmup_steps": 1}
    assert config["policy"]["max_reasoning_tokens"] == 0 and config["args"]["lr"] == 3e-3
    checkpoint = run / "checkpoints" / "epoch03"
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == ["epoch01", "epoch02", "epoch03"]
    assert (checkpoint / "config.json").is_file() and not (checkpoint / "platt.json").exists()
    with pytest.raises(SystemExit):                           # a second run does not write into the directory
        sft.main(["--smoke", "--out", str(run)])
    with pytest.raises(SystemExit):                           # reasoning must be off outside --smoke
        sft.main(["--out", str(tmp_path / "x"), "--train-data", str(tmp_path / "none.jsonl")])

    # (b) RL starts from the SFT checkpoint; with beta > 0 the frozen reference is that checkpoint, so the KL of
    # the first pass (taken before any update) is 0
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"policy": {"max_reasoning_tokens": 0}, "grpo": {"beta": 0.1}}))
    rl = tmp_path / "rl"
    train.main(["--smoke", "--model", str(checkpoint), "--config", str(cfg), "--out", str(rl), "--batches", "2"])
    rl_rows = _rows(rl)
    first_val = next(r for r in rl_rows if r["split"] == "val")
    # the same model on the same games and seeds as the SFT run's last evaluation
    assert first_val["batch"] == 0 and first_val["win_rate"] == pytest.approx(val_rows[-1]["win_rate"])
    rl_train = [r for r in rl_rows if r["split"] == "train"]
    assert len(rl_train) == 2 and rl_train[0]["passes"][0]["kl"] == pytest.approx(0.0, abs=1e-6)
    assert all(math.isfinite(p["loss"]) for r in rl_train for p in r["passes"])
    assert json.loads((rl / "config.json").read_text())["args"]["model"] == str(checkpoint)

    # (c) the final evaluation of the checkpoint
    out = tmp_path / "final"
    assert final_eval.main(["--smoke", "--model", str(checkpoint), "--config", str(cfg), "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    records = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert summary["seeds"] == [101, 102, 103] and summary["episodes"] == len(records) == 4 * 3
    assert sorted({r["sample_seed"] for r in records}) == [101, 102, 103] and summary["games"] == 4
    assert summary["wins"] == sum(r["z"] for r in records) and summary["win_rate"] == summary["wins"] / 12
    assert summary["wilson95"] == pytest.approx(final_eval.wilson(summary["wins"], 12))
    assert summary["mean_length"] == pytest.approx(sum(r["length"] for r in records) / 12)
    assert sum(summary["action_types"].values()) == sum(r["length"] for r in records)
    assert set(summary["action_types"]) <= {"open", "take", "smash", "wait"}
    assert {"forecast_brier", "forecast_brier_const"} <= set(summary) and "commit" in summary["meta"]
    for rec in records:
        assert set(rec) == {"game", "env_seed", "sample_seed", "z", "length", "steps"}
        assert all(set(step) == {"t", "command", "s", "p", "s_options"} for step in rec["steps"])
    with pytest.raises(SystemExit):
        final_eval.main(["--smoke", "--model", str(checkpoint), "--out", str(out)])
    capsys.readouterr()

    # the report generator reads the SFT run
    reports = tmp_path / "reports"
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_run.py"), str(run), "--id", "T0007", "--title", "sft",
                    "--out", str(reports)], check=True)
    report = json.loads((reports / "T0007.json").read_text())
    assert report["progress"] == {"batches_done": 3, "batches_planned": 3, "last_batch": 3}
    assert [r["epoch"] for r in report["sft"]] == [1, 2, 3] and report["sft"][0]["train_acc"] is not None
    assert report["train_first"] is None and report["anomalies"] == [] and len(report["val"]) == 3
    text = (reports / "T0007.md").read_text()
    assert "## Training (SFT, one row per epoch)" in text and "| epoch | sft_loss | train_acc |" in text
    assert "3 / 3 epochs (SFT)" in text and "24 expert states from 8 games" in text


def test_summarize_flags_an_incomplete_sft_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"args": {"epochs": 5}, "policy": {}, "grpo": {}}))
    lines = [{"batch": 1, "split": "train", "sft_loss": 1.0, "train_acc": 0.5, "grad_norm": 2.0},
             {"batch": 1, "split": "val", "win_rate": 0.25},
             {"batch": 2, "split": "train", "sft_loss": float("nan"), "train_acc": 0.5, "grad_norm": 2.0}]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    out = tmp_path / "reports"
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_run.py"), str(run), "--id", "X", "--out", str(out)],
                   check=True)
    anomalies = json.loads((out / "X.json").read_text())["anomalies"]
    assert anomalies == ["epoch 2: sft_loss = nan", "incomplete: 2 of 5 epochs"]


# ---- 5. Wilson interval and aggregation over training seeds

def test_wilson_interval():
    import final_eval

    z = final_eval.Z95
    assert z == pytest.approx(1.96, abs=1e-3)
    low, high = final_eval.wilson(7, 20)                      # textbook value for 7 / 20
    assert low == pytest.approx(0.18119, abs=1e-5) and high == pytest.approx(0.56715, abs=1e-5)
    # by hand: (p + z^2/2n -+ z sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)
    p, n = 0.35, 20
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    assert [low, high] == pytest.approx([(p + z * z / (2 * n) - half) / (1 + z * z / n),
                                         (p + z * z / (2 * n) + half) / (1 + z * z / n)])
    assert final_eval.wilson(0, 10) == pytest.approx([0.0, z * z / (10 + z * z)])       # no wins: [0, z^2/(n+z^2)]
    assert final_eval.wilson(10, 10) == pytest.approx([10 / (10 + z * z), 1.0])
    with pytest.raises(ValueError):
        final_eval.wilson(0, 0)


def test_final_summary_counts():
    import final_eval

    records = [{"game": "a", "z": 1, "length": 2, "steps": [{"command": "take key"}, {"command": "open chest"}]},
               {"game": "a", "z": 0, "length": 1, "steps": [{"command": "smash chest"}]},
               {"game": "b", "z": 0, "length": 3, "steps": [{"command": "open box"}, {"command": "wait"},
                                                            {"command": "open drawer"}]}]
    s = final_eval.summarize(records, {"forecast_auc": 0.75, "forecast_auc_t0": None, "forecast_brier": 0.2,
                                       "forecast_brier_const": 0.25, "win_rate": 1 / 3})
    assert s["episodes"] == 3 and s["wins"] == 1 and s["win_rate"] == pytest.approx(1 / 3) and s["games"] == 2
    assert s["mean_length"] == 2.0 and s["action_types"] == {"open": 3, "take": 1, "smash": 1, "wait": 1}
    assert s["forecast_auc"] == 0.75 and s["forecast_brier"] == 0.2 and "forecast_auc_t0" not in s
    assert "forecast_auc" not in final_eval.summarize(records, {})                   # a policy without forecasts


def test_aggregate_final(tmp_path, capsys):
    import aggregate_final as agg
    import final_eval

    def summary(wins, auc):
        return {"episodes": 120, "wins": wins, "win_rate": wins / 120, "wilson95": final_eval.wilson(wins, 120),
                "mean_length": 10.0 + wins / 10, "seeds": [101, 102, 103], "action_types": {"take": wins, "eat": 1},
                **({} if auc is None else {"forecast_auc": auc})}

    for name, wins, auc in (("s17", 30, 0.6), ("s23", 60, 0.7), ("s29", 90, None)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(summary(wins, auc)))
    out = tmp_path / "agg" / "c.json"
    assert agg.main([f"s17={tmp_path / 's17'}", f"s23={tmp_path / 's23' / 'summary.json'}", str(tmp_path / "s29"),
                     "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert list(result["runs"]) == ["s17", "s23", "s29"]
    assert [r["win_rate"] for r in result["runs"].values()] == [0.25, 0.5, 0.75]
    win = result["stats"]["win_rate"]
    assert win["mean"] == pytest.approx(0.5) and win["std"] == pytest.approx(0.25) and win["n"] == 3   # n - 1
    assert result["stats"]["mean_length"]["mean"] == pytest.approx(16.0)
    assert result["stats"]["mean_length"]["std"] == pytest.approx(3.0)
    auc = result["stats"]["forecast_auc"]                    # the run without forecasts is left out
    assert auc["n"] == 2 and auc["mean"] == pytest.approx(0.65) and auc["std"] == pytest.approx(math.sqrt(0.005))
    assert result["stats"]["forecast_brier"] == {"mean": None, "std": None, "n": 0}
    assert result["pooled"]["wins"] == 180 and result["pooled"]["episodes"] == 360
    assert result["pooled"]["wilson95"] == pytest.approx(final_eval.wilson(180, 360))
    assert result["action_types"] == {"take": 180, "eat": 3}
    text = capsys.readouterr().out
    assert "| s23 | 60/120 | 0.5000 |" in text and "| mean | 180/360 | 0.5000 |" in text and "| std | | 0.2500 |" in text
    assert agg.mean_std([0.3]) == (0.3, None) and agg.mean_std([]) == (None, None)
    with pytest.raises(SystemExit):
        agg.main([f"a={tmp_path / 's17'}", f"a={tmp_path / 's23'}"])
