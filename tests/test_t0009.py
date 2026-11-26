"""T0009: data parallelism over two processes (gloo on the CPU) gives the update of one process."""
import copy
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from jevtw import dist as dist_mod
from jevtw.config import GRPOConfig, PolicyConfig
from jevtw.envs import KeyChestEnv
from jevtw.policy import LMPolicy
from jevtw.rollout import collect_group
from jevtw.testing import ByteTokenizer, tiny_model
from jevtw.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]
POLICY = dict(max_reasoning_tokens=4, history_window=3, yes_text="Y", no_text="N",
              confidence_prompt="\nWill you win? Y or N:", platt=True)
GRPO = dict(lam=0.3, alpha=0.1, beta=0.04, cal_loss="bce", cal_grad="platt_only", optimization_passes=2,
            learning_rate=1e-3, warmup_batches=2)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _setup(policy_kw, grpo_kw):
    torch.manual_seed(0)
    model = tiny_model(seed=0)
    reference = copy.deepcopy(model).requires_grad_(False)
    policy = LMPolicy(model, ByteTokenizer(), PolicyConfig(**{**POLICY, **policy_kw}))
    return policy, reference, GRPOConfig(**{**GRPO, **grpo_kw})


def _groups(policy):
    return [collect_group(policy, KeyChestEnv, f"keychest-{g}", g, list(range(100 * g, 100 * g + 4)), 3)
            for g in range(3)]


def _train(policy, reference, cfg, groups, dist):
    trainer = Trainer(policy, cfg, reference_model=reference, dist=dist)
    metrics = trainer.step(copy.deepcopy(groups))
    params = {n: p.detach().clone() for n, p in policy.model.named_parameters()}
    params.update({f"platt{i}": p.detach().clone() for i, p in enumerate(policy.platt_parameters())})
    return metrics, params


def _worker(rank, world, port, path, policy_kw, grpo_kw):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port))
    torch.set_num_threads(1)
    d = dist_mod.setup()
    try:
        policy, reference, cfg = _setup(policy_kw, grpo_kw)
        groups = pickle.loads(Path(path, "groups.pkl").read_bytes())
        metrics, params = _train(policy, reference, cfg, groups, d)
        torch.save({"metrics": metrics, "params": params}, Path(path, f"rank{rank}.pt"))
    finally:
        dist_mod.teardown()


def test_balance_is_deterministic_and_greedy():
    owner = dist_mod.balance([5, 1, 4, 4, 2], 2)
    assert owner == dist_mod.balance([5, 1, 4, 4, 2], 2)
    loads = [sum(c for c, o in zip([5, 1, 4, 4, 2], owner) if o == r) for r in range(2)]
    assert sorted(loads) == [8, 8]
    assert dist_mod.balance([3, 2], 1) == [0, 0]


def test_single_process_helpers_are_identity():
    d = dist_mod.Dist()
    assert not d.enabled and d.main
    assert d.run_split([1, 2, 3], lambda part: [x * 2 for x in part]) == [2, 4, 6]
    assert d.sum_floats([1.5, 2]) == [1.5, 2]


VERBAL = {"platt": False, "confidence_mode": "verbal", "forecast": False}


@pytest.mark.parametrize("policy_kw,grpo_kw", [
    ({}, {}), ({"platt": False}, {"alpha": 0.0, "cal_grad": "full"}),
    (VERBAL, {"alpha": 0.0, "cal_grad": "full", "adv_norm": "mean"}),                              # E0012 group B
    (VERBAL, {"alpha": 0.0, "cal_grad": "full", "adv_norm": "mean", "lam": 0.0, "conf_advantage": "decoupled",
              "conf_coef": 0.3}),                                                                   # E0012 group C
], ids=["logit-platt", "logit", "verbal-shared", "verbal-decoupled"])
def test_two_processes_equal_one(tmp_path, policy_kw, grpo_kw):
    policy, reference, cfg = _setup(policy_kw, grpo_kw)
    groups = _groups(policy)
    assert any(ep.advantage is None for g in groups for ep in g)   # not trained yet
    (tmp_path / "groups.pkl").write_bytes(pickle.dumps(groups))
    single_metrics, single_params = _train(policy, reference, cfg, groups, None)

    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path), policy_kw, grpo_kw), nprocs=2, join=True)
    for rank in range(2):
        out = torch.load(tmp_path / f"rank{rank}.pt", weights_only=False)
        m = out["metrics"]
        for key in ("win_rate", "mean_return", "mean_bonus", "active_groups", "lr"):
            assert m[key] == single_metrics[key], key
        assert m.get("platt_b") == single_metrics.get("platt_b")
        for p_single, p_multi in zip(single_metrics["passes"], m["passes"]):
            for key in ("loss", "grad_norm", "kl", "cal_loss", "clip_frac", "kl_old", "conf_loss", "conf_kl"):
                if p_single.get(key) is not None:
                    assert p_multi[key] == pytest.approx(p_single[key], rel=1e-5, abs=1e-8), key
        # atol = 1% of the learning rate: Adam divides by sqrt(v), so an element whose gradient is ~0 can move by
        # a visible fraction of lr when the summation order changes its rounding.
        for name, value in single_params.items():
            torch.testing.assert_close(out["params"][name], value, rtol=1e-5, atol=1e-5, msg=lambda m, name=name: f"{name}: {m}")
    # both processes hold the same parameters
    a, b = (torch.load(tmp_path / f"rank{r}.pt", weights_only=False)["params"] for r in range(2))
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_torchrun_smoke_matches_single_process(tmp_path):
    torchrun = shutil.which("torchrun") or str(Path(sys.executable).with_name("torchrun"))
    if not Path(torchrun).exists():
        pytest.skip("torchrun not available")
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"grpo": {"parallel_episodes": 1, "beta": 0.04}}))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    base = ["-m", "jevtw.train", "--smoke", "--config", str(config), "--no-save"]
    for cmd, out in (([sys.executable], "one"), ([torchrun, "--standalone", "--nproc-per-node=2"], "two")):
        proc = subprocess.run([*cmd, *base, "--out", str(tmp_path / out)], cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-3000:]
    one = [json.loads(x) for x in (tmp_path / "one" / "metrics.jsonl").read_text().splitlines()]
    two = [json.loads(x) for x in (tmp_path / "two" / "metrics.jsonl").read_text().splitlines()]
    assert len(one) == len(two) > 0
    for a, b in zip(one, two):
        assert (a["batch"], a["split"]) == (b["batch"], b["split"])
        assert a["win_rate"] == b["win_rate"] and a["decisions"] == b["decisions"]
        if a["split"] == "train":
            assert b["passes"][0]["loss"] == pytest.approx(a["passes"][0]["loss"], rel=1e-4, abs=1e-7)
        else:
            assert b["forecast_brier"] == pytest.approx(a["forecast_brier"], rel=1e-4)
    assert json.loads((tmp_path / "two" / "config.json").read_text())["meta"]["world_size"] == 2
