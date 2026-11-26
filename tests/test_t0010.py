"""T0010: build_sft_data.py skips redundant `open X` of the walkthrough by default (--keep-redundant-open restores the
old output byte for byte), and jevtw.sft pools several data files with a level label per state, in one process
and in two (torchrun)."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_sft_data as bsd  # noqa: E402
import diagnose_history as dh  # noqa: E402
from jevtw import sft  # noqa: E402
from jevtw.envs import Observation  # noqa: E402
from jevtw.testing import ByteTokenizer  # noqa: E402

BEFORE_T0010 = "7c90c6859d71e97aa4ee2061b96104f3e1c2ccee"     # the last commit with the old build_sft_data.py


class FakeEnv:
    """A fridge that has to be opened once; the walkthrough of tw-cooking opens it again later (E0011)."""
    objective = "cook"

    def __init__(self, path):
        self.open, self.won = False, False

    def _keep(self, cmd):
        return cmd != "inventory"

    def _obs(self):
        cmds = (["close fridge", "take a", "take b"] if self.open else ["open fridge"]) + ["eat meal"]
        return Observation(text=f"open={self.open}", commands=[] if self.won else cmds, done=self.won, won=self.won,
                           feedback="")

    def reset(self, seed):
        return self._obs()

    def step(self, cmd):
        self.open = self.open or cmd == "open fridge"
        self.won = cmd == "eat meal"
        return self._obs()

    def close(self):
        pass


def _fake_games(directory: Path) -> Path:
    """Two games: `r.z8` with a redundant second `open fridge`, `s.z8` without."""
    directory.mkdir(parents=True)
    for name, walk in (("r", ["inventory", "open fridge", "take a", "open fridge", "take b", "eat meal"]),
                       ("s", ["open fridge", "take a", "take b", "eat meal"])):
        (directory / f"{name}.z8").write_bytes(b"")
        (directory / f"{name}.json").write_text(json.dumps({"metadata": {"walkthrough": walk}}))
    return directory


def _old_module(tmp_path):
    """scripts/build_sft_data.py as it was before T0010, from git."""
    proc = subprocess.run(["git", "show", f"{BEFORE_T0010}:scripts/build_sft_data.py"], cwd=ROOT, capture_output=True,
                          text=True)
    if proc.returncode != 0:
        pytest.skip(f"git show failed: {proc.stderr[-200:]}")
    path = tmp_path / "build_sft_data_before_t0010.py"
    path.write_text(proc.stdout)
    spec = importlib.util.spec_from_file_location("build_sft_data_before_t0010", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- 1. the redundant open is skipped and the data follows the walkthrough

def test_build_skips_redundant_open(tmp_path, monkeypatch):
    monkeypatch.setattr(dh, "TextWorldEnv", FakeEnv)
    games = _fake_games(tmp_path / "games")
    out = tmp_path / "sft" / "train.jsonl"
    assert bsd.main(["--games", str(games), "--out", str(out)]) == 0
    rows = sft.read_rows(out)
    by_game = {g: [r for r in rows if r["game"] == g] for g in ("r.z8", "s.z8")}
    for game, mine in by_game.items():               # both are the walkthrough without inventory and the second open
        assert [r["command"] for r in mine] == ["open fridge", "take a", "take b", "eat meal"], game
        assert [r["t"] for r in mine] == [0, 1, 2, 3]
        assert [len(r["history"]) for r in mine] == [0, 1, 2, 3]
        assert all(r["obs"]["commands"][r["a_star"]] == r["command"] for r in mine)
    assert by_game["r.z8"] == [{**r, "game": "r.z8"} for r in by_game["s.z8"]]
    stats = json.loads(Path(str(out) + ".stats.json").read_text())
    assert stats["games"] == stats["games_used"] == 2 and stats["dropped"] == [] and stats["states"] == 8
    assert stats["redundant_open_skipped"] == 1 and stats["games_with_redundant_open"] == 1

    # --keep-redundant-open: the old behaviour, the game with the redundant open is dropped
    kept = tmp_path / "sft" / "keep.jsonl"
    assert bsd.main(["--games", str(games), "--out", str(kept), "--keep-redundant-open"]) == 0
    stats = json.loads(Path(str(kept) + ".stats.json").read_text())
    assert stats["games_used"] == 1 and stats["states"] == 4 and "redundant_open_skipped" not in stats
    assert stats["dropped"] == [{"game": "r.z8", "problem": "'open fridge' is not admissible at step 2", "won": False}]


# ---- 2. --keep-redundant-open writes exactly what the script wrote before T0010

def test_keep_redundant_open_is_byte_identical_to_before(tmp_path, monkeypatch, capsys):
    old = _old_module(tmp_path)
    monkeypatch.setattr(dh, "TextWorldEnv", FakeEnv)
    games = _fake_games(tmp_path / "games")
    assert old.main(["--games", str(games), "--out", str(tmp_path / "old.jsonl")]) == 0
    assert bsd.main(["--games", str(games), "--out", str(tmp_path / "new.jsonl"), "--keep-redundant-open"]) == 0
    for suffix in ("", ".stats.json"):
        assert (tmp_path / f"new.jsonl{suffix}").read_bytes() == (tmp_path / f"old.jsonl{suffix}").read_bytes()
    assert (tmp_path / "new.jsonl").stat().st_size > 0


def test_keep_redundant_open_is_byte_identical_on_a_real_game(cooking_game, tmp_path, capsys):
    old = _old_module(tmp_path)
    assert old.main(["--games", str(cooking_game.parent), "--out", str(tmp_path / "old.jsonl")]) == 0
    assert bsd.main(["--games", str(cooking_game.parent), "--out", str(tmp_path / "new.jsonl"),
                     "--keep-redundant-open"]) == 0
    for suffix in ("", ".stats.json"):
        assert (tmp_path / f"new.jsonl{suffix}").read_bytes() == (tmp_path / f"old.jsonl{suffix}").read_bytes()
    # this game has no redundant open: the default output has the same states
    assert bsd.main(["--games", str(cooking_game.parent), "--out", str(tmp_path / "skip.jsonl")]) == 0
    assert (tmp_path / "skip.jsonl").read_bytes() == (tmp_path / "old.jsonl").read_bytes()
    assert json.loads((tmp_path / "skip.jsonl.stats.json").read_text())["redundant_open_skipped"] == 0


# ---- 3. several data files: pooled, labelled by level, reported per level

def _two_files(tmp_path):
    rows = sft.keychest_expert_rows(range(8))
    a, b = tmp_path / "K1_train.jsonl", tmp_path / "K2_train.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in rows[:15]))
    b.write_text("".join(json.dumps(r) + "\n" for r in rows[15:]))
    return a, b, len(rows)


def test_read_levels_pools_and_labels(tmp_path):
    a, b, n = _two_files(tmp_path)
    assert sft.level_label("data/sft/C3_train.jsonl") == "C3" and sft.level_label("x/C1.jsonl") == "C1"
    rows = sft.read_levels([a, b])
    assert len(rows) == n == len(sft.read_rows(a)) + len(sft.read_rows(b))
    assert [r["level"] for r in rows] == ["K1"] * 15 + ["K2"] * (n - 15)
    tagged = tmp_path / "tagged.jsonl"                          # a row that names its level keeps it
    tagged.write_text(json.dumps({**sft.read_rows(a)[0], "level": "C9"}) + "\n")
    assert sft.read_levels([tagged])[0]["level"] == "C9"


def test_sft_run_reports_each_level(tmp_path, capsys):
    a, b, n = _two_files(tmp_path)
    sft.main(["--smoke", "--train-data", str(a), str(b), "--epochs", "2", "--batch-decisions", "4", "--no-save",
              "--out", str(tmp_path / "run")])
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["sft"]["states"] == n and config["sft"]["levels"] == {"K1": 15, "K2": n - 15}
    train = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in train if r["split"] == "train"]
    assert len(train) == 2
    for r in train:
        assert r["decisions_K1"] == 15 and r["decisions_K2"] == n - 15 and r["decisions"] == n
        pooled = (r["sft_loss_K1"] * 15 + r["sft_loss_K2"] * (n - 15)) / n
        assert pooled == pytest.approx(r["sft_loss"], rel=1e-9)
        assert (r["train_acc_K1"] * 15 + r["train_acc_K2"] * (n - 15)) / n == pytest.approx(r["train_acc"])

    # one file: the log has no per-level keys, as before T0010
    sft.main(["--smoke", "--train-data", str(a), "--epochs", "1", "--no-save", "--out", str(tmp_path / "one")])
    line = json.loads((tmp_path / "one" / "metrics.jsonl").read_text().splitlines()[0])
    assert not any(k.endswith("_K1") for k in line) and line["decisions"] == 15


# ---- 4. two processes (torchrun) give the result of one

def test_torchrun_sft_matches_single_process(tmp_path):
    torchrun = shutil.which("torchrun") or str(Path(sys.executable).with_name("torchrun"))
    if not Path(torchrun).exists():
        pytest.skip("torchrun not available")
    a, b, n = _two_files(tmp_path)
    env = {**os.environ, "OMP_NUM_THREADS": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    base = ["-m", "jevtw.sft", "--smoke", "--train-data", str(a), str(b), "--epochs", "2", "--batch-decisions", "6",
            "--no-save"]
    for cmd, out in (([sys.executable], "one"), ([torchrun, "--standalone", "--nproc-per-node=2"], "two")):
        proc = subprocess.run([*cmd, *base, "--out", str(tmp_path / out)], cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-3000:]
    one = [json.loads(x) for x in (tmp_path / "one" / "metrics.jsonl").read_text().splitlines()]
    two = [json.loads(x) for x in (tmp_path / "two" / "metrics.jsonl").read_text().splitlines()]
    assert len(one) == len(two) == 4                            # train + val per epoch
    for x, y in zip(one, two):
        assert (x["epoch"], x["split"]) == (y["epoch"], y["split"])
        if x["split"] == "train":
            for key in ("sft_loss", "sft_loss_K1", "sft_loss_K2", "grad_norm"):
                assert y[key] == pytest.approx(x[key], rel=1e-4, abs=1e-7), key
            for key in ("train_acc", "train_acc_K1", "train_acc_K2", "decisions", "decisions_K1", "steps"):
                assert y[key] == x[key], key
        else:
            assert y["win_rate"] == x["win_rate"] and y["decisions"] == x["decisions"]
    assert json.loads((tmp_path / "two" / "config.json").read_text())["meta"]["world_size"] == 2
