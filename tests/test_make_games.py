"""scripts/make_games.py: the tw-make command line and the manifest, without calling tw-make."""
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_games", ROOT / "scripts/make_games.py")
make_games = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_games)


def run(monkeypatch, tmp_path, *extra):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    make_games.main(["--out", str(tmp_path), "--train", "2", "--valid", "1", "--test", "0", *extra])
    return calls, json.loads((tmp_path / "manifest.json").read_text())


def flags(cmd):
    return [c for c in cmd if c in ("--open", "--cook", "--cut")]


def test_default_skills_and_name_match_the_old_behaviour(monkeypatch, tmp_path):
    calls, manifest = run(monkeypatch, tmp_path)
    assert len(calls) == 3 and all(flags(c) == ["--open", "--cook", "--cut"] for c in calls)
    assert all(m["skills"] == ["open", "cook", "cut"] for m in manifest)
    assert all(Path(m["path"]).name.startswith("cook_r") for m in manifest)


def test_no_skills(monkeypatch, tmp_path):
    calls, manifest = run(monkeypatch, tmp_path, "--recipe", "1", "--take", "1", "--go", "1", "--skills")
    assert all(flags(c) == [] for c in calls) and all(m["skills"] == [] for m in manifest)
    assert all(m["recipe"] == m["take"] == m["go"] == 1 for m in manifest)


def test_some_skills_and_name_prefix(monkeypatch, tmp_path):
    calls, manifest = run(monkeypatch, tmp_path, "--skills", "cut", "open", "--name", "L1")
    assert all(flags(c) == ["--open", "--cut"] for c in calls)
    assert all(m["skills"] == ["open", "cut"] for m in manifest)
    assert [Path(m["path"]).name[:3] for m in manifest] == ["L1_"] * 3
    assert [m["split"] for m in manifest] == ["train", "train", "valid"]
    assert all(c[c.index("--output") + 1] == m["path"] for c, m in zip(calls, manifest))


def test_same_seed_draws_do_not_depend_on_skills(monkeypatch, tmp_path):
    _, a = run(monkeypatch, tmp_path / "a", "--skills")
    _, b = run(monkeypatch, tmp_path / "b", "--skills", "open", "cut")
    assert [(m["seed"], m["recipe_seed"]) for m in a] == [(m["seed"], m["recipe_seed"]) for m in b]
