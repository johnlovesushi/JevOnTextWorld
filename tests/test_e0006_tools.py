"""E0006: scripts/diagnose_history.py walks the walkthrough and scores a* like LMPolicy.decide does."""
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from jevtw.policy import LMPolicy, build_prompt
from jevtw.testing import ByteTokenizer, tiny_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def game(tmp_path_factory):
    """A cooking game with its .json next to it (the walkthrough is read from there)."""
    pytest.importorskip("textworld")
    if shutil.which("tw-make") is None:
        pytest.skip("tw-make not available")
    path = tmp_path_factory.mktemp("tw6") / "cook.z8"
    proc = subprocess.run(["tw-make", "tw-cooking", "--recipe", "1", "--take", "1", "--cut", "--split", "valid",
                           "--seed", "3", "--output", str(path), "-f"], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not path.with_suffix(".json").is_file():
        pytest.skip(f"tw-make failed: {proc.stderr[-300:]}")
    return path


def test_walk_follows_the_walkthrough_and_wins(game):
    import diagnose_history as dh

    items = list(dh.walk(game))
    *states, end = items
    assert end["won"] and end["problem"] is None and end["steps"] == len(states)
    assert "inventory" in end["skipped"]                      # an information command, not an option of the agent
    full = dh.walkthrough(game)
    assert [s[4] for s in states] == [c for c in full if c not in end["skipped"]]
    for t, objective, history, obs, a_star in states:
        assert len(history) == t and a_star in obs.commands and objective
        assert [c for c, _ in history] == [s[4] for s in states[:t]]
    assert states[-1][4] == "eat meal"


def test_variants_and_scores(game):
    import diagnose_history as dh

    assert {n: (c.history_mode, c.history_window) for n, c in ((n, dh.variant_config(n)) for n in dh.VARIANTS)} == \
        {"V1": ("full", 8), "V2": ("feedback", 8), "V3": ("full", 1), "V4": ("feedback", 1), "V5": ("full", 0)}
    model, tokenizer = tiny_model(seed=0), ByteTokenizer()
    out = io.StringIO()
    yn = {"yes_text": "Y", "no_text": "N"}                    # single tokens for the byte tokenizer
    summary = dh.run(model, tokenizer, "tiny", {"L": game.parent}, out, **yn)
    rows = [json.loads(line) for line in out.getvalue().splitlines()]
    ends = [r for r in rows if "end" in r]
    rows = [r for r in rows if "end" not in r]
    n = summary["L"]["states"]
    assert summary["L"] == {"games": 1, "states": n, "walkthroughs_won": 1, "problems": 0} and len(ends) == 1
    assert len(rows) == 5 * n and {r["variant"] for r in rows} == set(dh.VARIANTS)

    states = [s for s in dh.walk(game) if not isinstance(s, dict)]
    for r in rows:
        t, objective, history, obs, a_star = states[r["t"]]
        assert (r["a_star"], r["type"], r["K"]) == (a_star, a_star.split()[0], len(obs.commands))
        # the same numbers as a decision of LMPolicy with this configuration
        policy = LMPolicy(model, tokenizer, dh.variant_config(r["variant"], **yn))
        d = policy.act(objective, history, obs, torch.Generator().manual_seed(0))
        with torch.no_grad():
            _, scores, _ = policy.decision_forward(model, d)
        log_q = torch.log_softmax(scores, dim=0)
        target = obs.commands.index(a_star)
        assert r["logq_star"] == pytest.approx(float(log_q[target]), abs=1e-5)
        assert r["rank_star"] == 1 + int((log_q > log_q[target]).sum())
        assert len(r["top"]) == min(5, r["K"]) and r["top"][0]["p"] == pytest.approx(float(log_q.max().exp()), abs=1e-5)
        assert [x["p"] for x in r["top"]] == sorted((x["p"] for x in r["top"]), reverse=True)
        assert r["prompt_tokens"] == len(policy.encode(build_prompt(objective, history, obs, policy.cfg)))
    # at t = 0 there is no history: all variants give the same prompt and the same numbers
    first = [r for r in rows if r["t"] == 0]
    assert len({r["prompt_tokens"] for r in first}) == 1
    assert max(r["logq_star"] for r in first) - min(r["logq_star"] for r in first) < 1e-5
    # later the layouts differ: feedback histories are shorter than full ones, no history is the shortest
    last = {r["variant"]: r["prompt_tokens"] for r in rows if r["t"] == n - 1}
    assert last["V5"] < last["V4"] < last["V2"] < last["V1"] and last["V4"] < last["V3"] < last["V1"]


def test_show_prints_prompts(game, capsys):
    import diagnose_history as dh

    dh.main(["--show", str(game), "--step", "1", "--variants", "V1", "V5"])
    text = capsys.readouterr().out
    assert text.count("=====") == 4 and "Recent steps:" in text.split("===== V5")[0]
    assert "Recent steps:" not in text.split("===== V5")[1]
