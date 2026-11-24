"""E0007: scripts/diagnose_cot.py samples reasoning on the walkthrough's prepare / eat states and records q."""
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

YN = {"yes_text": "Y", "no_text": "N"}                         # single tokens for the byte tokenizer
FIELDS = {"level", "game", "t", "type", "a_star", "K", "commands", "setting", "history_mode", "prompt_format",
          "max_reasoning_tokens", "k", "seed", "prompt_tokens", "reasoning", "reasoning_tokens", "reasoning_end",
          "q_star", "logq_star", "rank_star", "top", "q_reprocess", "action", "action_is_star",
          "action_is_reprocess", "notices_done", "plans_process"}


@pytest.fixture(scope="module")
def game(tmp_path_factory):
    pytest.importorskip("textworld")
    if shutil.which("tw-make") is None:
        pytest.skip("tw-make not available")
    path = tmp_path_factory.mktemp("tw7") / "cook.z8"
    proc = subprocess.run(["tw-make", "tw-cooking", "--recipe", "1", "--take", "1", "--cut", "--split", "valid",
                           "--seed", "3", "--output", str(path), "-f"], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not path.with_suffix(".json").is_file():
        pytest.skip(f"tw-make failed: {proc.stderr[-300:]}")
    return path


def test_settings_and_seeds():
    import diagnose_cot as dc

    got = {n: (c.history_mode, c.prompt_format, c.max_reasoning_tokens, c.history_window, c.forecast)
           for n, c in ((n, dc.setting_config(n)) for n in dc.SETTINGS)}
    assert got == {"full_plain": ("full", "plain", 320, 8, False), "full_chat": ("full", "chat_nothink", 128, 8, False),
                   "feedback_plain": ("feedback", "plain", 320, 8, False),
                   "feedback_chat": ("feedback", "chat_nothink", 128, 8, False)}
    seeds = {dc.sample_seed(g, t, h, f, k) for g in ("a", "b") for t in (2, 3) for h in ("full", "feedback")
             for f in ("plain", "chat_nothink") for k in range(5)}
    assert len(seeds) == 80 and all(0 <= s < 2 ** 63 for s in seeds)
    assert dc.sample_seed("a", 2, "full", "plain", 0) == 2629554514891596493      # stable across processes


def test_keyword_tags():
    import diagnose_cot as dc

    def t(text):
        d = dc.tags(text)
        return d["notices_done"], d["plans_process"]

    assert t("The red onion is already fried, so I prepare the meal.") == (True, False)
    assert t("The carrots are chopped and have already been roasted.") == (True, False)
    assert t("The onion is fried. Still, I should fry it as the recipe says.") == (True, True)
    assert t("The recipe says fry the red onion. I need to cook the red onion with the stove.") == (False, True)
    assert t("Next, slice the pepper.") == (False, True) and t("Let's dice it") == (False, True)
    assert t("I should read the cookbook. I have a fried red onion.") == (False, False)   # "cookbook" is not "cook"
    assert t("I must take the knife. Then we cook.") == (False, False)                    # not the same sentence
    assert dc.is_reprocess("cook red onion with stove") and dc.is_reprocess("slice carrot with knife")
    assert not dc.is_reprocess("prepare meal") and not dc.is_reprocess("take cookbook from counter")


def test_run_on_one_game(game):
    import diagnose_cot as dc

    model, tokenizer = tiny_model(seed=0), ByteTokenizer()
    out = io.StringIO()
    summary = dc.run(model, tokenizer, {"L": game.parent}, out, samples=2, cap=6, batch_states=2, **YN)
    rows = [json.loads(line) for line in out.getvalue().splitlines()]
    found = list(dc.states(game.parent))
    assert [s[5] for s in found] == ["prepare meal", "eat meal"]
    assert len(rows) == 2 * 4 * 2 and all(v["rows"] == 4 for v in summary.values())
    by = {(r["setting"], r["t"], r["k"]): r for r in rows}
    assert len(by) == len(rows)
    for r in rows:
        assert set(r) == FIELDS
        assert r["type"] == r["a_star"].split()[0] and r["type"] in ("prepare", "eat")
        assert r["K"] == len(r["commands"]) and r["a_star"] in r["commands"] and r["action"] in r["commands"]
        assert 1 <= r["reasoning_tokens"] <= 6 and r["reasoning_end"] in ("answer", "eos", "cap")
        assert r["max_reasoning_tokens"] == 6 and r["seed"] == dc.sample_seed(
            r["game"], r["t"], r["history_mode"], r["prompt_format"], r["k"])
        assert 0 < r["q_star"] <= 1 and r["q_star"] == pytest.approx(torch.tensor(r["logq_star"]).exp().item())
        assert [x["p"] for x in r["top"]] == sorted((x["p"] for x in r["top"]), reverse=True) and len(r["top"]) == 5
        assert (r["rank_star"] == 1) == (r["top"][0]["command"] == r["a_star"])
        assert 0 <= r["q_reprocess"] <= 1 and r["action_is_star"] == (r["action"] == r["a_star"])
        assert r["action_is_reprocess"] == dc.is_reprocess(r["action"])
        assert {"notices_done": r["notices_done"], "plans_process": r["plans_process"]} == dc.tags(r["reasoning"])
    # the feedback prompt is the shorter one; a state's prompt does not depend on k
    last = found[-1][1]
    assert by["feedback_plain", last, 0]["prompt_tokens"] < by["full_plain", last, 0]["prompt_tokens"]
    assert by["full_chat", last, 0]["prompt_tokens"] == by["full_chat", last, 1]["prompt_tokens"]
    assert len({r["reasoning"] for r in rows}) > 4                             # different seeds, different texts

    # Every row is what LMPolicy.act gives for that state with that seed (serial sampling, no batch).
    checked = 0
    for name in ("full_plain", "feedback_chat"):
        policy = LMPolicy(model, tokenizer, dc.setting_config(name, 6, **YN))
        for gname, t, objective, history, obs, a_star in found:
            for k in range(2):
                seed = dc.sample_seed(gname, t, policy.cfg.history_mode, policy.cfg.prompt_format, k)
                d = policy.act(objective, history, obs, torch.Generator().manual_seed(seed))
                r = by[name, t, k]
                assert tokenizer.decode(d.reasoning_ids) == r["reasoning"] and d.reasoning_end == r["reasoning_end"]
                assert d.command == r["action"] and len(d.prompt_ids) == r["prompt_tokens"]
                top = {x["command"]: x["p"] for x in r["top"]}
                if d.command in top:                                   # the action's log q is the last behaviour logp
                    assert d.behavior_logps[-1] == pytest.approx(torch.tensor(top[d.command]).log().item(), abs=1e-4)
                    checked += 1
                assert policy.encode(build_prompt(objective, history, obs, policy.cfg)) == d.prompt_ids
    assert checked > 0

    # a different batch size gives the same rows
    out2 = io.StringIO()
    dc.run(model, tokenizer, {"L": game.parent}, out2, samples=2, cap=6, batch_states=1, settings=("full_plain",), **YN)
    for line in out2.getvalue().splitlines():
        r = json.loads(line)
        ref = by["full_plain", r["t"], r["k"]]
        assert r["reasoning"] == ref["reasoning"] and r["action"] == ref["action"]
        assert r["q_star"] == pytest.approx(ref["q_star"], rel=1e-4)


def _row(setting, game, t, k, level="L0", kind="prepare", q=0.5, rank=1, notices=False, plans=False, re=0.1):
    return {"level": level, "game": game, "t": t, "type": kind, "a_star": f"{kind} meal", "K": 9, "setting": setting,
            "k": k, "reasoning": f"text {setting} {game} {k}", "reasoning_tokens": 10 + k, "reasoning_end": "answer",
            "q_star": q, "rank_star": rank, "top": [{"command": f"{kind} meal", "p": q}, {"command": "cook x with stove", "p": re}],
            "q_reprocess": re, "action": f"{kind} meal" if rank == 1 else "cook x with stove", "action_is_star": rank == 1,
            "action_is_reprocess": rank != 1, "notices_done": notices, "plans_process": plans}


def test_analyze_cot(tmp_path, monkeypatch):
    import analyze_cot as ac

    rows = []
    for s in ac.SETTINGS:
        good = s.startswith("full")
        for g in range(6):
            for k in range(5):
                rows.append(_row(s, f"g{g}.z8", 2, k, q=0.9 if good else 0.1, rank=1 if good else 2,
                                 notices=good, plans=not good, re=0.05 if good else 0.8))
        rows.append(_row(s, "g0.z8", 3, 0, level="L1", kind="eat", q=0.4))
    cot = tmp_path / "cot.jsonl"
    cot.write_text("".join(json.dumps(r) + "\n" for r in rows))
    teacher = tmp_path / "teacher.jsonl"
    top = [{"command": "prepare meal", "p": 0.25}, {"command": "cook x with stove", "p": 0.5},
           {"command": "chop x with knife", "p": 0.125}]
    teacher.write_text("".join(json.dumps(r) + "\n" for r in [
        {"model": "m", "level": "L0", "game": "g0.z8", "t": 2, "type": "prepare", "variant": "V1", "logq_star": -1.386294,
         "rank_star": 2, "top": top},
        {"model": "m", "level": "L0", "game": "g0.z8", "t": 0, "type": "examine", "variant": "V1", "logq_star": 0.0,
         "rank_star": 1, "top": top},                                      # not a state of the CoT run
        {"model": "other", "level": "L0", "game": "g0.z8", "t": 2, "type": "prepare", "variant": "V1", "logq_star": 0.0,
         "rank_star": 1, "top": top},
        {"model": "m", "level": "L0", "game": "g0.z8", "end": {"won": True, "problem": None}}]))
    args = ["--cot", str(cot), "--teacher", str(teacher), "--teacher-model", "m", "--json", str(tmp_path / "o.json"),
            "--reading", str(tmp_path / "reading.md"), "--texts", str(tmp_path / "texts.md")]
    monkeypatch.setattr(ac, "EXAMPLE", ("g0.z8", 2))                                              # the report's example state is not in this data
    result = ac.main(args)
    cell = {(r["level"], r["type"], r["setting"]): r for r in result["master"]}
    assert cell["L0", "prepare", "full_plain"]["n"] == 30 and cell["L0", "prepare", "full_plain"]["states"] == 6
    assert cell["L0", "prepare", "full_plain"]["mean_q_star"] == pytest.approx(0.9)
    assert cell["L0", "prepare", "feedback_chat"]["top1"] == 0 and cell["L0", "prepare", "feedback_chat"]["mean_q_reprocess"] == pytest.approx(0.8)
    assert cell["L0", "prepare", "full_chat"]["ends"]["answer"] == 1 and cell["L1", "eat", "full_chat"]["n"] == 1
    none = cell["L0", "prepare", "full_none"]
    assert none["n"] == 1 and none["mean_q_star"] == pytest.approx(0.25, abs=1e-5) and none["top1"] == 0
    assert none["mean_q_reprocess_top5"] == pytest.approx(0.625)
    cond = {(r["scope"], r["setting"], r["notices_done"]): r for r in result["by_notices_done"]}
    assert cond["L0", "full_plain", True]["n"] == 30 and cond["L0", "full_plain", False]["n"] == 0
    assert cond["all", "full_plain", False]["n"] == 1 and cond["L0", "full_plain", False]["mean_q_star"] is None
    spread = {(r["level"], r["type"], r["setting"]): r["states_by_top1_count"] for r in result["per_state"]}
    assert spread["L0", "prepare", "full_plain"] == {"5": 6} and spread["L0", "prepare", "feedback_plain"] == {"0": 6}
    # the reading sample: 20 L0 prepare reasonings per setting, the same ones on every call
    sample = result["reading_sample"]
    assert len(sample) == 80 == len(set(sample)) and all("|g" in k and k.split("|")[2] == "2" for k in sample)
    assert [sum(k.startswith(s + "|") for k in sample) for s in ac.SETTINGS] == [20] * 4
    assert ac.main(args[:2])["reading_sample"] == sample
    assert (tmp_path / "reading.md").read_text().count("#### ") == 80
    text = (tmp_path / "texts.md").read_text()
    assert text.count("**") // 2 == 20 + 2 * 4 * 2 and "text feedback_chat g0.z8 4" in text
    # labels: the classes of the reader against the keyword tags
    labels = {k: ("A" if k.startswith("full") else "B") for k in sample}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    reading = ac.main(args + ["--labels", str(tmp_path / "labels.json")])["reading"]
    assert reading["counts"]["full_plain"] == {"A": 20} and reading["counts"]["feedback_chat"] == {"B": 20}
    assert reading["notices_done_vs_AC"] == {"tp": 40, "fp": 0, "fn": 0, "tn": 40}
    assert reading["plans_process_vs_B"] == {"tp": 40, "fp": 0, "fn": 0, "tn": 40}
    assert reading["tags_by_class"]["B"]["mean_q_reprocess"] == pytest.approx(0.8)
    (tmp_path / "labels.json").write_text(json.dumps(dict(list(labels.items())[:5])))
    with pytest.raises(SystemExit):
        ac.main(args + ["--labels", str(tmp_path / "labels.json")])
