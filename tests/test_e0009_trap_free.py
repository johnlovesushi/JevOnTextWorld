"""E0009: scripts/analyze_trap_free.py classifies processing errors from the commands of an episode."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RECIPE = {"red potato": {"cook": "roast", "cut": "dice"}}


def test_classify():
    import analyze_trap_free as tf

    good = ["examine cookbook", "take red potato from counter", "cook red potato with oven", "take knife from table",
            "dice red potato with knife", "drop knife", "prepare meal", "eat meal"]
    assert tf.classify(good, RECIPE) == [[] for _ in good]
    assert tf.progress(good, RECIPE) == {"needs_cook": True, "needs_cut": True, "cook_done": True, "cut_done": True,
                                        "took_knife": True, "prepared": True}
    bad = ["cook red potato with stove",            # recipe says roast
           "cook red potato with oven",             # second cooking, even with the right appliance
           "slice red potato with knife",           # no knife yet, and the wrong cut
           "take knife from table", "put knife on counter",
           "dice red potato with knife",            # knife put down again; second cut
           "take knife from counter", "chop banana with knife", "cook banana with stove"]
    kinds = tf.classify(bad, RECIPE)
    assert kinds == [["wrong_cook"], ["recook"], ["cut_no_knife", "wrong_cut"], [], [], ["cut_no_knife", "recut"], [],
                     ["other_item"], ["other_item"]]
    assert tf.progress(bad, RECIPE) == {"needs_cook": True, "needs_cut": True, "cook_done": False, "cut_done": False,
                                       "took_knife": True, "prepared": False}
    # an ingredient the recipe does not cut: any cut is wrong
    assert tf.classify(["take knife from table", "chop pork chop with knife", "cook pork chop with BBQ"],
                       {"pork chop": {"cook": "grill", "cut": None}}) == [[], ["wrong_cut"], []]


def test_errors_from_files(tmp_path, capsys):
    import analyze_trap_free as tf

    games = tmp_path / "games"
    games.mkdir()
    (games / "g1.json").write_text(json.dumps({"metadata": {"ingredients": [["red potato", "roasted", "diced"]]}}))
    (games / "g2.json").write_text(json.dumps({"metadata": {"ingredients": [["carrot", "raw", "sliced"]]}}))
    assert tf.recipe_of(games / "g2.json") == {"carrot": {"cook": None, "cut": "slice"}}

    def ep(game, z, cs, length=None):
        return {"game": game, "env_seed": 0, "sample_seed": 17, "z": z, "length": length or len(cs),
                "steps": [{"t": t, "command": c, "s": 0.0, "p": 0.5, "s_options": [0.0]} for t, c in enumerate(cs)]}

    eps = [ep("g1.z8", 1, ["cook red potato with oven", "take knife from table", "dice red potato with knife",
                           "prepare meal", "eat meal"]),
           ep("g1.z8", 0, ["cook red potato with oven", "cook red potato with oven"]),
           ep("g2.z8", 0, ["take knife from table", "cook carrot with stove"]),
           ep("g2.z8", 0, ["examine cookbook"] * 30)]
    path = tmp_path / "batch0000.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in eps) + "\n")
    e = tf.errors(path, games)
    assert e["episodes"] == 4 and e["lost_before_limit"] == 2
    assert e["commands"]["recook"] == 1 and e["commands"]["wrong_cook"] == 1 and e["episodes_with"]["wrong_cut"] == 0
    assert dict(e["last_command_kind_when_lost"]) == {"recook": 1, "wrong_cook": 1}
    assert e["reached"]["needs_cook"] == 2 and e["reached"]["cook_done"] == 1      # the recooked potato does not count
    assert e["reached"]["needs_cut"] == 4 and e["reached"]["cut_done"] == 1 and e["reached"]["took_knife"] == 2
    out = tmp_path / "out.json"
    tf.main(["--games", str(games), "--episodes", f"x={path}", "--json", str(out)])
    assert json.loads(out.read_text())["x"]["behaviour"]["endings"] == {"win": 1, "lost": 2, "ran out of steps": 1}
    assert "Processing errors: x" in capsys.readouterr().out


def test_teacher_tables(tmp_path, capsys):
    import math

    import analyze_trap_free as tf

    def row(game, t, a_star, variant, q, rank):
        return {"model": "m", "level": "C0", "game": game, "t": t, "a_star": a_star, "type": a_star.split()[0], "K": 4,
                "variant": variant, "prompt_tokens": 100, "logq_star": math.log(q), "rank_star": rank}

    rows = [row("g1", 0, "take knife from table", "V1", 0.5, 1), row("g1", 1, "slice carrot with knife", "V1", 0.5, 1),
            row("g1", 2, "drop knife", "V1", 0.1, 3), row("g2", 0, "take carrot from fridge", "V1", 0.2, 2),
            row("g2", 1, "dice carrot with knife", "V1", 1.0, 1)]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows + [{"model": "m", "level": "C0", "game": "g1", "end": {}}]) + "\n")
    assert tf.action_type("chop x with knife") == "cut" and tf.action_type("take knife from table") == "take knife"
    t = tf.teacher(path)
    assert t["overall"][0]["n"] == 5 and t["overall"][0]["top1"] == 3 / 5
    types = {r["type"]: r for r in t["by_type"]}
    assert set(types) == {"take knife", "cut", "drop", "take"} and types["cut"]["n"] == 2 and types["cut"]["mean_q"] == 0.75
    p = t["path"][0]
    assert p["games"] == 2 and math.isclose(p["mean_path_prob"], (0.025 + 0.2) / 2)
    assert math.isclose(p["mean_path_prob_needed"], (0.25 + 0.2) / 2) and p["all_rank1"] == 0 and p["all_rank1_needed"] == 0.5
    tf.main(["--teacher", str(path)])
    assert "Whole walkthrough" in capsys.readouterr().out
