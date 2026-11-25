"""E0009: scripts/check_games.py maps recipe directions to walkthrough commands and plays the walkthrough."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RECIPE = ("\nRecipe #1\n---------\nGather all following ingredients and follow the directions.\n\nIngredients:\n"
          "red potato\n\nDirections:\ndice the red potato\n  roast the red potato\n  prepare meal\n")
WALK = ["inventory", "examine cookbook", "take red potato from counter", "cook red potato with oven",
        "take knife from counter", "dice red potato with knife", "drop knife", "prepare meal", "eat meal"]


def metadata(walk=WALK, ingredients=(("red potato", "roasted", "diced"),)):
    return {"recipe": RECIPE, "walkthrough": list(walk), "ingredients": [list(i) for i in ingredients],
            "entities": ["kitchen", "stove", "oven"], "settings": {"seed": 1, "recipe_seed": 2}}


def test_directions_and_commands():
    import check_games as cg

    assert cg.directions(RECIPE) == ["dice the red potato", "roast the red potato", "prepare meal"]
    assert cg.needed_command("prepare meal") is None
    assert cg.needed_command("fry the red hot pepper") == ("fry", "red hot pepper", "cook red hot pepper with stove")
    assert cg.needed_command("grill the pork chop")[2] == "cook pork chop with BBQ"
    assert cg.needed_command("slice the block of cheese")[2] == "slice block of cheese with knife"
    with pytest.raises(ValueError):
        cg.needed_command("boil the egg")


def test_check_recipe_finds_missing_steps_and_mismatches():
    import check_games as cg

    good = cg.check_recipe(metadata())
    assert good["missing"] == [] and good["mismatch"] == [] and good["verbs"] == ["dice", "roast"]
    # the trap of the old levels: the recipe asks for a step that the walkthrough never does
    no_cut = cg.check_recipe(metadata([c for c in WALK if not c.startswith("dice")]))
    assert no_cut["missing"] == ["dice the red potato -> dice red potato with knife"]
    # the wrong appliance does not count
    stove = cg.check_recipe(metadata([c.replace("oven", "stove") for c in WALK]))
    assert stove["missing"] == ["roast the red potato -> cook red potato with oven"]
    other = cg.check_recipe(metadata(ingredients=(("red potato", "roasted", "uncut"),)))
    assert other["missing"] == [] and other["mismatch"]


def test_summary_and_exit_status(tmp_path, capsys):
    import check_games as cg

    for name, walk in (("a.z8", WALK), ("b.z8", [c for c in WALK if not c.startswith("cook")])):
        (tmp_path / name).write_bytes(b"")
        (tmp_path / name).with_suffix(".json").write_text(json.dumps({"metadata": metadata(walk)}))
    out = tmp_path / "out.json"
    assert cg.main(["--games", f"S={tmp_path}", "--no-play", "--out", str(out)]) == 1
    s = json.loads(out.read_text())["S"]["summary"]
    assert s["games"] == 2 and s["fail1"] == ["b.z8"] and s["fail2"] == []
    assert s["verbs"] == {"fry": 0, "roast": 2, "grill": 0, "chop": 0, "slice": 0, "dice": 2}
    assert s["walkthrough_len"] == {"8": 1, "9": 1} and s["leaves_room"] == 0 and s["has_bbq"] == 0
    assert "FAIL b.z8" in capsys.readouterr().out
    (tmp_path / "b.z8").unlink()
    assert cg.main(["--games", f"S={tmp_path}", "--no-play"]) == 0


def test_real_game_passes(tmp_path):
    pytest.importorskip("textworld")
    if shutil.which("tw-make") is None:
        pytest.skip("tw-make not available")
    import check_games as cg

    path = tmp_path / "cook.z8"
    proc = subprocess.run(["tw-make", "tw-cooking", "--recipe", "1", "--take", "1", "--go", "1", "--cook", "--cut",
                           "--split", "valid", "--seed", "3", "--output", str(path), "-f"],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not path.with_suffix(".json").is_file():
        pytest.skip(f"tw-make failed: {proc.stderr[-300:]}")
    row = cg.check_game(path)
    assert row["ok1"] and row["ok2"] and row["won"] and row["verbs"]
    assert row["agent_steps"] == row["walkthrough_len"] - len(row["skipped"])
