"""E0011: scripts/analyze_ladder.py (failure types and back-and-forth walking), on hand-made episodes."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_ladder  # noqa: E402


def episode(z, commands):
    return {"z": z, "length": len(commands), "steps": [{"t": t, "command": c} for t, c in enumerate(commands)]}


def test_go_stats_counts_returns_and_loops():
    g = analyze_ladder.go_stats(["go east", "go west", "go east", "go west", "take apple", "go north", "go south"])
    assert g == {"go": 6, "returns": 4, "loops": 1, "loop_commands": 4}
    # a walk without turning back, and `go` commands separated by another command
    assert analyze_ladder.go_stats(["go east", "go east", "go north"])["returns"] == 0
    assert analyze_ladder.go_stats(["go east", "open door", "go west"]) == {"go": 2, "returns": 0, "loops": 0,
                                                                            "loop_commands": 0}
    # exactly three alternating commands are a loop, two are only a return
    assert analyze_ladder.go_stats(["go north", "go south", "go north"])["loops"] == 1
    assert analyze_ladder.go_stats(["go north", "go south"])["loops"] == 0


def test_analyze_splits_losses_by_H_and_last_command():
    records = [episode(1, ["take apple", "eat meal"]),
               episode(0, ["take apple", "cook apple with stove"]),
               episode(0, ["go east", "go west", "go east", "examine cookbook"]),
               episode(0, ["take apple", "eat apple"])]
    r = analyze_ladder.analyze(records, max_actions=4)
    assert (r["episodes"], r["wins"], r["losses"], r["lost_at_H"], r["lost_early"]) == (4, 1, 3, 1, 2)
    assert r["last_command"] == {"cook": 1, "eat": 1}
    assert (r["returns"], r["loops"], r["loop_commands"], r["episodes_with_loop"]) == (2, 1, 3, 1)
    assert r["go_share"] == 3 / 10


def test_main_writes_json_and_table(tmp_path, capsys):
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in [episode(0, ["go east", "go west"]), episode(1, ["eat meal"])]))
    out = tmp_path / "out.json"
    assert analyze_ladder.main([f"x={path}", "--max-actions", "2", "--json", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert saved["max_actions"] == 2 and saved["runs"]["x"]["lost_at_H"] == 1
    assert "| x | 1 / 2 |" in capsys.readouterr().out
