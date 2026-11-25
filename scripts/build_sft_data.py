"""T0007: expert states for supervised fine-tuning, from the walkthroughs of a directory of games.

    python scripts/build_sft_data.py --games data/games_C1/train --out data/sft/C1_train.jsonl

Every game is played along its walkthrough exactly as scripts/diagnose_history.py does (`walk`: commands the agent
is never offered, such as `inventory`, are skipped). Each state the expert passes becomes one line with the raw
material of a prompt, not the prompt itself, so that `jevtw.sft` can build it under any PolicyConfig:
  game, t, objective, history (per past step: the command and every field of the Observation that followed),
  obs (the current Observation), a_star (index of the walkthrough command among obs.commands), command.
A game whose walkthrough cannot be followed to a win contributes no states and is listed in the statistics.
The statistics (games, states, states per game, action types = first word of a*) are printed and written to
<out>.stats.json. An existing --out is not overwritten. CPU only."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_history import GAME_SUFFIXES, walk  # noqa: E402
from jevtw.sft import state_row  # noqa: E402


def game_rows(path: Path) -> tuple[list[dict], dict]:
    """(the states of one walkthrough, the final dict of `walk`)."""
    rows, end = [], None
    for item in walk(path):
        if isinstance(item, dict):
            end = item
        else:
            t, objective, history, obs, a_star = item
            rows.append(state_row(path.name, t, objective, history, obs, a_star))
    return rows, end


def build(directory, limit=None) -> tuple[list[dict], dict]:
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix in GAME_SUFFIXES)[:limit]
    if not paths:
        raise FileNotFoundError(f"no {GAME_SUFFIXES} games in {directory}")
    rows, dropped, per_game = [], [], Counter()
    for path in paths:
        mine, end = game_rows(path)
        if end["problem"] is not None or not end["won"]:
            dropped.append({"game": path.name, "problem": end["problem"], "won": end["won"]})
            continue
        rows += mine
        per_game[len(mine)] += 1
    types = Counter(r["command"].split()[0] for r in rows)
    stats = {"directory": str(directory), "games": len(paths), "games_used": len(paths) - len(dropped),
             "dropped": dropped, "states": len(rows),
             "states_per_game": {str(k): v for k, v in sorted(per_game.items())},
             "action_types": dict(types.most_common()),
             "mean_options": sum(len(r["obs"]["commands"]) for r in rows) / len(rows) if rows else None}
    return rows, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", required=True, help="directory of .z8 / .ulx games with their .json")
    ap.add_argument("--out", required=True, help="jsonl to write; the statistics go to <out>.stats.json")
    ap.add_argument("--limit", type=int, help="only the first N games")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        ap.error(f"{out} already exists; choose a new file")
    rows, stats = build(args.games, args.limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))
    print(f"wrote {out} ({len(rows)} states)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
