"""E0009 part A: check generated cooking games without a model.

    python scripts/check_games.py --games C0/valid=data/games_C0/valid C0/test=data/games_C0/test \
        --out runs/E0009/check_games.json

For every game (its .z8 / .ulx and the .json next to it):
  1. every line of the recipe's "Directions:" except `prepare meal` has its processing command in the
     walkthrough (metadata.walkthrough): "fry the X" <-> `cook X with stove`, "roast the X" <-> `cook X with oven`,
     "grill the X" <-> `cook X with BBQ`, "chop / slice / dice the X" <-> `chop / slice / dice X with knife`.
     The directions are also compared with metadata.ingredients ([name, cooked state, cut state]), which is the
     generator's own record of what the recipe needs;
  2. the walkthrough is played in TextWorldEnv (scripts/diagnose_history.py, `walk`) and has to end in a win;
  3. statistics: walkthrough length, number of directions per verb, games whose walkthrough leaves the first room
     (`go ...`) and games with a BBQ.
The exit status is 1 if any game fails check 1 or 2.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

COOK_TOOL = {"fry": "stove", "roast": "oven", "grill": "BBQ"}
CUT_VERBS = ("chop", "slice", "dice")
VERBS = (*COOK_TOOL, *CUT_VERBS)
# metadata.ingredients states -> the verb of the direction
STATE_VERB = {"fried": "fry", "roasted": "roast", "grilled": "grill", "chopped": "chop", "sliced": "slice",
              "diced": "dice"}
NO_STATE = ("raw", "uncut", None, "")
GAME_SUFFIXES = (".z8", ".ulx")


def directions(recipe: str) -> list[str]:
    """The lines under "Directions:" of a recipe text."""
    if "Directions:" not in recipe:
        raise ValueError("recipe has no Directions")
    return [line.strip() for line in recipe.split("Directions:", 1)[1].splitlines() if line.strip()]


def needed_command(direction: str):
    """(verb, ingredient, walkthrough command) of one direction; None for `prepare meal`."""
    if direction == "prepare meal":
        return None
    verb, _, rest = direction.partition(" ")
    if verb not in VERBS or not rest.startswith("the "):
        raise ValueError(f"unknown direction: {direction!r}")
    item = rest[len("the "):]
    if verb in COOK_TOOL:
        return verb, item, f"cook {item} with {COOK_TOOL[verb]}"
    return verb, item, f"{verb} {item} with knife"


def check_recipe(metadata: dict) -> dict:
    """Check 1 on a game's metadata. `missing`: directions whose command is not in the walkthrough; `mismatch`:
    differences between the directions and metadata.ingredients."""
    steps = [needed_command(d) for d in directions(metadata["recipe"])]
    steps = [s for s in steps if s is not None]
    walk_cmds = list(metadata["walkthrough"])
    missing = [f"{verb} the {item} -> {cmd}" for verb, item, cmd in steps if cmd not in walk_cmds]
    from_directions = sorted((item, verb) for verb, item, _ in steps)
    from_metadata = sorted((name, STATE_VERB[state]) for name, *states in metadata.get("ingredients", [])
                           for state in states if state not in NO_STATE)
    mismatch = []
    if "ingredients" in metadata and from_directions != from_metadata:
        mismatch = [f"directions {from_directions} != metadata.ingredients {from_metadata}"]
    return {"verbs": [verb for verb, _, _ in steps], "commands": [cmd for _, _, cmd in steps],
            "missing": missing, "mismatch": mismatch, "has_prepare": "prepare meal" in directions(metadata["recipe"])}


def check_game(path: Path, play: bool = True) -> dict:
    metadata = json.loads(path.with_suffix(".json").read_text())["metadata"]
    row = {"game": path.name, **check_recipe(metadata)}
    walk_cmds = list(metadata["walkthrough"])
    row["walkthrough"] = walk_cmds
    row["walkthrough_len"] = len(walk_cmds)
    row["leaves_room"] = any(c.startswith("go ") for c in walk_cmds)
    row["has_bbq"] = "BBQ" in metadata.get("entities", [])
    row["seeds"] = [metadata["settings"].get("seed"), metadata["settings"].get("recipe_seed")]
    if play:
        from diagnose_history import walk

        *states, end = list(walk(path))
        row.update(won=bool(end["won"]), problem=end["problem"], agent_steps=end["steps"], skipped=end["skipped"])
    row["ok1"] = not row["missing"] and not row["mismatch"]
    row["ok2"] = (row["won"] and row["problem"] is None) if play else None
    return row


def summarize(rows: list[dict]) -> dict:
    verbs = Counter(v for r in rows for v in r["verbs"])
    out = {"games": len(rows), "fail1": [r["game"] for r in rows if not r["ok1"]],
           "fail2": [r["game"] for r in rows if r["ok2"] is False],
           "walkthrough_len": dict(sorted(Counter(r["walkthrough_len"] for r in rows).items())),
           "verbs": {v: verbs.get(v, 0) for v in VERBS},
           "directions_per_game": dict(sorted(Counter(len(r["verbs"]) for r in rows).items())),
           "leaves_room": sum(r["leaves_room"] for r in rows), "has_bbq": sum(r["has_bbq"] for r in rows)}
    if rows and "agent_steps" in rows[0]:
        out["agent_steps"] = dict(sorted(Counter(r["agent_steps"] for r in rows).items()))
    return out


def run(sets: dict, play: bool = True) -> dict:
    result = {}
    for name, directory in sets.items():
        paths = sorted(p for p in Path(directory).iterdir() if p.suffix in GAME_SUFFIXES)
        rows = [check_game(p, play) for p in paths]
        result[name] = {"directory": str(directory), "summary": summarize(rows), "games": rows}
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", required=True, help="NAME=directory pairs")
    ap.add_argument("--out", help="write the full result (one row per game) as JSON")
    ap.add_argument("--no-play", action="store_true", help="skip check 2 (do not start TextWorld)")
    args = ap.parse_args(argv)
    if args.out and Path(args.out).exists():
        ap.error(f"{args.out} already exists; choose a new file")
    result = run(dict(pair.split("=", 1) for pair in args.games), play=not args.no_play)
    failed = False
    for name, block in result.items():
        s = block["summary"]
        print(f"{name}: {json.dumps(s)}")
        for r in block["games"]:
            if not r["ok1"] or r["ok2"] is False:
                failed = True
                print(f"  FAIL {r['game']}: missing={r['missing']} mismatch={r['mismatch']} "
                      f"won={r.get('won')} problem={r.get('problem')}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
