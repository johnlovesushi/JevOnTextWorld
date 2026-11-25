"""E0011 part B: how the episodes of an evaluation are lost, and how often the agent walks back and forth.

    python scripts/analyze_ladder.py --max-actions 38 base=runs/E0011-C2-base/forecasts/batch0000.jsonl \
        sft=runs/E0011-C2-sft/forecasts/batch0000.jsonl --json reports/E0011/failures_C2.json

Each argument is NAME=<episodes jsonl in the --save-forecasts format> (per episode: z, length, steps with the
chosen command). Per file it reports:
  episodes, wins, losses
  lost_at_H        lost episodes with length >= --max-actions (the episode ran out of steps)
  lost_early       lost episodes that ended before (the game declared the loss)
  last_command     first word of the last command of the lost_early episodes -> count
  go_share         share of all commands that are `go ...`
  returns          pairs of consecutive commands `go d`, `go opposite(d)`: the agent enters a room and leaves it
                   at once through the door it came in by
  loops            maximal runs of at least 3 consecutive `go` commands that alternate between a direction and
                   its opposite (A -> B -> A -> B ...); `loop_commands` is the number of commands inside them and
                   `episodes_with_loop` the number of episodes with at least one
Standard library only."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def direction(command: str) -> str | None:
    """The direction of a `go <direction>` command, None for any other command."""
    words = command.split()
    return words[1] if len(words) == 2 and words[0] == "go" and words[1] in OPPOSITE else None


def go_stats(commands: list[str]) -> dict:
    """returns, loops and loop_commands of one episode (see the module docstring)."""
    dirs = [direction(c) for c in commands]
    returns = sum(a is not None and b == OPPOSITE.get(a) for a, b in zip(dirs, dirs[1:]))
    loops, loop_commands, run = 0, 0, 0
    for i, d in enumerate(dirs):
        if d is not None and i > 0 and dirs[i - 1] == OPPOSITE[d]:
            run += 1                       # run = number of consecutive alternating `go` commands so far
        else:
            if run >= 3:
                loops, loop_commands = loops + 1, loop_commands + run
            run = 1 if d is not None else 0
    if run >= 3:
        loops, loop_commands = loops + 1, loop_commands + run
    return {"go": sum(d is not None for d in dirs), "returns": returns, "loops": loops,
            "loop_commands": loop_commands}


def analyze(records: list[dict], max_actions: int) -> dict:
    lost = [r for r in records if not r["z"]]
    at_h = [r for r in lost if r["length"] >= max_actions]
    early = [r for r in lost if r["length"] < max_actions]
    last = Counter(r["steps"][-1]["command"].split()[0] for r in early if r["steps"])
    per_episode = [go_stats([s["command"] for s in r["steps"]]) for r in records]
    commands = sum(len(r["steps"]) for r in records)
    go = sum(g["go"] for g in per_episode)
    return {"episodes": len(records), "wins": len(records) - len(lost), "losses": len(lost),
            "lost_at_H": len(at_h), "lost_early": len(early), "last_command": dict(last.most_common()),
            "commands": commands, "go": go, "go_share": go / commands if commands else None,
            "returns": sum(g["returns"] for g in per_episode), "loops": sum(g["loops"] for g in per_episode),
            "loop_commands": sum(g["loop_commands"] for g in per_episode),
            "episodes_with_loop": sum(g["loops"] > 0 for g in per_episode)}


def markdown(results: dict[str, dict]) -> str:
    head = ["run", "wins", "lost", "lost at H", "lost early", "last command of the early losses", "go share",
            "returns", "loops", "commands in loops", "episodes with a loop"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in results.items():
        last = ", ".join(f"{k} {v}" for k, v in r["last_command"].items()) or "–"
        share = "–" if r["go_share"] is None else f'{r["go_share"]:.3f}'
        lines.append("| " + " | ".join(str(x) for x in [
            name, f'{r["wins"]} / {r["episodes"]}', r["losses"], r["lost_at_H"], r["lost_early"], last, share,
            r["returns"], r["loops"], r["loop_commands"], r["episodes_with_loop"]]) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", help="NAME=<episodes jsonl>")
    ap.add_argument("--max-actions", type=int, required=True, help="H of the evaluation")
    ap.add_argument("--json", help="write the numbers here")
    args = ap.parse_args(argv)
    results = {}
    for item in args.episodes:
        name, sep, path = item.partition("=")
        if not sep:
            ap.error(f"expected NAME=path, got {item!r}")
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        results[name] = analyze(records, args.max_actions)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"max_actions": args.max_actions, "runs": results}, indent=1) + "\n")
    print(markdown(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
