"""Pick random reasonings from a file written by `jevtw.train --save-reasonings` (E0004).

    python scripts/sample_reasonings.py runs/E0004b/reasonings/batch0000.jsonl [--seed 0] [--chars 400]

Prints one reasoning from a won episode and one from a lost episode (when both exist), each cut to --chars
characters, as markdown. The choice depends only on the file and --seed.
"""
import argparse
import json
import random
from pathlib import Path


def pick(records: list[dict], seed: int) -> list[dict]:
    """One record with z = 1 and one with z = 0, drawn uniformly among the decisions that have reasoning tokens."""
    rng = random.Random(seed)
    chosen = []
    for z in (1, 0):
        pool = [r for r in records if r["z"] == z and r["reasoning_tokens"] > 0]
        if pool:
            chosen.append(rng.choice(pool))
    return chosen


def render(record: dict, chars: int) -> str:
    text = record["reasoning"]
    if len(text) > chars:
        text = text[:chars] + " […]"
    text = text.replace("```", "'''")
    outcome = "won" if record["z"] else "lost"
    return (f"**{outcome} episode** — {record['game']}, sample seed {record['sample_seed']}, step {record['t']} of "
            f"{record['length']}, {record['reasoning_tokens']} tokens, ended by `{record['reasoning_end']}`, "
            f"command `{record['command']}`\n\n```\n{text}\n```")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chars", type=int, default=400)
    args = ap.parse_args(argv)
    records = [json.loads(line) for line in Path(args.file).read_text().splitlines()]
    chosen = pick(records, args.seed)
    if not chosen:
        print("(no decision with reasoning tokens)")
    print("\n\n".join(render(r, args.chars) for r in chosen))


if __name__ == "__main__":
    main()
