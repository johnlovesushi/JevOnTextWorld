"""Per-step table from a forecasts file written by `jevtw.train --save-forecasts` (E0005).

    python scripts/forecast_by_step.py runs/E0005d/forecasts/batch0000.jsonl [--steps 11] [--value s|p]

For t = 0 .. steps-1: number of won / lost episodes that reach step t and the mean of s (or p) at that step.
Prints a markdown table."""
import argparse
import json
from pathlib import Path


def by_step(records: list[dict], steps: int, value: str = "s") -> list[dict]:
    rows = []
    for t in range(steps):
        won = [r["steps"][t][value] for r in records if r["z"] == 1 and len(r["steps"]) > t]
        lost = [r["steps"][t][value] for r in records if r["z"] == 0 and len(r["steps"]) > t]
        rows.append({"t": t, "n_win": len(won), "mean_win": sum(won) / len(won) if won else None,
                     "n_lose": len(lost), "mean_lose": sum(lost) / len(lost) if lost else None})
    return rows


def markdown(rows: list[dict], value: str = "s") -> str:
    fmt = (lambda x: "–" if x is None else f"{x:.3f}")
    lines = [f"| t | n (won) | mean {value} (won) | n (lost) | mean {value} (lost) |", "|---|---|---|---|---|"]
    lines += [f"| {r['t']} | {r['n_win']} | {fmt(r['mean_win'])} | {r['n_lose']} | {fmt(r['mean_lose'])} |"
              for r in rows]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--value", choices=("s", "p"), default="s")
    args = ap.parse_args(argv)
    records = [json.loads(line) for line in Path(args.path).read_text().splitlines() if line.strip()]
    print(markdown(by_step(records, args.steps, args.value), args.value))


if __name__ == "__main__":
    main()
