"""T0007: combine the final evaluations (scripts/final_eval.py) of several training seeds of one method.

    python scripts/aggregate_final.py s17=runs/E0010c-s17/final_test s23=runs/E0010c-s23/final_test \
        s29=runs/E0010c-s29/final_test --out reports/E0010/c_final.json

Each argument is NAME=<directory with summary.json> (or the summary.json itself; without NAME= the directory name
is used). For win_rate, mean_length and the forecast metrics it reports the value of every run, their mean and
their standard deviation over runs (sample standard deviation, n - 1 in the denominator; None for a single run).
`pooled` adds up wins and episodes over the runs and gives the Wilson interval of that total, which ignores that
the runs are different models. Standard library only."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_eval import CALIB_KEYS, FORECAST_KEYS, wilson  # noqa: E402

KEYS = ("win_rate", "mean_length", *FORECAST_KEYS, *CALIB_KEYS)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, None
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def aggregate(summaries: dict[str, dict]) -> dict:
    """{name: summary.json content} -> per-run numbers, mean and std per metric, pooled counts."""
    runs = {name: {"episodes": s["episodes"], "wins": s["wins"], "wilson95": s.get("wilson95"),
                   "seeds": s.get("seeds"), **{k: s.get(k) for k in KEYS}} for name, s in summaries.items()}
    stats = {}
    for k in KEYS:
        values = [r[k] for r in runs.values()]
        m, sd = mean_std(values)
        stats[k] = {"mean": m, "std": sd, "n": sum(v is not None for v in values)}
    wins, n = sum(r["wins"] for r in runs.values()), sum(r["episodes"] for r in runs.values())
    actions = {}
    for s in summaries.values():
        for kind, count in (s.get("action_types") or {}).items():
            actions[kind] = actions.get(kind, 0) + count
    return {"runs": runs, "stats": stats,
            "pooled": {"wins": wins, "episodes": n, "win_rate": wins / n, "wilson95": wilson(wins, n)},
            "action_types": dict(sorted(actions.items(), key=lambda kv: -kv[1]))}


def _f(x, digits=4):
    return "–" if x is None else f"{x:.{digits}f}"


def markdown(result: dict) -> str:
    keys = [k for k in KEYS if k not in CALIB_KEYS or result["stats"][k]["n"]]   # CALIB_KEYS only when present
    head = ["run", "wins", *keys, "Wilson 95%"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in result["runs"].items():
        ci = r.get("wilson95")
        lines.append("| " + " | ".join([name, f'{r["wins"]}/{r["episodes"]}', *[_f(r[k]) for k in keys],
                                        "–" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"]) + " |")
    pooled = result["pooled"]
    lines.append("| mean | " + f'{pooled["wins"]}/{pooled["episodes"]} | '
                 + " | ".join(_f(result["stats"][k]["mean"]) for k in keys)
                 + f' | [{pooled["wilson95"][0]:.3f}, {pooled["wilson95"][1]:.3f}] (pooled) |')
    lines.append("| std | | " + " | ".join(_f(result["stats"][k]["std"]) for k in keys) + " | |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="NAME=<final_eval directory or summary.json>")
    ap.add_argument("--out", help="write the result as JSON")
    args = ap.parse_args(argv)
    summaries = {}
    for item in args.runs:
        name, _, path = item.rpartition("=")
        path = Path(path)
        if path.is_dir():
            path = path / "summary.json"
        name = name or path.parent.name
        if name in summaries:
            ap.error(f"duplicate name {name!r}")
        summaries[name] = json.loads(path.read_text())
    result = aggregate(summaries)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
