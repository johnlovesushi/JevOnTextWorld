"""E0006: tables from scripts/diagnose_history.py output and from saved evaluation episodes.

    python scripts/analyze_history.py --teacher runs/E0006/teacher_forced.jsonl \
        --episodes full=runs/E0005a/forecasts/batch0000.jsonl feedback=runs/E0005h/forecasts/batch0000.jsonl \
        --json reports/E0006.json

Prints markdown tables and (with --json) writes all numbers. Needs neither a model nor a GPU.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path



def variant_key(name: str):
    return int(name[1:]) if name[1:].isdigit() else 0, name          # V2 before V10


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def stats(rows) -> dict:
    return {"n": len(rows), "mean_logq": mean(r["logq_star"] for r in rows),
            "top1": mean(r["rank_star"] == 1 for r in rows), "mean_rank": mean(r["rank_star"] for r in rows),
            "mean_K": mean(r["K"] for r in rows), "mean_prompt_tokens": mean(r["prompt_tokens"] for r in rows)}


def teacher_tables(path, optional=("drop",)) -> dict:
    """`optional`: action types of the walkthrough that are not needed to win (on L1 `prepare meal` is already
    admissible, and wins, at the `drop knife` step); "without_optional" leaves those states out."""
    rows, ends = [], []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        (ends if "end" in r else rows).append(r)
    groups, by_type, by_step = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        groups[r["model"], r["level"], r["variant"]].append(r)
        by_type[r["model"], r["level"], r["type"], r["variant"]].append(r)
        by_step[r["model"], r["level"], r["t"], r["variant"]].append(r)
    out = {"walks": {}, "overall": [], "by_type": [], "by_step": [], "after_t0": [], "without_optional": [],
           "optional_types": list(optional)}
    for e in ends:
        w = out["walks"].setdefault(f'{e["model"]}|{e["level"]}', {"games": 0, "won": 0, "problems": []})
        w["games"] += 1
        w["won"] += e["end"]["won"]
        if e["end"]["problem"]:
            w["problems"].append([e["game"], e["end"]["problem"]])
    for (model, level, variant), g in sorted(groups.items(), key=lambda kv: (*kv[0][:2], variant_key(kv[0][2]))):
        out["overall"].append({"model": model, "level": level, "variant": variant, **stats(g)})
        # t = 0 has no history, so every variant sees the same prompt there; the later steps carry the difference
        out["after_t0"].append({"model": model, "level": level, "variant": variant,
                                **stats([r for r in g if r["t"] > 0])})
        out["without_optional"].append({"model": model, "level": level, "variant": variant,
                                        **stats([r for r in g if r["type"] not in optional])})
    for (model, level, kind, variant), g in sorted(by_type.items()):
        out["by_type"].append({"model": model, "level": level, "type": kind, "variant": variant, **stats(g)})
    for (model, level, t, variant), g in sorted(by_step.items()):
        out["by_step"].append({"model": model, "level": level, "t": t, "variant": variant, **stats(g)})
    # states where V1 ranks a* first and V2 does not (candidates for the prompt comparison in the report)
    index = {(r["model"], r["level"], r["game"], r["t"], r["variant"]): r for r in rows}
    out["v1_hit_v2_miss"] = [
        {"model": m, "level": lv, "game": g, "t": t, "a_star": r["a_star"],
         "V1": {"logq_star": r["logq_star"], "rank_star": r["rank_star"], "top": r["top"]},
         "V2": {k: index[m, lv, g, t, "V2"][k] for k in ("logq_star", "rank_star", "top")}}
        for (m, lv, g, t, v), r in sorted(index.items())
        if v == "V1" and r["rank_star"] == 1 and index[m, lv, g, t, "V2"]["rank_star"] > 1]
    return out


def f(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def print_teacher(t: dict) -> None:
    for key, title in (("overall", "all states"), ("without_optional", f'without {t["optional_types"]} states')):
        print(f"\n## Teacher-forced: {title}\n")
        print("| model | level | variant | states | mean log q(a*) | top-1 | mean rank | mean K | prompt tokens |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in t[key]:
            print(f'| `{r["model"]}` | {r["level"]} | {r["variant"]} | {r["n"]} | {f(r["mean_logq"])} | '
                  f'{f(r["top1"])} | {f(r["mean_rank"], 2)} | {f(r["mean_K"], 1)} | {f(r["mean_prompt_tokens"], 0)} |')
    for key, title, col in (("by_type", "by action type", "type"), ("by_step", "by step", "t")):
        print(f"\n## Teacher-forced: top-1 rate {title} (mean log q in brackets)\n")
        cells = defaultdict(dict)
        VARIANTS = sorted({r["variant"] for r in t[key]}, key=variant_key)
        for r in t[key]:
            cells[r["model"], r["level"], r[col]][r["variant"]] = r
        print(f"| model | level | {col} | states | " + " | ".join(VARIANTS) + " |")
        print("|---|---|---|---|" + "---|" * len(VARIANTS))
        for (model, level, k), c in cells.items():
            print(f"| `{model}` | {level} | {k} | {c[VARIANTS[0]]['n']} | "
                  + " | ".join(f'{f(c[v]["top1"], 2)} ({f(c[v]["mean_logq"], 2)})' for v in VARIANTS) + " |")
    print(f'\nstates with V1 rank 1 and V2 rank > 1: {len(t["v1_hit_v2_miss"])}; by (model, level, a* type): '
          f'{dict(Counter((r["model"], r["level"], r["a_star"].split()[0]) for r in t["v1_hit_v2_miss"]))}')


def item_of(command: str):
    """("take", item) or ("release", item) or None, for take / drop / put / insert commands."""
    words = command.split()
    if words[0] == "take":
        rest = command[5:]
        return "take", rest.split(" from ")[0]
    if words[0] == "drop":
        return "release", command[5:]
    if words[0] in ("put", "insert"):
        rest = command[len(words[0]) + 1:]
        return "release", rest.split(" on ")[0].split(" into ")[0]
    return None


def longest_run(commands, target) -> int:
    best = run = 0
    for c in commands:
        run = run + 1 if c == target else 0
        best = max(best, run)
    return best


def behaviour(path, max_actions=30) -> dict:
    episodes = [json.loads(line) for line in Path(path).read_text().splitlines()]
    commands = Counter()
    endings = Counter()
    last_lost, lost_step, after_take = Counter(), Counter(), Counter()
    cookbook_runs, take_release, wasted = [], 0, []
    prepare = eat = prepare_eps = eat_eps = took_first = 0
    for e in episodes:
        cs = [s["command"] for s in e["steps"]]
        commands.update(cs)
        if e["z"]:
            endings["win"] += 1
        elif e["length"] < max_actions:
            endings["lost"] += 1
            last_lost[cs[-1]] += 1
            lost_step[len(cs)] += 1
        else:
            endings["ran out of steps"] += 1
        cookbook_runs.append(longest_run(cs, "examine cookbook"))
        held, pairs = set(), 0
        for c in cs:
            kind = item_of(c)
            if kind and kind[0] == "take":
                held.add(kind[1])
            elif kind and kind[1] in held:
                held.discard(kind[1])
                pairs += 1
        take_release += pairs > 0
        prepare += cs.count("prepare meal")
        eat += cs.count("eat meal")
        prepare_eps += "prepare meal" in cs
        eat_eps += "eat meal" in cs
        # the command chosen right after the first `take`: on L0 the walkthrough says `prepare meal` there
        takes = [i for i, c in enumerate(cs) if c.startswith("take ")]
        if takes and takes[0] + 1 < len(cs):
            took_first += 1
            after_take[cs[takes[0] + 1]] += 1
        wasted.append(sum(c == "examine cookbook" for c in cs) / len(cs))
    n = len(episodes)
    return {"file": str(path), "episodes": n, "endings": dict(endings), "mean_length": mean(e["length"] for e in episodes),
            "lost_at_step": dict(sorted(lost_step.items())), "last_command_when_lost": last_lost.most_common(),
            "top_commands": commands.most_common(10), "total_commands": sum(commands.values()),
            "examine_cookbook_share": commands["examine cookbook"] / sum(commands.values()),
            "episodes_with_3_cookbook_in_a_row": sum(r >= 3 for r in cookbook_runs),
            "mean_longest_cookbook_run": mean(cookbook_runs),
            "episodes_take_then_release_same_item": take_release,
            "prepare_meal": {"count": prepare, "episodes": prepare_eps}, "eat_meal": {"count": eat, "episodes": eat_eps},
            "after_first_take": {"episodes": took_first, "commands": after_take.most_common(8)}}


def print_behaviour(name: str, b: dict) -> None:
    n = b["episodes"]
    print(f'\n## Behaviour: {name} (`{b["file"]}`, {n} episodes, mean length {b["mean_length"]:.1f})\n')
    print(f'- endings: {b["endings"]}')
    print(f'- lost at step (number of commands): {b["lost_at_step"]}')
    print(f'- last command of lost episodes: {b["last_command_when_lost"]}')
    print(f'- `prepare meal`: {b["prepare_meal"]}; `eat meal`: {b["eat_meal"]}')
    print(f'- `examine cookbook`: {b["examine_cookbook_share"]:.1%} of all commands; three or more in a row in '
          f'{b["episodes_with_3_cookbook_in_a_row"]} episodes; mean longest run {b["mean_longest_cookbook_run"]:.1f}')
    print(f'- took an item and later dropped / put / inserted it: {b["episodes_take_then_release_same_item"]} episodes')
    print(f'- command right after the first `take`: {b["after_first_take"]}')
    print("\n| command | count | share |\n|---|---|---|")
    for c, k in b["top_commands"]:
        print(f'| {c} | {k} | {k / b["total_commands"]:.1%} |')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", help="teacher_forced.jsonl of scripts/diagnose_history.py")
    ap.add_argument("--episodes", nargs="*", default=[], help="NAME=forecasts.jsonl pairs")
    ap.add_argument("--max-actions", type=int, default=30)
    ap.add_argument("--json", help="write all numbers here")
    args = ap.parse_args(argv)
    result = {}
    if args.teacher:
        result["teacher_forced"] = teacher_tables(args.teacher)
        print_teacher(result["teacher_forced"])
    for pair in args.episodes:
        name, path = pair.split("=", 1)
        result.setdefault("behaviour", {})[name] = behaviour(path, args.max_actions)
        print_behaviour(name, result["behaviour"][name])
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    main()
