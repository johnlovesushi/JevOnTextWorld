"""E0009 part D: processing errors in sampled episodes on the trap-free levels (C0, C1).

    python scripts/analyze_trap_free.py --games data/games_C0/valid \
        --episodes a=runs/E0009a/forecasts/batch0000.jsonl b=runs/E0009b/forecasts/batch0000.jsonl \
        --json reports/E0009_behaviour_C0.json

For every episode file (jevtw.train --save-forecasts) it prints the E0006 section 4 numbers
(scripts/analyze_history.py, `behaviour`: endings, last command of lost episodes, top 10 commands) and the errors
that only exist when the recipe asks for cooking and cutting. What the recipe needs is read from the game's
.json (metadata.ingredients: [name, cooked state, cut state]); the commands of an episode are replayed on a small
state (is the knife held, what was done to every item) without starting the game:
  recook        `cook X with ...` when X (a recipe ingredient) was already cooked in this episode
  wrong_cook    the first cooking of a recipe ingredient uses another appliance than the recipe's
                (fry = stove, roast = oven, grill = BBQ)
  wrong_cut     a recipe ingredient is cut with another verb than the recipe's (chop / slice / dice)
  recut         a recipe ingredient is cut a second time
  cut_no_knife  a cut command while the knife is not held (`take knife` not yet chosen, or dropped / put down)
  other_item    cooking or cutting an item that is not in the recipe
Counts are per command and per episode; for lost episodes the kind of the last command is reported too.

    python scripts/analyze_trap_free.py --teacher runs/E0009/teacher_forced.jsonl --json reports/E0009_teacher.json

Part C, from scripts/diagnose_history.py output: master table, table by action type (`take knife ...` apart from
the other `take`, chop / slice / dice as one type "cut") and the probability of the whole walkthrough, i.e. the
product of q(a*) over the states of a game, averaged over games (also without the `drop knife` state, which a
win does not need).
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_history import behaviour, item_of, mean, print_behaviour, stats, variant_key  # noqa: E402
from check_games import COOK_TOOL, CUT_VERBS, STATE_VERB  # noqa: E402

KINDS = ("recook", "wrong_cook", "wrong_cut", "recut", "cut_no_knife", "other_item")
TOOL_VERB = {tool: verb for verb, tool in COOK_TOOL.items()}


def recipe_of(game_json: Path) -> dict:
    """{ingredient: {"cook": verb or None, "cut": verb or None}}"""
    metadata = json.loads(game_json.read_text())["metadata"]
    out = {}
    for name, *states in metadata["ingredients"]:
        verbs = [STATE_VERB[s] for s in states if s in STATE_VERB]
        out[name] = {"cook": next((v for v in verbs if v in COOK_TOOL), None),
                     "cut": next((v for v in verbs if v in CUT_VERBS), None)}
    return out


def classify(commands: list[str], recipe: dict) -> list[list[str]]:
    """For every command the list of error kinds it shows (empty for most commands)."""
    knife, cooked, cut, out = False, set(), set(), []
    for c in commands:
        kinds, words = [], c.split()
        moved = item_of(c)
        if moved and moved[1] == "knife":
            knife = moved[0] == "take"
        if words[0] == "cook" and " with " in c:
            item, tool = c[len("cook "):].rsplit(" with ", 1)
            if item not in recipe:
                kinds.append("other_item")
            elif item in cooked:
                kinds.append("recook")
            elif TOOL_VERB.get(tool) != recipe[item]["cook"]:
                kinds.append("wrong_cook")
            cooked.add(item)
        elif words[0] in CUT_VERBS and " with " in c:
            item = c[len(words[0]) + 1:].rsplit(" with ", 1)[0]
            if not knife:
                kinds.append("cut_no_knife")
            if item not in recipe:
                kinds.append("other_item")
            elif item in cut:
                kinds.append("recut")
            elif words[0] != recipe[item]["cut"]:
                kinds.append("wrong_cut")
            cut.add(item)
        out.append(kinds)
    return out


def progress(commands: list[str], recipe: dict) -> dict:
    """Does the recipe need cooking / cutting, and did the episode do all of it without a processing error on
    a recipe ingredient."""
    kinds = classify(commands, recipe)
    cook_ok = cut_ok = 0
    for c, k in zip(commands, kinds):
        words = c.split()
        if k or " with " not in c:
            continue
        cook_ok += words[0] == "cook"
        cut_ok += words[0] in CUT_VERBS
    need_cook = sum(v["cook"] is not None for v in recipe.values())
    need_cut = sum(v["cut"] is not None for v in recipe.values())
    flat = {k for ks in kinds for k in ks}
    return {"needs_cook": need_cook > 0, "needs_cut": need_cut > 0,
            "cook_done": need_cook > 0 and cook_ok >= need_cook and not flat & {"wrong_cook", "recook"},
            "cut_done": need_cut > 0 and cut_ok >= need_cut and not flat & {"wrong_cut", "recut"},
            "took_knife": any(c.startswith("take knife") for c in commands),
            "prepared": "prepare meal" in commands}


def errors(path, games_dir, max_actions=30) -> dict:
    episodes = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    commands_n, episodes_n, last_kind = Counter(), Counter(), Counter()
    reached = Counter()
    lost = 0
    for e in episodes:
        recipe = recipe_of((Path(games_dir) / e["game"]).with_suffix(".json"))
        cs = [s["command"] for s in e["steps"]]
        kinds = classify(cs, recipe)
        commands_n.update(k for ks in kinds for k in ks)
        episodes_n.update({k for ks in kinds for k in ks})
        reached.update(k for k, v in progress(cs, recipe).items() if v)
        if not e["z"] and e["length"] < max_actions:
            lost += 1
            last_kind[", ".join(kinds[-1]) or f"no processing error ({cs[-1].split()[0]} ...)"] += 1
    return {"file": str(path), "episodes": len(episodes), "lost_before_limit": lost,
            "commands": {k: commands_n.get(k, 0) for k in KINDS}, "episodes_with": {k: episodes_n.get(k, 0) for k in KINDS},
            "last_command_kind_when_lost": last_kind.most_common(), "reached": dict(reached)}


def print_errors(name: str, e: dict) -> None:
    print(f'\n### Processing errors: {name} ({e["episodes"]} episodes, {e["lost_before_limit"]} lost before the step limit)\n')
    print("| error | commands | episodes |\n|---|---|---|")
    for k in KINDS:
        print(f'| {k} | {e["commands"][k]} | {e["episodes_with"][k]} |')
    print(f'\n- kind of the last command of lost episodes: {e["last_command_kind_when_lost"]}')
    print(f'- episodes that did a step correctly: {e["reached"]}')


def action_type(a_star: str) -> str:
    first = a_star.split()[0]
    if first in CUT_VERBS:
        return "cut"
    if a_star.startswith("take knife"):
        return "take knife"
    return first


def teacher(path, optional=("drop",)) -> dict:
    rows = [r for r in map(json.loads, Path(path).read_text().splitlines()) if "end" not in r]
    groups, by_type, games = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        groups[r["level"], r["variant"]].append(r)
        by_type[r["level"], action_type(r["a_star"]), r["variant"]].append(r)
        games[r["level"], r["variant"], r["game"]].append(r)
    out = {"overall": [], "by_type": [], "path": []}
    for (level, variant), g in sorted(groups.items(), key=lambda kv: (kv[0][0], variant_key(kv[0][1]))):
        out["overall"].append({"level": level, "variant": variant, **stats(g), "mean_q": mean(math.exp(r["logq_star"]) for r in g)})
        mine = [v for (lv, va, _), v in games.items() if (lv, va) == (level, variant)]
        prob = [math.exp(sum(r["logq_star"] for r in v)) for v in mine]
        needed = [math.exp(sum(r["logq_star"] for r in v if r["type"] not in optional)) for v in mine]
        out["path"].append({"level": level, "variant": variant, "games": len(mine), "mean_path_prob": mean(prob),
                            "mean_path_prob_needed": mean(needed), "all_rank1": mean(all(r["rank_star"] == 1 for r in v) for v in mine),
                            "all_rank1_needed": mean(all(r["rank_star"] == 1 for r in v if r["type"] not in optional) for v in mine)})
    for (level, kind, variant), g in sorted(by_type.items(), key=lambda kv: (kv[0][0], kv[0][1], variant_key(kv[0][2]))):
        out["by_type"].append({"level": level, "type": kind, "variant": variant, **stats(g),
                               "mean_q": mean(math.exp(r["logq_star"]) for r in g)})
    return out


def print_teacher(t: dict) -> None:
    f = (lambda x, d=3: "-" if x is None else f"{x:.{d}f}")
    print("\n| level | variant | states | mean q(a*) | mean log q(a*) | rank-1 rate | mean rank | mean K | prompt tokens |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in t["overall"]:
        print(f'| {r["level"]} | {r["variant"]} | {r["n"]} | {f(r["mean_q"])} | {f(r["mean_logq"])} | {f(r["top1"])} | '
              f'{f(r["mean_rank"], 2)} | {f(r["mean_K"], 1)} | {f(r["mean_prompt_tokens"], 0)} |')
    variants = sorted({r["variant"] for r in t["by_type"]}, key=variant_key)
    cells = defaultdict(dict)
    for r in t["by_type"]:
        cells[r["level"], r["type"]][r["variant"]] = r
    print("\nBy action type: rank-1 rate (mean q(a*))\n")
    print("| level | type | states | " + " | ".join(variants) + " |\n|---|---|---|" + "---|" * len(variants))
    for (level, kind), c in cells.items():
        print(f"| {level} | {kind} | {c[variants[0]]['n']} | "
              + " | ".join(f'{f(c[v]["top1"], 2)} ({f(c[v]["mean_q"], 2)})' for v in variants) + " |")
    print("\nWhole walkthrough (per game, then the mean over games)\n")
    print("| level | variant | games | mean prod q(a*) | same without `drop` | all states rank 1 | same without `drop` |")
    print("|---|---|---|---|---|---|---|")
    for r in t["path"]:
        print(f'| {r["level"]} | {r["variant"]} | {r["games"]} | {f(r["mean_path_prob"], 4)} | '
              f'{f(r["mean_path_prob_needed"], 4)} | {f(r["all_rank1"])} | {f(r["all_rank1_needed"])} |')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", help="directory of the games the episodes were played on")
    ap.add_argument("--episodes", nargs="*", default=[], help="NAME=forecasts.jsonl pairs")
    ap.add_argument("--teacher", help="teacher_forced.jsonl of scripts/diagnose_history.py")
    ap.add_argument("--max-actions", type=int, default=30)
    ap.add_argument("--json", help="write all numbers here")
    args = ap.parse_args(argv)
    result = {}
    if args.teacher:
        result["teacher"] = teacher(args.teacher)
        print_teacher(result["teacher"])
    if args.episodes and not args.games:
        ap.error("--episodes needs --games")
    for pair in args.episodes:
        name, path = pair.split("=", 1)
        b = behaviour(path, args.max_actions)
        e = errors(path, args.games, args.max_actions)
        print_behaviour(name, b)
        print_errors(name, e)
        result[name] = {"behaviour": b, "errors": e}
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
