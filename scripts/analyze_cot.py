"""E0007: tables and text dumps from scripts/diagnose_cot.py output.

    python scripts/analyze_cot.py --cot runs/E0007/cot.jsonl --teacher runs/E0006/teacher_forced.jsonl \
        --json reports/E0007.json --reading reading.md [--labels labels.json] --texts texts.md

Prints markdown tables. `--reading` writes the fixed random sample for the human reading (L0 `prepare` states,
20 reasonings per setting); `--labels` is a JSON object {"setting|game|t|k": "A".."E"} with the classes given by
the reader, compared here with the keyword tags; `--texts` writes the full reasoning texts for the report.
Needs neither a model nor a GPU.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SETTINGS = ("full_plain", "full_chat", "feedback_plain", "feedback_chat")
NO_REASONING = {"V1": "full_none", "V2": "feedback_none"}        # E0006 variants on the same states
ENDS = ("answer", "eos", "cap")
REPROCESS = ("cook", "chop", "slice", "dice", "fry", "roast", "grill")
READ_SEED, READ_N = 7, 20
EXAMPLE = ("cook_r1_t1_g1_0000.z8", 2)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def key(r) -> str:
    return f'{r["setting"]}|{r["game"]}|{r["t"]}|{r["k"]}'


def stats(rows) -> dict:
    out = {"n": len(rows), "states": len({(r["game"], r["t"]) for r in rows}),
           "mean_q_star": mean(r["q_star"] for r in rows), "top1": mean(r["rank_star"] == 1 for r in rows),
           "action_is_star": mean(r["action_is_star"] for r in rows),
           "mean_q_reprocess": mean(r["q_reprocess"] for r in rows),
           "action_is_reprocess": mean(r["action_is_reprocess"] for r in rows),
           "notices_done": mean(r["notices_done"] for r in rows), "plans_process": mean(r["plans_process"] for r in rows),
           "mean_reasoning_tokens": mean(r["reasoning_tokens"] for r in rows)}
    ends = Counter(r["reasoning_end"] for r in rows)
    out["ends"] = {e: ends[e] / len(rows) for e in ENDS} if rows else {}
    out["ends"]["other"] = sum(v for e, v in ends.items() if e not in ENDS) / len(rows) if rows else None
    return out


def teacher_stats(rows) -> dict:
    """E0006 rows keep the five most likely options only, so the re-process mass is a lower bound (top 5)."""
    import math
    return {"n": len(rows), "states": len(rows), "mean_q_star": mean(math.exp(r["logq_star"]) for r in rows),
            "top1": mean(r["rank_star"] == 1 for r in rows),
            "mean_q_reprocess_top5": mean(sum(x["p"] for x in r["top"] if x["command"].split()[0] in REPROCESS)
                                          for r in rows)}


def master(cot, teacher) -> list:
    groups = defaultdict(list)
    for r in cot:
        groups[r["level"], r["type"], r["setting"]].append(r)
    out = [{"level": lv, "type": ty, "setting": s, **stats(g)} for (lv, ty, s), g in groups.items()]
    states = {(r["level"], r["game"], r["t"]) for r in cot}
    tg = defaultdict(list)
    for r in teacher:
        if r.get("variant") in NO_REASONING and (r["level"], r["game"], r["t"]) in states:
            tg[r["level"], r["type"], NO_REASONING[r["variant"]]].append(r)
    out += [{"level": lv, "type": ty, "setting": s, **teacher_stats(g)} for (lv, ty, s), g in tg.items()]
    order = ("full_none", "full_plain", "full_chat", "feedback_none", "feedback_plain", "feedback_chat")
    return sorted(out, key=lambda r: (r["level"], -ord(r["type"][0]), order.index(r["setting"])))


def conditional(cot, tag) -> list:
    out = []
    for scope in ("all", "L0", "L1", "prepare", "eat"):
        for s in SETTINGS:
            rows = [r for r in cot if r["setting"] == s and scope in ("all", r["level"], r["type"])]
            for value in (True, False):
                g = [r for r in rows if r[tag] is value]
                out.append({"scope": scope, "setting": s, tag: value, "n": len(g),
                            "mean_q_star": mean(r["q_star"] for r in g), "top1": mean(r["rank_star"] == 1 for r in g),
                            "action_is_star": mean(r["action_is_star"] for r in g),
                            "mean_q_reprocess": mean(r["q_reprocess"] for r in g)})
    return out


def per_state(cot) -> list:
    """How many of a state's samples pick a*: the spread over reasonings of the same state."""
    out = []
    for lv in ("L0", "L1"):
        for ty in ("prepare", "eat"):
            for s in SETTINGS:
                hits = defaultdict(list)
                for r in cot:
                    if (r["level"], r["type"], r["setting"]) == (lv, ty, s):
                        hits[r["game"], r["t"]].append(r["rank_star"] == 1)
                c = Counter(sum(v) for v in hits.values())
                out.append({"level": lv, "type": ty, "setting": s, "states": len(hits),
                            "states_by_top1_count": {str(k): c[k] for k in sorted(c)}})
    return out


def reading_sample(cot, type="prepare") -> list:
    """READ_N reasonings per setting on the L0 states of one action type. The `prepare` sample is the one the
    issue asks for; the `eat` sample was added on the user's request and has its own seed string."""
    out = []
    for s in SETTINGS:
        rows = sorted((r for r in cot if r["setting"] == s and r["level"] == "L0" and r["type"] == type), key=key)
        seed = f"{READ_SEED}|{s}" if type == "prepare" else f"{READ_SEED}|{s}|{type}"
        out += random.Random(seed).sample(rows, min(READ_N, len(rows)))
    return out


def block(r, top=3) -> str:
    tops = "; ".join(f'`{x["command"]}` {x["p"]:.3f}' for x in r["top"][:top])
    return (f'**{r["setting"]} · {r["game"]} step {r["t"]} · k={r["k"]}** — {r["reasoning_tokens"]} tokens, ends with '
            f'`{r["reasoning_end"]}`; tags: notices_done={r["notices_done"]}, plans_process={r["plans_process"]}\n\n'
            f'```text\n{r["reasoning"]}\n```\n\n'
            f'After the reasoning: {tops}. q(a\\*) = {r["q_star"]:.3f} (a\\* = `{r["a_star"]}`), re-process mass '
            f'{r["q_reprocess"]:.3f}; sampled action `{r["action"]}`.\n')


def texts(cot) -> str:
    l0 = sorted({(r["game"], r["t"]) for r in cot if r["level"] == "L0" and r["type"] == "prepare"} - {EXAMPLE})
    others = random.Random(READ_SEED).sample(l0, 2)
    parts = [f"### `{EXAMPLE[0]}` step {EXAMPLE[1]}: all four settings, five reasonings each\n"]
    for s in SETTINGS:
        parts += [block(r) for r in sorted(cot, key=key) if r["setting"] == s and (r["game"], r["t"]) == EXAMPLE]
    for game, t in others:
        parts.append(f"### `{game}` step {t}: two reasonings per setting (k = 0, 1)\n")
        for s in SETTINGS:
            parts += [block(r) for r in sorted(cot, key=key)
                      if r["setting"] == s and (r["game"], r["t"]) == (game, t) and r["k"] < 2]
    return "\n".join(parts)


def labels_table(sample, labels) -> dict:
    out = {"counts": {}, "tags_by_class": {}, "rows": []}
    for s in SETTINGS:
        out["counts"][s] = dict(Counter(labels[key(r)] for r in sample if r["setting"] == s))
    by = defaultdict(list)
    for r in sample:
        c = labels[key(r)]
        by[c].append(r)
        out["rows"].append({"key": key(r), "class": c, "notices_done": r["notices_done"],
                            "plans_process": r["plans_process"], "q_star": r["q_star"],
                            "q_reprocess": r["q_reprocess"], "action": r["action"]})
    for c, g in sorted(by.items()):
        out["tags_by_class"][c] = {"n": len(g), "notices_done": sum(r["notices_done"] for r in g),
                                   "plans_process": sum(r["plans_process"] for r in g),
                                   "mean_q_star": mean(r["q_star"] for r in g),
                                   "mean_q_reprocess": mean(r["q_reprocess"] for r in g)}
    # the tags as classifiers: notices_done for "the reader says the reasoning notices it is done" (A or C),
    # plans_process for "the reader says the reasoning plans to process" (B)
    for tag, positive in (("notices_done", ("A", "C")), ("plans_process", ("B",))):
        tp = sum(r[tag] and labels[key(r)] in positive for r in sample)
        fp = sum(r[tag] and labels[key(r)] not in positive for r in sample)
        fn = sum(not r[tag] and labels[key(r)] in positive for r in sample)
        out[f"{tag}_vs_{''.join(positive)}"] = {"tp": tp, "fp": fp, "fn": fn, "tn": len(sample) - tp - fp - fn}
    return out


def f(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def print_master(rows) -> None:
    print("\n## Master table\n")
    print("| level | type | setting | n | mean q(a*) | a* rank 1 | sampled = a* | re-process mass | sampled re-process | "
          "notices_done | plans_process | tokens | answer / eos / cap |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "ends" in r:
            print(f'| {r["level"]} | {r["type"]} | {r["setting"]} | {r["n"]} | {f(r["mean_q_star"], 3)} | {f(r["top1"])} | '
                  f'{f(r["action_is_star"])} | {f(r["mean_q_reprocess"], 3)} | {f(r["action_is_reprocess"])} | '
                  f'{f(r["notices_done"])} | {f(r["plans_process"])} | {f(r["mean_reasoning_tokens"], 0)} | '
                  + " / ".join(f(r["ends"][e]) for e in ENDS) + " |")
        else:
            print(f'| {r["level"]} | {r["type"]} | {r["setting"]} (E0006) | {r["n"]} | {f(r["mean_q_star"], 3)} | '
                  f'{f(r["top1"])} | - | ≥ {f(r["mean_q_reprocess_top5"], 3)} | - | - | - | 0 | - |')


def print_conditional(rows, tag) -> None:
    print(f"\n## By `{tag}`\n")
    print(f"| scope | setting | {tag} | n | mean q(a*) | a* rank 1 | sampled = a* | re-process mass |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f'| {r["scope"]} | {r["setting"]} | {r[tag]} | {r["n"]} | {f(r["mean_q_star"], 3)} | {f(r["top1"])} | '
              f'{f(r["action_is_star"])} | {f(r["mean_q_reprocess"], 3)} |')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cot", required=True)
    ap.add_argument("--teacher", help="runs/E0006/teacher_forced.jsonl (no-reasoning V1 / V2 on the same states)")
    ap.add_argument("--teacher-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--json")
    ap.add_argument("--reading", help="write the reading sample here (markdown)")
    ap.add_argument("--labels", help="JSON {key: class} for the reading sample")
    ap.add_argument("--labels-eat", help="JSON {key: class} for the L0 `eat` reading sample (added after the issue)")
    ap.add_argument("--texts", help="write the full-text section here (markdown)")
    args = ap.parse_args(argv)
    cot = load(args.cot)
    teacher = [r for r in load(args.teacher) if "end" not in r and r["model"] == args.teacher_model] if args.teacher else []
    result = {"rows": len(cot), "master": master(cot, teacher),
              "by_notices_done": conditional(cot, "notices_done"), "by_plans_process": conditional(cot, "plans_process"),
              "per_state": per_state(cot)}
    print(f"{len(cot)} rows")
    print_master(result["master"])
    print_conditional(result["by_notices_done"], "notices_done")
    print_conditional(result["by_plans_process"], "plans_process")
    print("\n## States by the number of samples (of 5) that rank a* first\n")
    for r in result["per_state"]:
        print(f'- {r["level"]} {r["type"]} {r["setting"]}: {r["states_by_top1_count"]}')
    sample = reading_sample(cot)
    result["reading_sample"] = [key(r) for r in sample]
    if args.reading:
        Path(args.reading).write_text("\n".join(f"#### {i + 1}. `{key(r)}`\n\n{block(r, 5)}" for i, r in enumerate(sample)))
    if args.labels:
        labels = json.loads(Path(args.labels).read_text())
        missing = [key(r) for r in sample if key(r) not in labels]
        if missing:
            ap.error(f"{len(missing)} sampled reasonings have no label, e.g. {missing[:3]}")
        result["reading"] = labels_table(sample, labels)
        print("\n## Human reading\n")
        print(json.dumps({k: v for k, v in result["reading"].items() if k != "rows"}, indent=1))
    if args.labels_eat:
        labels = json.loads(Path(args.labels_eat).read_text())
        sample = reading_sample(cot, "eat")
        missing = [key(r) for r in sample if key(r) not in labels]
        if missing:
            ap.error(f"{len(missing)} sampled eat reasonings have no label, e.g. {missing[:3]}")
        result["reading_eat_sample"] = [key(r) for r in sample]
        result["reading_eat"] = labels_table(sample, labels)
        print("\n## Human reading, eat\n")
        print(json.dumps({k: v for k, v in result["reading_eat"].items() if k != "rows"}, indent=1))
    if args.texts:
        Path(args.texts).write_text(texts(cot))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    main()
