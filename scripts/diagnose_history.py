"""E0006: score the walkthrough action under several history layouts, teacher-forced.

    python scripts/diagnose_history.py --out runs/E0006/teacher_forced.jsonl \
        --models Qwen/Qwen3-1.7B runs/E0003c/checkpoints/batch0100 --tokenizer Qwen/Qwen3-1.7B \
        --games L0=data/games_L0/valid L1=data/games_L1/valid

Every game is played along its walkthrough (the game's .json, metadata.walkthrough; the pure information
commands that TextWorldEnv filters out are skipped). At every state on that path the prompt is built once per
variant (V1-V5: history_mode x history_window, see VARIANTS; V6-V14: changes to the V2 prompt made in this
script, see SCRIPT_VARIANTS) and `LMPolicy.decision_forward` scores all options
without reasoning and without the forecast. All variants see the same states, so differences come from the
prompt alone. One JSON line per (model, level, game, step, variant).

    python scripts/diagnose_history.py --show data/games_L0/valid/<game>.z8 --step 2 --variants V1 V2

prints the prompts of one state (no model needed).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevtw.config import PolicyConfig  # noqa: E402
from jevtw.envs import TextWorldEnv  # noqa: E402
from jevtw.policy import Decision, LMPolicy, build_prompt  # noqa: E402

# name -> (history_mode, history_window). V5 has no history at all, so its history_mode does not matter.
VARIANTS = {"V1": ("full", 8), "V2": ("feedback", 8), "V3": ("full", 1), "V4": ("feedback", 1), "V5": ("full", 0)}
# Variants built in this script, not by jevtw (asked for by the user after the first report): the V2 prompt
# (feedback, window 8) with one or more changes. Keys: see `script_prompt`.
SCRIPT_VARIANTS = {"V6": {"dedup": True}, "V7": {"inventory": True}, "V8": {"hint": True},
                   "V9": {"dedup": True, "inventory": True, "hint": True},
                   # second follow-up: parts of the full observation put back into every past step (R room
                   # description, I inventory, N notes = the recipe). With V2 (nothing), V7 (I, in its own layout)
                   # and V1 (R + I + N, which is history_mode "full") this is the 2 x 2 x 2 design.
                   "V10": {"parts": ("notes",)}, "V11": {"parts": ("room",)},
                   "V12": {"parts": ("inventory", "notes")}, "V13": {"parts": ("room", "notes")},
                   "V14": {"parts": ("room", "inventory")}}
PARTS = ("room", "inventory", "notes")
ALL_VARIANTS = [*VARIANTS, *SCRIPT_VARIANTS]
HINT = ("Ingredients in your inventory show their state (for example fried or sliced). "
        "Do not repeat a step that is already done.")
GAME_SUFFIXES = (".z8", ".ulx")
TOP = 5


def variant_config(name: str, **overrides) -> PolicyConfig:
    """`overrides` is only for tests (a tokenizer whose yes / no texts differ); it never touches the prompt."""
    mode, window = VARIANTS.get(name, VARIANTS["V2"])         # the script variants change the V2 prompt
    return PolicyConfig(**{**overrides, "max_reasoning_tokens": 0, "forecast": False, "history_mode": mode,
                           "history_window": window})


def inventory_line(text: str) -> str:
    """The "Inventory: ..." paragraph of an observation text ("" if it has none)."""
    if "Inventory:" not in text:
        return ""
    return "Inventory:" + text.split("Inventory:", 1)[1].split("\n\n", 1)[0].rstrip()


def split_observation(obs) -> dict:
    """The pieces TextWorldEnv._observe joins into obs.text: "head" (what full history shows first: the feedback,
    or the whole room description after a move), "room" (the room description; "" when the head already is
    it), "inventory" and "notes" ("" before the cookbook was read). The pieces joined by blank lines are
    obs.text again."""
    rest, notes = obs.text, ""
    if "\n\nNotes:\n" in rest:
        rest, notes = rest.split("\n\nNotes:\n", 1)
        notes = "Notes:\n" + notes
    head, inventory = rest.rsplit("\n\nInventory: ", 1)
    inventory = "Inventory: " + inventory
    room = ""
    if obs.feedback and head.startswith(obs.feedback + "\n\n"):
        head, room = obs.feedback, head[len(obs.feedback) + 2:]
    return {"head": head, "room": room, "inventory": inventory, "notes": notes}


def step_text(after, parts) -> str:
    """A past step under `parts`: its feedback, then the chosen pieces of its full observation. With a room
    description that is the feedback itself (after a move) "room" shows that description, as full history does."""
    pieces = split_observation(after)
    first = pieces["head"] if "room" in parts else after.feedback
    return "\n\n".join([first, *(pieces[k] for k in PARTS if k in parts and pieces[k])])


def script_prompt(objective, history, obs, cfg: PolicyConfig, dedup=False, inventory=False, hint=False,
                  parts=()) -> str:
    """The prompt of jevtw.policy.build_prompt for history_mode "feedback" without reasoning (with no flag set
    it is the same string; a test checks this), with these changes:
      dedup      the current observation does not start with the feedback of the last command again (that line
                 is already the end of "Recent steps")
      inventory  every past step also shows the inventory after it
      hint       the sentence HINT follows the objective
      parts      every past step also shows these pieces of its full observation (see `step_text`); all of
                 PARTS gives the prompt of history_mode "full".
    """
    lines = [f"Objective: {objective}", ""]
    if hint:
        lines += [HINT, ""]
    recent = list(history)[-cfg.history_window:] if cfg.history_window > 0 else []
    if recent:
        lines.append("Recent steps:")
        for cmd, after in recent:
            lines += [f"> {cmd}", step_text(after, parts)]
            if inventory and inventory_line(after.text):
                lines.append(inventory_line(after.text))
            lines.append("")
    text = obs.text
    if dedup and recent and obs.feedback and text.startswith(obs.feedback):
        text = text[len(obs.feedback):].lstrip("\n")
    lines += ["Current observation:", text, "", "Options:"]
    lines += [f"{j + 1}. {c}" for j, c in enumerate(obs.commands)]
    lines += ["", "Reply with the number of one option."]
    return "\n".join(lines)


def variant_prompt(name: str, objective, history, obs, cfg: PolicyConfig) -> str:
    if name in SCRIPT_VARIANTS:
        return script_prompt(objective, history, obs, cfg, **SCRIPT_VARIANTS[name])
    return build_prompt(objective, history, obs, cfg)


def walkthrough(game_path: Path) -> list[str]:
    return list(json.loads(game_path.with_suffix(".json").read_text())["metadata"]["walkthrough"])


def walk(game_path: Path, skip_redundant_open: bool = False):
    """Yield (t, objective, history, obs, a_star) for every state on the walkthrough, then a final dict
    {"won", "skipped", "problem", "steps", "redundant"}. `history` is the list of (command, observation after it)
    so far. The walk stops early (problem != None) if the next walkthrough command is not among the admissible
    commands. With `skip_redundant_open`, an `open X` that is not admissible while `close X` is (X is already
    open: tw-cooking walkthroughs with several ingredients repeat `open fridge` and `open <door>`) is left out
    and listed under "redundant" instead."""
    env = TextWorldEnv(game_path)
    try:
        obs = env.reset(0)
        history, skipped, redundant, problem, t = [], [], [], None, 0
        for command in walkthrough(game_path):
            if not env._keep(command):            # look / inventory / examine X: not an option of the agent
                skipped.append(command)
                continue
            if obs.done:
                problem = f"game over before {command!r}"
                break
            if (skip_redundant_open and command not in obs.commands and command.startswith("open ")
                    and "close " + command[len("open "):] in obs.commands):
                redundant.append(command)
                continue
            if command not in obs.commands:
                problem = f"{command!r} is not admissible at step {t}"
                break
            yield t, env.objective, list(history), obs, command
            after = env.step(command)
            history.append((command, after))
            obs, t = after, t + 1
        yield {"won": bool(obs.won), "skipped": skipped, "problem": problem, "steps": t, "redundant": redundant}
    finally:
        env.close()


@torch.no_grad()
def option_logq(policy: LMPolicy, prompt: str, n_options: int) -> torch.Tensor:
    """log q over the options after `prompt`, exactly as LMPolicy.decide computes it without reasoning."""
    prompt_ids = policy.encode(prompt)
    context = policy._action_context(prompt_ids, [])
    options = [policy.encode(f" {j + 1}") + [policy.eos_id] for j in range(n_options)]
    d = Decision(command="", action=-1, prompt_ids=prompt_ids, reasoning_ids=[], action_context_ids=context,
                 option_ids=options, behavior_logps=[])
    _, scores, _ = policy.decision_forward(policy.model, d, forecast=False)
    return torch.log_softmax(scores.float(), dim=0).cpu()


def score_state(policies: dict, objective, history, obs, a_star) -> list[dict]:
    rows, target = [], obs.commands.index(a_star)
    for name, policy in policies.items():
        prompt = variant_prompt(name, objective, history, obs, policy.cfg)
        log_q = option_logq(policy, prompt, len(obs.commands))
        order = torch.argsort(log_q, descending=True).tolist()
        rows.append({"variant": name, "history_mode": policy.cfg.history_mode,
                     "history_window": policy.cfg.history_window, "prompt_tokens": len(policy.encode(prompt)),
                     "logq_star": float(log_q[target]), "rank_star": order.index(target) + 1,
                     "top": [{"command": obs.commands[j], "p": float(log_q[j].exp())} for j in order[:TOP]]})
    return rows


def run(model, tokenizer, model_name: str, levels: dict, out, limit=None, variants=tuple(VARIANTS),
        **overrides) -> dict:
    policies = {name: LMPolicy(model, tokenizer, variant_config(name, **overrides)) for name in variants}
    summary = {}
    for level, directory in levels.items():
        paths = sorted(p for p in Path(directory).iterdir() if p.suffix in GAME_SUFFIXES)[:limit]
        states = problems = wins = 0
        for path in paths:
            for item in walk(path):
                if isinstance(item, dict):
                    wins += item["won"]
                    problems += item["problem"] is not None
                    out.write(json.dumps({"model": model_name, "level": level, "game": path.name, "end": item}) + "\n")
                    continue
                t, objective, history, obs, a_star = item
                states += 1
                for row in score_state(policies, objective, history, obs, a_star):
                    out.write(json.dumps({"model": model_name, "level": level, "game": path.name, "t": t,
                                          "a_star": a_star, "type": a_star.split()[0], "K": len(obs.commands),
                                          **row}) + "\n")
            out.flush()
        summary[level] = {"games": len(paths), "states": states, "walkthroughs_won": wins, "problems": problems}
        print(f"{model_name} {level}: {summary[level]}", flush=True)
    return summary


def show(game: str, step: int, variants) -> None:
    for item in walk(Path(game)):
        if isinstance(item, dict) or item[0] != step:
            continue
        t, objective, history, obs, a_star = item
        for name in variants:
            cfg = variant_config(name)
            extra = f", {SCRIPT_VARIANTS[name]}" if name in SCRIPT_VARIANTS else ""
            print(f"===== {name} (history_mode={cfg.history_mode}, history_window={cfg.history_window}{extra}), "
                  f"step {t}, a* = {a_star!r} (option {obs.commands.index(a_star) + 1}) =====")
            print(variant_prompt(name, objective, history, obs, cfg))
            print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-1.7B"])
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--games", nargs="+", default=["L0=data/games_L0/valid", "L1=data/games_L1/valid"],
                    help="LEVEL=directory pairs")
    ap.add_argument("--limit", type=int, help="only the first N games of each directory")
    ap.add_argument("--show", help="a game file: print the prompts of --step instead of scoring")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=ALL_VARIANTS,
                    help="default V1-V5 (jevtw configurations); V6-V14 are the script variants of the V2 prompt")
    args = ap.parse_args(argv)
    if args.show:
        show(args.show, args.step, args.variants)
        return
    if not args.out:
        ap.error("--out is required")
    out_path = Path(args.out)
    if out_path.exists():
        ap.error(f"{out_path} already exists; choose a new file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    levels = dict(pair.split("=", 1) for pair in args.games)
    with out_path.open("w") as out:
        for name in args.models:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
            run(model, tokenizer, name, levels, out, args.limit, tuple(args.variants))
            del model
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
