"""E0006: score the walkthrough action under several history layouts, teacher-forced.

    python scripts/diagnose_history.py --out runs/E0006/teacher_forced.jsonl \
        --models Qwen/Qwen3-1.7B runs/E0003c/checkpoints/batch0100 --tokenizer Qwen/Qwen3-1.7B \
        --games L0=data/games_L0/valid L1=data/games_L1/valid

Every game is played along its walkthrough (the game's .json, metadata.walkthrough; the pure information
commands that TextWorldEnv filters out are skipped). At every state on that path the prompt is built once per
variant (V1-V5: history_mode x history_window, see VARIANTS) and `LMPolicy.decision_forward` scores all options
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
GAME_SUFFIXES = (".z8", ".ulx")
TOP = 5


def variant_config(name: str, **overrides) -> PolicyConfig:
    """`overrides` is only for tests (a tokenizer whose yes / no texts differ); it never touches the prompt."""
    mode, window = VARIANTS[name]
    return PolicyConfig(**{**overrides, "max_reasoning_tokens": 0, "forecast": False, "history_mode": mode,
                           "history_window": window})


def walkthrough(game_path: Path) -> list[str]:
    return list(json.loads(game_path.with_suffix(".json").read_text())["metadata"]["walkthrough"])


def walk(game_path: Path):
    """Yield (t, objective, history, obs, a_star) for every state on the walkthrough, then a final dict
    {"won", "skipped", "problem"}. `history` is the list of (command, observation after it) so far. The walk
    stops early (problem != None) if the next walkthrough command is not among the admissible commands."""
    env = TextWorldEnv(game_path)
    try:
        obs = env.reset(0)
        history, skipped, problem, t = [], [], None, 0
        for command in walkthrough(game_path):
            if not env._keep(command):            # look / inventory / examine X: not an option of the agent
                skipped.append(command)
                continue
            if obs.done:
                problem = f"game over before {command!r}"
                break
            if command not in obs.commands:
                problem = f"{command!r} is not admissible at step {t}"
                break
            yield t, env.objective, list(history), obs, command
            after = env.step(command)
            history.append((command, after))
            obs, t = after, t + 1
        yield {"won": bool(obs.won), "skipped": skipped, "problem": problem, "steps": t}
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
        prompt = build_prompt(objective, history, obs, policy.cfg)
        log_q = option_logq(policy, prompt, len(obs.commands))
        order = torch.argsort(log_q, descending=True).tolist()
        rows.append({"variant": name, "history_mode": policy.cfg.history_mode,
                     "history_window": policy.cfg.history_window, "prompt_tokens": len(policy.encode(prompt)),
                     "logq_star": float(log_q[target]), "rank_star": order.index(target) + 1,
                     "top": [{"command": obs.commands[j], "p": float(log_q[j].exp())} for j in order[:TOP]]})
    return rows


def run(model, tokenizer, model_name: str, levels: dict, out, limit=None, **overrides) -> dict:
    policies = {name: LMPolicy(model, tokenizer, variant_config(name, **overrides)) for name in VARIANTS}
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
            print(f"===== {name} (history_mode={VARIANTS[name][0]}, history_window={VARIANTS[name][1]}), "
                  f"step {t}, a* = {a_star!r} (option {obs.commands.index(a_star) + 1}) =====")
            print(build_prompt(objective, history, obs, variant_config(name)))
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
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
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
            run(model, tokenizer, name, levels, out, args.limit)
            del model
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
