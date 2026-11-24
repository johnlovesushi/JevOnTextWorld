"""Prompt length along the walkthrough of every game in a directory, for each history_mode (CPU, tokenizer only).

  python scripts/prompt_tokens.py data/games_L0/train data/games/train --model Qwen/Qwen3-1.7B

Prints one JSON line per (directory, history_mode): mean / p90 / max prompt tokens over all decisions, and the
same for the last decision of each game (where the history is longest)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevtw.config import HISTORY_MODES, Config  # noqa: E402
from jevtw.envs import TextWorldEnv  # noqa: E402
from jevtw.policy import build_prompt  # noqa: E402
from jevtw.train import GAME_SUFFIXES  # noqa: E402


def walkthrough_prompts(path: Path, configs: dict) -> tuple[dict, bool]:
    """The prompt of every decision along the walkthrough, per history mode, and whether the walkthrough won."""
    import textworld

    raw = textworld.start(str(path), request_infos=textworld.EnvInfos(extras=["walkthrough"]))
    commands = raw.reset()["extra.walkthrough"]
    raw.close()
    env = TextWorldEnv(path)
    obs, history = env.reset(0), []
    prompts = {mode: [] for mode in configs}
    for command in commands:
        if not env._keep(command):        # information-only commands are not offered to the agent
            continue
        if obs.done:
            break
        for mode, cfg in configs.items():
            prompts[mode].append(build_prompt(env.objective, history, obs, cfg))
        obs = env.step(command)
        history.append((command, obs))
    env.close()
    return prompts, obs.won


def stats(values: list[int]) -> dict:
    ordered = sorted(values)
    return {"n": len(ordered), "mean": round(sum(ordered) / len(ordered), 1),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], "max": ordered[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directories", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision")
    ap.add_argument("--config", help="policy settings other than history_mode (default: PolicyConfig defaults)")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    base = (Config.load(args.config) if args.config else Config()).policy
    configs = {mode: type(base)(**{**base.__dict__, "history_mode": mode}) for mode in HISTORY_MODES}
    for directory in args.directories:
        games = sorted(p for p in Path(directory).iterdir() if p.suffix in GAME_SUFFIXES)
        tokens = {mode: [] for mode in configs}
        last = {mode: [] for mode in configs}
        won = 0
        for path in games:
            prompts, game_won = walkthrough_prompts(path, configs)
            won += game_won
            for mode, texts in prompts.items():
                counts = [len(ids) for ids in tokenizer(texts)["input_ids"]]
                tokens[mode] += counts
                last[mode].append(counts[-1])
        for mode in configs:
            print(json.dumps({"games": directory, "n_games": len(games), "walkthroughs_won": won,
                              "history_mode": mode, "history_window": base.history_window,
                              "max_reasoning_tokens": base.max_reasoning_tokens,
                              "prompt_tokens": stats(tokens[mode]), "last_decision_prompt_tokens": stats(last[mode])}),
                  flush=True)


if __name__ == "__main__":
    main()
