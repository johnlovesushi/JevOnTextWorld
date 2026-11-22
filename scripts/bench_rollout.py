"""Sampling speed of jevtw.rollout.collect_batch: one training batch of rollouts (no update) per --parallel value.

  python scripts/bench_rollout.py --config configs/experiments/E0001a.json --train-games data/games/train \
      --parallel 1 8 32

The games and sample seeds are drawn exactly as jevtw.train draws them for its first batch with the same --seed,
and are the same for every --parallel value. Prints one JSON line per value: seconds per decision, the time
split, reasoning statistics, win rate and peak GPU memory."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevtw.config import Config  # noqa: E402
from jevtw.policy import LMPolicy  # noqa: E402
from jevtw.rollout import collect_batch, rollout_metrics  # noqa: E402
from jevtw.train import textworld_games  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--train-games", required=True)
    ap.add_argument("--parallel", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = Config.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype).to(device).eval()
    policy = LMPolicy(model, tokenizer, cfg.policy)
    games, g = textworld_games(args.train_games), cfg.grpo
    base = torch.cuda.memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
    print(json.dumps({"model": args.model, "device": str(device), "dtype": str(dtype), "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                      "model_gib": round(base, 3), "config": args.config, "seed": args.seed,
                      "episodes": g.games_per_batch * g.group_size, "max_actions": g.max_actions,
                      "max_reasoning_tokens": cfg.policy.max_reasoning_tokens}), flush=True)

    for parallel in args.parallel:
        rng = random.Random(args.seed)                  # the same jobs for every value
        jobs = []
        for name, make_env, env_seed in (rng.choice(games) for _ in range(g.games_per_batch)):
            jobs += [(name, make_env, env_seed, rng.randrange(2 ** 31)) for _ in range(g.group_size)]
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        row = {"parallel": parallel}
        stats, start = {}, time.time()
        try:
            episodes = collect_batch(policy, jobs, g.max_actions, parallel, stats)
            wall = time.time() - start
            row.update(wall_s=round(wall, 2), win_rate=sum(ep.z for ep in episodes) / len(episodes),
                       mean_length=sum(ep.length for ep in episodes) / len(episodes),
                       max_context_tokens=max(len(d.action_context_ids) for ep in episodes for d in ep.decisions),
                       **rollout_metrics(episodes, stats, wall, policy.eos_id))
        except torch.OutOfMemoryError as e:
            row["error"] = str(e)[:300]
        if device.type == "cuda":
            row["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
