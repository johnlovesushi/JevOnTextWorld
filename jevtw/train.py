"""Training loop: sample games, roll out G episodes per game, one GRPO update per batch, periodic evaluation.

  # TextWorld (GPU): games made by scripts/make_games.py
  python -m jevtw.train --config configs/grpo.json --model Qwen/Qwen3-1.7B \
      --train-games data/games/train --val-games data/games/valid --out runs/grpo
  # CPU smoke run: tiny random model, byte tokenizer, KeyChest POMDP
  python -m jevtw.train --smoke --out runs/smoke

Outputs in --out: config.json, metrics.jsonl (one line per batch / evaluation), checkpoints/batchNNNN/."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import torch

from .config import Config
from .envs import KeyChestEnv, TextWorldEnv
from .policy import LMPolicy
from .rollout import collect_batch, rollout_metrics
from .trainer import Trainer

GAME_SUFFIXES = (".z8", ".ulx")


def textworld_games(directory: str) -> list[tuple[str, object, int]]:
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix in GAME_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no {GAME_SUFFIXES} games in {directory}")
    return [(p.name, (lambda p=p: TextWorldEnv(p)), 0) for p in paths]


def keychest_games(seeds) -> list[tuple[str, object, int]]:
    return [(f"keychest-{s}", KeyChestEnv, s) for s in seeds]


def run_metadata(cfg: Config, args) -> dict:
    """What a report needs to identify the run: code version, host, library versions, device."""
    root = Path(__file__).resolve().parents[1]

    def git(*cmd):
        try:
            return subprocess.run(["git", "-C", str(root), *cmd], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    import transformers

    meta = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "host": platform.node(),
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return {"meta": meta, "args": vars(args), **cfg.to_dict()}


def evaluate(policy: LMPolicy, games, seeds, max_actions: int, parallel: int = 1) -> dict:
    """One sampled episode per (game, seed). Reports the win rate, how well p_t forecast z, and sampling speed."""
    jobs = [(name, make_env, env_seed, s) for name, make_env, env_seed in games for s in seeds]
    stats, start = {}, time.time()
    episodes = collect_batch(policy, jobs, max_actions, parallel, stats)
    sampling = rollout_metrics(episodes, stats, time.time() - start, policy.eos_id)
    pairs = [(d.p, ep.z) for ep in episodes for d in ep.decisions if d.p is not None]
    return {"episodes": len(episodes), "win_rate": sum(ep.z for ep in episodes) / len(episodes),
            "mean_length": sum(ep.length for ep in episodes) / len(episodes),
            "forecast_brier": (sum((p - z) ** 2 for p, z in pairs) / len(pairs)) if pairs else None,
            "mean_forecast": (sum(p for p, _ in pairs) / len(pairs)) if pairs else None,
            **sampling}


def load(args, cfg: Config):
    if args.smoke:
        from .testing import ByteTokenizer, tiny_model

        cfg.policy.yes_text, cfg.policy.no_text = "Y", "N"
        cfg.policy.confidence_prompt = "\nWill you win? Y or N:"
        return tiny_model(seed=args.seed), ByteTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype).to(device)
    model.eval()                          # dropout off: sampling and training forwards use the same function
    return model, AutoTokenizer.from_pretrained(args.model, revision=args.revision)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON with optional 'policy' and 'grpo' sections")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision")
    ap.add_argument("--train-games")
    ap.add_argument("--val-games")
    ap.add_argument("--val-limit", type=int, help="evaluate on the first N validation games only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[17, 23, 29])
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-save", action="store_true", help="do not write checkpoints")
    ap.add_argument("--smoke", action="store_true", help="tiny random model on KeyChest, CPU")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    if args.smoke:
        cfg.policy.max_reasoning_tokens = min(cfg.policy.max_reasoning_tokens, 8)
        cfg.grpo.games_per_batch, cfg.grpo.group_size, cfg.grpo.max_actions = 2, 4, 4
        args.batches, args.eval_every, args.eval_seeds = min(args.batches, 3), 2, [0, 1]
        args.model = "tiny-random-llama (smoke)"
    model, tokenizer = load(args, cfg)
    if args.smoke:
        train_games, val_games = keychest_games(range(8)), keychest_games(range(100, 104))
    else:
        if not args.train_games:
            ap.error("--train-games is required unless --smoke")
        train_games = textworld_games(args.train_games)
        val_games = textworld_games(args.val_games) if args.val_games else []
    if args.val_limit is not None:
        val_games = val_games[:args.val_limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(run_metadata(cfg, args), indent=2))
    metrics_file = (out / "metrics.jsonl").open("a")

    def log(record):
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        print(json.dumps(record), flush=True)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    reference = None
    if cfg.grpo.beta > 0:
        reference = copy.deepcopy(model).requires_grad_(False)
    policy = LMPolicy(model, tokenizer, cfg.policy)
    trainer = Trainer(policy, cfg.grpo, reference_model=reference)
    g = cfg.grpo

    def run_eval(batch):
        if val_games:
            start = time.time()
            result = evaluate(policy, val_games, args.eval_seeds, g.max_actions, g.parallel_episodes)
            log({"batch": batch, "split": "val", "eval_s": round(time.time() - start, 2), **result})

    run_eval(0)
    for batch in range(1, args.batches + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        jobs = []
        for name, make_env, env_seed in (rng.choice(train_games) for _ in range(g.games_per_batch)):
            jobs += [(name, make_env, env_seed, rng.randrange(2 ** 31)) for _ in range(g.group_size)]
        stats = {}
        episodes = collect_batch(policy, jobs, g.max_actions, g.parallel_episodes, stats)
        groups = [episodes[i:i + g.group_size] for i in range(0, len(episodes), g.group_size)]
        rollout_time = time.time() - start
        metrics = trainer.step(groups)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        log({"batch": batch, "split": "train", "rollout_s": round(rollout_time, 2),
             "update_s": round(time.time() - start - rollout_time, 2),
             "reasoning_tokens": sum(len(d.reasoning_ids) for ep in episodes for d in ep.decisions),
             "gpu_peak_gb": None if peak is None else round(peak, 2),
             **rollout_metrics(episodes, stats, rollout_time, policy.eos_id), **metrics})
        if batch % args.eval_every == 0 or batch == args.batches:
            run_eval(batch)
            if not (args.smoke or args.no_save):
                model.save_pretrained(out / "checkpoints" / f"batch{batch:04d}")
    metrics_file.close()


if __name__ == "__main__":
    main()
