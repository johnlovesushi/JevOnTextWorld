"""T0007: the final evaluation of one model on a test set, with fixed sampling seeds.

    python scripts/final_eval.py --config configs/experiments/<ID>.json --model runs/<ID>/checkpoints/<ckpt> \
        --tokenizer Qwen/Qwen3-1.7B --games data/games_<level>/test --out runs/<ID>/final_test

Every game is played once per seed (default 101 102 103, never used for training or validation), sampled from the
policy exactly as `jevtw.train.evaluate` does. Use the test set only when an issue asks for a final evaluation
("最终评测"); nothing may be selected on it.

Outputs in --out (which must not hold a summary yet):
  summary.json    episodes, wins, win_rate, wilson95 (Wilson score interval of the win rate, z = 1.96; it treats the
                  episodes as independent, although the seeds share the games), mean_length (commands per episode),
                  action_types (first word of every chosen command -> count), and with forecasts forecast_auc,
                  forecast_auc_t0, forecast_brier, forecast_brier_const; also the seeds, the arguments, the
                  configuration and the code version
  episodes.jsonl  one line per episode in the --save-forecasts format (game, env_seed, sample_seed, z, length,
                  steps: t, command, s, p, s_options)
A checkpoint directory with platt.json turns Platt scaling on and loads a, b, like jevtw.train.
Several GPUs (T0009): `torchrun --standalone --nproc-per-node=2 scripts/final_eval.py ...` splits the episodes over
the processes (same episodes and seeds as one process); only process 0 writes."""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = (101, 102, 103)
Z95 = 1.959963984540054
FORECAST_KEYS = ("forecast_auc", "forecast_auc_t0", "forecast_brier", "forecast_brier_const")


def wilson(wins: int, n: int, z: float = Z95) -> list[float]:
    """Wilson score interval [low, high] of a proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(records: list[dict], metrics: dict) -> dict:
    """The numbers of summary.json from the episode records and the result of `train.evaluate`."""
    wins, n = sum(r["z"] for r in records), len(records)
    types = Counter(step["command"].split()[0] for r in records for step in r["steps"])
    out = {"episodes": n, "wins": wins, "win_rate": wins / n, "wilson95": wilson(wins, n),
           "mean_length": sum(r["length"] for r in records) / n, "games": len({r["game"] for r in records}),
           "action_types": dict(types.most_common())}
    out.update({k: metrics[k] for k in FORECAST_KEYS if metrics.get(k) is not None})
    return out


def main(argv=None) -> int:
    import torch

    from jevtw import dist as dist_mod
    from jevtw.config import Config
    from jevtw.policy import PLATT_FILE, LMPolicy
    from jevtw.train import evaluate, keychest_games, load, run_metadata, textworld_games

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON with optional 'policy' and 'grpo' sections")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision")
    ap.add_argument("--tokenizer", help="tokenizer name or directory (default: the same as --model)")
    ap.add_argument("--games", help="directory of test games")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="sampling seeds, one episode each")
    ap.add_argument("--seed", type=int, default=17, help="only used by --smoke without a checkpoint (model init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny random model (or a smoke checkpoint) on KeyChest, CPU")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if (out / "summary.json").exists() or (out / "episodes.jsonl").exists():
        ap.error(f"{out} already holds a final evaluation")
    if not args.smoke and not args.games:
        ap.error("--games is required unless --smoke")

    cfg = Config.load(args.config) if args.config else Config()
    if args.smoke:
        cfg.policy.max_reasoning_tokens = min(cfg.policy.max_reasoning_tokens, 8)
        cfg.grpo.max_actions = 4
    platt_file = Path(args.model) / PLATT_FILE
    if platt_file.is_file() and not cfg.policy.platt:
        print(f"{platt_file} found: turning PolicyConfig.platt on", flush=True)
        cfg.policy.platt = True
    dist = dist_mod.setup()               # before loading: process r puts the model on cuda:r
    model, tokenizer = load(args, cfg)
    games = keychest_games(range(200, 204)) if args.smoke else textworld_games(args.games)
    policy = LMPolicy(model, tokenizer, cfg.policy)
    if policy.load_platt(args.model):
        print(f"loaded Platt parameters a = {float(policy.platt_a):.6g}, b = {float(policy.platt_b):.6g}", flush=True)

    try:
        if dist.main:
            out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed)
        start = time.time()
        metrics = evaluate(policy, games, args.seeds, cfg.grpo.max_actions, cfg.grpo.parallel_episodes,
                           forecasts_path=out / "episodes.jsonl", dist=dist)
        if dist.main:
            write_summary(out, args, cfg, metrics, start, run_metadata)
    finally:
        dist_mod.teardown()
    return 0


def write_summary(out, args, cfg, metrics, start, run_metadata):
    records = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines() if line.strip()]
    summary = {**summarize(records, metrics), "seeds": args.seeds, "eval_s": round(time.time() - start, 2),
               "gpu_peak_gb": metrics.get("gpu_peak_gb"), **run_metadata(cfg, args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("episodes", "wins", "win_rate", "wilson95", "mean_length",
                                              "action_types", *[f for f in FORECAST_KEYS if f in summary])}, indent=1))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
