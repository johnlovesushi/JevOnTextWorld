"""Training loop: sample games, roll out G episodes per game, one GRPO update per batch, periodic evaluation.

  # TextWorld (GPU): games made by scripts/make_games.py
  python -m jevtw.train --config configs/grpo.json --model Qwen/Qwen3-1.7B \
      --train-games data/games/train --val-games data/games/valid --out runs/grpo
  # CPU smoke run: tiny random model, byte tokenizer, KeyChest POMDP
  python -m jevtw.train --smoke --out runs/smoke

  # Evaluate a checkpoint only, and dump every forecast
  python -m jevtw.train --config <the run's config> --model runs/<ID>/checkpoints/batch0040 --batches 0 \
      --val-games data/games/valid --train-games data/games/train --save-forecasts --out runs/<new ID>

Outputs in --out: config.json, metrics.jsonl (one line per batch / evaluation), checkpoints/batchNNNN/ (model,
tokenizer and, with Platt scaling, platt.json), and with --save-forecasts forecasts/batchNNNN.jsonl.
With --wandb-project the same lines also go to Weights & Biases (optional; wandb is only imported then)."""
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
from .diagnostics import forecast_metrics, write_forecasts
from .envs import KeyChestEnv, TextWorldEnv
from .policy import PLATT_FILE, LMPolicy
from .rollout import collect_batch, rollout_metrics
from .trainer import Trainer

GAME_SUFFIXES = (".z8", ".ulx")
WANDB_INIT_TIMEOUT = 60                   # seconds an online wandb.init may take before falling back to offline


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


def evaluate(policy: LMPolicy, games, seeds, max_actions: int, parallel: int = 1,
             forecasts_path: Path | None = None) -> dict:
    """One sampled episode per (game, seed). Reports the win rate, how well the forecast predicted z
    (`diagnostics.forecast_metrics`) and sampling speed; writes every forecast to `forecasts_path` when given."""
    jobs = [(name, make_env, env_seed, s) for name, make_env, env_seed in games for s in seeds]
    stats, start = {}, time.time()
    episodes = collect_batch(policy, jobs, max_actions, parallel, stats)
    sampling = rollout_metrics(episodes, stats, time.time() - start, policy.eos_id)
    if forecasts_path is not None:
        write_forecasts(forecasts_path, episodes)
    return {"episodes": len(episodes), "win_rate": sum(ep.z for ep in episodes) / len(episodes),
            "mean_length": sum(ep.length for ep in episodes) / len(episodes),
            **forecast_metrics(episodes), **sampling}


def save_checkpoint(directory: Path, model, tokenizer, policy: LMPolicy) -> None:
    """Model weights, the tokenizer (so that the directory can be given to --model on its own) and platt.json."""
    model.save_pretrained(directory)
    if hasattr(tokenizer, "save_pretrained"):         # the smoke run's ByteTokenizer has nothing to save
        tokenizer.save_pretrained(directory)
    policy.save_platt(directory)


def wandb_record(record: dict) -> dict:
    """One metrics.jsonl line as a flat W&B row. Evaluation lines get the prefix "val/"; the per-pass list of a
    training line becomes pass1/<key> (first pass) and pass_last/<key>. The step is record["batch"]."""
    if record.get("split") == "val":
        return {f"val/{k}": v for k, v in record.items() if k not in ("batch", "split")}
    row = {k: v for k, v in record.items() if k not in ("batch", "split", "passes")}
    passes = record.get("passes") or []
    if passes:
        row.update({f"pass1/{k}": v for k, v in passes[0].items()})
        row.update({f"pass_last/{k}": v for k, v in passes[-1].items()})
    return row


def start_wandb(project: str, name: str, mode: str, config: dict, directory: Path):
    """wandb.init that cannot stop the training: when the online initialisation fails (no network, no login,
    timeout), warn and continue with an offline run (`wandb sync` uploads it later)."""
    import wandb  # optional dependency, imported only when --wandb-project is given

    kwargs = dict(project=project, name=name, config=config, dir=str(directory))
    if mode == "online":
        try:
            return wandb.init(mode="online", settings=wandb.Settings(init_timeout=WANDB_INIT_TIMEOUT), **kwargs)
        except Exception as error:  # noqa: BLE001 - whatever went wrong, training goes on
            print(f"WARNING: wandb online init failed ({type(error).__name__}: {error}); logging offline instead",
                  flush=True)
    return wandb.init(mode="offline", **kwargs)


def load(args, cfg: Config):
    if args.smoke:
        from .testing import ByteTokenizer, tiny_model

        cfg.policy.yes_text, cfg.policy.no_text = "Y", "N"
        cfg.policy.confidence_prompt = "\nWill you win? Y or N:"
        cfg.policy.chat_yes_text, cfg.policy.chat_no_text = "Y", "N"
        if Path(args.model).is_dir():     # a checkpoint of an earlier smoke run
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval(), ByteTokenizer()
        return tiny_model(seed=args.seed), ByteTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype).to(device)
    model.eval()                          # dropout off: sampling and training forwards use the same function
    # --revision names a revision of --model; a tokenizer given separately is loaded at its default revision.
    if args.tokenizer:
        return model, AutoTokenizer.from_pretrained(args.tokenizer)
    return model, AutoTokenizer.from_pretrained(args.model, revision=args.revision)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON with optional 'policy' and 'grpo' sections")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision")
    ap.add_argument("--tokenizer", help="tokenizer name or directory (default: the same as --model)")
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
    ap.add_argument("--save-forecasts", action="store_true",
                    help="every evaluation writes forecasts/batchNNNN.jsonl: one line per episode with s and p of "
                         "every step")
    ap.add_argument("--wandb-project", help="also log to this Weights & Biases project (off when not given)")
    ap.add_argument("--wandb-run-name", help="default: the name of the --out directory")
    ap.add_argument("--wandb-mode", choices=("online", "offline"), default="online",
                    help="online falls back to offline when wandb.init fails")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    if args.smoke:
        cfg.policy.max_reasoning_tokens = min(cfg.policy.max_reasoning_tokens, 8)
        cfg.grpo.games_per_batch, cfg.grpo.group_size, cfg.grpo.max_actions = 2, 4, 4
        args.batches, args.eval_every, args.eval_seeds = min(args.batches, 3), 2, [0, 1]
        if not Path(args.model).is_dir():
            args.model = "tiny-random-llama (smoke)"
    # A checkpoint with platt.json was trained with Platt scaling: its forecasts need a and b.
    platt_file = Path(args.model) / PLATT_FILE
    if platt_file.is_file() and not cfg.policy.platt:
        print(f"{platt_file} found: turning PolicyConfig.platt on", flush=True)
        cfg.policy.platt = True
    model, tokenizer = load(args, cfg)
    if args.smoke:
        train_games, val_games = keychest_games(range(8)), keychest_games(range(100, 104))
    else:
        if not args.train_games and args.batches > 0:
            ap.error("--train-games is required unless --smoke or --batches 0")
        train_games = textworld_games(args.train_games) if args.train_games else []
        val_games = textworld_games(args.val_games) if args.val_games else []
    if args.val_limit is not None:
        val_games = val_games[:args.val_limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_config = run_metadata(cfg, args)
    (out / "config.json").write_text(json.dumps(run_config, indent=2))
    metrics_file = (out / "metrics.jsonl").open("a")
    wandb_run = None
    if args.wandb_project:
        wandb_run = start_wandb(args.wandb_project, args.wandb_run_name or out.resolve().name, args.wandb_mode,
                                run_config, out)

    def log(record):
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        print(json.dumps(record), flush=True)
        if wandb_run is not None:
            wandb_run.log(wandb_record(record), step=record["batch"])

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    reference = None
    if cfg.grpo.beta > 0:
        reference = copy.deepcopy(model).requires_grad_(False)
    policy = LMPolicy(model, tokenizer, cfg.policy)
    if policy.load_platt(args.model):
        print(f"loaded Platt parameters a = {float(policy.platt_a):.6g}, b = {float(policy.platt_b):.6g}", flush=True)
    trainer = Trainer(policy, cfg.grpo, reference_model=reference)
    g = cfg.grpo

    def run_eval(batch):
        if val_games:
            start = time.time()
            path = out / "forecasts" / f"batch{batch:04d}.jsonl" if args.save_forecasts else None
            result = evaluate(policy, val_games, args.eval_seeds, g.max_actions, g.parallel_episodes, path)
            log({"batch": batch, "split": "val", "eval_s": round(time.time() - start, 2), **result})

    try:
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
                if not args.no_save:
                    save_checkpoint(out / "checkpoints" / f"batch{batch:04d}", model, tokenizer, policy)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    metrics_file.close()


if __name__ == "__main__":
    main()
