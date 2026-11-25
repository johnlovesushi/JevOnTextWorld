"""Supervised fine-tuning on expert states (T0007): maximise log q(a* | h_t) on the states of the walkthroughs.

  python scripts/build_sft_data.py --games data/games_C1/train --out data/sft/C1_train.jsonl
  python -m jevtw.sft --config configs/experiments/<ID>.json --model Qwen/Qwen3-1.7B \
      --train-data data/sft/C1_train.jsonl --val-games data/games_C1/valid --out runs/<ID>
  # CPU smoke run: tiny random model, byte tokenizer, the KeyChest expert
  python -m jevtw.sft --smoke --out runs/sft_smoke

The loss of one state is -log q(a*), where q = softmax over the option scores l(a^j) of
`LMPolicy.decision_forward`: the same action distribution the rollouts sample from and GRPO trains. There is no
reasoning (the configuration must have max_reasoning_tokens = 0) and the forecast p_t is not trained. The prompt of
a state is `build_prompt` under the run's PolicyConfig, built from the raw material in the data file, so one data
file serves every history_mode / history_window / prompt_format.

One optimizer step = `--batch-decisions` states: the gradients of loss / (number of states in the step) are
accumulated state by state, as `Trainer.step` does, then clipped to GRPOConfig.grad_clip. The learning rate of step
k (from 1) is lr * min(1, k / ceil(warmup_frac * total steps)), constant afterwards.

Outputs in --out: config.json, metrics.jsonl (per epoch one train line with sft_loss and train_acc, the means over
the states as they were visited, and one val line from `train.evaluate`), checkpoints/epochNN/ (model and
tokenizer; a directory that `jevtw.train --model` and scripts/final_eval.py load).

Several GPUs (T0009): `torchrun --standalone --nproc-per-node=2 -m jevtw.sft ...`. Every process builds the same
steps (same seed); the states of a step are split by token count, the gradients summed, then every process clips and
steps the same way. Evaluations are split by episodes. Only process 0 writes.

Limitation: the data holds only states the expert visits. A policy trained on it has never seen the states that
its own mistakes lead to, so errors compound during a rollout; the training accuracy is an upper bound on nothing
about the win rate."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from . import dist as dist_mod
from .config import Config
from .envs import KeyChestEnv, Observation
from .policy import Decision, LMPolicy
from .train import evaluate, keychest_games, load, run_metadata, textworld_games


# ---------------------------------------------------------------- data

def state_row(game: str, t: int, objective: str, history, obs: Observation, command: str) -> dict:
    """One expert state as plain JSON: everything `build_prompt` can use, and the index of a* among the options."""
    return {"game": game, "t": t, "objective": objective,
            "history": [{"command": c, "obs": asdict(o)} for c, o in history],
            "obs": asdict(obs), "a_star": obs.commands.index(command), "command": command}


def row_state(row: dict):
    """(objective, history, obs, a_star) with `Observation` objects again."""
    history = [(h["command"], Observation(**h["obs"])) for h in row["history"]]
    return row["objective"], history, Observation(**row["obs"]), row["a_star"]


def read_rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def keychest_expert_rows(seeds) -> list[dict]:
    """The KeyChest expert: open the container that holds the key, take the key, open the chest. It reads the
    hidden state for its first command, which the observation does not determine: on that state no policy can
    do better than the more frequent container."""
    rows = []
    for seed in seeds:
        env = KeyChestEnv()
        obs = env.reset(seed)
        history = []
        for t, command in enumerate((f"open {env._state['key_in']}", "take key", "open chest")):
            rows.append(state_row(f"keychest-{seed}", t, env.objective, list(history), obs, command))
            obs = env.step(command)
            history.append((command, obs))
        assert obs.won
    return rows


# ---------------------------------------------------------------- loss

def expert_decision(policy: LMPolicy, objective: str, history, obs: Observation, a_star: int) -> Decision:
    """The decision `LMPolicy.decide` builds for this state without reasoning, with a_t = a*."""
    prompt_ids = policy.encode_prompt(objective, history, obs)
    options = [policy.encode(f" {j + 1}") + [policy.eos_id] for j in range(len(obs.commands))]
    return Decision(command=obs.commands[a_star], action=a_star, prompt_ids=prompt_ids, reasoning_ids=[],
                    action_context_ids=policy._action_context(prompt_ids, []), option_ids=options, behavior_logps=[])


def sft_loss(policy: LMPolicy, d: Decision) -> tuple[torch.Tensor, bool]:
    """(-log q(a*), is a* the most likely option) for one state; differentiable."""
    _, scores, _ = policy.decision_forward(policy.model, d, forecast=False)
    log_q = torch.log_softmax(scores.float(), dim=0)
    return -log_q[d.action], int(torch.argmax(log_q)) == d.action


def warmup_steps(total_steps: int, warmup_frac: float) -> int:
    return math.ceil(warmup_frac * total_steps)


def lr_factor(step: int, warmup: int) -> float:
    """Factor of optimizer step `step` (from 1)."""
    return min(1.0, step / warmup) if warmup > 0 else 1.0


class SFTTrainer:
    def __init__(self, policy: LMPolicy, lr: float, weight_decay: float, grad_clip: float, total_steps: int,
                 warmup_frac: float, dist: dist_mod.Dist | None = None):
        self.policy, self.lr, self.grad_clip = policy, lr, grad_clip
        self.dist = dist if dist is not None else dist_mod.Dist()
        self.params = [p for p in policy.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay)
        self.warmup = warmup_steps(total_steps, warmup_frac)
        self.steps_done = 0

    def step(self, decisions: list[Decision]) -> dict:
        """One optimizer step on the mean loss of `decisions`."""
        factor = lr_factor(self.steps_done + 1, self.warmup)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr * factor
        self.optimizer.zero_grad(set_to_none=True)
        total, correct = 0.0, 0
        owner = dist_mod.balance([len(d.action_context_ids) + sum(len(o) for o in d.option_ids) for d in decisions],
                                 self.dist.world)
        for d, r in zip(decisions, owner):
            if r != self.dist.rank:
                continue
            loss, hit = sft_loss(self.policy, d)
            (loss / len(decisions)).backward()
            total += float(loss.detach())
            correct += hit
        if self.dist.enabled:
            self.dist.all_reduce_grads(self.params)
            total, correct = self.dist.sum_floats([total, correct])
            correct = int(correct)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip))
        if math.isfinite(grad_norm):
            self.optimizer.step()
        self.steps_done += 1
        return {"loss_sum": total, "correct": correct, "n": len(decisions), "grad_norm": grad_norm,
                "lr": self.lr * factor}


def save_checkpoint(directory: Path, model, tokenizer) -> None:
    """Model and tokenizer. No platt.json: SFT does not train the forecast, and a platt.json with a = 1, b = 0
    would make a later GRPO run skip the data-dependent start of b."""
    model.save_pretrained(directory)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(directory)


# ---------------------------------------------------------------- entry point

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON with optional 'policy' and 'grpo' sections (the prompt, H, grad_clip)")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision")
    ap.add_argument("--tokenizer", help="tokenizer name or directory (default: the same as --model)")
    ap.add_argument("--train-data", help="jsonl of scripts/build_sft_data.py")
    ap.add_argument("--val-games")
    ap.add_argument("--val-limit", type=int, help="evaluate on the first N validation games only")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-decisions", type=int, default=32, help="states per optimizer step")
    ap.add_argument("--warmup-frac", type=float, default=0.1, help="share of all steps with a linear warmup")
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[17, 23, 29])
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-save", action="store_true", help="do not write checkpoints")
    ap.add_argument("--smoke", action="store_true", help="tiny random model on the KeyChest expert, CPU")
    args = ap.parse_args(argv)
    dist = dist_mod.setup()
    try:
        _run(ap, args, dist)
    finally:
        dist_mod.teardown()


def _run(ap, args, dist: dist_mod.Dist):
    if args.batch_decisions < 1 or args.epochs < 0 or not 0.0 <= args.warmup_frac <= 1.0:
        ap.error("need --batch-decisions >= 1, --epochs >= 0 and 0 <= --warmup-frac <= 1")

    cfg = Config.load(args.config) if args.config else Config()
    if args.smoke:
        cfg.policy.max_reasoning_tokens, cfg.grpo.max_actions = 0, 4
        args.eval_seeds = [0, 1]
        if not Path(args.model).is_dir():
            args.model = "tiny-random-llama (smoke)"
    if cfg.policy.max_reasoning_tokens != 0:
        ap.error("SFT trains the action after the prompt only: the configuration needs policy.max_reasoning_tokens = 0")
    out = Path(args.out)
    if (out / "metrics.jsonl").exists():
        ap.error(f"{out} already holds a run")
    model, tokenizer = load(args, cfg)
    if args.smoke:
        rows = read_rows(args.train_data) if args.train_data else keychest_expert_rows(range(8))
        val_games = keychest_games(range(100, 104))
    else:
        if not args.train_data:
            ap.error("--train-data is required unless --smoke")
        rows = read_rows(args.train_data)
        val_games = textworld_games(args.val_games) if args.val_games else []
    if args.val_limit is not None:
        val_games = val_games[:args.val_limit]
    if not rows:
        ap.error("no training states")

    steps_per_epoch = math.ceil(len(rows) / args.batch_decisions)
    total_steps = steps_per_epoch * args.epochs
    if dist.main:
        out.mkdir(parents=True, exist_ok=True)
    run_config = run_metadata(cfg, args)
    run_config["meta"]["world_size"] = dist.world
    run_config["sft"] = {"states": len(rows), "games": len({r["game"] for r in rows}),
                         "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                         "warmup_steps": warmup_steps(total_steps, args.warmup_frac)}
    metrics_file = None
    if dist.main:
        (out / "config.json").write_text(json.dumps(run_config, indent=2))
        metrics_file = (out / "metrics.jsonl").open("a")

    def log(record):
        if not dist.main:
            return
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        print(json.dumps(record), flush=True)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    policy = LMPolicy(model, tokenizer, cfg.policy)
    trainer = SFTTrainer(policy, args.lr, args.weight_decay, cfg.grpo.grad_clip, total_steps, args.warmup_frac,
                         dist=dist)
    g = cfg.grpo
    for epoch in range(1, args.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        order = list(range(len(rows)))
        rng.shuffle(order)
        loss_sum, correct, norms, lr = 0.0, 0, [], None
        for i in range(0, len(order), args.batch_decisions):
            decisions = [expert_decision(policy, *row_state(rows[j])) for j in order[i:i + args.batch_decisions]]
            stats = trainer.step(decisions)
            loss_sum += stats["loss_sum"]
            correct += stats["correct"]
            norms.append(stats["grad_norm"])
            lr = stats["lr"]
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        log({"batch": epoch, "epoch": epoch, "split": "train", "sft_loss": loss_sum / len(rows),
             "train_acc": correct / len(rows), "decisions": len(rows), "steps": trainer.steps_done,
             "grad_norm": sum(norms) / len(norms), "max_grad_norm": max(norms), "lr": lr,
             "update_s": round(time.time() - start, 2), "gpu_peak_gb": None if peak is None else round(peak, 2)})
        if val_games:
            start = time.time()
            result = evaluate(policy, val_games, args.eval_seeds, g.max_actions, g.parallel_episodes, dist=dist)
            log({"batch": epoch, "epoch": epoch, "split": "val", "eval_s": round(time.time() - start, 2), **result})
        if not args.no_save and dist.main:
            save_checkpoint(out / "checkpoints" / f"epoch{epoch:02d}", model, tokenizer)
        dist.barrier()
    if metrics_file is not None:
        metrics_file.close()


if __name__ == "__main__":
    main()
