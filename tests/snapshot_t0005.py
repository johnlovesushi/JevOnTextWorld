"""Numbers of the default configuration on a fixed tiny scenario. tests/snapshots/t0005_default.json was written by this
file at the commit before T0005 (python tests/snapshot_t0005.py); test_t0005.py recomputes and compares."""
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevtw.config import GRPOConfig, PolicyConfig  # noqa: E402
from jevtw.envs import KeyChestEnv  # noqa: E402
from jevtw.policy import LMPolicy  # noqa: E402
from jevtw.rollout import collect_batch  # noqa: E402
from jevtw.testing import ByteTokenizer, tiny_model  # noqa: E402
from jevtw.trainer import Trainer  # noqa: E402

PATH = Path(__file__).parent / "snapshots" / "t0005_default.json"


def compute() -> dict:
    cfg = PolicyConfig(max_reasoning_tokens=6, history_window=3, yes_text="Y", no_text="N",
                       confidence_prompt="\nWill you win? Y or N:")
    model = tiny_model(seed=0)
    policy = LMPolicy(model, ByteTokenizer(), cfg)
    grpo = GRPOConfig(games_per_batch=2, group_size=4, max_actions=4, lam=0.3, alpha=0.1, beta=0.04,
                      learning_rate=1e-3, parallel_episodes=4)
    jobs = [(f"keychest-{g}", KeyChestEnv, g, 1000 + 10 * g + i) for g in range(2) for i in range(4)]
    episodes = collect_batch(policy, jobs, grpo.max_actions, grpo.parallel_episodes)
    decisions = [d for ep in episodes for d in ep.decisions]
    out = {"commands": [d.command for d in decisions],
           "option_scores": [policy.decision_forward(model, d, forecast=True)[1].tolist() for d in decisions],
           "behavior_logps": [d.behavior_logps for d in decisions],
           "p": [d.p for d in decisions], "p_options": [d.p_options for d in decisions]}
    with torch.no_grad():
        scored = [policy.score(d, forecast=True) for d in decisions]
    out["score_logps"] = [lp.tolist() for lp, _ in scored]
    out["score_p"] = [float(policy.forecast_prob(s)) for _, s in scored]      # score returns s since T0005
    trainer = Trainer(policy, grpo, reference_model=copy.deepcopy(model).requires_grad_(False))
    metrics = trainer.step([episodes[:4], episodes[4:]])
    out["passes"] = metrics["passes"]
    out["train"] = {k: metrics[k] for k in ("win_rate", "mean_return", "mean_bonus", "mean_forecast",
                                            "sampling_vs_old_max_abs_logp_diff")}
    out["param_sum"] = float(sum(p.double().sum() for p in model.parameters()))
    return out


if __name__ == "__main__":
    PATH.write_text(json.dumps(compute(), indent=1))
    print(f"wrote {PATH}")
