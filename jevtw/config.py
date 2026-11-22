"""Configuration. Defaults reproduce plain multi-step GRPO (lam = alpha = 0, note section 3)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

NORMALIZATIONS = ("token", "episode")


@dataclass
class PolicyConfig:
    """How the language model sees a step and how c_t, a_t and p_t are produced (eqs. 4-5)."""

    max_reasoning_tokens: int = 320      # |c_t| cap; 0 = choose directly after the prompt
    reasoning_prefix: str = "Reasoning:"
    answer_prefix: str = "Answer:"        # sampling c_t stops once the reasoning ends with this, or at EOS
    history_window: int = 8               # past (command, observation) pairs shown; h_t is truncated to this
    # p_t = sigmoid(l(yes) - l(no)) read after this question; it is never written into the history.
    confidence_prompt: str = "\nQuestion: will you complete the objective? Answer yes or no:"
    yes_text: str = " yes"                # must encode to exactly one token
    no_text: str = " no"
    forecast: bool = True                 # compute p_t during rollouts (needed when lam > 0, logged otherwise)
    # Ignored since T0001 (all options are scored in one suffix batch on a shared prefix); kept so that old
    # configuration files still load.
    option_batch_size: int = 16


@dataclass
class GRPOConfig:
    """Trajectory-level GRPO plus calibration (eqs. 6-9)."""

    games_per_batch: int = 4
    group_size: int = 8                   # G, rollouts per game from the same initial state
    max_actions: int = 50                 # H
    lam: float = 0.0                      # lambda in R_i = z_i + lambda * b_i, 0 <= lambda < 1
    alpha: float = 0.0                    # weight of the Brier loss L_cal
    beta: float = 0.0                     # weight of the reference-policy KL (optional)
    clip_eps: float = 0.2                 # epsilon in eq. (8)
    adv_delta: float = 1e-8               # delta in eq. (7)
    # "token": divide by N = sum_{i,t}(|c_t^i| + 1) within each game, as in eq. (8).
    # "episode": average the terms within each episode, then over the G episodes (rlcd-bench baselines).
    normalization: str = "token"
    optimization_passes: int = 2
    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.lam < 1.0:
            raise ValueError(f"lam must satisfy 0 <= lam < 1 (eq. 6), got {self.lam}")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {self.normalization!r}")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for group-relative advantages")


@dataclass
class Config:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        def build(kind, values):
            known = {f.name for f in fields(kind)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"unknown {kind.__name__} keys: {sorted(unknown)}")
            return kind(**values)

        return cls(policy=build(PolicyConfig, data.get("policy", {})), grpo=build(GRPOConfig, data.get("grpo", {})))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)
