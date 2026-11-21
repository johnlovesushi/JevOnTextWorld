"""Clipped GRPO loss (eq. 8), Brier calibration loss (eq. 9) and the optional reference KL.

The total loss is a sum over decisions with weights fixed before optimisation, so it can be accumulated one
decision at a time (bounded memory) and still equal

    L = mean over games [ L_GRPO + alpha * L_cal + beta * L_KL ].

A "term" is one sampled decision u_{i,t,k}: a reasoning token of c_t, or the categorical action a_t. A decision
(one environment step) has |c_t| + 1 terms.

Normalisation of the policy terms within a game (`normalization`):
  token    1 / N with N = sum_{i,t} (|c_t^i| + 1), exactly eq. (8): long episodes and long rationales weigh more
  episode  1 / (G * terms of episode i): every episode weighs the same (the rlcd-bench baselines choice)
L_KL uses the same weights as L_GRPO. L_cal divides by sum_i T_i, one term per decision, as in eq. (9)."""
from __future__ import annotations

from typing import Sequence

import torch

TermCounts = Sequence[Sequence[Sequence[int]]]   # [game][episode][decision] -> |c_t| + 1


def policy_weights(term_counts: TermCounts, normalization: str = "token") -> list[list[list[float]]]:
    """Weight of every policy term of each decision, already divided by the number of games."""
    n_games = len(term_counts)
    out = []
    for game in term_counts:
        if normalization == "token":
            n = sum(sum(ep) for ep in game)
            out.append([[1.0 / (n * n_games)] * len(ep) for ep in game])
        elif normalization == "episode":
            out.append([[1.0 / (sum(ep) * len(game) * n_games)] * len(ep) for ep in game])
        else:
            raise ValueError(f"unknown normalization {normalization!r}")
    return out


def calibration_weights(term_counts: TermCounts) -> list[list[list[float]]]:
    """Weight of each decision's Brier term: 1 / (sum_i T_i * number of games)."""
    n_games = len(term_counts)
    out = []
    for game in term_counts:
        total_steps = sum(len(ep) for ep in game)
        out.append([[1.0 / (total_steps * n_games)] * len(ep) for ep in game])
    return out


def clipped_surrogate(logp: torch.Tensor, old_logp: torch.Tensor, advantage: float, eps: float) -> torch.Tensor:
    """Per-term -min(rho * A, clip(rho, 1 - eps, 1 + eps) * A), rho = exp(logp - old_logp)."""
    ratio = torch.exp(logp - old_logp)
    return -torch.minimum(ratio * advantage, ratio.clamp(1.0 - eps, 1.0 + eps) * advantage)


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-term non-negative estimator of KL(pi_theta || pi_ref): r - log r - 1 with r = pi_ref / pi_theta."""
    log_r = ref_logp - logp
    return torch.exp(log_r) - log_r - 1.0


def brier(p: torch.Tensor, z: float) -> torch.Tensor:
    return (p - z) ** 2


def decision_loss(logps: torch.Tensor, old_logps: torch.Tensor, advantage: float, weight: float, eps: float, *,
                  ref_logps: torch.Tensor | None = None, beta: float = 0.0,
                  p: torch.Tensor | None = None, z: float | None = None, cal_weight: float = 0.0,
                  alpha: float = 0.0) -> tuple[torch.Tensor, dict]:
    """One decision's contribution to L. logps / old_logps / ref_logps: (|c_t| + 1,) FP32, the last entry being the
    action. Returns the weighted scalar and unweighted diagnostics."""
    surrogate = clipped_surrogate(logps, old_logps, advantage, eps)
    loss = weight * surrogate.sum()
    with torch.no_grad():
        ratio = torch.exp(logps - old_logps)
        stats = {"policy": float(loss.detach()),
                 "clipped": int(((ratio < 1 - eps) | (ratio > 1 + eps)).sum()), "terms": int(logps.numel())}
    if beta > 0:
        if ref_logps is None:
            raise ValueError("beta > 0 needs reference log-probabilities")
        kl = kl_k3(logps, ref_logps)
        loss = loss + beta * weight * kl.sum()
        stats["kl"] = float(kl.detach().sum())
    if alpha > 0:
        if p is None or z is None:
            raise ValueError("alpha > 0 needs the forecast p and the outcome z")
        cal = brier(p, z)
        loss = loss + alpha * cal_weight * cal
        stats["brier"] = float(cal.detach())
    return loss, stats
