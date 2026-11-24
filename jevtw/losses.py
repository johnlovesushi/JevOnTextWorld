"""Clipped GRPO loss (eq. 8), calibration loss (eq. 9: Brier, or cross-entropy) and the optional reference KL.

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
import torch.nn.functional as F

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
    """Weight of each decision's calibration term: 1 / (sum_i T_i * number of games)."""
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


def tis_weights(old_logp: torch.Tensor, behavior_logp: torch.Tensor, cap: float) -> torch.Tensor:
    """Truncated importance weights min(pi_old / mu, cap) per term: mu is the probability the term was sampled
    with, pi_old the one the training forward assigns to it under the same parameters."""
    return torch.exp(old_logp - behavior_logp).clamp(max=cap)


def brier(p: torch.Tensor, z: float) -> torch.Tensor:
    return (p - z) ** 2


def bce(logit: torch.Tensor, z: float) -> torch.Tensor:
    """-z log p - (1 - z) log(1 - p) with p = sigmoid(logit), computed from the logit (stable for large |logit|).
    Its gradient with respect to the logit is p - z; that of the Brier score is 2 (p - z) p (1 - p)."""
    return F.binary_cross_entropy_with_logits(logit, torch.as_tensor(float(z), dtype=logit.dtype, device=logit.device))


def calibration_loss(kind: str, logit: torch.Tensor, z: float) -> torch.Tensor:
    """The calibration term of one decision from its (Platt-scaled) forecast logit."""
    if kind == "brier":
        return brier(torch.sigmoid(logit), z)
    if kind == "bce":
        return bce(logit, z)
    raise ValueError(f"unknown calibration loss {kind!r}")


def decision_loss(logps: torch.Tensor, old_logps: torch.Tensor, advantage: float, weight: float, eps: float, *,
                  ref_logps: torch.Tensor | None = None, beta: float = 0.0,
                  p: torch.Tensor | None = None, z: float | None = None, cal_weight: float = 0.0,
                  alpha: float = 0.0, tis_weights: torch.Tensor | None = None,
                  cal_loss: str = "brier", logit: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """One decision's contribution to L. logps / old_logps / ref_logps: (|c_t| + 1,) FP32, the last entry being the
    action. `tis_weights` (same shape, constants) multiply the policy terms only, never the KL or the calibration
    term. The calibration term is `cal_loss` of the forecast: "brier" takes the probability `p`, "bce" the
    (Platt-scaled) `logit`. Returns the weighted scalar and unweighted diagnostics."""
    surrogate = clipped_surrogate(logps, old_logps, advantage, eps)
    if tis_weights is not None:
        surrogate = surrogate * tis_weights.detach()
    loss = weight * surrogate.sum()
    with torch.no_grad():
        ratio = torch.exp(logps - old_logps)
        stats = {"policy": float(loss.detach()),
                 "clipped": int(((ratio < 1 - eps) | (ratio > 1 + eps)).sum()), "terms": int(logps.numel()),
                 # k3 with r = pi_theta / pi_old: the terms were sampled from (about) pi_old, so the mean of this
                 # over terms estimates KL(pi_old || pi_theta). Exactly 0 while theta = theta_old.
                 "kl_old": float(kl_k3(old_logps, logps).sum())}
    if beta > 0:
        if ref_logps is None:
            raise ValueError("beta > 0 needs reference log-probabilities")
        kl = kl_k3(logps, ref_logps)
        loss = loss + beta * weight * kl.sum()
        stats["kl"] = float(kl.detach().sum())
    if alpha > 0:
        if cal_loss == "brier":
            if p is None or z is None:
                raise ValueError("alpha > 0 needs the forecast p and the outcome z")
            cal = brier(p, z)
        elif cal_loss == "bce":
            if logit is None or z is None:
                raise ValueError("alpha > 0 with cal_loss = 'bce' needs the forecast logit and the outcome z")
            cal = bce(logit, z)
        else:
            raise ValueError(f"unknown calibration loss {cal_loss!r}")
        loss = loss + alpha * cal_weight * cal
        stats["cal"] = float(cal.detach())
        if cal_loss == "brier":
            stats["brier"] = stats["cal"]
    return loss, stats
