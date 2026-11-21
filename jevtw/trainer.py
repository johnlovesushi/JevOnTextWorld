"""One GRPO update from groups of complete episodes (section 3).

1. R_i = z_i + lam * b_i with b_i from the rollout-time forecasts (eq. 6); advantages within each game (eq. 7).
2. Fixed old-policy log-probabilities: every decision is re-scored once under theta_old with the same forward used
   for training, so the first ratio is exactly 1 (the sampling-time log-probs are kept only for diagnostics).
3. `optimization_passes` passes. Each pass accumulates the gradient decision by decision with weights fixed in
   advance, so the result equals mean over games [L_GRPO + alpha L_cal + beta L_KL] (eqs. 8-9), then steps."""
from __future__ import annotations

import math
from typing import Sequence

import torch

from .config import GRPOConfig
from .losses import calibration_weights, decision_loss, policy_weights
from .policy import LMPolicy
from .rewards import brier_bonus, episode_return, group_advantages
from .rollout import Episode


def assign_returns(groups: Sequence[Sequence[Episode]], cfg: GRPOConfig) -> None:
    for group in groups:
        for ep in group:
            if cfg.lam > 0:
                forecasts = [d.p for d in ep.decisions]
                if any(p is None for p in forecasts):
                    raise ValueError("lam > 0 needs rollout forecasts (PolicyConfig.forecast = True)")
                ep.bonus = brier_bonus(forecasts, ep.z)
            elif ep.decisions[0].p is not None:
                ep.bonus = brier_bonus([d.p for d in ep.decisions], ep.z)   # logged only
            ep.ret = episode_return(ep.z, ep.bonus if cfg.lam > 0 else 0.0, cfg.lam)
        for ep, a in zip(group, group_advantages([ep.ret for ep in group], cfg.adv_delta)):
            ep.advantage = a


class Trainer:
    def __init__(self, policy: LMPolicy, cfg: GRPOConfig, reference_model=None, optimizer=None):
        if cfg.beta > 0 and reference_model is None:
            raise ValueError("beta > 0 needs a frozen reference model")
        self.policy, self.cfg, self.reference = policy, cfg, reference_model
        params = [p for p in policy.model.parameters() if p.requires_grad]
        self.optimizer = optimizer or torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def _active(self, ep: Episode) -> bool:
        return ep.advantage != 0 or self.cfg.alpha > 0 or self.cfg.beta > 0

    @torch.no_grad()
    def _fix_old(self, groups):
        max_diff = 0.0
        for group in groups:
            for ep in group:
                if not self._active(ep):
                    continue
                for d in ep.decisions:
                    old, _ = self.policy.score(d)
                    d.old_logps = old.tolist()
                    max_diff = max(max_diff, max(abs(a - b) for a, b in zip(d.old_logps, d.behavior_logps)))
                    if self.cfg.beta > 0:
                        ref, _ = self.policy.score(d, model=self.reference)
                        d.ref_logps = ref.tolist()
        return max_diff

    def step(self, groups: Sequence[Sequence[Episode]]) -> dict:
        cfg = self.cfg
        assign_returns(groups, cfg)
        counts = [[[d.n_terms for d in ep.decisions] for ep in group] for group in groups]
        w_pol = policy_weights(counts, cfg.normalization)
        w_cal = calibration_weights(counts)
        max_diff = self._fix_old(groups)
        device = self.policy.device
        passes = []
        for _ in range(cfg.optimization_passes):
            self.optimizer.zero_grad(set_to_none=True)
            total, clipped, terms, kl, cal = 0.0, 0, 0, 0.0, 0.0
            for g, group in enumerate(groups):
                for i, ep in enumerate(group):
                    if not self._active(ep):
                        continue
                    for t, d in enumerate(ep.decisions):
                        logps, p = self.policy.score(d, forecast=cfg.alpha > 0)
                        old = torch.tensor(d.old_logps, device=device)
                        ref = torch.tensor(d.ref_logps, device=device) if cfg.beta > 0 else None
                        loss, stats = decision_loss(logps, old, ep.advantage, w_pol[g][i][t], cfg.clip_eps,
                                                    ref_logps=ref, beta=cfg.beta, p=p, z=float(ep.z),
                                                    cal_weight=w_cal[g][i][t], alpha=cfg.alpha)
                        loss.backward()
                        total += float(loss.detach())
                        clipped += stats["clipped"]
                        terms += stats["terms"]
                        kl += stats.get("kl", 0.0) * w_pol[g][i][t]
                        cal += stats.get("brier", 0.0) * w_cal[g][i][t]
            params = [q for q in self.policy.model.parameters() if q.grad is not None]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)) if params else 0.0
            if params and math.isfinite(grad_norm):
                self.optimizer.step()
            passes.append({"loss": total, "grad_norm": grad_norm, "clip_frac": clipped / max(terms, 1),
                           "kl": kl, "cal_loss": cal})
        eps = [ep for group in groups for ep in group]
        forecasts = [d.p for ep in eps for d in ep.decisions if d.p is not None]
        return {
            "win_rate": sum(ep.z for ep in eps) / len(eps),
            "mean_return": sum(ep.ret for ep in eps) / len(eps),
            "mean_bonus": (sum(ep.bonus for ep in eps) / len(eps)) if eps[0].bonus is not None else None,
            "mean_length": sum(ep.length for ep in eps) / len(eps),
            "mixed_groups": sum(len({ep.z for ep in group}) > 1 for group in groups),
            "active_groups": sum(any(ep.advantage != 0 for ep in group) for group in groups),
            "mean_forecast": (sum(forecasts) / len(forecasts)) if forecasts else None,
            "sampling_vs_old_max_abs_logp_diff": max_diff,
            "passes": passes,
        }
