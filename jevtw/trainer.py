"""One GRPO update from groups of complete episodes (section 3).

1. R_i = z_i + lam * b_i with b_i from the rollout-time forecasts (eq. 6); advantages within each game (eq. 7).
2. Fixed old-policy log-probabilities: every decision is re-scored once under theta_old with the same forward used
   for training, so the first ratio is exactly 1. Old, reference and training forwards all use the sequences the
   rollout used (`forecast = policy.cfg.forecast`: with the confidence question after every option when the rollout
   read forecasts), so that the sampling-time log-probs come from the same function on the same input. Those are
   used for diagnostics and, when `tis_cap` is set, for the truncated importance weights.
3. `optimization_passes` passes. Each pass accumulates the gradient decision by decision with weights fixed in
   advance, so the result equals mean over games [L_GRPO + alpha L_cal + beta L_KL] (eqs. 8-9), then steps.
   The learning rate of batch b is learning_rate * min(1, b / warmup_batches), the same for all passes.

Calibration (T0005): L_cal is the Brier score or the cross-entropy (`cal_loss`) of p = sigmoid(a s + b), where s is
the forecast logit and a, b exist only with Platt scaling (`PolicyConfig.platt`; otherwise p = sigmoid(s)). a and b
form their own parameter group (lr `platt_lr`, no weight decay, no warmup, not part of the gradient clipping);
before the first update b is set from the first batch. `cal_grad = "platt_only"` detaches s in L_cal.

Data parallelism (T0009, `jevtw.dist`): every process gets the whole batch (gathered after sampling), computes the
returns, advantages, Platt start and weights from it, and owns a share of the active decisions (`dist.balance` by
token count). Steps 2 and 3 run on the own decisions only; the gradients are summed over the processes before the
clipping and the step, and the logged sums (loss, clipped terms, KL, ...) are summed too."""
from __future__ import annotations

import math
from typing import Sequence

import torch

from .config import GRPOConfig
from .dist import Dist, balance
from .losses import (calibration_loss, calibration_weights, conf_term_loss, decision_loss, policy_weights,
                     tis_weights)
from .policy import LMPolicy
from .rewards import brier_bonus, decoupled_conf_advantage, episode_return, group_advantages
from .rollout import Episode


def assign_returns(groups: Sequence[Sequence[Episode]], cfg: GRPOConfig) -> None:
    """R_i, b_i and the advantages. With verbal confidence (decisions carry conf_bin) b_i uses the sampled c_t
    in place of p_t, and with conf_advantage = "decoupled" every decision also gets A^c_t (`conf_adv`)."""
    for group in groups:
        for ep in group:
            verbal = ep.decisions[0].conf_bin is not None
            if verbal:
                ep.bonus = brier_bonus([d.conf for d in ep.decisions], ep.z)
                if cfg.conf_advantage == "decoupled":
                    for d in ep.decisions:
                        d.conf_adv = decoupled_conf_advantage(d.conf_logps, d.conf_bin, ep.z)
            elif cfg.lam > 0:
                forecasts = [d.p for d in ep.decisions]
                if any(p is None for p in forecasts):
                    raise ValueError("lam > 0 needs rollout forecasts (PolicyConfig.forecast = True)")
                ep.bonus = brier_bonus(forecasts, ep.z)
            elif ep.decisions[0].p is not None:
                ep.bonus = brier_bonus([d.p for d in ep.decisions], ep.z)   # logged only
            ep.ret = episode_return(ep.z, ep.bonus if cfg.lam > 0 else 0.0, cfg.lam)
        for ep, a in zip(group, group_advantages([ep.ret for ep in group], cfg.adv_delta, cfg.adv_norm)):
            ep.advantage = a


class Trainer:
    def __init__(self, policy: LMPolicy, cfg: GRPOConfig, reference_model=None, optimizer=None,
                 dist: Dist | None = None):
        if cfg.beta > 0 and reference_model is None:
            raise ValueError("beta > 0 needs a frozen reference model")
        if cfg.cal_grad == "platt_only" and not policy.cfg.platt:
            raise ValueError('cal_grad = "platt_only" needs Platt scaling (PolicyConfig.platt = True)')
        self.policy, self.cfg, self.reference = policy, cfg, reference_model
        self.dist = dist if dist is not None else Dist()
        self.verbal = policy.cfg.confidence_mode == "verbal"
        if self.verbal and cfg.tis_cap is not None:
            raise ValueError("tis_cap is not implemented for the verbal confidence term")
        params = [p for p in policy.model.parameters() if p.requires_grad]
        if optimizer is None:
            if policy.cfg.platt:
                optimizer = torch.optim.AdamW(
                    [{"params": params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
                     {"params": policy.platt_parameters(), "lr": cfg.platt_lr, "weight_decay": 0.0, "platt": True}])
            else:
                optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.optimizer = optimizer
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.batches_done = 0             # completed calls of step(); drives the warmup

    def _set_learning_rate(self) -> float:
        """Learning-rate factor of the batch being processed (batch b = batches_done + 1)."""
        warmup = self.cfg.warmup_batches
        factor = min(1.0, (self.batches_done + 1) / warmup) if warmup > 0 else 1.0
        if warmup > 0:                    # without warmup the optimizer is left untouched
            for group, base in zip(self.optimizer.param_groups, self._base_lrs):
                if not group.get("platt"):
                    group["lr"] = base * factor
        return factor

    def _conf_advantage(self, ep: Episode, d) -> float:
        return d.conf_adv if self.cfg.conf_advantage == "decoupled" else ep.advantage

    def _active(self, ep: Episode) -> bool:
        if ep.advantage != 0 or self.cfg.alpha > 0 or self.cfg.beta > 0:
            return True
        return self.verbal and self.cfg.conf_coef > 0 and any(self._conf_advantage(ep, d) != 0 for d in ep.decisions)

    def _score(self, d, model=None):
        """(logps, s, log pi^c(m_t) or None) of one decision, with the sequences the rollout used."""
        forecast = self.policy.cfg.forecast
        if self.verbal:
            logps, s, log_pc = self.policy.score_with_confidence(d, model=model, forecast=forecast)
            return logps, s, log_pc[d.conf_bin]
        logps, s = self.policy.score(d, model=model, forecast=forecast)
        return logps, s, None

    def _owners(self, groups) -> dict:
        """(g, i, t) -> process that scores this decision, for the decisions of active episodes. Cost = tokens of
        the folded forward (action context + options)."""
        keys = [(g, i, t) for g, group in enumerate(groups) for i, ep in enumerate(group) if self._active(ep)
                for t in range(len(ep.decisions))]
        extra = (len(self.policy.verbal_prompt_ids) + sum(len(b) for b in self.policy.bin_ids)) if self.verbal else 0
        costs = []
        for g, i, t in keys:
            d = groups[g][i].decisions[t]
            costs.append(len(d.action_context_ids) + sum(len(o) for o in d.option_ids) + extra)
        return dict(zip(keys, balance(costs, self.dist.world)))

    @torch.no_grad()
    def _fix_old(self, groups, owner):
        max_diff, sum_diff, terms = 0.0, 0.0, 0
        for g, group in enumerate(groups):
            for i, ep in enumerate(group):
                if not self._active(ep):
                    continue
                for t, d in enumerate(ep.decisions):
                    if owner[(g, i, t)] != self.dist.rank:
                        continue
                    old, _, conf_old = self._score(d)
                    d.old_logps = old.tolist()
                    diffs = [abs(a - b) for a, b in zip(d.old_logps, d.behavior_logps)]
                    if conf_old is not None:
                        d.conf_old_logp = float(conf_old)
                        diffs.append(abs(d.conf_old_logp - d.conf_behavior_logp))
                    max_diff, sum_diff, terms = max(max_diff, max(diffs)), sum_diff + sum(diffs), terms + len(diffs)
                    if self.cfg.beta > 0:
                        ref, _, conf_ref = self._score(d, model=self.reference)
                        d.ref_logps = ref.tolist()
                        if conf_ref is not None:
                            d.conf_ref_logp = float(conf_ref)
        (max_diff,) = self.dist.max_floats([max_diff])
        sum_diff, terms = self.dist.sum_floats([sum_diff, terms])
        return max_diff, (sum_diff / terms if terms else 0.0)

    def _init_platt(self, groups) -> None:
        """Before the first update: b = logit(clip(w, 0.05, 0.95)) - mean s, with w the decision-weighted win rate
        and the mean over all decisions of this batch. The batch was sampled with b = 0, so its stored p (used
        for the bonus b_i and the logs) are recomputed from the stored s with the new b."""
        policy = self.policy
        if not policy.cfg.platt or policy.platt_initialized:
            return
        pairs = [(d.forecast_logit, ep.z) for group in groups for ep in group for d in ep.decisions]
        if not pairs or any(s is None for s, _ in pairs):
            return
        policy.init_platt_bias(sum(z for _, z in pairs) / len(pairs), sum(s for s, _ in pairs) / len(pairs))
        with torch.no_grad():
            for group in groups:
                for ep in group:
                    for d in ep.decisions:
                        s = torch.tensor(d.forecast_logits_options, dtype=torch.float32, device=policy.device)
                        d.p_options = [float(v) for v in policy.forecast_prob(s)]
                        d.p = d.p_options[d.action]

    @torch.no_grad()
    def _rollout_cal_loss(self, groups, w_cal) -> float | None:
        """The calibration loss in use (`cal_loss`) on the rollout-time logits with the current a, b, weighted
        like L_cal. Logged for every batch, also when alpha = 0."""
        total = 0.0
        for g, group in enumerate(groups):
            for i, ep in enumerate(group):
                for t, d in enumerate(ep.decisions):
                    if d.forecast_logit is None:
                        return None
                    logit = self.policy.calibrated_logit(torch.tensor(d.forecast_logit, dtype=torch.float32,
                                                                      device=self.policy.device))
                    total += w_cal[g][i][t] * float(calibration_loss(self.cfg.cal_loss, logit, float(ep.z)))
        return total

    def step(self, groups: Sequence[Sequence[Episode]]) -> dict:
        cfg = self.cfg
        self._init_platt(groups)
        assign_returns(groups, cfg)
        counts = [[[d.n_terms for d in ep.decisions] for ep in group] for group in groups]
        w_pol = policy_weights(counts, cfg.normalization)
        w_cal = calibration_weights(counts)
        rollout_cal = self._rollout_cal_loss(groups, w_cal)
        platt_start = [float(v) for v in self.policy.platt_parameters()]
        owner = self._owners(groups)
        max_diff, mean_diff = self._fix_old(groups, owner)
        lr_factor = self._set_learning_rate()
        device = self.policy.device
        forecast = self.policy.cfg.forecast
        if cfg.alpha > 0 and not forecast:
            raise ValueError("alpha > 0 needs forecasts (PolicyConfig.forecast = True)")
        passes = []
        for _ in range(cfg.optimization_passes):
            self.optimizer.zero_grad(set_to_none=True)
            total, clipped, terms, kl, cal, kl_old, tis_sum, tis_capped = 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0
            conf_clipped, conf_terms, conf_kl, conf_loss = 0, 0, 0.0, 0.0
            for g, group in enumerate(groups):
                for i, ep in enumerate(group):
                    if not self._active(ep):
                        continue
                    for t, d in enumerate(ep.decisions):
                        if owner[(g, i, t)] != self.dist.rank:
                            continue
                        # Same sequence as the rollout and as _fix_old; the forecast enters the loss only when
                        # alpha > 0.
                        logps, s, conf_lp = self._score(d)
                        p = logit = None
                        if cfg.alpha > 0:
                            logit = self.policy.calibrated_logit(s.detach() if cfg.cal_grad == "platt_only" else s)
                            p = torch.sigmoid(logit)
                        old = torch.tensor(d.old_logps, device=device)
                        ref = torch.tensor(d.ref_logps, device=device) if cfg.beta > 0 else None
                        tis = None
                        if cfg.tis_cap is not None:
                            behavior = torch.tensor(d.behavior_logps, device=device)
                            tis = tis_weights(old, behavior, cfg.tis_cap)
                            tis_sum += float(tis.sum())
                            tis_capped += int((torch.exp(old - behavior) > cfg.tis_cap).sum())
                        loss, stats = decision_loss(logps, old, ep.advantage, w_pol[g][i][t], cfg.clip_eps,
                                                    ref_logps=ref, beta=cfg.beta,
                                                    p=p, z=float(ep.z),
                                                    cal_weight=w_cal[g][i][t], alpha=cfg.alpha, tis_weights=tis,
                                                    cal_loss=cfg.cal_loss, logit=logit)
                        if conf_lp is not None:
                            c_loss, c_stats = conf_term_loss(
                                conf_lp, d.conf_old_logp, self._conf_advantage(ep, d), w_pol[g][i][t],
                                cfg.clip_eps, cfg.conf_coef, ref_logp=d.conf_ref_logp, beta=cfg.beta)
                            loss = loss + c_loss
                            conf_loss += float(c_loss.detach())
                            conf_clipped += c_stats["clipped"]
                            conf_terms += 1
                            conf_kl += c_stats.get("kl", 0.0) * w_pol[g][i][t]
                        loss.backward()
                        total += float(loss.detach())
                        clipped += stats["clipped"]
                        terms += stats["terms"]
                        kl += stats.get("kl", 0.0) * w_pol[g][i][t]
                        cal += stats.get("cal", 0.0) * w_cal[g][i][t]
                        kl_old += stats["kl_old"]
            if self.dist.enabled:
                self.dist.all_reduce_grads([q for q in self.policy.model.parameters() if q.requires_grad]
                                           + self.policy.platt_parameters())
                (total, clipped, terms, kl, cal, kl_old, tis_sum, tis_capped, conf_clipped, conf_terms, conf_kl,
                 conf_loss) = self.dist.sum_floats([total, clipped, terms, kl, cal, kl_old, tis_sum, tis_capped,
                                                    conf_clipped, conf_terms, conf_kl, conf_loss])
                clipped, terms, tis_capped = int(clipped), int(terms), int(tis_capped)
            params = [q for q in self.policy.model.parameters() if q.grad is not None]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)) if params else 0.0
            if params and math.isfinite(grad_norm):
                self.optimizer.step()
            record = {"loss": total, "grad_norm": grad_norm, "clip_frac": clipped / max(terms, 1),
                      "kl": kl, "cal_loss": cal, "kl_old": kl_old / max(terms, 1)}
            if self.verbal:               # the confidence part of "loss"; kl above covers the action terms only
                record.update({"conf_loss": conf_loss, "conf_clip_frac": conf_clipped / max(conf_terms, 1),
                               "conf_kl": conf_kl})
            if cfg.tis_cap is not None:
                record["tis_mean_w"] = tis_sum / max(terms, 1)
                record["tis_capped_frac"] = tis_capped / max(terms, 1)
            passes.append(record)
        self.batches_done += 1
        eps = [ep for group in groups for ep in group]
        forecasts = [d.p for ep in eps for d in ep.decisions if d.p is not None]
        logits = [d.forecast_logit for ep in eps for d in ep.decisions if d.forecast_logit is not None]
        verbal = {}
        if self.verbal:
            ds = [d for ep in eps for d in ep.decisions]
            verbal = {"mean_conf": sum(d.conf for d in ds) / len(ds),
                      "mean_conf_hat": sum(d.conf_mean for d in ds) / len(ds),
                      "conf_entropy": sum(d.conf_entropy for d in ds) / len(ds),
                      "conf_brier": sum((d.conf - ep.z) ** 2 for ep in eps for d in ep.decisions) / len(ds)}
            if cfg.conf_advantage == "decoupled":
                verbal["mean_abs_conf_adv"] = sum(abs(d.conf_adv) for d in ds) / len(ds)
        platt = {}
        if platt_start:                   # a, b the batch was trained from (after the data-dependent start of b)
            platt = {"platt_a": platt_start[0], "platt_b": platt_start[1]}
        return {
            "win_rate": sum(ep.z for ep in eps) / len(eps),
            "mean_return": sum(ep.ret for ep in eps) / len(eps),
            "mean_bonus": (sum(ep.bonus for ep in eps) / len(eps)) if eps[0].bonus is not None else None,
            "mean_length": sum(ep.length for ep in eps) / len(eps),
            "mixed_groups": sum(len({ep.z for ep in group}) > 1 for group in groups),
            "active_groups": sum(any(ep.advantage != 0 for ep in group) for group in groups),
            "mean_forecast": (sum(forecasts) / len(forecasts)) if forecasts else None,
            "mean_forecast_logit": (sum(logits) / len(logits)) if logits else None,
            "cal_loss": rollout_cal, "cal_loss_kind": cfg.cal_loss, **platt, **verbal,
            "sampling_vs_old_max_abs_logp_diff": max_diff,
            "behavior_old_mean_abs_diff": mean_diff,
            "lr": cfg.learning_rate * lr_factor,
            "passes": passes,
        }
