"""Forecast diagnostics for evaluations (T0005): how well the forecast logit s_t separates wins from losses, and a
per-episode dump of every forecast."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .rollout import Episode


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Area under the ROC curve: the probability that a positive (label 1) has a higher score than a negative,
    ties counting 1/2. Computed from mid-ranks (Mann-Whitney U). None when one of the classes is empty."""
    positives = sum(1 for z in labels if z)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum, i = 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mid_rank = (i + j) / 2 + 1                       # ranks start at 1; tied scores share their mean rank
        rank_sum += mid_rank * sum(1 for m in range(i, j + 1) if labels[order[m]])
        i = j + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def forecast_metrics(episodes: Sequence[Episode]) -> dict:
    """Forecast quality over one evaluation. Every decision is one sample with label z of its episode, except
    forecast_auc_t0, which uses only the first decision of every episode.
      forecast_brier        mean (p_t - z)^2 over decisions
      forecast_brier_const  w (1 - w) with w the decision-weighted win rate: the Brier score of always forecasting
                            w, over the same samples as forecast_brier
      forecast_auc(_t0)     AUC of s_t as a score for z (None when all episodes won or all lost)
      forecast_s_mean_win / _lose   mean s_t over the decisions of won / lost episodes"""
    rows = [(d.p, d.forecast_logit, ep.z, t) for ep in episodes for t, d in enumerate(ep.decisions)
            if d.p is not None]
    keys = ("forecast_brier", "forecast_brier_const", "mean_forecast", "mean_forecast_logit", "forecast_auc",
            "forecast_auc_t0", "forecast_s_mean_win", "forecast_s_mean_lose")
    if not rows:
        return dict.fromkeys(keys)
    n = len(rows)
    w = sum(z for _, _, z, _ in rows) / n
    win = [s for _, s, z, _ in rows if z]
    lose = [s for _, s, z, _ in rows if not z]
    first = [(s, z) for _, s, z, t in rows if t == 0]
    extra = prediction_metrics([(d.p, d.forecast_logit, ep.z, t, ep.game, k) for k, ep in enumerate(episodes)
                                for t, d in enumerate(ep.decisions) if d.p is not None], "forecast_")
    return {**extra, "forecast_brier": sum((p - z) ** 2 for p, _, z, _ in rows) / n,
            "forecast_brier_const": w * (1 - w),
            "mean_forecast": sum(p for p, _, _, _ in rows) / n,
            "mean_forecast_logit": sum(s for _, s, _, _ in rows) / n,
            "forecast_auc": auc([s for _, s, _, _ in rows], [z for _, _, z, _ in rows]),
            "forecast_auc_t0": auc([s for s, _ in first], [z for _, z in first]),
            "forecast_s_mean_win": (sum(win) / len(win)) if win else None,
            "forecast_s_mean_lose": (sum(lose) / len(lose)) if lose else None}


def logistic_fit(features: Sequence[Sequence[float]], labels: Sequence[int], ridge: float = 1e-4,
                 iters: int = 50) -> list[float]:
    """Linear predictor of a logistic regression of `labels` on `features` (with intercept; features standardised;
    Newton's method with a small ridge on the slopes so that separable data stays finite)."""
    import torch

    x = torch.tensor([list(f) for f in features], dtype=torch.float64)
    y = torch.tensor(list(labels), dtype=torch.float64)
    x = (x - x.mean(0)) / x.std(0, unbiased=False).clamp(min=1e-12)
    x = torch.cat([torch.ones(len(y), 1, dtype=torch.float64), x], dim=1)
    w = torch.zeros(x.shape[1], dtype=torch.float64)
    reg = torch.full((x.shape[1],), ridge, dtype=torch.float64)
    reg[0] = 0.0
    for _ in range(iters):
        p = torch.sigmoid(x @ w)
        grad = x.T @ (p - y) + reg * w
        hess = x.T @ (x * (p * (1 - p))[:, None]) + torch.diag(reg) + 1e-10 * torch.eye(x.shape[1], dtype=torch.float64)
        step = torch.linalg.solve(hess, grad)
        w = w - step
        if float(step.abs().max()) < 1e-10:
            break
    return (x @ w).tolist()


def prediction_metrics(rows: Sequence[tuple], prefix: str) -> dict:
    """Section 6 of E0012 for one kind of forecast. rows: (prediction in [0, 1], score for the AUC, z, t, game,
    episode). Every decision is one sample with the label z of its episode.
      offset          |mean prediction - decision-weighted win rate|
      logloss         cross-entropy with the prediction clipped to [0.01, 0.99]
      ece             10 equal-width bins, decision-weighted
      auc_within      AUC within each game that has both wins and losses among its episodes, averaged over those games
      auc_t_only / auc_t_plus / auc_step_gain
                      AUC of a logistic regression of z on the step t alone, on t + logit(clipped prediction), and
                      the difference (the forecast's separation beyond what the step index gives)
      saturation      fraction of predictions <= 0.005 or >= 0.995"""
    keys = ("offset", "logloss", "ece", "auc_within", "auc_t_only", "auc_t_plus", "auc_step_gain", "saturation")
    if not rows:
        return {prefix + k: None for k in keys}
    n = len(rows)
    preds = [r[0] for r in rows]
    labels = [r[2] for r in rows]
    w = sum(labels) / n
    clip = [min(max(p, 0.01), 0.99) for p in preds]
    out = {"offset": abs(sum(preds) / n - w),
           "logloss": -sum(z * math.log(p) + (1 - z) * math.log(1 - p) for p, z in zip(clip, labels)) / n,
           "saturation": sum(p <= 0.005 or p >= 0.995 for p in preds) / n}
    bins = defaultdict(list)
    for p, z in zip(preds, labels):
        bins[min(int(p * 10), 9)].append((p, z))
    out["ece"] = sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(z for _, z in b) / len(b))
                     for b in bins.values())
    by_game = defaultdict(list)
    for r in rows:
        by_game[r[4]].append(r)
    within = [auc([r[1] for r in g], [r[2] for r in g]) for g in by_game.values()]
    within = [a for a in within if a is not None]
    out["auc_within"] = sum(within) / len(within) if within else None
    if 0 < sum(labels) < n:
        t_only = logistic_fit([[r[3]] for r in rows], labels)
        t_plus = logistic_fit([[r[3], math.log(p / (1 - p))] for r, p in zip(rows, clip)], labels)
        out["auc_t_only"], out["auc_t_plus"] = auc(t_only, labels), auc(t_plus, labels)
        out["auc_step_gain"] = out["auc_t_plus"] - out["auc_t_only"]
    else:
        out.update(auc_t_only=None, auc_t_plus=None, auc_step_gain=None)
    return {prefix + k: out[k] for k in keys}


def verbal_metrics(episodes: Sequence[Episode]) -> dict:
    """Verbal confidence (T0008) over one evaluation: c-hat_t (expected confidence) is the prediction for the
    calibration numbers and the AUC; the sampled c_t gets its own Brier score and saturation.
      conf_brier / conf_brier_sampled / conf_brier_const, mean_conf_hat / mean_conf, conf_auc / conf_auc_t0,
      conf_entropy (mean entropy of pi^c, nats), conf_saturation_sampled (c_t in {0, 1}) and the
      `prediction_metrics` of c-hat with prefix conf_."""
    rows = [(d.conf_mean, d.conf_mean, ep.z, t, ep.game, k, d.conf, d.conf_entropy)
            for k, ep in enumerate(episodes) for t, d in enumerate(ep.decisions) if d.conf_bin is not None]
    keys = ("conf_brier", "conf_brier_sampled", "conf_brier_const", "mean_conf_hat", "mean_conf", "conf_auc",
            "conf_auc_t0", "conf_entropy", "conf_saturation_sampled")
    if not rows:
        return {}
    n = len(rows)
    w = sum(r[2] for r in rows) / n
    first = [r for r in rows if r[3] == 0]
    out = {"conf_brier": sum((r[0] - r[2]) ** 2 for r in rows) / n,
           "conf_brier_sampled": sum((r[6] - r[2]) ** 2 for r in rows) / n,
           "conf_brier_const": w * (1 - w),
           "mean_conf_hat": sum(r[0] for r in rows) / n,
           "mean_conf": sum(r[6] for r in rows) / n,
           "conf_auc": auc([r[1] for r in rows], [r[2] for r in rows]),
           "conf_auc_t0": auc([r[1] for r in first], [r[2] for r in first]),
           "conf_entropy": sum(r[7] for r in rows) / n,
           "conf_saturation_sampled": sum(r[6] in (0.0, 1.0) for r in rows) / n}
    assert set(out) == set(keys)
    return {**out, **prediction_metrics([r[:6] for r in rows], "conf_")}


def forecast_records(episodes: Sequence[Episode]) -> list[dict]:
    """One record per episode: game, env_seed, sample_seed, z, length and, per step, the command, the forecast
    logit s and probability p of the chosen option and the logits s_options of all options; with verbal confidence
    also c (sampled), c_hat and pi_c (the M probabilities)."""
    return [{"game": ep.game, "env_seed": ep.env_seed, "sample_seed": ep.sample_seed, "z": ep.z, "length": ep.length,
             "steps": [_step_record(t, d) for t, d in enumerate(ep.decisions)]}
            for ep in episodes]


def _step_record(t: int, d) -> dict:
    out = {"t": t, "command": d.command, "s": d.forecast_logit, "p": d.p, "s_options": d.forecast_logits_options}
    if d.conf_bin is not None:            # verbal confidence: c_t, c-hat_t, pi^c
        out.update({"c": d.conf, "c_hat": d.conf_mean, "pi_c": [math.exp(v) for v in d.conf_logps]})
    return out


def write_forecasts(path: Path, episodes: Sequence[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in forecast_records(episodes):
            f.write(json.dumps(record) + "\n")


def reasoning_records(episodes: Sequence[Episode], decode) -> list[dict]:
    """One record per decision: the episode it belongs to (game, env_seed, sample_seed, z, length), the step t,
    the command, how the reasoning ended, its length in tokens and its text (`decode(reasoning_ids)`)."""
    return [{"game": ep.game, "env_seed": ep.env_seed, "sample_seed": ep.sample_seed, "z": ep.z, "length": ep.length,
             "t": t, "command": d.command, "reasoning_end": d.reasoning_end,
             "reasoning_tokens": len(d.reasoning_ids), "reasoning": decode(d.reasoning_ids)}
            for ep in episodes for t, d in enumerate(ep.decisions)]


def write_reasonings(path: Path, episodes: Sequence[Episode], decode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in reasoning_records(episodes, decode):
            f.write(json.dumps(record) + "\n")
