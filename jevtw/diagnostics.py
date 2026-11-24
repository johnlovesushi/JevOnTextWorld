"""Forecast diagnostics for evaluations (T0005): how well the forecast logit s_t separates wins from losses, and a
per-episode dump of every forecast."""
from __future__ import annotations

import json
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
    return {"forecast_brier": sum((p - z) ** 2 for p, _, z, _ in rows) / n,
            "forecast_brier_const": w * (1 - w),
            "mean_forecast": sum(p for p, _, _, _ in rows) / n,
            "mean_forecast_logit": sum(s for _, s, _, _ in rows) / n,
            "forecast_auc": auc([s for _, s, _, _ in rows], [z for _, _, z, _ in rows]),
            "forecast_auc_t0": auc([s for s, _ in first], [z for _, z in first]),
            "forecast_s_mean_win": (sum(win) / len(win)) if win else None,
            "forecast_s_mean_lose": (sum(lose) / len(lose)) if lose else None}


def forecast_records(episodes: Sequence[Episode]) -> list[dict]:
    """One record per episode: game, env_seed, sample_seed, z, length and, per step, the command, the forecast
    logit s and probability p of the chosen option and the logits s_options of all options."""
    return [{"game": ep.game, "env_seed": ep.env_seed, "sample_seed": ep.sample_seed, "z": ep.z, "length": ep.length,
             "steps": [{"t": t, "command": d.command, "s": d.forecast_logit, "p": d.p,
                        "s_options": d.forecast_logits_options} for t, d in enumerate(ep.decisions)]}
            for ep in episodes]


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
