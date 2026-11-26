"""Episode returns and group-relative advantages, eqs. (6)-(7)."""
from __future__ import annotations

import math
from typing import Sequence


def brier_bonus(forecasts: Sequence[float], z: int) -> float:
    """b_i = 1 - (1/T_i) sum_t (p_t - z)^2, in [0, 1]. One forecast per decision of the episode."""
    if z not in (0, 1):
        raise ValueError(f"z must be 0 or 1, got {z}")
    if not forecasts:
        raise ValueError("an episode needs at least one decision")
    for p in forecasts:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"forecast {p} is not a probability")
    return 1.0 - sum((p - z) ** 2 for p in forecasts) / len(forecasts)


def episode_return(z: int, b: float, lam: float) -> float:
    """R_i = z_i + lam * b_i. With 0 <= lam < 1 and b in [0, 1], every win (R >= 1) beats every loss (R <= lam)."""
    if not 0.0 <= lam < 1.0:
        raise ValueError(f"lam must satisfy 0 <= lam < 1, got {lam}")
    return float(z) + lam * b


def group_advantages(returns: Sequence[float], delta: float = 1e-8, norm: str = "std") -> list[float]:
    """A_i = (R_i - mean R) / (sd(R) + delta) over one game's G rollouts (norm "std", eq. 7; sd is the population
    sd, ddof = 0), or A_i = R_i - mean R (norm "mean").

    A group whose returns are all equal gets zero advantages."""
    if len(returns) < 2:
        raise ValueError("need at least two rollouts per group")
    mean = sum(returns) / len(returns)
    if norm == "mean":
        return [r - mean for r in returns]
    if norm != "std":
        raise ValueError(f"unknown advantage normalisation {norm!r}")
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
    return [(r - mean) / (sd + delta) for r in returns]


def score(c: float, z: int) -> float:
    """S(c, z) = 1 - (c - z)^2, the per-decision term of b_i (E0012 section 2.3)."""
    return 1.0 - (c - z) ** 2


def bin_values(m: int) -> list[float]:
    """v_m = m / (M - 1), m = 0 .. M - 1: the values of the verbal confidence answers."""
    return [i / (m - 1) for i in range(m)]


def decoupled_conf_advantage(conf_logps: Sequence[float], conf_bin: int, z: int) -> float:
    """A^c_t = S(v_{m_t}, z) - sum_m pi^c_old(m) S(v_m, z) (E0012 eq. 2.5b). `conf_logps` = log pi^c_old over the M
    bins (normalised), `conf_bin` = m_t. The baseline is the expected score under pi^c_old given the outcome z;
    it does not depend on m_t, so the estimator stays unbiased (m_t does not influence z)."""
    values = bin_values(len(conf_logps))
    baseline = sum(math.exp(lp) * score(v, z) for lp, v in zip(conf_logps, values))
    return score(values[conf_bin], z) - baseline
