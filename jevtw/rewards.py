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


def group_advantages(returns: Sequence[float], delta: float = 1e-8) -> list[float]:
    """A_i = (R_i - mean R) / (sd(R) + delta) over one game's G rollouts. sd is the population sd (ddof = 0).

    A group whose returns are all equal gets zero advantages."""
    if len(returns) < 2:
        raise ValueError("need at least two rollouts per group")
    mean = sum(returns) / len(returns)
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
    return [(r - mean) / (sd + delta) for r in returns]
