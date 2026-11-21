"""Complete episodes tau = (h_t, c_t, a_t, p_t)_{t < T} with terminal outcome z (section 2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .policy import Decision, LMPolicy


@dataclass
class Episode:
    game: str
    env_seed: int                         # the same for the whole group: same initial state
    sample_seed: int                      # differs per rollout
    decisions: list[Decision]
    won: bool
    finished: bool                        # the game ended (win or loss) rather than reaching max_actions
    ret: float | None = field(default=None)          # R_i, set by the trainer
    bonus: float | None = field(default=None)        # b_i
    advantage: float | None = field(default=None)    # A_i

    @property
    def z(self) -> int:
        return int(self.won)

    @property
    def length(self) -> int:
        return len(self.decisions)


def run_episode(policy: LMPolicy, env, game: str, env_seed: int, sample_seed: int, max_actions: int) -> Episode:
    """Play one episode until the game ends or max_actions (H) commands were issued; z = 1 only for a win."""
    obs = env.reset(env_seed)
    generator = torch.Generator().manual_seed(sample_seed)
    history: list[tuple[str, str]] = []
    decisions = []
    while not obs.done and len(decisions) < max_actions:
        decision = policy.act(env.objective, history, obs, generator)
        decisions.append(decision)
        obs = env.step(decision.command)
        history.append((decision.command, obs.text))
    if not decisions:
        raise RuntimeError(f"{game}: the episode ended before the first decision")
    return Episode(game=game, env_seed=env_seed, sample_seed=sample_seed, decisions=decisions, won=obs.won,
                   finished=obs.done)


def collect_group(policy: LMPolicy, make_env: Callable[[], object], game: str, env_seed: int,
                  sample_seeds: list[int], max_actions: int) -> list[Episode]:
    """G rollouts of one game from the same initial state with the current (rollout) policy theta_old."""
    env = make_env()
    try:
        return [run_episode(policy, env, game, env_seed, s, max_actions) for s in sample_seeds]
    finally:
        if hasattr(env, "close"):
            env.close()
