"""Complete episodes tau = (h_t, c_t, a_t, p_t)_{t < T} with terminal outcome z (section 2).

`run_episode` plays one episode at a time and is the reference. `collect_batch` advances up to `parallel` episodes
in lockstep and samples their reasoning in one batch; every episode still has its own environment and its own
random generator, and is sampled from exactly pi_theta."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from .envs import Observation
from .policy import Decision, LMPolicy

TIMING_KEYS = ("prompt_s", "gen_s", "score_s", "env_s")


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
    history: list[tuple[str, Observation]] = []
    decisions = []
    while not obs.done and len(decisions) < max_actions:
        decision = policy.act(env.objective, history, obs, generator)
        decisions.append(decision)
        obs = env.step(decision.command)
        history.append((decision.command, obs))
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


@dataclass
class _Run:
    """One episode in progress inside collect_batch."""
    index: int
    game: str
    env_seed: int
    sample_seed: int
    env: object
    generator: torch.Generator
    obs: object = None
    history: list = field(default_factory=list)
    decisions: list = field(default_factory=list)


def collect_batch(policy: LMPolicy, jobs: Sequence[tuple[str, Callable[[], object], int, int]], max_actions: int,
                  parallel: int, stats: dict | None = None) -> list[Episode]:
    """Play one episode per job (game, make_env, env_seed, sample_seed); the result is in the order of `jobs`.

    At most `parallel` episodes are in progress; when one ends, the next job in the queue takes its place. One
    lockstep step: build the prompts of all running episodes, sample their reasoning in one batch
    (`LMPolicy.sample_reasoning_batch`), then per episode score the options / sample the action / read p_t
    (`LMPolicy.decide`, one `decision_forward` each) and finally step the environments one after another.
    Each episode owns an environment instance and a torch.Generator on the model's device seeded with its
    sample_seed, so what an episode samples does not depend on which other episodes run next to it (up to
    floating-point effects of batching).

    `stats`, when given, accumulates the seconds spent in each part under TIMING_KEYS (env_s includes creating
    and resetting environments)."""
    if parallel < 1:
        raise ValueError("parallel must be at least 1")
    timer = stats if stats is not None else {}
    for key in TIMING_KEYS:
        timer.setdefault(key, 0.0)
    clock = time.perf_counter
    episodes: list[Episode | None] = [None] * len(jobs)
    pending = deque(enumerate(jobs))
    active: list[_Run] = []

    def close(run: _Run):
        if hasattr(run.env, "close"):
            run.env.close()

    try:
        while pending or active:
            start = clock()
            while pending and len(active) < parallel:
                index, (game, make_env, env_seed, sample_seed) = pending.popleft()
                generator = torch.Generator(device=policy.device).manual_seed(sample_seed)
                run = _Run(index, game, env_seed, sample_seed, make_env(), generator)
                active.append(run)
                run.obs = run.env.reset(env_seed)
                if run.obs.done:
                    raise RuntimeError(f"{game}: the episode ended before the first decision")
            timer["env_s"] += clock() - start

            start = clock()
            prompts = [policy.encode_prompt(run.env.objective, run.history, run.obs) for run in active]
            timer["prompt_s"] += clock() - start

            start = clock()
            reasoning = policy.sample_reasoning_batch(prompts, [run.generator for run in active])
            timer["gen_s"] += clock() - start

            start = clock()
            for run, prompt, (tokens, logps) in zip(active, prompts, reasoning):
                run.decisions.append(policy.decide(prompt, tokens, logps, run.obs, run.generator))
            timer["score_s"] += clock() - start

            start = clock()
            running = []
            for run in active:
                command = run.decisions[-1].command
                run.obs = run.env.step(command)
                run.history.append((command, run.obs))
                if run.obs.done or len(run.decisions) >= max_actions:
                    episodes[run.index] = Episode(game=run.game, env_seed=run.env_seed, sample_seed=run.sample_seed,
                                                  decisions=run.decisions, won=run.obs.won, finished=run.obs.done)
                    close(run)
                else:
                    running.append(run)
            active = running
            timer["env_s"] += clock() - start
    finally:
        for run in active:                # only non-empty when an exception interrupted the loop
            close(run)
    return episodes


REASONING_ENDS = ("answer", "eos", "cap", "cap_in_think")       # disjoint; the fractions sum to 1


def reasoning_end(d: Decision, eos_id: int) -> str | None:
    """How c_t ended: "eos", "answer" (it ends with "Answer:"), "cap" (max_reasoning_tokens reached, so the answer
    prefix was forced), "cap_in_think" (the cap was reached inside the thinking block of prompt_format
    "chat_think", so "</think>" was forced too), or None when there is no reasoning. Decisions made by
    `LMPolicy.decide` carry the answer (`Decision.reasoning_end`, from `LMPolicy._finish`); for a hand-built
    Decision it is inferred from the ids, which cannot tell "cap_in_think" from "cap"."""
    if not d.reasoning_ids:
        return None
    if d.reasoning_end is not None:
        return d.reasoning_end
    if d.reasoning_ids[-1] == eos_id:
        return "eos"
    # Nothing dropped and nothing appended to prompt + c_t: see LMPolicy._action_context.
    return "answer" if len(d.action_context_ids) == len(d.prompt_ids) + len(d.reasoning_ids) else "cap"


def rollout_metrics(episodes: Sequence[Episode], stats: dict, wall_s: float, eos_id: int) -> dict:
    """Sampling speed and reasoning statistics of a set of episodes, for the training and evaluation logs.
    sec_per_decision is the wall time of sampling divided by the number of decisions; the reasoning_end_*
    fractions are over the decisions that have reasoning (None without reasoning)."""
    decisions = [d for ep in episodes for d in ep.decisions]
    ends = [e for e in (reasoning_end(d, eos_id) for d in decisions) if e is not None]
    n = len(decisions)
    out = {"decisions": n, "sec_per_decision": wall_s / n,
           **{k: round(stats.get(k, 0.0), 3) for k in ("gen_s", "score_s", "env_s", "prompt_s")},
           "mean_reasoning_tokens": sum(len(d.reasoning_ids) for d in decisions) / n,
           "mean_options": sum(len(d.option_ids) for d in decisions) / n}
    for kind in REASONING_ENDS:
        out[f"reasoning_end_{kind}"] = (ends.count(kind) / len(ends)) if ends else None
    return out
