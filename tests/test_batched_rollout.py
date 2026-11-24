"""T0002: lockstep batched rollouts sample from exactly pi_theta and agree with the serial reference."""
import pytest
import torch

from jevtw.envs import KeyChestEnv, Observation, TextWorldEnv
from jevtw.policy import LMPolicy
from jevtw.rollout import TIMING_KEYS, collect_batch, reasoning_end, rollout_metrics, run_episode
from jevtw.testing import ByteTokenizer, tiny_model


class EarlyStopPolicy(LMPolicy):
    """A random model never writes "Answer:" by itself. Here every token divisible by 5 counts as the end of the
    reasoning, so that rows of a batch stop at different steps and all three endings occur."""

    def _ends_with_answer(self, tokens):
        return bool(tokens) and tokens[-1] != self.eos_id and tokens[-1] % 5 == 0


@pytest.fixture
def stop_policy(policy_cfg):
    model = tiny_model(seed=0)
    with torch.no_grad():                 # make EOS likely enough to be sampled now and then
        model.get_output_embeddings().weight[ByteTokenizer.eos_token_id] *= 12.0
    return EarlyStopPolicy(model, ByteTokenizer(), policy_cfg)


class CountingEnv(KeyChestEnv):
    """KeyChest that records how many instances are open at once."""
    open_now = 0
    max_open = 0

    def reset(self, seed):
        cls = CountingEnv
        cls.open_now += 1
        cls.max_open = max(cls.max_open, cls.open_now)
        return super().reset(seed)

    def close(self):
        CountingEnv.open_now -= 1


def _jobs(n, make_env=KeyChestEnv):
    return [(f"keychest-{i % 4}", make_env, i % 4, 1000 + i) for i in range(n)]


def _same(a, b):
    return (a.game, a.env_seed, a.sample_seed, a.won, a.finished) == (b.game, b.env_seed, b.sample_seed, b.won,
                                                                      b.finished) and \
        [(d.command, d.action, d.reasoning_ids, d.action_context_ids) for d in a.decisions] == \
        [(d.command, d.action, d.reasoning_ids, d.action_context_ids) for d in b.decisions]


def _assert_on_policy(policy, episodes):
    for ep in episodes:
        for d in ep.decisions:
            with torch.no_grad():
                logps, p = policy.score(d, forecast=True)
            assert torch.allclose(logps, torch.tensor(d.behavior_logps), atol=1e-4)
            assert p.item() == pytest.approx(d.p, abs=1e-4)


def test_batched_reasoning_is_on_policy_with_left_padding(stop_policy):
    """Prompts of different lengths in one batch, rows stopping at different steps and in all three ways: the
    recorded log-probs are those of the unpadded single-sequence training forward."""
    policy = stop_policy
    observations = [Observation("You are in a room. " * (i + 1), [f"option {j}" for j in range(2 + i)])
                    for i in range(6)]
    prompts = [policy.encode_prompt("Win.", [("wait", Observation("Time passes.", []))] * (i % 3), obs)
               for i, obs in enumerate(observations)]
    assert len({len(p) for p in prompts}) == len(prompts)
    seen = set()
    for seed in range(4):
        generators = [torch.Generator().manual_seed(100 * seed + i) for i in range(len(prompts))]
        sampled = policy.sample_reasoning_batch(prompts, generators)
        lengths = [len(tokens) for tokens, _ in sampled]
        assert all(1 <= n <= policy.cfg.max_reasoning_tokens for n in lengths)
        for prompt, obs, (tokens, logps), generator in zip(prompts, observations, sampled, generators):
            d = policy.decide(prompt, tokens, logps, obs, generator)
            seen.add(reasoning_end(d, policy.eos_id))
            with torch.no_grad():
                scored, _ = policy.score(d)
            assert torch.allclose(scored, torch.tensor(d.behavior_logps), atol=1e-4)
    assert seen == {"answer", "eos", "cap"}


def test_batch_row_does_not_depend_on_other_rows(stop_policy):
    """The same prompt and seed give the same tokens alone and next to other (longer and shorter) prompts."""
    policy = stop_policy
    prompts = [policy.encode(text) for text in ("short", "a much longer prompt than the others " * 3, "medium one")]
    for i, prompt in enumerate(prompts):
        alone = policy.sample_reasoning_batch([prompt], [torch.Generator().manual_seed(7)])[0]
        generators = [torch.Generator().manual_seed(7 if j == i else 50 + j) for j in range(3)]
        together = policy.sample_reasoning_batch(prompts, generators)[i]
        assert alone[0] == together[0]
        assert alone[1] == pytest.approx(together[1], abs=1e-4)


@pytest.mark.parametrize("which", ["policy", "stop_policy"])
def test_parallel_one_equals_run_episode(which, request):
    policy = request.getfixturevalue(which)
    jobs = _jobs(5)
    batch = collect_batch(policy, jobs, max_actions=4, parallel=1)
    for (game, make_env, env_seed, sample_seed), ep in zip(jobs, batch):
        reference = run_episode(policy, make_env(), game, env_seed, sample_seed, 4)
        assert _same(ep, reference)
        assert [d.behavior_logps for d in ep.decisions] == [d.behavior_logps for d in reference.decisions]


def test_collect_batch_is_reproducible_and_on_policy(stop_policy):
    jobs = _jobs(6)
    first = collect_batch(stop_policy, jobs, max_actions=4, parallel=4)
    second = collect_batch(stop_policy, jobs, max_actions=4, parallel=4)
    assert all(_same(a, b) for a, b in zip(first, second))
    assert [[d.behavior_logps for d in ep.decisions] for ep in first] == \
        [[d.behavior_logps for d in ep.decisions] for ep in second]
    _assert_on_policy(stop_policy, first)


def test_queue_keeps_order_and_content(stop_policy):
    """More jobs than slots, episodes ending after different numbers of steps: the output follows `jobs`, every
    episode equals its serial rollout, and never more than `parallel` environments are open."""
    CountingEnv.open_now = CountingEnv.max_open = 0
    jobs = _jobs(9, CountingEnv)
    stats = {}
    episodes = collect_batch(stop_policy, jobs, max_actions=5, parallel=3, stats=stats)
    assert CountingEnv.max_open == 3 and CountingEnv.open_now == 0
    assert len(episodes) == len(jobs) and len({ep.length for ep in episodes}) > 1
    for (game, _, env_seed, sample_seed), ep in zip(jobs, episodes):
        assert _same(ep, run_episode(stop_policy, KeyChestEnv(), game, env_seed, sample_seed, 5))
        assert ep.finished or ep.length == 5
    _assert_on_policy(stop_policy, episodes)

    assert set(stats) == set(TIMING_KEYS) and all(v >= 0 for v in stats.values())
    metrics = rollout_metrics(episodes, stats, wall_s=2.0, eos_id=stop_policy.eos_id)
    n = sum(ep.length for ep in episodes)
    assert metrics["decisions"] == n and metrics["sec_per_decision"] == pytest.approx(2.0 / n)
    assert metrics["mean_options"] >= 4 and 1 <= metrics["mean_reasoning_tokens"] <= 6
    assert sum(metrics[f"reasoning_end_{k}"] for k in ("answer", "eos", "cap")) == pytest.approx(1.0)


def test_environments_are_closed_on_error(policy):
    CountingEnv.open_now = CountingEnv.max_open = 0

    class Broken(CountingEnv):
        def step(self, command):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        collect_batch(policy, _jobs(4, Broken), max_actions=3, parallel=2)
    assert CountingEnv.open_now == 0


def test_no_reasoning_batch(tokenizer, policy_cfg):
    from jevtw.config import PolicyConfig

    policy = LMPolicy(tiny_model(seed=1), tokenizer, PolicyConfig(**{**policy_cfg.__dict__, "max_reasoning_tokens": 0}))
    stats = {}
    episodes = collect_batch(policy, _jobs(4), max_actions=3, parallel=4, stats=stats)
    assert all(d.reasoning_ids == [] and len(d.behavior_logps) == 1 for ep in episodes for d in ep.decisions)
    metrics = rollout_metrics(episodes, stats, 1.0, policy.eos_id)
    assert metrics["mean_reasoning_tokens"] == 0 and metrics["reasoning_end_cap"] is None


def test_collect_batch_plays_textworld(cooking_game, policy):
    jobs = [(cooking_game.name, lambda: TextWorldEnv(cooking_game), 0, seed) for seed in range(3)]
    episodes = collect_batch(policy, jobs, max_actions=3, parallel=2)
    assert len(episodes) == 3 and all(1 <= ep.length <= 3 for ep in episodes)
    for (game, make_env, env_seed, sample_seed), ep in zip(jobs, episodes):
        env = make_env()
        assert _same(ep, run_episode(policy, env, game, env_seed, sample_seed, 3))
        env.close()
    _assert_on_policy(policy, episodes)
