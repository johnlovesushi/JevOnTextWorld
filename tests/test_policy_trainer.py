"""Policy (eqs. 4-5), rollouts and one GRPO update on the tiny random model and the KeyChest POMDP."""
import copy

import pytest
import torch

from jevtw.config import GRPOConfig, PolicyConfig
from jevtw.envs import KeyChestEnv, Observation
from jevtw.policy import LMPolicy, build_prompt
from jevtw.rollout import collect_group, run_episode
from jevtw.testing import tiny_model
from jevtw.trainer import Trainer, assign_returns


def test_prompt_shows_window_and_numbered_options(policy_cfg):
    history = [(f"cmd{i}", f"obs{i}") for i in range(5)]
    text = build_prompt("Win.", history, Observation("here", ["open box", "wait"]), policy_cfg)
    assert "cmd1" not in text and "cmd2" in text and "cmd4" in text           # window of 3
    assert "1. open box\n2. wait" in text and text.endswith(policy_cfg.reasoning_prefix)


def test_sampling_and_scoring_agree(policy):
    """Log-probs recorded with the KV cache equal the full-sequence training forward; p_t is a probability."""
    env = KeyChestEnv()
    obs = env.reset(0)
    for seed in range(3):
        d = policy.act(env.objective, [], obs, torch.Generator().manual_seed(seed))
        assert d.command in obs.commands and d.n_terms == len(d.reasoning_ids) + 1 <= 7
        assert d.action_context_ids[-len(policy.answer_ids):] == policy.answer_ids
        with torch.no_grad():
            logps, p = policy.score(d, forecast=True)
        assert torch.allclose(logps, torch.tensor(d.behavior_logps), atol=1e-4)
        assert 0.0 <= d.p <= 1.0 and p.item() == pytest.approx(d.p, abs=1e-5)


def test_action_distribution_is_softmax_over_options(policy):
    env = KeyChestEnv()
    obs = env.reset(1)
    d = policy.act(env.objective, [], obs, torch.Generator().manual_seed(0))
    with torch.no_grad():
        scores = policy._option_scores(policy.model, d.action_context_ids, d.option_ids)
    assert scores.shape == (len(obs.commands),)
    assert d.behavior_logps[-1] == pytest.approx(float(torch.log_softmax(scores, 0)[d.action]), abs=1e-5)


def test_option_scores_only_project_option_positions(policy):
    """Regression (E0001 OOM): scoring must not build full-vocabulary logits for every context position, and the
    scores must equal the ones read from the full logits."""
    d = policy.act("goal", [], Observation("room", "nothing", [f"cmd {i}" for i in range(5)]), torch.Generator().manual_seed(2))
    context, options = d.action_context_ids, d.option_ids + [policy.encode(" 12") + [policy.eos_id]]
    projected = []
    hook = policy.model.get_output_embeddings().register_forward_hook(
        lambda module, args, output: projected.append(output.numel() // output.shape[-1]))
    try:
        with torch.no_grad():
            scores = policy._option_scores(policy.model, context, options)
    finally:
        hook.remove()
    assert sum(projected) == sum(len(o) for o in options) < len(context)
    n = len(context)
    with torch.no_grad():
        for j, opt in enumerate(options):
            ids = torch.tensor([context + opt])
            logprobs = torch.log_softmax(policy.model(input_ids=ids).logits[0].float(), dim=-1)
            expected = sum(logprobs[n - 1 + k, t] for k, t in enumerate(opt))
            assert scores[j].item() == pytest.approx(expected.item(), abs=1e-4)


def test_no_reasoning_mode(tokenizer, policy_cfg):
    cfg = PolicyConfig(**{**policy_cfg.__dict__, "max_reasoning_tokens": 0})
    pol = LMPolicy(tiny_model(seed=1), tokenizer, cfg)
    env = KeyChestEnv()
    d = pol.act(env.objective, [], env.reset(0), torch.Generator().manual_seed(0))
    assert d.reasoning_ids == [] and d.n_terms == 1 and len(d.behavior_logps) == 1


def test_group_shares_initial_state_and_episode_fields(policy):
    group = collect_group(policy, KeyChestEnv, "keychest-5", env_seed=5, sample_seeds=[0, 1, 2], max_actions=4)
    assert len(group) == 3 and all(ep.env_seed == 5 for ep in group)
    for ep in group:
        assert 1 <= ep.length <= 4 and ep.z in (0, 1)
        assert ep.finished or ep.length == 4
        assert all(d.p is not None for d in ep.decisions)
    again = run_episode(policy, KeyChestEnv(), "keychest-5", 5, 1, 4)        # same seeds -> same rollout
    assert [d.command for d in again.decisions] == [d.command for d in group[1].decisions]


def _groups(policy, n_games=2, group_size=4, max_actions=3):
    return [collect_group(policy, KeyChestEnv, f"keychest-{g}", g, list(range(100 * g, 100 * g + group_size)),
                          max_actions) for g in range(n_games)]


def test_assign_returns_with_and_without_calibration(policy):
    groups = _groups(policy)
    assign_returns(groups, GRPOConfig(lam=0.0))
    assert all(ep.ret == ep.z for g in groups for ep in g) and groups[0][0].bonus is not None
    assign_returns(groups, GRPOConfig(lam=0.5))
    for g in groups:
        for ep in g:
            assert ep.ret == pytest.approx(ep.z + 0.5 * ep.bonus)
        assert sum(ep.advantage for ep in g) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("normalization", ["token", "episode"])
def test_first_pass_loss_matches_formula(policy, normalization):
    """At the first pass rho = 1, so L = mean_g[-sum_i A_i * mass_i + alpha * Brier_g]."""
    groups = _groups(policy)
    cfg = GRPOConfig(lam=0.5, alpha=0.3, normalization=normalization, optimization_passes=1, learning_rate=0.0)
    metrics = Trainer(policy, cfg).step(groups)
    expected = 0.0
    for g in groups:
        n = sum(d.n_terms for ep in g for d in ep.decisions)
        mass = [sum(d.n_terms for d in ep.decisions) / n if normalization == "token" else 1 / len(g) for ep in g]
        cal = sum((d.p - ep.z) ** 2 for ep in g for d in ep.decisions) / sum(ep.length for ep in g)
        expected += (-sum(m * ep.advantage for m, ep in zip(mass, g)) + 0.3 * cal) / len(groups)
    first = metrics["passes"][0]
    assert first["loss"] == pytest.approx(expected, rel=1e-4, abs=1e-6)
    assert first["clip_frac"] == 0.0 and metrics["sampling_vs_old_max_abs_logp_diff"] < 1e-4


def test_update_changes_parameters_and_noop_when_no_signal(tokenizer, policy_cfg):
    pol = LMPolicy(tiny_model(seed=2), tokenizer, policy_cfg)
    groups = _groups(pol)
    before = copy.deepcopy(pol.model.state_dict())
    metrics = Trainer(pol, GRPOConfig(lam=0.5, alpha=0.1, learning_rate=1e-3)).step(groups)
    assert all(torch.isfinite(torch.tensor(p["loss"])) for p in metrics["passes"])
    assert metrics["passes"][1]["loss"] != metrics["passes"][0]["loss"]
    assert any(not torch.equal(before[k], v) for k, v in pol.model.state_dict().items())

    # Plain GRPO (lam = alpha = beta = 0) with identical outcomes in every group: zero advantages, no update.
    for g in groups:
        for ep in g:
            ep.won = False
    before = copy.deepcopy(pol.model.state_dict())
    metrics = Trainer(pol, GRPOConfig(learning_rate=1e-3)).step(groups)
    assert metrics["active_groups"] == 0 and metrics["passes"][0]["loss"] == 0.0
    assert all(torch.equal(before[k], v) for k, v in pol.model.state_dict().items())


def test_kl_needs_reference_and_is_zero_against_itself(policy):
    with pytest.raises(ValueError):
        Trainer(policy, GRPOConfig(beta=0.1))
    ref = copy.deepcopy(policy.model)
    metrics = Trainer(policy, GRPOConfig(beta=0.1, optimization_passes=1, learning_rate=0.0), ref).step(
        _groups(policy))
    assert metrics["passes"][0]["kl"] == pytest.approx(0.0, abs=1e-6)
