"""Policy (eqs. 4-5), rollouts and one GRPO update on the tiny random model and the KeyChest POMDP."""
import copy

import pytest
import torch

from jevtw.config import GRPOConfig, PolicyConfig
from jevtw.envs import KeyChestEnv, Observation
from jevtw.policy import Decision, LMPolicy, build_prompt
from jevtw.readout import yes_no_probability
from jevtw.rollout import collect_group, run_episode
from jevtw.testing import tiny_model
from jevtw.trainer import Trainer, assign_returns


def count_head_rows(monkeypatch) -> list[int]:
    """Record how many hidden-state rows every call of the FP32 output layer (LMPolicy._head) projects."""
    projected, original = [], LMPolicy._head

    def spy(model, hidden, rows=None):
        projected.append(hidden.shape[0])
        return original(model, hidden, rows)

    monkeypatch.setattr(LMPolicy, "_head", staticmethod(spy))
    return projected


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


# ---- Reference implementation (the pre-T0001 code): one full-context forward per quantity, full logits. ----

def ref_option_scores(policy, model, context, options):
    scores, n = [], len(context)
    for opt in options:
        logprobs = torch.log_softmax(model(input_ids=torch.tensor([context + opt])).logits[0].float(), dim=-1)
        scores.append(sum(logprobs[n - 1 + i, t] for i, t in enumerate(opt)))
    return torch.stack(scores)


def ref_reasoning_logps(policy, model, d):
    if not d.reasoning_ids:
        return torch.zeros(0)
    n = len(d.prompt_ids)
    logits = model(input_ids=torch.tensor([d.prompt_ids + d.reasoning_ids])).logits[0, n - 1:-1].float()
    return torch.log_softmax(logits, dim=-1).gather(1, torch.tensor(d.reasoning_ids)[:, None])[:, 0]


def ref_forecasts(policy, model, d):
    out = []
    for opt in d.option_ids:
        ids = d.action_context_ids + opt + policy.confidence_prompt_ids
        out.append(yes_no_probability(model(input_ids=torch.tensor([ids])).logits[0, -1], policy.yes_id, policy.no_id))
    return torch.stack(out)


def _manual_decision(policy, ending: str, n_options: int = 12, action: int = 10) -> Decision:
    """A hand-built decision (no sampling). `ending` is how the reasoning stopped."""
    prompt = policy.encode("Objective: open the chest.\nOptions:\n1. a\n2. b\nReasoning:")
    reasoning = {"answer": policy.encode(" the key first\n" + policy.cfg.answer_prefix),
                 "eos": policy.encode(" take key") + [policy.eos_id],
                 "cap": policy.encode(" I think"),
                 "none": []}[ending]
    context = policy._action_context(prompt, reasoning)
    expected_n = {"answer": len(prompt) + len(reasoning),
                  "eos": len(prompt) + len(reasoning) - 1 + len(policy.answer_ids),
                  "cap": len(prompt) + len(reasoning) + len(policy.answer_ids),
                  "none": len(prompt) + len(policy.answer_ids)}[ending]
    assert len(context) == expected_n
    options = [policy.encode(f" {j + 1}") + [policy.eos_id] for j in range(n_options)]   # one- and two-digit numbers
    assert {len(o) for o in options} == {3, 4}
    return Decision(command="x", action=action, prompt_ids=prompt, reasoning_ids=reasoning,
                    action_context_ids=context, option_ids=options, behavior_logps=[])


ENDINGS = ["answer", "eos", "cap", "none"]


@pytest.mark.parametrize("ending", ENDINGS)
def test_decision_forward_matches_reference(policy, ending):
    """Shared-prefix scoring returns the same reasoning log-probs, l(a^j) and p_j as full-context forwards."""
    d = _manual_decision(policy, ending)
    with torch.no_grad():
        lp, scores, p = policy.decision_forward(policy.model, d, forecast=True)
        _, scores_no_forecast, none = policy.decision_forward(policy.model, d, forecast=False)
        assert none is None and torch.allclose(scores, scores_no_forecast, atol=1e-5)   # appending z changes nothing
        assert lp.shape == (len(d.reasoning_ids),) and scores.shape == p.shape == (12,)
        assert torch.allclose(lp, ref_reasoning_logps(policy, policy.model, d), atol=1e-4)
        assert torch.allclose(scores, ref_option_scores(policy, policy.model, d.action_context_ids, d.option_ids),
                              atol=1e-4)
        assert torch.allclose(p, ref_forecasts(policy, policy.model, d), atol=1e-4)
        logps, p_a = policy.score(d, forecast=True)
        assert logps.shape == (d.n_terms,) and p_a.item() == pytest.approx(p[d.action].item())
        assert logps[-1].item() == pytest.approx(torch.log_softmax(scores, 0)[d.action].item(), abs=1e-6)


@pytest.mark.parametrize("ending", ENDINGS)
def test_decision_forward_gradient_matches_reference(policy, ending):
    d = _manual_decision(policy, ending)
    params = [w for w in policy.model.parameters() if w.requires_grad]

    logps, p_a = policy.score(d, forecast=True)
    new = torch.autograd.grad(logps.sum() + p_a, params, allow_unused=True)

    log_q = torch.log_softmax(ref_option_scores(policy, policy.model, d.action_context_ids, d.option_ids), dim=0)
    ref_scalar = (ref_reasoning_logps(policy, policy.model, d).sum() + log_q[d.action]
                  + ref_forecasts(policy, policy.model, d)[d.action])
    ref = torch.autograd.grad(ref_scalar, params, allow_unused=True)

    assert (logps.sum() + p_a).item() == pytest.approx(ref_scalar.item(), abs=1e-4)
    flat = lambda gs: torch.cat([torch.zeros_like(w).flatten() if g is None else g.flatten()
                                 for g, w in zip(gs, params)])
    g_new, g_ref = flat(new), flat(ref)
    assert g_ref.norm() > 0 and (g_new - g_ref).norm() / g_ref.norm() < 1e-4


@pytest.mark.parametrize("forecast", [True, False])
def test_decision_forward_work_is_one_prefix_plus_suffixes(policy, forecast, monkeypatch):
    """Regression for the E0001 OOMs: the context is embedded once, and the output layer only sees the positions
    that are read (never the whole context)."""
    d = _manual_decision(policy, "cap")
    n, c, z = len(d.action_context_ids), len(d.reasoning_ids), len(policy.confidence_prompt_ids)
    embedded, projected = [], count_head_rows(monkeypatch)
    hooks = [policy.model.get_input_embeddings().register_forward_hook(
                 lambda module, args, output: embedded.append(args[0].numel()))]
    try:
        with torch.no_grad():
            policy.decision_forward(policy.model, d, forecast=forecast)
    finally:
        for h in hooks:
            h.remove()
    assert len(embedded) == 1                                  # one forward
    assert sum(embedded) == (n - 1) + sum(1 + len(o) + (z if forecast else 0) for o in d.option_ids)
    assert projected == [c + sum(len(o) for o in d.option_ids)] + ([len(d.option_ids)] if forecast else [])
    assert sum(embedded) < len(d.option_ids) * n               # the old cost: one full context per option


def test_action_distribution_is_softmax_over_options(policy):
    env = KeyChestEnv()
    obs = env.reset(1)
    d = policy.act(env.objective, [], obs, torch.Generator().manual_seed(0))
    with torch.no_grad():
        scores = ref_option_scores(policy, policy.model, d.action_context_ids, d.option_ids)
        forecasts = ref_forecasts(policy, policy.model, d)
    assert scores.shape == (len(obs.commands),)
    assert d.behavior_logps[-1] == pytest.approx(float(torch.log_softmax(scores, 0)[d.action]), abs=1e-4)
    assert len(d.p_options) == len(obs.commands) and d.p == d.p_options[d.action]
    assert torch.allclose(torch.tensor(d.p_options), forecasts, atol=1e-4)


def test_sampling_prefill_projects_one_position_per_step(policy, monkeypatch):
    """Sampling c_t never builds logits for the prompt positions: one output-layer row per sampled token, plus the
    rows of the single decision_forward call."""
    obs = Observation("a room", [f"cmd {i}" for i in range(5)])
    projected = count_head_rows(monkeypatch)
    d = policy.act("goal", [], obs, torch.Generator().manual_seed(2))
    c = len(d.reasoning_ids)
    assert d.command in obs.commands and len(d.option_ids) == 5
    assert sum(projected) == c + c + sum(len(o) + 1 for o in d.option_ids) < len(d.prompt_ids)


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
