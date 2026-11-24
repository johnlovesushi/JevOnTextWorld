"""T0003: FP32 output layer, identical sequences for sampling / old / reference / training, truncated importance
sampling, learning-rate warmup and the update-size metrics."""
import copy

import pytest
import torch
import torch.nn.functional as F

from jevtw.config import GRPOConfig, PolicyConfig
from jevtw.envs import KeyChestEnv
from jevtw.losses import clipped_surrogate, decision_loss, tis_weights
from jevtw.policy import LMPolicy
from jevtw.rollout import collect_group
from jevtw.testing import tiny_model
from jevtw.trainer import Trainer


def _groups(policy, n_games=2, group_size=4, max_actions=3):
    return [collect_group(policy, KeyChestEnv, f"keychest-{g}", g, list(range(100 * g, 100 * g + group_size)),
                          max_actions) for g in range(n_games)]


def _no_reasoning_policy(tokenizer, policy_cfg, seed=3, **overrides):
    cfg = PolicyConfig(**{**policy_cfg.__dict__, "max_reasoning_tokens": 0, **overrides})
    return LMPolicy(tiny_model(seed=seed), tokenizer, cfg)


# ---- (a) FP32 output layer

def test_logprobs_use_fp32_head_on_bf16_model(tokenizer, policy_cfg):
    """With bf16 parameters, _logprobs is bit for bit log_softmax(F.linear(h.float(), W.float())), not the bf16
    logits cast to FP32 afterwards."""
    model = tiny_model(seed=0).to(torch.bfloat16)
    policy = LMPolicy(model, tokenizer, policy_cfg)
    hidden = torch.randn(5, 32, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16) * 4
    weight = model.get_output_embeddings().weight
    assert weight.dtype == torch.bfloat16
    got = policy._logprobs(model, hidden)
    expected = torch.log_softmax(F.linear(hidden.float(), weight.float()), dim=-1)
    assert got.dtype == torch.float32 and torch.equal(got, expected)
    rounded = torch.log_softmax(model.get_output_embeddings()(hidden).float(), dim=-1)     # the pre-T0003 formula
    assert not torch.equal(got, rounded)
    # The forecast reads the same FP32 layer: selecting the yes / no rows of W gives the same two logits.
    yes_no = policy._head(model, hidden, rows=[policy.yes_id, policy.no_id])
    full = policy._head(model, hidden)
    assert yes_no.dtype == torch.float32
    assert torch.allclose(yes_no, full[:, [policy.yes_id, policy.no_id]], atol=1e-6, rtol=0)


def test_fp32_head_passes_gradients_to_bf16_parameters(tokenizer, policy_cfg):
    model = tiny_model(seed=0).to(torch.bfloat16)
    policy = LMPolicy(model, tokenizer, policy_cfg)
    d = policy.act("goal", [], KeyChestEnv().reset(0), torch.Generator().manual_seed(0))
    logps, p = policy.score(d, forecast=True)
    (logps.sum() + p).backward()
    weight = model.get_output_embeddings().weight
    assert logps.dtype == p.dtype == torch.float32
    assert weight.grad is not None and weight.grad.dtype == torch.bfloat16 and weight.grad.abs().sum() > 0


# ---- (b) same sequences everywhere

@pytest.mark.parametrize("alpha, beta", [(0.0, 0.0), (0.1, 0.05)])
def test_old_logps_equal_sampling_logps_and_forecast_flag_is_shared(tokenizer, policy_cfg, monkeypatch, alpha, beta):
    """No reasoning, forecast on: the recomputed old log-probs are bit for bit the sampling-time ones (FP32 CPU),
    because _fix_old, the reference and the training forward all score the sequence the rollout scored."""
    policy = _no_reasoning_policy(tokenizer, policy_cfg)
    assert policy.cfg.forecast
    groups = _groups(policy)
    calls, original = [], LMPolicy.score

    def spy(self, d, model=None, forecast=False):
        calls.append((model is not None, forecast))
        return original(self, d, model=model, forecast=forecast)

    monkeypatch.setattr(LMPolicy, "score", spy)
    reference = copy.deepcopy(policy.model).requires_grad_(False) if beta > 0 else None
    cfg = GRPOConfig(lam=0.5, alpha=alpha, beta=beta, optimization_passes=2, learning_rate=0.0)
    metrics = Trainer(policy, cfg, reference_model=reference).step(groups)

    decisions = [d for g in groups for ep in g for d in ep.decisions]
    assert decisions and all(d.old_logps == d.behavior_logps for d in decisions)
    assert metrics["sampling_vs_old_max_abs_logp_diff"] == 0.0 and metrics["behavior_old_mean_abs_diff"] == 0.0
    assert metrics["passes"][0]["clip_frac"] == 0.0
    # old (+ reference) once per decision, then one training forward per decision and pass
    assert len(calls) == len(decisions) * ((2 if beta > 0 else 1) + 2)
    assert sum(is_ref for is_ref, _ in calls) == (len(decisions) if beta > 0 else 0)
    assert all(forecast is policy.cfg.forecast for _, forecast in calls)


def test_forecast_flag_follows_policy_config_when_off(tokenizer, policy_cfg, monkeypatch):
    policy = _no_reasoning_policy(tokenizer, policy_cfg, forecast=False)
    groups = _groups(policy)
    seen, original = [], LMPolicy.score
    monkeypatch.setattr(LMPolicy, "score", lambda self, d, model=None, forecast=False: (
        seen.append(forecast), original(self, d, model=model, forecast=forecast))[1])
    Trainer(policy, GRPOConfig(optimization_passes=1, learning_rate=0.0)).step(groups)
    assert seen == [False] * len(seen)
    with pytest.raises(ValueError):
        Trainer(policy, GRPOConfig(alpha=0.1)).step(groups)


def test_alpha_zero_keeps_forecast_out_of_the_loss(policy):
    """forecast=True in the forward, but with alpha = 0 the loss has no Brier term."""
    groups = _groups(policy)
    metrics = Trainer(policy, GRPOConfig(lam=0.0, alpha=0.0, optimization_passes=1, learning_rate=0.0)).step(groups)
    assert metrics["passes"][0]["cal_loss"] == 0.0


# ---- (c) truncated importance sampling

def test_tis_weights_and_loss_by_hand():
    logps = torch.tensor([-1.0, -0.5, -2.0], requires_grad=True)
    old = torch.tensor([-1.2, -0.5, -1.0])
    behavior = torch.tensor([-1.0, -1.5, -1.1])
    w = tis_weights(old, behavior, cap=2.0)
    expected_w = torch.tensor([torch.exp(torch.tensor(-0.2)).item(), 2.0, torch.exp(torch.tensor(0.1)).item()])
    assert torch.allclose(w, expected_w) and not w.requires_grad

    advantage, eps, weight = 0.7, 0.2, 0.25
    plain, _ = decision_loss(logps, old, advantage, weight, eps)
    off, _ = decision_loss(logps, old, advantage, weight, eps, tis_weights=None)
    assert torch.equal(plain, off)                                 # tis off: nothing changes
    ratio = torch.exp(logps.detach() - old)
    surrogate = -torch.minimum(ratio * advantage, ratio.clamp(1 - eps, 1 + eps) * advantage)
    assert torch.allclose(clipped_surrogate(logps, old, advantage, eps), surrogate)
    loss, stats = decision_loss(logps, old, advantage, weight, eps, tis_weights=w)
    assert loss.item() == pytest.approx(weight * float((surrogate * expected_w).sum()), rel=1e-6)

    # The weights are constants: no gradient reaches them, and they never touch the KL or Brier terms.
    w_leaf = w.clone().requires_grad_(True)
    ref = torch.tensor([-1.1, -0.6, -1.9])
    p = torch.tensor(0.8, requires_grad=True)
    with_tis, s1 = decision_loss(logps, old, advantage, weight, eps, ref_logps=ref, beta=0.3, p=p, z=1.0,
                                 cal_weight=0.5, alpha=0.2, tis_weights=w_leaf)
    without, s0 = decision_loss(logps, old, advantage, weight, eps, ref_logps=ref, beta=0.3, p=p, z=1.0,
                                cal_weight=0.5, alpha=0.2)
    with_tis.backward()
    assert w_leaf.grad is None
    assert s1["kl"] == s0["kl"] and s1["brier"] == s0["brier"]
    extra = (with_tis - without).item()
    assert extra == pytest.approx(weight * float((surrogate * (expected_w - 1)).sum()), rel=1e-5)


def test_trainer_tis_off_is_unchanged_and_on_matches_formula(tokenizer, policy_cfg):
    """tis_cap=None: no TIS metrics and the same loss as before. With a cap, the first-pass loss is
    -sum_u weight_u * A_u * w_u (rho = 1), with w_u from the recorded sampling log-probs."""
    def run(tis_cap, perturb):
        policy = LMPolicy(tiny_model(seed=4), tokenizer, policy_cfg)
        groups = _groups(policy)
        if perturb:                        # pretend the sampler's probabilities differed from the training forward
            gen = torch.Generator().manual_seed(0)
            for g in groups:
                for ep in g:
                    for d in ep.decisions:
                        d.behavior_logps = [v + float(torch.randn((), generator=gen)) for v in d.behavior_logps]
        cfg = GRPOConfig(lam=0.5, optimization_passes=1, learning_rate=0.0, tis_cap=tis_cap)
        return groups, Trainer(policy, cfg).step(groups)

    _, base = run(None, perturb=False)
    _, off = run(None, perturb=True)                               # behavior log-probs are ignored when TIS is off
    assert off["passes"][0]["loss"] == base["passes"][0]["loss"]
    assert "tis_mean_w" not in off["passes"][0] and off["behavior_old_mean_abs_diff"] > 0.1

    cap = 1.5
    groups, on = run(cap, perturb=True)
    expected, weights, capped = 0.0, [], 0
    for g in groups:
        n = sum(d.n_terms for ep in g for d in ep.decisions)
        for ep in g:
            for d in ep.decisions:
                for old, behavior in zip(d.old_logps, d.behavior_logps):
                    raw = float(torch.exp(torch.tensor(old - behavior)))
                    weights.append(min(raw, cap))
                    capped += raw > cap
                    expected += -ep.advantage * weights[-1] / (n * len(groups))
    first = on["passes"][0]
    assert first["loss"] == pytest.approx(expected, rel=1e-4, abs=1e-7)
    assert first["loss"] != pytest.approx(base["passes"][0]["loss"], rel=1e-3)
    assert first["tis_mean_w"] == pytest.approx(sum(weights) / len(weights), rel=1e-5)
    assert first["tis_capped_frac"] == pytest.approx(capped / len(weights)) and 0 < capped < len(weights)


def test_tis_cap_and_warmup_validation():
    assert GRPOConfig().tis_cap is None and GRPOConfig().warmup_batches == 0
    with pytest.raises(ValueError):
        GRPOConfig(tis_cap=0.0)
    with pytest.raises(ValueError):
        GRPOConfig(warmup_batches=-1)


# ---- (d) warmup

def test_warmup_learning_rate_per_batch_and_same_for_all_passes(tokenizer, policy_cfg):
    policy = LMPolicy(tiny_model(seed=5), tokenizer, policy_cfg)
    groups = _groups(policy)
    cfg = GRPOConfig(lam=0.5, learning_rate=1e-3, warmup_batches=4, optimization_passes=3)
    trainer = Trainer(policy, cfg)
    used, original = [], trainer.optimizer.step

    def step(*args, **kwargs):
        used.append([g["lr"] for g in trainer.optimizer.param_groups])
        return original(*args, **kwargs)

    trainer.optimizer.step = step
    reported = [trainer.step(groups)["lr"] for _ in range(6)]
    expected = [1e-3 * min(1.0, b / 4) for b in range(1, 7)]
    assert reported == pytest.approx(expected) and trainer.batches_done == 6
    assert len(used) == 18
    for b in range(6):
        assert used[3 * b:3 * b + 3] == [[pytest.approx(expected[b])]] * 3


def test_no_warmup_leaves_the_optimizer_alone(tokenizer, policy_cfg):
    policy = LMPolicy(tiny_model(seed=5), tokenizer, policy_cfg)
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.123)
    trainer = Trainer(policy, GRPOConfig(lam=0.5, learning_rate=1e-3), optimizer=optimizer)
    metrics = trainer.step(_groups(policy))
    assert optimizer.param_groups[0]["lr"] == 0.123 and metrics["lr"] == 1e-3


# ---- (e) kl_old

def test_kl_old_is_zero_at_first_pass_and_positive_after_an_update(tokenizer, policy_cfg):
    policy = LMPolicy(tiny_model(seed=6), tokenizer, policy_cfg)
    metrics = Trainer(policy, GRPOConfig(lam=0.5, learning_rate=1e-2, optimization_passes=2)).step(_groups(policy))
    first, second = metrics["passes"]
    assert first["kl_old"] == 0.0 and first["clip_frac"] == 0.0
    assert second["kl_old"] > 0.0
