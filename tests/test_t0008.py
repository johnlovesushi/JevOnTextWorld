"""T0008: verbal confidence (constrained sampling of " 0" .. " 100" after the chosen option), shared and decoupled
confidence advantages, mean-centred advantages and the section-6 diagnostics of E0012."""
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from jevtw.config import GRPOConfig, PolicyConfig
from jevtw.diagnostics import auc, forecast_records, logistic_fit, prediction_metrics, verbal_metrics
from jevtw.envs import KeyChestEnv
from jevtw.losses import conf_term_loss
from jevtw.policy import LMPolicy
from jevtw.rewards import bin_values, decoupled_conf_advantage, group_advantages, score
from jevtw.rollout import collect_group
from jevtw.testing import ByteTokenizer, tiny_model
from jevtw.trainer import Trainer, assign_returns

from test_policy_trainer import ENDINGS, _manual_decision, ref_forecasts, ref_option_scores, ref_reasoning_logps

ROOT = Path(__file__).resolve().parents[1]


def _verbal_cfg(policy_cfg, **kw):
    return PolicyConfig(**{**policy_cfg.__dict__, "confidence_mode": "verbal", "verbal_prompt": "\nConf:", **kw})


def flat_model(seed=0):
    """The tiny random model with a near-uniform output layer that favours " ", the digits and EOS. Under the byte
    tokenizer the bins are 2-4 bytes + EOS, so a plain random model puts almost all of pi^c on the shortest one
    (" 0"); restricting the mass to the 12 tokens the answers use spreads pi^c over the bins (entropy ~1.6 nats)."""
    model = tiny_model(seed=seed)
    old = model.lm_head
    head = torch.nn.Linear(old.in_features, old.out_features, bias=True)
    with torch.no_grad():
        head.weight.copy_(old.weight * 0.05)
        head.bias.fill_(-8.0)
        head.bias[[ord(" "), *range(ord("0"), ord("9") + 1), ByteTokenizer.eos_token_id]] = 0.0
    model.lm_head = head
    return model


def _groups(policy, n_games=2, group_size=4, max_actions=3):
    return [collect_group(policy, KeyChestEnv, f"keychest-{g}", g, list(range(100 * g, 100 * g + group_size)),
                          max_actions) for g in range(n_games)]


def ref_conf(policy, model, d, j):
    """Naive reference: one full forward per answer m of x + y^j + u + y^m, summing the log-probs of y^m."""
    out = []
    prefix = d.action_context_ids + d.option_ids[j] + policy.verbal_prompt_ids
    for y in policy.bin_ids:
        logprobs = torch.log_softmax(model(input_ids=torch.tensor([prefix + y])).logits[0].float(), dim=-1)
        out.append(sum(logprobs[len(prefix) - 1 + i, t] for i, t in enumerate(y)))
    return torch.stack(out)


# ---- 1. the folded tree equals full-context forwards (numbers and gradients)

def test_bins_and_prompt(tokenizer, policy_cfg):
    policy = LMPolicy(tiny_model(seed=0), tokenizer, _verbal_cfg(policy_cfg))
    texts = [tokenizer.decode(b) for b in policy.bin_ids]
    assert texts == [f" {10 * m}" for m in range(11)] and all(b[-1] == tokenizer.eos_token_id for b in policy.bin_ids)
    assert bin_values(11)[3] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        PolicyConfig(confidence_mode="verbal", forecast_format="no_eos")
    with pytest.raises(ValueError):
        PolicyConfig(confidence_mode="words")


@pytest.mark.parametrize("ending", ENDINGS)
@pytest.mark.parametrize("forecast", [False, True])
def test_verbal_forward_matches_reference(tokenizer, policy_cfg, ending, forecast):
    model = tiny_model(seed=1)
    policy = LMPolicy(model, tokenizer, _verbal_cfg(policy_cfg))
    d = _manual_decision(policy, ending, n_options=12, action=10)
    with torch.no_grad():
        r_lp, scores, s, conf = policy.verbal_forward(model, d, forecast, [10, 3])
        torch.testing.assert_close(scores, ref_option_scores(policy, model, d.action_context_ids, d.option_ids),
                                   atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(r_lp, ref_reasoning_logps(policy, model, d), atol=2e-5, rtol=1e-5)
        for b, j in enumerate([10, 3]):
            torch.testing.assert_close(conf[b], ref_conf(policy, model, d, j), atol=2e-5, rtol=1e-5)
        if forecast:
            p = ref_forecasts(policy, model, d)
            torch.testing.assert_close(torch.sigmoid(s), p, atol=2e-5, rtol=1e-5)
            _, _, s_plain = policy.decision_forward(model, d, forecast=True)
            torch.testing.assert_close(s, s_plain, atol=2e-5, rtol=1e-5)
        else:
            assert s is None


def test_verbal_forward_matches_reference_101_bins(tokenizer, policy_cfg):
    """M = 101 (E0012d / e): answers " 0" .. " 100" in steps of 1, of 1-3 digits, all scored in the folded tree."""
    model = tiny_model(seed=1)
    policy = LMPolicy(model, tokenizer, _verbal_cfg(policy_cfg, confidence_bins=101))
    assert [tokenizer.decode(b) for b in policy.bin_ids] == [f" {m}" for m in range(101)]
    assert bin_values(101)[73] == pytest.approx(0.73)
    d = _manual_decision(policy, ENDINGS[0], n_options=12, action=10)
    with torch.no_grad():
        _, scores, _, conf = policy.verbal_forward(model, d, False, range(12))     # as in the rollout (`decide`)
        torch.testing.assert_close(scores, ref_option_scores(policy, model, d.action_context_ids, d.option_ids),
                                   atol=2e-5, rtol=1e-5)
        for j in (10, 3):
            torch.testing.assert_close(conf[j], ref_conf(policy, model, d, j), atol=2e-5, rtol=1e-5)
        _, _, log_pc = policy.score_with_confidence(d)                             # as in training
        torch.testing.assert_close(log_pc, torch.log_softmax(conf[10], 0), atol=2e-5, rtol=1e-5)


def test_verbal_forward_chunks_the_output_layer(tokenizer, policy_cfg, monkeypatch):
    """E0012d/e ran out of GPU memory: the rollout's `verbal_forward` with M = 101 sent every confidence token of all K
    options to the output layer in one call (~400 K rows x the vocabulary in fp32). The rows now go in chunks of
    LOGPROB_CHUNK; the numbers and gradients are those of one unchunked call."""
    model = tiny_model(seed=3)
    policy = LMPolicy(model, tokenizer, _verbal_cfg(policy_cfg, confidence_bins=101))
    d = _manual_decision(policy, ENDINGS[0], n_options=12, action=5)
    d.conf_bin = 42
    calls = []                                                              # rows of every full-vocabulary call
    head = LMPolicy._head

    def spy(model, hidden, rows=None):
        if rows is None:
            calls.append(hidden.shape[0])
        return head(model, hidden, rows)

    monkeypatch.setattr(LMPolicy, "_head", staticmethod(spy))

    def run(chunk):
        monkeypatch.setattr(LMPolicy, "LOGPROB_CHUNK", chunk, raising=False)
        calls.clear()
        with torch.no_grad():
            out = policy.verbal_forward(model, d, True, range(12))          # all K branches, as in `decide`
        sampled = list(calls)
        model.zero_grad()
        logps, _, log_pc = policy.score_with_confidence(d)                  # one branch, with gradients (training)
        (logps.sum() + 0.3 * log_pc[42]).backward()
        return out, sampled, {n: p.grad.clone() for n, p in model.named_parameters()}

    (r1, s1, _, c1), rows_one, g1 = run(10 ** 9)
    assert len(rows_one) == 1 and rows_one[0] > 12 * 101                   # before the fix: one call with every row
    (r2, s2, _, c2), rows_chunked, g2 = run(97)
    assert max(rows_chunked) <= 97 and sum(rows_chunked) == rows_one[0] and len(rows_chunked) > 1
    torch.testing.assert_close(s2, s1, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(c2, c1, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(r2, r1, atol=1e-6, rtol=1e-6)
    for n in g1:
        torch.testing.assert_close(g2[n], g1[n], atol=1e-6, rtol=1e-5, msg=lambda m, n=n: f"{n}: {m}")


def test_verbal_forward_gradients_match_reference(tokenizer, policy_cfg):
    model = tiny_model(seed=2)
    policy = LMPolicy(model, tokenizer, _verbal_cfg(policy_cfg))
    d = _manual_decision(policy, "eos", n_options=12, action=4)
    d.conf_bin = 7
    logps, _, log_pc = policy.score_with_confidence(d)
    (logps.sum() + 0.7 * log_pc[7]).backward()
    fast = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()
    ref_lq = torch.log_softmax(ref_option_scores(policy, model, d.action_context_ids, d.option_ids), 0)[4]
    ref_lpc = torch.log_softmax(ref_conf(policy, model, d, 4), 0)[7]
    (ref_reasoning_logps(policy, model, d).sum() + ref_lq + 0.7 * ref_lpc).backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(fast[n], p.grad, atol=1e-5, rtol=1e-4, msg=lambda m, n=n: f"{n}: {m}")


# ---- 2. sampling records pi^c, and score() returns it

def test_sampled_confidence_equals_score(tokenizer, policy_cfg):
    policy = LMPolicy(flat_model(), tokenizer, _verbal_cfg(policy_cfg))
    groups = _groups(policy)
    seen_bins = set()
    for ep in (ep for g in groups for ep in g):
        for d in ep.decisions:
            assert len(d.conf_logps) == 11 and sum(math.exp(v) for v in d.conf_logps) == pytest.approx(1.0)
            assert d.conf_behavior_logp == d.conf_logps[d.conf_bin] and d.conf == d.conf_bin / 10
            assert 0 <= d.conf_mean <= 1 and 0 <= d.conf_entropy <= math.log(11) + 1e-9
            with torch.no_grad():
                logps, s, log_pc = policy.score_with_confidence(d, forecast=True)
                plain, s_plain = policy.score(d, forecast=True)
            torch.testing.assert_close(log_pc, torch.tensor(d.conf_logps), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(logps, torch.tensor(d.behavior_logps), atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(logps, plain, atol=1e-5, rtol=1e-5)
            assert float(s) == pytest.approx(float(s_plain), abs=1e-5) and d.p is not None
            seen_bins.add(d.conf_bin)
    assert len(seen_bins) > 1                       # a near-uniform random model draws several bins


def test_logit_mode_draws_nothing_extra(tokenizer, policy_cfg):
    """confidence_mode = "logit" draws no m_t: the same seeds give the same episodes as an explicit default."""
    a = LMPolicy(tiny_model(seed=0), tokenizer, policy_cfg)
    b = LMPolicy(tiny_model(seed=0), tokenizer, PolicyConfig(**{**policy_cfg.__dict__, "confidence_mode": "logit"}))
    ga, gb = _groups(a), _groups(b)
    for ea, eb in zip((e for g in ga for e in g), (e for g in gb for e in g)):
        assert [d.command for d in ea.decisions] == [d.command for d in eb.decisions]
        assert all(d.conf_bin is None for d in ea.decisions)


# ---- 3. returns and advantages

def test_decoupled_advantage_by_hand():
    logps = [math.log(v) for v in (0.2, 0.3, 0.5)]           # M = 3: v = 0, 0.5, 1
    # z = 1: S = 0, 0.75, 1; baseline 0.2*0 + 0.3*0.75 + 0.5*1 = 0.725
    assert decoupled_conf_advantage(logps, 2, 1) == pytest.approx(1 - 0.725)
    assert decoupled_conf_advantage(logps, 0, 1) == pytest.approx(0 - 0.725)
    # z = 0: S = 1, 0.75, 0; baseline 0.2 + 0.225 = 0.425
    assert decoupled_conf_advantage(logps, 1, 0) == pytest.approx(0.75 - 0.425)


def test_mean_advantages():
    assert group_advantages([1.0, 0.0, 0.0, 0.3], norm="mean") == pytest.approx([0.675, -0.325, -0.325, -0.025])
    assert group_advantages([0.2, 0.2], norm="mean") == [0.0, 0.0]
    with pytest.raises(ValueError):
        GRPOConfig(adv_norm="max")


def test_assign_returns_uses_sampled_confidence(tokenizer, policy_cfg):
    policy = LMPolicy(tiny_model(seed=0), tokenizer, _verbal_cfg(policy_cfg, forecast=False))
    groups = _groups(policy)
    cfg = GRPOConfig(lam=0.3, adv_norm="mean", conf_advantage="decoupled")
    assign_returns(groups, cfg)
    for group in groups:
        mean = sum(ep.ret for ep in group) / len(group)
        for ep in group:
            b = 1 - sum((d.conf - ep.z) ** 2 for d in ep.decisions) / ep.length
            assert ep.bonus == pytest.approx(b) and ep.ret == pytest.approx(ep.z + 0.3 * b)
            assert ep.advantage == pytest.approx(ep.ret - mean)
            for d in ep.decisions:
                assert d.conf_adv == pytest.approx(decoupled_conf_advantage(d.conf_logps, d.conf_bin, ep.z))
                assert d.p is None


# ---- 4. both estimators have the same expectation; proposition 2

def test_shared_and_decoupled_estimators_agree_in_expectation():
    """One decision with pi^c = softmax(theta) over M = 11 bins and an outcome z ~ Bernoulli(p) independent of m
    (as on KeyChest, where c_t never enters the history). With G rollouts per group:
      shared     grad log pi(m_i) (R_i - mean R), R = z + lam S(c, z)   -> (1 - 1/G) lam grad E[S]
      decoupled  grad log pi(m_i) (S(c_i, z_i) - sum_m pi(m) S(v_m, z_i)) -> grad E[S]
    Monte Carlo over 200k groups against the exact gradient; the decoupled estimator also has the lower variance."""
    gen = torch.Generator().manual_seed(0)
    m, p, lam, g, n = 11, 0.3, 0.3, 8, 200_000
    theta = torch.randn(m, generator=gen, dtype=torch.float64)
    v = torch.linspace(0, 1, m, dtype=torch.float64)
    pi = torch.softmax(theta, 0)
    expected_s = 1 - (v - p) ** 2 - p * (1 - p)                       # E_z[S(v_m, z)], proposition 2
    th = theta.clone().requires_grad_(True)
    (torch.softmax(th, 0) * expected_s).sum().backward()
    exact = th.grad

    bins = torch.multinomial(pi, n * g, replacement=True, generator=gen).view(n, g)
    z = (torch.rand(n, g, generator=gen, dtype=torch.float64) < p).double()
    s = 1 - (v[bins] - z) ** 2
    grad_logpi = torch.nn.functional.one_hot(bins, m).double() - pi           # d log pi(m) / d theta
    r = z + lam * s
    shared = (grad_logpi * (r - r.mean(1, keepdim=True))[..., None]).mean(1)          # per group, per bin
    baseline = (pi[None, None] * (1 - (v[None, None] - z[..., None]) ** 2)).sum(-1)
    decoupled = (grad_logpi * (s - baseline)[..., None]).mean(1)
    se = decoupled.std(0) / math.sqrt(n)
    assert torch.all((decoupled.mean(0) - exact).abs() < 5 * se + 1e-4)
    scaled = shared / ((1 - 1 / g) * lam)
    se_s = scaled.std(0) / math.sqrt(n)
    assert torch.all((scaled.mean(0) - exact).abs() < 5 * se_s + 1e-4)
    assert float(decoupled.var(0).sum()) < float(scaled.var(0).sum())


def test_proposition_2():
    values = bin_values(11)
    for p in [i / 37 for i in range(38)]:
        expected = [p * score(v, 1) + (1 - p) * score(v, 0) for v in values]
        assert expected == pytest.approx([1 - (v - p) ** 2 - p * (1 - p) for v in values])
        best = max(range(11), key=lambda i: expected[i])
        assert abs(values[best] - p) <= 1 / (2 * 10) + 1e-12


# ---- 5. one update

@pytest.mark.parametrize("mode", ["shared", "decoupled"])
def test_verbal_trainer_step(tokenizer, policy_cfg, mode):
    policy = LMPolicy(flat_model(), tokenizer, _verbal_cfg(policy_cfg, forecast=False))
    reference = copy.deepcopy(policy.model).requires_grad_(False)
    cfg = GRPOConfig(lam=0.3 if mode == "shared" else 0.0, beta=0.04, adv_norm="mean", conf_advantage=mode,
                     conf_coef=0.3 if mode == "decoupled" else 1.0, learning_rate=1e-3, optimization_passes=2)
    groups = _groups(policy, n_games=2, group_size=4)
    before = {n: q.detach().clone() for n, q in policy.model.named_parameters()}
    m = Trainer(policy, cfg, reference_model=reference).step(groups)
    assert m["sampling_vs_old_max_abs_logp_diff"] < 1e-4
    first = m["passes"][0]
    assert first["conf_clip_frac"] == 0.0 and first["kl"] == pytest.approx(0, abs=1e-9)
    assert first["conf_kl"] == pytest.approx(0, abs=1e-9) and math.isfinite(first["loss"])
    assert {"mean_conf", "mean_conf_hat", "conf_entropy", "conf_brier"} <= set(m)
    assert ("mean_abs_conf_adv" in m) == (mode == "decoupled")
    assert any(not torch.equal(before[n], q) for n, q in policy.model.named_parameters())


def test_confidence_term_gradient(tokenizer, policy_cfg):
    """With zero action advantage the only policy gradient is weight * kappa * A^c * grad log pi^c(m_t)."""
    logp = torch.tensor(-1.2, dtype=torch.float64, requires_grad=True)
    loss, stats = conf_term_loss(logp, -1.2, advantage=0.5, weight=0.1, eps=0.2, coef=0.3)
    loss.backward()
    assert float(logp.grad) == pytest.approx(-0.1 * 0.3 * 0.5) and stats["clipped"] == 0
    logp2 = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)       # rho = e^0.7 > 1.2 with A > 0
    loss2, stats2 = conf_term_loss(logp2, -1.2, advantage=0.5, weight=0.1, eps=0.2, coef=0.3)
    loss2.backward()
    assert float(logp2.grad) == 0.0 and stats2["clipped"] == 1


def test_decoupled_zero_action_advantage_still_trains_confidence(tokenizer, policy_cfg):
    """All episodes lose: the action advantage is 0 everywhere, A^c is not, so the episodes stay active."""
    policy = LMPolicy(tiny_model(seed=0), tokenizer, _verbal_cfg(policy_cfg, forecast=False))
    groups = _groups(policy, n_games=1, group_size=4, max_actions=1)
    for ep in groups[0]:
        ep.won = False
    trainer = Trainer(policy, GRPOConfig(conf_advantage="decoupled", adv_norm="mean", learning_rate=1e-3),
                      reference_model=None)
    m = trainer.step(groups)
    assert m["active_groups"] == 0 and m["passes"][0]["grad_norm"] > 0


# ---- 6. diagnostics

def test_prediction_metrics_by_hand():
    rows = [(0.9, 0.9, 1, 0, "a", 0), (0.8, 0.8, 1, 1, "a", 0), (0.2, 0.2, 0, 0, "a", 1),
            (0.05, 0.05, 0, 0, "b", 2), (0.95, 0.95, 0, 1, "b", 2)]
    m = prediction_metrics(rows, "x_")
    assert m["x_offset"] == pytest.approx(abs(2.9 / 5 - 2 / 5))
    # bins: 0 -> (0.05, 0), 2 -> (0.2, 0), 8 -> (0.8, 1), 9 -> (0.9, 1), (0.95, 0)
    ece = (0.05 + 0.2 + 0.2 + 2 * abs(0.925 - 0.5)) / 5
    assert m["x_ece"] == pytest.approx(ece)
    assert m["x_auc_within"] == pytest.approx(1.0)            # only game a has wins and losses
    assert m["x_saturation"] == 0.0
    clip = [0.9, 0.8, 0.2, 0.05, 0.95]
    ll = -(math.log(0.9) + math.log(0.8) + math.log(0.8) + math.log(0.95) + math.log(0.05)) / 5
    assert m["x_logloss"] == pytest.approx(ll) and clip
    assert m["x_auc_step_gain"] == pytest.approx(m["x_auc_t_plus"] - m["x_auc_t_only"])


def test_logistic_fit_separates():
    t = list(range(20))
    labels = [int(x >= 10) for x in t]
    assert auc(logistic_fit([[x] for x in t], labels), labels) == 1.0
    labels = [int(x % 2 == 0) for x in t]                      # unrelated to t: little separation
    assert auc(logistic_fit([[x] for x in t], labels), labels) < 0.7


def test_verbal_metrics_and_records(tokenizer, policy_cfg):
    policy = LMPolicy(flat_model(), tokenizer, _verbal_cfg(policy_cfg))
    groups = _groups(policy)
    eps = [ep for g in groups for ep in g]
    eps[0].won = True                                           # both classes
    m = verbal_metrics(eps)
    ds = [(d, ep.z) for ep in eps for d in ep.decisions]
    assert m["conf_brier"] == pytest.approx(sum((d.conf_mean - z) ** 2 for d, z in ds) / len(ds))
    assert m["conf_brier_sampled"] == pytest.approx(sum((d.conf - z) ** 2 for d, z in ds) / len(ds))
    assert m["mean_conf_hat"] == pytest.approx(sum(d.conf_mean for d, _ in ds) / len(ds))
    assert m["conf_entropy"] > 1.0 and m["conf_auc"] is not None and "conf_ece" in m
    rec = forecast_records(eps[:1])[0]["steps"][0]
    assert rec["c"] == eps[0].decisions[0].conf and len(rec["pi_c"]) == 11 and rec["p"] is not None
    assert verbal_metrics([]) == {}


# ---- 7. --smoke for the three groups of E0012

GROUPS = {
    "A": {"policy": {"platt": True},
          "grpo": {"alpha": 0.1, "cal_loss": "bce", "cal_grad": "platt_only", "adv_norm": "mean", "beta": 0.04}},
    "B": {"policy": {"confidence_mode": "verbal", "forecast": False},
          "grpo": {"lam": 0.3, "adv_norm": "mean", "beta": 0.04, "conf_advantage": "shared"}},
    "C": {"policy": {"confidence_mode": "verbal", "forecast": False},
          "grpo": {"adv_norm": "mean", "beta": 0.04, "conf_advantage": "decoupled", "conf_coef": 0.3}},
}


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_smoke_groups(tmp_path, group):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(GROUPS[group]))
    env = {**os.environ, "OMP_NUM_THREADS": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    proc = subprocess.run([sys.executable, "-m", "jevtw.train", "--smoke", "--config", str(config), "--no-save",
                           "--batches", "1", "--out", str(tmp_path / "run")], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    train = [x for x in lines if x["split"] == "train"]
    val = [x for x in lines if x["split"] != "train"]
    assert len(train) == 1 and val and all(math.isfinite(p["loss"]) for p in train[0]["passes"])
    if group == "A":
        assert val[-1]["forecast_ece"] is not None and "conf_brier" not in val[-1]
    else:
        assert val[-1]["conf_entropy"] > 0 and val[-1]["forecast_brier"] is None
        assert train[0]["mean_conf"] is not None
