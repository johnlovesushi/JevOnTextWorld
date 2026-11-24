"""T0005: forecast logits and diagnostics, forecast formats, cross-entropy calibration, Platt scaling, checkpoints
with tokenizer / platt.json."""
import copy
import itertools
import json
import math
import random

import pytest
import torch

from jevtw import train
from jevtw.config import Config, GRPOConfig, PolicyConfig
from jevtw.diagnostics import auc, forecast_metrics, forecast_records
from jevtw.envs import KeyChestEnv
from jevtw.losses import bce, brier, calibration_weights, decision_loss
from jevtw.policy import PLATT_FILE, Decision, LMPolicy, chat_forecast_suffix
from jevtw.rollout import Episode, collect_group
from jevtw.testing import tiny_model
from jevtw.trainer import Trainer

from snapshot_t0005 import PATH as SNAPSHOT, compute as snapshot_compute
from test_policy_trainer import ENDINGS, _manual_decision, ref_option_scores

FORMATS = ("after_eos", "no_eos", "chat")


def _policy(tokenizer, policy_cfg, seed=0, model=None, **overrides):
    cfg = PolicyConfig(**{**policy_cfg.__dict__, "chat_yes_text": "Y", "chat_no_text": "N", **overrides})
    return LMPolicy(tiny_model(seed=seed) if model is None else model, tokenizer, cfg)


def _groups(policy, n_games=2, group_size=4, max_actions=3):
    return [collect_group(policy, KeyChestEnv, f"keychest-{g}", g, list(range(100 * g, 100 * g + group_size)),
                          max_actions) for g in range(n_games)]


def _logit(w):
    return math.log(w / (1 - w))


# ---- 1. defaults reproduce the code before T0005

def test_default_configuration_reproduces_pre_t0005_numbers():
    """tests/snapshots/t0005_default.json was written by the code before T0005 (commit 59ceadc): option scores l_j,
    sampling log-probs, p_t / p_j, the losses of both passes of one update and the updated parameters."""
    old, new = json.loads(SNAPSHOT.read_text()), json.loads(json.dumps(snapshot_compute()))
    assert new["commands"] == old["commands"]

    worst = {}

    def walk(a, b, path):
        """Every number of the old file against the new one (new code may add keys, never change old ones)."""
        if isinstance(a, dict):
            assert set(a) <= set(b), (path, set(a) - set(b))
            for key in a:
                walk(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for u, v in zip(a, b):
                walk(u, v, path)
        elif not isinstance(a, str):
            worst[path] = max(worst.get(path, 0.0), abs(a - b))

    walk(old, new, "")
    # Identical arithmetic: exactly equal on the machine that wrote the file (same torch build and thread count).
    # The tolerance only allows for another BLAS or library version on another machine (1.9e-6 was measured with
    # transformers 5.24).
    assert max(worst.values()) <= 1e-5, {k: v for k, v in worst.items() if v > 0}


def test_new_options_default_to_old_behaviour(policy):
    p, g = PolicyConfig(), GRPOConfig()
    assert (p.forecast_format, p.platt, g.cal_loss, g.cal_grad, g.platt_lr) == ("after_eos", False, "brier", "full", 1e-2)
    assert policy.platt_a is None and policy.platt_parameters() == []
    assert policy.forecast_tail_ids == policy.confidence_prompt_ids
    s = torch.tensor([-3.0, 0.2, 40.0])
    assert torch.equal(policy.forecast_prob(s), torch.sigmoid(s)) and policy.calibrated_logit(s) is s
    assert len(Trainer(policy, g).optimizer.param_groups) == 1
    for bad in ({"policy": {"forecast_format": "x"}}, {"grpo": {"cal_loss": "x"}}, {"grpo": {"cal_grad": "x"}},
                {"grpo": {"platt_lr": 0}}):
        with pytest.raises(ValueError):
            Config.from_dict(bad)


def test_decision_stores_logits_and_probabilities(policy):
    env = KeyChestEnv()
    obs = env.reset(0)
    d = policy.act(env.objective, [], obs, torch.Generator().manual_seed(0))
    assert len(d.forecast_logits_options) == len(obs.commands)
    assert d.forecast_logit == d.forecast_logits_options[d.action]
    assert d.p_options == [float(torch.sigmoid(torch.tensor(s))) for s in d.forecast_logits_options]
    with torch.no_grad():
        _, _, s = policy.decision_forward(policy.model, d, forecast=True)
        _, s_t = policy.score(d, forecast=True)
    assert s.tolist() == pytest.approx(d.forecast_logits_options, abs=1e-5) and s_t.item() == s[d.action].item()


# ---- 2. forecast formats

def _naive_logits(policy, d):
    """One full sequence per option, full logits: the reference for s_j."""
    out = []
    for opt in d.option_ids:
        ids = policy._forecast_sequence(d.action_context_ids, opt)
        logits = policy.model(input_ids=torch.tensor([ids])).logits[0, -1].float()
        out.append(logits[policy.yes_id] - logits[policy.no_id])
    return torch.stack(out)


@pytest.mark.parametrize("ending", ENDINGS)
def test_forecast_formats(tokenizer, policy_cfg, ending):
    model = tiny_model(seed=0)
    policies = {f: _policy(tokenizer, policy_cfg, model=model, forecast_format=f) for f in FORMATS}
    d = _manual_decision(policies["after_eos"], ending)
    x, eos = d.action_context_ids, tokenizer.eos_token_id
    z = policies["after_eos"].confidence_prompt_ids
    chat = tokenizer(chat_forecast_suffix(policy_cfg.forecast_question))["input_ids"]
    opt = d.option_ids[3]
    assert opt[-1] == eos
    assert policies["after_eos"]._forecast_sequence(x, opt) == x + opt + z
    assert policies["no_eos"]._forecast_sequence(x, opt) == x + opt[:-1] + z and eos not in opt[:-1]
    assert policies["chat"]._forecast_sequence(x, opt) == x + opt + chat

    with torch.no_grad():
        _, plain, none = policies["after_eos"].decision_forward(model, d, forecast=False)
        reference = ref_option_scores(policies["after_eos"], model, x, d.option_ids)
        results = {f: policies[f].decision_forward(model, d, forecast=True) for f in FORMATS}
        assert none is None
        for f in FORMATS:
            reasoning_lp, scores, s = results[f]
            # l_j does not depend on the format (and still contains the EOS term)
            # Same arithmetic on the same rows, but the rows are padded to another length and batched with other
            # rows, so BLAS may round differently: measured 0 or 1 float32 ulp (1.9e-6 at |l_j| ~ 17).
            assert torch.allclose(scores, results["after_eos"][1], atol=1e-5, rtol=0)
            assert torch.allclose(scores, plain, atol=1e-5, rtol=0)
            assert torch.allclose(scores, reference, atol=1e-4)
            assert torch.allclose(reasoning_lp, results["after_eos"][0], atol=1e-5, rtol=0)
            assert s.shape == (len(d.option_ids),)
            assert torch.allclose(s, _naive_logits(policies[f], d), atol=1e-4, rtol=0), f
        # the three formats really read different things
        assert not torch.allclose(results["no_eos"][2], results["after_eos"][2], atol=1e-3)
        assert not torch.allclose(results["chat"][2], results["after_eos"][2], atol=1e-3)


def test_after_eos_option_scores_are_bit_identical_to_the_default_path(tokenizer, policy_cfg):
    """forecast_format = "after_eos" is the default path: same tensors, bit for bit."""
    model = tiny_model(seed=0)
    default, explicit = _policy(tokenizer, policy_cfg, model=model), _policy(tokenizer, policy_cfg, model=model,
                                                                            forecast_format="after_eos")
    d = _manual_decision(default, "cap")
    with torch.no_grad():
        a, b = default.decision_forward(model, d, forecast=True), explicit.decision_forward(model, d, forecast=True)
    assert all(torch.equal(u, v) for u, v in zip(a, b))


@pytest.mark.parametrize("fmt", ["no_eos", "chat"])
def test_forecast_format_gradient_matches_naive(tokenizer, policy_cfg, fmt):
    policy = _policy(tokenizer, policy_cfg, forecast_format=fmt)
    d = _manual_decision(policy, "eos")
    params = [w for w in policy.model.parameters() if w.requires_grad]
    logps, s = policy.score(d, forecast=True)
    new = torch.autograd.grad(logps[-1] + s, params, allow_unused=True)
    log_q = torch.log_softmax(ref_option_scores(policy, policy.model, d.action_context_ids, d.option_ids), dim=0)
    ref = torch.autograd.grad(log_q[d.action] + _naive_logits(policy, d)[d.action], params, allow_unused=True)
    flat = lambda gs: torch.cat([torch.zeros_like(w).flatten() if g is None else g.flatten()
                                 for g, w in zip(gs, params)])
    assert flat(ref).norm() > 0 and (flat(new) - flat(ref)).norm() / flat(ref).norm() < 1e-4


@pytest.mark.parametrize("fmt", FORMATS)
def test_rollout_and_training_use_the_same_forecast_sequence(tokenizer, policy_cfg, fmt):
    policy = _policy(tokenizer, policy_cfg, forecast_format=fmt)
    env = KeyChestEnv()
    d = policy.act(env.objective, [], env.reset(0), torch.Generator().manual_seed(1))
    assert d.confidence_ids == policy._forecast_sequence(d.action_context_ids, d.option_ids[d.action])
    with torch.no_grad():
        logps, s = policy.score(d, forecast=True)
        assert torch.allclose(_naive_logits(policy, d), torch.tensor(d.forecast_logits_options), atol=1e-4)
    assert s.item() == pytest.approx(d.forecast_logit, abs=1e-5)
    assert torch.allclose(logps, torch.tensor(d.behavior_logps), atol=1e-4)


def test_chat_format_needs_single_token_answers(tokenizer, policy_cfg):
    with pytest.raises(ValueError, match="exactly one"):          # "Yes" is three byte tokens
        LMPolicy(tiny_model(), tokenizer, PolicyConfig(**{**policy_cfg.__dict__, "forecast_format": "chat"}))
    assert chat_forecast_suffix("Q?") == ("\n<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n"
                                          "<think>\n\n</think>\n\n")


# ---- 3. cross-entropy calibration

@pytest.mark.parametrize("z", [0.0, 1.0])
@pytest.mark.parametrize("u", [-25.0, -2.0, 0.3, 4.0, 25.0])
def test_bce_value_and_gradient(u, z):
    logit = torch.tensor(u, requires_grad=True)
    loss = bce(logit, z)
    p = 1 / (1 + math.exp(-u))
    expected = math.log1p(math.exp(-u)) if z == 1.0 else math.log1p(math.exp(u))   # -log p, -log(1 - p)
    assert loss.item() == pytest.approx(expected, rel=1e-5, abs=1e-9)
    (grad,) = torch.autograd.grad(loss, logit)
    assert grad.item() == pytest.approx(p - z, abs=1e-6)                           # d/du = p - z
    logit2 = torch.tensor(u, requires_grad=True)
    (grad_brier,) = torch.autograd.grad(brier(torch.sigmoid(logit2), z), logit2)
    assert grad_brier.item() == pytest.approx(2 * (p - z) * p * (1 - p), abs=1e-6)
    if u == 25.0 and z == 0.0:                                    # the E0003d situation: p ~ 1 on a lost episode
        assert grad.item() > 0.99 and abs(grad_brier.item()) < 1e-9


def test_decision_loss_with_bce():
    logps, old = torch.tensor([-1.0, -0.5]), torch.tensor([-1.0, -0.5])
    logit = torch.tensor(1.5, requires_grad=True)
    kw = dict(z=0.0, cal_weight=0.25, alpha=2.0)
    base, _ = decision_loss(logps, old, 0.7, 0.1, 0.2)
    loss, stats = decision_loss(logps, old, 0.7, 0.1, 0.2, cal_loss="bce", logit=logit, **kw)
    assert (loss - base).item() == pytest.approx(2.0 * 0.25 * math.log1p(math.exp(1.5)), rel=1e-6)
    assert stats["cal"] == pytest.approx(math.log1p(math.exp(1.5)), rel=1e-6) and "brier" not in stats
    loss_b, stats_b = decision_loss(logps, old, 0.7, 0.1, 0.2, p=torch.sigmoid(logit), **kw)
    assert stats_b["cal"] == stats_b["brier"] == pytest.approx((1 / (1 + math.exp(-1.5))) ** 2, rel=1e-6)
    with pytest.raises(ValueError):
        decision_loss(logps, old, 0.7, 0.1, 0.2, cal_loss="bce", **kw)


@pytest.mark.parametrize("kind", ["brier", "bce"])
def test_trainer_calibration_loss_and_logged_values(tokenizer, policy_cfg, kind):
    policy = _policy(tokenizer, policy_cfg, seed=2)
    groups = _groups(policy)
    cfg = GRPOConfig(games_per_batch=2, group_size=4, max_actions=3, alpha=0.5, cal_loss=kind, learning_rate=1e-3)
    counts = [[[d.n_terms for d in ep.decisions] for ep in group] for group in groups]
    w = calibration_weights(counts)
    f = (lambda s, z: (1 / (1 + math.exp(-s)) - z) ** 2) if kind == "brier" else \
        (lambda s, z: math.log1p(math.exp(-s)) if z else math.log1p(math.exp(s)))
    expected = sum(w[g][i][t] * f(d.forecast_logit, ep.z) for g, group in enumerate(groups)
                   for i, ep in enumerate(group) for t, d in enumerate(ep.decisions))
    metrics = Trainer(policy, cfg).step(groups)
    assert metrics["cal_loss_kind"] == kind
    assert metrics["cal_loss"] == pytest.approx(expected, rel=1e-5)               # rollout forecasts
    assert metrics["passes"][0]["cal_loss"] == pytest.approx(expected, rel=1e-4)  # same function, training forward
    decisions = [d for group in groups for ep in group for d in ep.decisions]
    assert metrics["mean_forecast_logit"] == pytest.approx(sum(d.forecast_logit for d in decisions) / len(decisions))
    assert "platt_a" not in metrics


def test_bonus_uses_brier_whatever_the_calibration_loss(tokenizer, policy_cfg):
    policy = _policy(tokenizer, policy_cfg, seed=2)
    groups = _groups(policy)
    Trainer(policy, GRPOConfig(games_per_batch=2, group_size=4, lam=0.5, alpha=0.5, cal_loss="bce",
                               learning_rate=1e-4)).step(groups)
    for ep in (ep for group in groups for ep in group):
        assert ep.bonus == pytest.approx(1 - sum((d.p - ep.z) ** 2 for d in ep.decisions) / ep.length)


# ---- 4. cal_grad = "platt_only"

def _updated(tokenizer, policy_cfg, groups, model, **grpo):
    policy = _policy(tokenizer, policy_cfg, model=copy.deepcopy(model), platt=True)
    cfg = GRPOConfig(games_per_batch=2, group_size=4, max_actions=3, learning_rate=1e-3, optimization_passes=1,
                     **grpo)
    Trainer(policy, cfg).step(copy.deepcopy(groups))
    return policy


def test_platt_only_calibration_gives_no_gradient_to_the_language_model(tokenizer, policy_cfg):
    model = tiny_model(seed=2)
    groups = _groups(_policy(tokenizer, policy_cfg, model=model, platt=True))
    none = _updated(tokenizer, policy_cfg, groups, model, alpha=0.0, beta=0.0)
    only = _updated(tokenizer, policy_cfg, groups, model, alpha=5.0, cal_loss="bce", cal_grad="platt_only")
    full = _updated(tokenizer, policy_cfg, groups, model, alpha=5.0, cal_loss="bce", cal_grad="full")
    same = lambda a, b: all(torch.equal(u, v) for u, v in zip(a.model.parameters(), b.model.parameters()))
    assert same(only, none)                       # the language model moved exactly as without a calibration term
    assert not same(full, none)
    start_b = float(none.platt_b)                 # set from the first batch in all three
    assert float(none.platt_a) == 1.0
    assert float(only.platt_a) != 1.0 and float(only.platt_b) != start_b          # a and b were trained
    assert float(full.platt_a) != 1.0

    # directly: the gradient of the calibration term
    policy = _policy(tokenizer, policy_cfg, model=copy.deepcopy(model), platt=True)
    d = groups[0][0].decisions[0]
    _, s = policy.score(d, forecast=True)
    cal = bce(policy.calibrated_logit(s.detach()), 0.0)
    params = [w for w in policy.model.parameters() if w.requires_grad]
    assert all(g is None for g in torch.autograd.grad(cal, params, allow_unused=True, retain_graph=True))
    ga, gb = torch.autograd.grad(cal, policy.platt_parameters())
    p = float(policy.forecast_prob(s))
    assert gb.item() == pytest.approx(p, abs=1e-6) and ga.item() == pytest.approx(p * float(s), abs=1e-5)


def test_platt_only_needs_platt(policy):
    with pytest.raises(ValueError, match="platt_only"):
        Trainer(policy, GRPOConfig(cal_grad="platt_only"))


# ---- 5. Platt scaling

def test_platt_forecast_is_sigmoid_of_a_s_plus_b(tokenizer, policy_cfg):
    policy = _policy(tokenizer, policy_cfg, platt=True)
    assert float(policy.platt_a) == 1.0 and float(policy.platt_b) == 0.0 and not policy.platt_initialized
    assert all(isinstance(q, torch.nn.Parameter) and q.dtype == torch.float32 for q in policy.platt_parameters())
    with torch.no_grad():
        policy.platt_a.fill_(0.5)
        policy.platt_b.fill_(-2.0)
    env = KeyChestEnv()
    d = policy.act(env.objective, [], env.reset(0), torch.Generator().manual_seed(0))
    expected = [1 / (1 + math.exp(-(0.5 * s - 2.0))) for s in d.forecast_logits_options]
    assert d.p_options == pytest.approx(expected, abs=1e-6) and d.p == d.p_options[d.action]


def test_platt_bias_start_from_first_batch(tokenizer, policy_cfg):
    policy = _policy(tokenizer, policy_cfg, seed=2, platt=True)
    groups = _groups(policy)
    pairs = [(d.forecast_logit, ep.z) for group in groups for ep in group for d in ep.decisions]
    win_rate = sum(z for _, z in pairs) / len(pairs)             # decision-weighted
    mean_s = sum(s for s, _ in pairs) / len(pairs)
    expected = _logit(min(max(win_rate, 0.05), 0.95)) - mean_s
    cfg = GRPOConfig(games_per_batch=2, group_size=4, lam=0.5, alpha=1.0, cal_loss="bce", learning_rate=1e-4)
    trainer = Trainer(policy, cfg)
    trainer._init_platt(groups)
    assert float(policy.platt_b) == pytest.approx(expected, abs=1e-6) and float(policy.platt_a) == 1.0
    assert policy.platt_initialized
    for d in (d for group in groups for ep in group for d in ep.decisions):        # stored p follow the new b
        assert d.p == pytest.approx(1 / (1 + math.exp(-(d.forecast_logit + expected))), abs=1e-6)
        assert d.p == d.p_options[d.action]
    metrics = trainer.step(groups)
    assert metrics["platt_a"] == 1.0 and metrics["platt_b"] == pytest.approx(expected, abs=1e-6)
    after = float(policy.platt_b)
    assert after != pytest.approx(expected, abs=1e-9)             # trained by the update ...
    metrics2 = trainer.step(_groups(policy))
    assert metrics2["platt_b"] == pytest.approx(after, abs=1e-7)  # ... and not set from data a second time


def test_platt_bias_start_clips_the_win_rate(tokenizer, policy_cfg):
    policy = _policy(tokenizer, policy_cfg, platt=True)
    assert policy.init_platt_bias(0.0, 12.0) == pytest.approx(_logit(0.05) - 12.0, abs=1e-5)
    assert policy.init_platt_bias(1.0, -3.0) == pytest.approx(_logit(0.95) + 3.0, abs=1e-5)
    assert policy.init_platt_bias(0.3, 12.0) == pytest.approx(_logit(0.3) - 12.0, abs=1e-5)
    assert float(policy.forecast_prob(torch.tensor(12.0))) == pytest.approx(0.3, abs=1e-5)


def test_platt_parameter_group(tokenizer, policy_cfg):
    policy = _policy(tokenizer, policy_cfg, platt=True)
    cfg = GRPOConfig(learning_rate=2e-6, weight_decay=0.1, platt_lr=3e-2, warmup_batches=4)
    trainer = Trainer(policy, cfg)
    lm, platt = trainer.optimizer.param_groups
    assert lm["lr"] == 2e-6 and lm["weight_decay"] == 0.1
    assert platt["lr"] == 3e-2 and platt["weight_decay"] == 0.0
    assert [id(q) for q in platt["params"]] == [id(policy.platt_a), id(policy.platt_b)]
    assert not {id(q) for q in lm["params"]} & {id(policy.platt_a), id(policy.platt_b)}
    trainer._set_learning_rate()                                  # warmup scales the language model's group only
    assert lm["lr"] == pytest.approx(2e-6 / 4) and platt["lr"] == 3e-2


def test_platt_json_round_trip(tokenizer, policy_cfg, tmp_path):
    policy = _policy(tokenizer, policy_cfg, platt=True)
    with torch.no_grad():
        policy.platt_a.fill_(0.37)
        policy.platt_b.fill_(-4.25)
    policy.save_platt(tmp_path)
    assert json.loads((tmp_path / PLATT_FILE).read_text()) == {"a": pytest.approx(0.37), "b": -4.25}
    other = _policy(tokenizer, policy_cfg, platt=True)
    assert other.load_platt(tmp_path) and other.platt_initialized
    assert float(other.platt_a) == float(policy.platt_a) and float(other.platt_b) == -4.25
    assert not other.load_platt(tmp_path / "missing")
    plain = _policy(tokenizer, policy_cfg)
    plain.save_platt(tmp_path / "missing")                        # nothing to save, nothing written
    assert not (tmp_path / "missing").exists() and not plain.load_platt(tmp_path)


# ---- 6. diagnostics

def _brute_auc(scores, labels):
    pairs = [(a, b) for (a, za), (b, zb) in itertools.product(zip(scores, labels), repeat=2) if za and not zb]
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)


def test_auc_equals_pairwise_count():
    rng = random.Random(0)
    for trial in range(200):
        n = rng.randint(2, 30)
        scores = [rng.choice([rng.random(), round(rng.random(), 1)]) for _ in range(n)]      # with ties
        labels = [int(rng.random() < 0.4) for _ in range(n)]
        if len(set(labels)) == 1:
            assert auc(scores, labels) is None
        else:
            assert auc(scores, labels) == pytest.approx(_brute_auc(scores, labels), abs=1e-12)
    assert auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0 and auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert auc([5, 5, 5], [0, 1, 1]) == 0.5 and auc([], []) is None


def _episode(game, z, logits, seed=0):
    decisions = [Decision(command=f"c{t}", action=0, prompt_ids=[1], reasoning_ids=[], action_context_ids=[1, 2],
                          option_ids=[[3]], behavior_logps=[0.0], p=1 / (1 + math.exp(-s)), p_options=[0.5, 0.5],
                          forecast_logit=s, forecast_logits_options=[s, -s]) for t, s in enumerate(logits)]
    return Episode(game=game, env_seed=7, sample_seed=seed, decisions=decisions, won=bool(z), finished=True)


def test_forecast_metrics():
    episodes = [_episode("a", 1, [2.0, 3.0, 1.0, 4.0]), _episode("b", 0, [2.5]), _episode("c", 0, [-1.0, 0.0]),
                _episode("d", 1, [0.5])]
    m = forecast_metrics(episodes)
    samples = [(s, ep.z) for ep in episodes for s in (d.forecast_logit for d in ep.decisions)]
    w = 5 / 8                                                     # 5 of the 8 decisions belong to won episodes
    assert m["forecast_brier_const"] == pytest.approx(w * (1 - w))
    assert m["forecast_brier_const"] != pytest.approx(0.25)       # the per-episode win rate is 1/2
    assert m["forecast_brier"] == pytest.approx(sum((1 / (1 + math.exp(-s)) - z) ** 2 for s, z in samples) / 8)
    # a constant forecast of w has exactly this Brier score on the same samples
    assert sum((w - z) ** 2 for _, z in samples) / 8 == pytest.approx(m["forecast_brier_const"])
    assert m["forecast_auc"] == pytest.approx(_brute_auc(*zip(*samples)))
    assert m["forecast_auc_t0"] == pytest.approx(_brute_auc([2.0, 2.5, -1.0, 0.5], [1, 0, 0, 1])) == 0.5
    assert m["forecast_s_mean_win"] == pytest.approx(10.5 / 5) and m["forecast_s_mean_lose"] == pytest.approx(0.5)
    assert m["mean_forecast_logit"] == pytest.approx(12.0 / 8)
    all_won = forecast_metrics(episodes[:1])
    assert all_won["forecast_auc"] is None and all_won["forecast_auc_t0"] is None
    assert all_won["forecast_s_mean_lose"] is None and all_won["forecast_brier_const"] == 0.0


def test_forecast_records_fields():
    (record,) = forecast_records([_episode("g.z8", 1, [2.0, -1.0], seed=23)])
    assert json.loads(json.dumps(record)) == record
    assert {k: record[k] for k in ("game", "env_seed", "sample_seed", "z", "length")} == \
        {"game": "g.z8", "env_seed": 7, "sample_seed": 23, "z": 1, "length": 2}
    assert [set(step) for step in record["steps"]] == [{"t", "command", "s", "p", "s_options"}] * 2
    assert record["steps"][1] == {"t": 1, "command": "c1", "s": -1.0, "p": pytest.approx(1 / (1 + math.e)),
                                  "s_options": [-1.0, 1.0]}


# ---- 7. end to end: train (smoke), save a checkpoint, evaluate it with --batches 0 --save-forecasts

def _rows(run):
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("platt", [False, True])
def test_checkpoint_then_forecast_dump(tmp_path, platt, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"policy": {"platt": platt},
                               "grpo": {"alpha": 0.5, "lam": 0.3, "cal_loss": "bce" if platt else "brier"}}))
    first, second = tmp_path / "train", tmp_path / "eval"
    train.main(["--smoke", "--out", str(first), "--config", str(cfg), "--batches", "2", "--save-forecasts"])
    checkpoint = first / "checkpoints" / "batch0002"
    assert (checkpoint / "config.json").is_file() and (checkpoint / PLATT_FILE).is_file() == platt
    rows = _rows(first)
    train_rows = [r for r in rows if r["split"] == "train"]
    assert all(r["cal_loss"] is not None and r["mean_forecast_logit"] is not None for r in train_rows)
    assert ("platt_b" in train_rows[0]) == platt
    assert sorted(p.name for p in (first / "forecasts").iterdir()) == ["batch0000.jsonl", "batch0002.jsonl"]

    # evaluation only; the Platt parameters come from the checkpoint even though the configuration is the default
    train.main(["--smoke", "--model", str(checkpoint), "--out", str(second), "--batches", "0", "--save-forecasts"])
    output = capsys.readouterr().out
    assert ("loaded Platt parameters" in output) == platt
    (row,) = _rows(second)
    last = [r for r in rows if r["split"] == "val"][-1]
    assert row["split"] == "val" and row["batch"] == 0
    for key in ("win_rate", "forecast_brier", "forecast_brier_const", "mean_forecast", "mean_forecast_logit",
                "forecast_s_mean_lose"):                           # the checkpoint is the model that was evaluated
        assert row[key] == pytest.approx(last[key], abs=1e-5), key
    assert {"forecast_auc", "forecast_auc_t0", "forecast_s_mean_win", "forecast_s_mean_lose"} <= set(row)
    assert json.loads((second / "config.json").read_text())["policy"]["platt"] == platt
    assert not (second / "checkpoints").exists()

    records = [json.loads(line) for line in (second / "forecasts" / "batch0000.jsonl").read_text().splitlines()]
    assert len(records) == row["episodes"] == 4 * 2               # 4 validation games x 2 evaluation seeds
    a, b = (json.loads((checkpoint / PLATT_FILE).read_text())[k] for k in "ab") if platt else (1.0, 0.0)
    for rec in records:
        assert set(rec) == {"game", "env_seed", "sample_seed", "z", "length", "steps"}
        assert rec["length"] == len(rec["steps"]) and rec["z"] in (0, 1) and rec["sample_seed"] in (0, 1)
        for t, step in enumerate(rec["steps"]):
            assert step["t"] == t and step["s"] in step["s_options"] and isinstance(step["command"], str)
            assert step["p"] == pytest.approx(1 / (1 + math.exp(-(a * step["s"] + b))), abs=1e-5)
    samples = [(step["s"], rec["z"]) for rec in records for step in rec["steps"]]
    if len({z for _, z in samples}) == 2:
        assert row["forecast_auc"] == pytest.approx(_brute_auc(*zip(*samples)))
    else:
        assert row["forecast_auc"] is None


def test_forecast_by_step_script(tmp_path, capsys):
    import sys
    sys.path.insert(0, str(SNAPSHOT.parents[2] / "scripts"))
    import forecast_by_step
    episodes = [_episode("a", 1, [2.0, 3.0]), _episode("b", 0, [1.0]), _episode("c", 1, [4.0])]
    rows = forecast_by_step.by_step(forecast_records(episodes), steps=3)
    assert rows[0] == {"t": 0, "n_win": 2, "mean_win": 3.0, "n_lose": 1, "mean_lose": 1.0}
    assert rows[1] == {"t": 1, "n_win": 1, "mean_win": 3.0, "n_lose": 0, "mean_lose": None}
    assert rows[2]["n_win"] == 0 and rows[2]["mean_win"] is None
    assert "| 1 | 1 | 3.000 | 0 | – |" in forecast_by_step.markdown(rows)
