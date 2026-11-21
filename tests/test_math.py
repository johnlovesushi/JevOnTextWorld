"""Readout (eq. 5), returns and advantages (eqs. 6-7), loss weights and terms (eqs. 8-9), Proposition 1."""
import math
import random

import pytest
import torch

from jevtw.config import Config, GRPOConfig
from jevtw.losses import (brier, calibration_weights, clipped_surrogate, decision_loss, kl_k3, policy_weights)
from jevtw.readout import single_token_id, yes_no_probability
from jevtw.rewards import brier_bonus, episode_return, group_advantages


# ------------------------------------------------------------------ eq. (5)

def test_readout_is_two_way_softmax_and_shift_invariant():
    logits = torch.randn(5, 20)
    p = yes_no_probability(logits, 3, 7)
    two_way = torch.softmax(logits[:, [3, 7]], dim=-1)[:, 0]
    assert torch.allclose(p, two_way, atol=1e-6)
    assert torch.allclose(yes_no_probability(logits + 10.0, 3, 7), p, atol=1e-6)


def test_readout_worked_example():
    logits = torch.zeros(10)
    logits[1], logits[2] = 3.2, 1.8
    assert yes_no_probability(logits, 1, 2).item() == pytest.approx(0.802, abs=1e-3)


def test_brier_gradient_on_logits():
    # d(p - z)^2 / d l_yes = 2 (p - z) p (1 - p), and the opposite sign for l_no.
    logits = torch.tensor([3.2, 1.8], requires_grad=True)
    p = yes_no_probability(logits, 0, 1)
    brier(p, 0.0).backward()
    expected = 2 * p.item() * p.item() * (1 - p.item())
    assert logits.grad[0].item() == pytest.approx(expected, rel=1e-5)
    assert logits.grad[1].item() == pytest.approx(-expected, rel=1e-5)


def test_single_token_id(tokenizer):
    assert single_token_id(tokenizer, "Y") == ord("Y")
    with pytest.raises(ValueError):
        single_token_id(tokenizer, " yes")


# ------------------------------------------------------------------ eqs. (6)-(7)

def test_bonus_and_return_worked_example():
    b1 = brier_bonus([0.5, 0.7, 0.9], 1)
    b2 = brier_bonus([0.6, 0.4, 0.1], 0)
    assert b1 == pytest.approx(1 - 0.35 / 3)
    assert b2 == pytest.approx(1 - 0.53 / 3)
    assert episode_return(1, b1, 0.5) == pytest.approx(1.4417, abs=1e-4)
    assert episode_return(0, b2, 0.5) == pytest.approx(0.4117, abs=1e-4)


def test_every_win_beats_every_loss():
    rng = random.Random(0)
    for _ in range(2000):
        lam = rng.uniform(0, 0.999)
        win = episode_return(1, brier_bonus([rng.random() for _ in range(rng.randint(1, 9))], 1), lam)
        loss = episode_return(0, brier_bonus([rng.random() for _ in range(rng.randint(1, 9))], 0), lam)
        assert win > loss


@pytest.mark.parametrize("lam", [-0.1, 1.0, 1.5])
def test_lambda_range(lam):
    with pytest.raises(ValueError):
        episode_return(1, 0.5, lam)
    with pytest.raises(ValueError):
        GRPOConfig(lam=lam)


def test_bonus_validation():
    assert brier_bonus([1.0, 1.0], 1) == 1.0
    assert brier_bonus([1.0], 0) == 0.0
    with pytest.raises(ValueError):
        brier_bonus([1.2], 1)
    with pytest.raises(ValueError):
        brier_bonus([0.5], 2)
    with pytest.raises(ValueError):
        brier_bonus([], 1)


def test_group_advantages():
    adv = group_advantages([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], delta=0.0)
    assert sum(adv) == pytest.approx(0.0, abs=1e-12)
    assert math.sqrt(sum(a * a for a in adv) / len(adv)) == pytest.approx(1.0)   # population sd
    assert group_advantages([0.3, 0.3, 0.3]) == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------ eqs. (8)-(9)

COUNTS = [[[3, 2], [4]], [[1, 1, 1], [5], [2]]]          # [game][episode][decision] -> |c_t| + 1


def test_token_weights_are_one_over_N_per_game():
    w = policy_weights(COUNTS, "token")
    assert w[0][0][0] == pytest.approx(1 / (9 * 2))      # N = 3 + 2 + 4
    assert w[1][1][0] == pytest.approx(1 / (10 * 2))     # N = 3 + 5 + 2
    total = sum(wt * n for g, wg in zip(COUNTS, w) for e, we in zip(g, wg) for n, wt in zip(e, we))
    assert total == pytest.approx(1.0)


def test_episode_weights_give_every_episode_equal_mass():
    w = policy_weights(COUNTS, "episode")
    for g, wg in zip(COUNTS, w):
        for e, we in zip(g, wg):
            assert sum(n * wt for n, wt in zip(e, we)) == pytest.approx(1 / (len(g) * len(COUNTS)))


def test_calibration_weights():
    w = calibration_weights(COUNTS)
    assert w[0][0][0] == pytest.approx(1 / (3 * 2))      # sum_i T_i = 2 + 1
    assert sum(x for wg in w for we in wg for x in we) == pytest.approx(1.0)


@pytest.mark.parametrize("ratio,adv,expected", [(1.0, 2.0, -2.0), (1.5, 1.0, -1.2), (0.5, 1.0, -0.5),
                                                (0.5, -1.0, 0.8), (1.5, -1.0, 1.5)])
def test_clipped_surrogate(ratio, adv, expected):
    old = torch.tensor([-1.0])
    out = clipped_surrogate(old + math.log(ratio), old, adv, 0.2)
    assert out.item() == pytest.approx(expected, rel=1e-5)


def test_kl_k3():
    a = torch.randn(50)
    assert torch.all(kl_k3(a, a + torch.randn(50)) >= 0)
    assert torch.allclose(kl_k3(a, a), torch.zeros(50))


@pytest.mark.parametrize("normalization", ["token", "episode"])
def test_accumulated_loss_equals_eq8_plus_eq9(normalization):
    torch.manual_seed(0)
    alpha, eps = 0.7, 0.2
    adv = [[0.5, -1.0], [1.2, 0.0, -0.4]]
    z = [[1, 0], [1, 0, 0]]
    data = [[[(torch.randn(n) * 0.1 - 2, torch.randn(n) * 0.1 - 2, torch.rand(())) for n in ep] for ep in game]
            for game in COUNTS]
    w_pol, w_cal = policy_weights(COUNTS, normalization), calibration_weights(COUNTS)
    accumulated = 0.0
    for g, game in enumerate(data):
        for i, ep in enumerate(game):
            for t, (lp, old, p) in enumerate(ep):
                loss, _ = decision_loss(lp, old, adv[g][i], w_pol[g][i][t], eps, p=p, z=float(z[g][i]),
                                        cal_weight=w_cal[g][i][t], alpha=alpha)
                accumulated += loss.item()
    # Direct evaluation, one game at a time, then the mean over games.
    per_game = []
    for g, game in enumerate(data):
        if normalization == "token":
            n = sum(sum(ep) for ep in COUNTS[g])
            grpo = sum(clipped_surrogate(lp, old, adv[g][i], eps).sum() for i, ep in enumerate(game)
                       for lp, old, _ in ep) / n
        else:
            grpo = sum(sum(clipped_surrogate(lp, old, adv[g][i], eps).sum() for lp, old, _ in ep) /
                       sum(COUNTS[g][i]) for i, ep in enumerate(game)) / len(game)
        cal = sum((p - z[g][i]) ** 2 for i, ep in enumerate(game) for _, _, p in ep) / \
            sum(len(ep) for ep in COUNTS[g])
        per_game.append(grpo + alpha * cal)
    assert accumulated == pytest.approx(float(sum(per_game) / len(per_game)), rel=1e-5)


def test_proposition_1_brier_minimizer_is_success_probability():
    """Minimising the Brier loss of a forecast against outcomes with success frequency 0.3 drives p to 0.3."""
    gen = torch.Generator().manual_seed(0)
    z = (torch.rand(4000, generator=gen) < 0.3).float()
    logits = torch.zeros(2, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=5.0)
    for _ in range(500):
        opt.zero_grad()
        p = yes_no_probability(logits, 0, 1)
        brier(p, z).mean().backward()
        opt.step()
    assert yes_no_probability(logits, 0, 1).item() == pytest.approx(z.mean().item(), abs=1e-3)


# ------------------------------------------------------------------ configuration

def test_config_roundtrip_and_validation(tmp_path):
    cfg = Config.from_dict({"grpo": {"lam": 0.3, "alpha": 0.1, "normalization": "episode"}})
    assert cfg.grpo.lam == 0.3 and cfg.policy.yes_text == " yes"
    assert Config.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        Config.from_dict({"grpo": {"lamda": 0.3}})
    with pytest.raises(ValueError):
        GRPOConfig(normalization="sequence")
    with pytest.raises(ValueError):
        GRPOConfig(alpha=-1)
