"""T0006: prompt formats "plain" / "chat_nothink" / "chat_think", the ways c_t can end in each, the forced suffix."""
import warnings

import pytest
import torch

from jevtw.config import Config, PolicyConfig
from jevtw.envs import KeyChestEnv, Observation
from jevtw.policy import Decision, LMPolicy, build_prompt, chat_prompt
from jevtw.rollout import REASONING_ENDS, collect_batch, reasoning_end, rollout_metrics, run_episode
from jevtw.testing import ByteTokenizer, tiny_model

from test_policy_trainer import ref_forecasts, ref_option_scores, ref_reasoning_logps

CHAT = ("chat_nothink", "chat_think")
HISTORY = [("open door", Observation("The door is open.\nYou see a hall.", [], feedback="You open the door."))]
OBS = Observation("You are in a hall.", ["go north", "take key"])

PLAIN = ("Objective: Win.\n\nRecent steps:\n> open door\nThe door is open.\nYou see a hall.\n\n"
         "Current observation:\nYou are in a hall.\n\nOptions:\n1. go north\n2. take key\n\n"
         'Think step by step, then write "Answer:" followed by the number of one option.\nReasoning:')
PLAIN_DIRECT = ("Objective: Win.\n\nRecent steps:\n> open door\nThe door is open.\nYou see a hall.\n\n"
                "Current observation:\nYou are in a hall.\n\nOptions:\n1. go north\n2. take key\n\n"
                "Reply with the number of one option.")
USER = ("<|im_start|>user\nObjective: Win.\n\nRecent steps:\n> open door\nThe door is open.\nYou see a hall.\n\n"
        "Current observation:\nYou are in a hall.\n\nOptions:\n1. go north\n2. take key\n\n"
        'First analyse the situation in at most three sentences, then write "Answer:" followed by the number of one '
        "option.<|im_end|>\n<|im_start|>assistant\n")
CHAT_SNAPSHOT = {"chat_nothink": USER + "<think>\n\n</think>\n\n", "chat_think": USER + "<think>\n"}


def _cfg(policy_cfg, **overrides):
    return PolicyConfig(**{**policy_cfg.__dict__, "chat_yes_text": "Y", "chat_no_text": "N", **overrides})


def _policy(policy_cfg, model=None, kind=LMPolicy, **overrides):
    return kind(tiny_model(seed=0) if model is None else model, ByteTokenizer(), _cfg(policy_cfg, **overrides))


# ---------------------------------------------------------------- configuration

def test_default_is_plain_and_values_are_checked():
    assert PolicyConfig().prompt_format == "plain"
    with pytest.raises(ValueError, match="prompt_format"):
        PolicyConfig(prompt_format="chat")


def test_chat_prompt_with_plain_forecast_warns():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # none of these may warn
        Config()
        Config.from_dict({"policy": {"prompt_format": "chat_think", "forecast_format": "chat"}})
        Config.from_dict({"policy": {"forecast_format": "chat"}})
    for fmt in CHAT:
        with pytest.warns(UserWarning, match="forecast_format"):
            cfg = Config.from_dict({"policy": {"prompt_format": fmt}})
        assert cfg.policy.prompt_format == fmt and cfg.policy.forecast_format == "after_eos"      # a warning, no error


# ---------------------------------------------------------------- prompt strings

def test_plain_prompt_is_unchanged():
    """Character for character what build_prompt returned before T0006 (defaults: plain, full history)."""
    assert build_prompt("Win.", HISTORY, OBS, PolicyConfig()) == PLAIN
    assert build_prompt("Win.", HISTORY, OBS, PolicyConfig(max_reasoning_tokens=0)) == PLAIN_DIRECT
    feedback = build_prompt("Win.", HISTORY, OBS, PolicyConfig(history_mode="feedback"))
    assert feedback == PLAIN.replace("The door is open.\nYou see a hall.", "You open the door.")


@pytest.mark.parametrize("fmt", CHAT)
def test_chat_prompt_snapshot(fmt):
    text = build_prompt("Win.", HISTORY, OBS, PolicyConfig(prompt_format=fmt))
    assert text == CHAT_SNAPSHOT[fmt]
    assert "Reasoning:" not in text and "Think step by step" not in text
    # Without reasoning the user turn is the plain direct prompt.
    direct = build_prompt("Win.", HISTORY, OBS, PolicyConfig(prompt_format=fmt, max_reasoning_tokens=0))
    assert direct == chat_prompt(PLAIN_DIRECT, think=fmt == "chat_think")


# ---------------------------------------------------------------- how c_t ends

# name -> (prompt_format, reasoning text, sampled EOS?, kind, what the action context adds after the body)
CASES = {
    "plain/answer": ("plain", " the key first\nAnswer:", False, "answer", ""),
    "plain/eos": ("plain", " take key", True, "eos", "\nAnswer:"),
    "plain/cap": ("plain", " I think", False, "cap", "\nAnswer:"),
    "nothink/answer": ("chat_nothink", "The key opens it.\nAnswer:", False, "answer", ""),
    "nothink/eos": ("chat_nothink", "Take the key.", True, "eos", "\nAnswer:"),
    "nothink/cap": ("chat_nothink", "The key is", False, "cap", "\nAnswer:"),
    # "</think>" in a format without a thinking block is ordinary text
    "nothink/cap_with_close_tag": ("chat_nothink", "odd </think> text", False, "cap", "\nAnswer:"),
    "think/answer": ("chat_think", "Key first.\n</think>\n\nAnswer:", False, "answer", ""),
    "think/answer_twice": ("chat_think", "Answer: maybe 2\n</think>\n\nAnswer:", False, "answer", ""),
    "think/eos_in_think": ("chat_think", "Key first.", True, "eos", "\n</think>\n\nAnswer:"),
    "think/eos_after_think": ("chat_think", "Key.\n</think>\n\nTake it.", True, "eos", "\nAnswer:"),
    "think/cap_in_think": ("chat_think", "The key is on the", False, "cap_in_think", "\n</think>\n\nAnswer:"),
    # "Answer:" inside the thinking block does not end c_t; at the cap the block is closed like any other
    "think/cap_in_think_after_answer": ("chat_think", "so\nAnswer:", False, "cap_in_think", "\n</think>\n\nAnswer:"),
    "think/cap_after_think": ("chat_think", "Key.\n</think>\n\nI will", False, "cap", "\nAnswer:"),
    "plain/none": ("plain", "", False, None, "\nAnswer:"),
    "nothink/none": ("chat_nothink", "", False, None, "\nAnswer:"),
    "think/none": ("chat_think", "", False, None, "\n</think>\n\nAnswer:"),
}


def _case(policy_cfg, name, n_options=12, action=10):
    fmt, text, eos, kind, added = CASES[name]
    policy = _policy(policy_cfg, prompt_format=fmt, max_reasoning_tokens=64)
    prompt = policy.encode_prompt("Win.", HISTORY, OBS)
    body = policy.encode(text)
    reasoning = body + [policy.eos_id] if eos else body
    context = policy._action_context(prompt, reasoning)
    options = [policy.encode(f" {j + 1}") + [policy.eos_id] for j in range(n_options)]
    d = Decision(command="x", action=action, prompt_ids=prompt, reasoning_ids=reasoning, action_context_ids=context,
                 option_ids=options, behavior_logps=[], reasoning_end=policy._finish(reasoning)[2])
    return policy, d, body, kind, added


@pytest.mark.parametrize("name", CASES)
def test_ending_kind_suffix_and_invariant(policy_cfg, name):
    policy, d, body, kind, added = _case(policy_cfg, name)
    x, c, n_prompt = d.action_context_ids, d.reasoning_ids, len(d.prompt_ids)
    assert d.reasoning_end == kind == reasoning_end(d, policy.eos_id)
    assert x == d.prompt_ids + body + policy.encode(added)
    assert policy.tokenizer.decode(x).endswith("Answer:")
    # T0001: the reasoning tokens are predicted inside the prefix x[0:n-1], from the prompt followed by c[:-1]
    assert n_prompt + len(c) <= len(x)
    assert x[:n_prompt + max(len(c) - 1, 0)] == d.prompt_ids + c[:max(len(c) - 1, 0)]
    # sampling stops exactly where the kind says: at "answer" the last token completes it, never before
    if kind == "answer":
        assert policy._answer_reached(c) and not any(policy._answer_reached(c[:i]) for i in range(1, len(c)))
    elif c:
        assert not any(policy._answer_reached(c[:i]) for i in range(1, len(body) + 1))
    # a thinking block is balanced in the text the options are scored after
    text = policy.tokenizer.decode(x)
    if policy.cfg.prompt_format == "chat_think":
        assert text.count("</think>") == 1 and text.index("</think>") < text.rindex("Answer:")


@pytest.mark.parametrize("name", CASES)
def test_decision_forward_matches_reference(policy_cfg, name):
    """Reasoning log-probs, l(a^j) and p_j equal one full-context forward per quantity, for every ending."""
    policy, d, *_ = _case(policy_cfg, name)
    with torch.no_grad():
        lp, scores, s = policy.decision_forward(policy.model, d, forecast=True)
        assert lp.shape == (len(d.reasoning_ids),) and scores.shape == s.shape == (12,)
        assert torch.allclose(lp, ref_reasoning_logps(policy, policy.model, d), atol=1e-4)
        assert torch.allclose(scores, ref_option_scores(policy, policy.model, d.action_context_ids, d.option_ids),
                              atol=1e-4)
        assert torch.allclose(policy.forecast_prob(s), ref_forecasts(policy, policy.model, d), atol=1e-4)


@pytest.mark.parametrize("name", ["think/answer", "think/eos_in_think", "think/cap_in_think", "think/cap_after_think"])
def test_gradient_matches_reference(policy_cfg, name):
    policy, d, *_ = _case(policy_cfg, name)
    params = [w for w in policy.model.parameters() if w.requires_grad]
    logps, s_a = policy.score(d, forecast=True)
    new = torch.autograd.grad(logps.sum() + policy.forecast_prob(s_a), params, allow_unused=True)
    log_q = torch.log_softmax(ref_option_scores(policy, policy.model, d.action_context_ids, d.option_ids), dim=0)
    ref = torch.autograd.grad(ref_reasoning_logps(policy, policy.model, d).sum() + log_q[d.action]
                              + ref_forecasts(policy, policy.model, d)[d.action], params, allow_unused=True)
    for a, b in zip(new, ref):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------- sampling

class StopPolicy(LMPolicy):
    """A random model writes neither "Answer:" nor "</think>". Here a token divisible by 5 counts as the end of
    "Answer:" and (in chat_think) any earlier token divisible by 3 as "</think>", so that sampling itself runs
    into every ending. Only the two text predicates are replaced; stopping and `_finish` are the real code."""

    def _ends_with_answer(self, tokens):
        return bool(tokens) and tokens[-1] != self.eos_id and tokens[-1] % 5 == 0

    def _think_open(self, tokens):
        return self.cfg.prompt_format == "chat_think" and not any(t % 3 == 0 and t < 256 for t in tokens)


def _stop_policy(policy_cfg, fmt):
    model = tiny_model(seed=0)
    with torch.no_grad():                 # make EOS likely enough to be sampled now and then
        model.get_output_embeddings().weight[ByteTokenizer.eos_token_id] *= 12.0
    return _policy(policy_cfg, model=model, kind=StopPolicy, prompt_format=fmt)


@pytest.mark.parametrize("fmt,expected", [("chat_nothink", {"answer", "eos", "cap"}),
                                          ("chat_think", {"answer", "eos", "cap", "cap_in_think"})])
def test_sampling_probabilities_equal_scoring(policy_cfg, fmt, expected):
    """Serial and batched sampling: the recorded log-probs are those of the training forward, in every ending."""
    policy = _stop_policy(policy_cfg, fmt)
    prompt = policy.encode_prompt("Win.", HISTORY, OBS)
    seen = set()
    for seed in range(40):
        g = torch.Generator().manual_seed(seed)
        if seed % 2:
            d = policy.act("Win.", HISTORY, OBS, g)
        else:
            (tokens, logps), = policy.sample_reasoning_batch([prompt], [g])
            d = policy.decide(prompt, tokens, logps, OBS, g)
        kind = reasoning_end(d, policy.eos_id)
        seen.add(kind)
        c, cap = d.reasoning_ids, policy.cfg.max_reasoning_tokens
        assert (kind in ("cap", "cap_in_think")) == (len(c) == cap and c[-1] != policy.eos_id
                                                     and not policy._answer_reached(c))
        assert not any(policy._answer_reached(c[:i]) for i in range(1, len(c)))       # it stopped at the first one
        assert d.action_context_ids == policy._action_context(d.prompt_ids, c)
        with torch.no_grad():
            logps, s = policy.score(d, forecast=True)
        assert torch.allclose(logps, torch.tensor(d.behavior_logps), atol=1e-4)
        assert policy.forecast_prob(s).item() == pytest.approx(d.p, abs=1e-4)
    assert seen == expected


def _same(a, b):
    key = lambda ep: [(d.command, d.action, d.reasoning_ids, d.action_context_ids, d.behavior_logps, d.reasoning_end)
                      for d in ep.decisions]
    return (a.game, a.env_seed, a.sample_seed, a.won, a.finished) == (b.game, b.env_seed, b.sample_seed, b.won,
                                                                      b.finished) and key(a) == key(b)


@pytest.mark.parametrize("fmt", ["plain", *CHAT])
@pytest.mark.parametrize("stopping", [False, True])
def test_parallel_one_equals_run_episode(policy_cfg, fmt, stopping):
    policy = _stop_policy(policy_cfg, fmt) if stopping else _policy(policy_cfg, prompt_format=fmt)
    jobs = [(f"keychest-{i % 4}", KeyChestEnv, i % 4, 1000 + i) for i in range(5)]
    batch = collect_batch(policy, jobs, max_actions=4, parallel=1)
    for (game, make_env, env_seed, sample_seed), ep in zip(jobs, batch):
        assert _same(ep, run_episode(policy, make_env(), game, env_seed, sample_seed, 4))
    if fmt != "plain":
        assert policy.tokenizer.decode(batch[0].decisions[0].prompt_ids).startswith("<|im_start|>user\nObjective:")


def test_rollout_metrics_report_cap_in_think(policy_cfg):
    policy = _stop_policy(policy_cfg, "chat_think")
    jobs = [(f"keychest-{i % 4}", KeyChestEnv, i % 4, 2000 + i) for i in range(8)]
    episodes = collect_batch(policy, jobs, max_actions=5, parallel=4)
    metrics = rollout_metrics(episodes, {}, wall_s=1.0, eos_id=policy.eos_id)
    assert REASONING_ENDS == ("answer", "eos", "cap", "cap_in_think")
    assert sum(metrics[f"reasoning_end_{k}"] for k in REASONING_ENDS) == pytest.approx(1.0)
    assert metrics["reasoning_end_cap_in_think"] > 0
    # an untrained model (no "</think>", no "Answer:") always runs into the cap inside the thinking block
    plain = _policy(policy_cfg, prompt_format="chat_think")
    only = rollout_metrics(collect_batch(plain, jobs[:2], max_actions=2, parallel=2), {}, 1.0, plain.eos_id)
    assert only["reasoning_end_cap_in_think"] + only["reasoning_end_eos"] == pytest.approx(1.0)
