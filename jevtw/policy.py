"""Language-model policy, eqs. (4)-(5).

At step t the model sees a prompt built from h_t, samples reasoning tokens c_t (temperature 1, full vocabulary, so the
sampling law is exactly pi_theta), then picks a_t from q_theta = softmax over the K_t candidate commands, where the
logit l(a^j) of option j is the log-likelihood of the continuation " j<EOS>" after "Answer:". The prompt is a text
to continue or a chat turn (`PolicyConfig.prompt_format`); `LMPolicy._finish` says how c_t ends. Afterwards it can read
the forecast logit s_t = l(yes) - l(no) after a confidence question that is never written into the history;
p_t = sigmoid(s_t), or sigmoid(a s_t + b) with Platt scaling (`forecast_prob`).

All three come from one forward in `decision_forward`: the shared prefix (the action context) followed by the K
suffixes "option j + confidence question", which are kept apart by the attention mask. Where the question stands
relative to the option is `PolicyConfig.forecast_format`.

Likelihood terms: every sampled reasoning token (its exact ids, including a sampled EOS or the tokens of "Answer:")
and the single categorical action. Prompts, observations and the forced "Answer:" have none. Token ids are stored
and reused verbatim, so scoring never re-tokenises generated text."""
from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .config import PolicyConfig
from .envs import Observation
from .readout import single_token_id


@dataclass
class Decision:
    command: str
    action: int                           # index of a_t among the options
    prompt_ids: list[int]
    reasoning_ids: list[int]              # c_t
    action_context_ids: list[int]         # prompt + c_t (minus a final EOS) + the forced suffix when needed
    option_ids: list[list[int]]           # " j<EOS>" for every option j
    behavior_logps: list[float]           # log-probs at sampling time: |c_t| reasoning tokens, then log q(a_t)
    p: float | None = None                # forecast at sampling time (theta_old), used for b_i
    p_options: list[float] | None = None  # p_j for every option j: "win probability if a^j were chosen" (Q_win)
    forecast_logit: float | None = None   # s_t = l(yes) - l(no) of the chosen option, before Platt scaling
    forecast_logits_options: list[float] | None = None    # s_j for every option j
    confidence_ids: list[int] | None = None
    # How c_t ended: "answer", "eos", "cap", "cap_in_think" (LMPolicy._finish); None without reasoning.
    reasoning_end: str | None = None
    old_logps: list[float] | None = field(default=None, repr=False)   # filled by the trainer
    ref_logps: list[float] | None = field(default=None, repr=False)
    # Verbal confidence (T0008, confidence_mode = "verbal"): m_t, log pi^c over the M bins at sampling time
    # (normalised; after the chosen option) and log pi^c(m_t) at sampling time.
    conf_bin: int | None = None
    conf_logps: list[float] | None = None
    conf_behavior_logp: float | None = None
    conf_old_logp: float | None = field(default=None, repr=False)     # filled by the trainer
    conf_ref_logp: float | None = field(default=None, repr=False)
    conf_adv: float | None = field(default=None, repr=False)          # A^c_t with conf_advantage = "decoupled"

    @property
    def n_terms(self) -> int:
        return len(self.reasoning_ids) + 1

    @property
    def conf(self) -> float | None:
        """c_t = v_{m_t} = m_t / (M - 1), the sampled (spoken) confidence."""
        return None if self.conf_bin is None else self.conf_bin / (len(self.conf_logps) - 1)

    @property
    def conf_mean(self) -> float | None:
        """c-hat_t = sum_m pi^c(m) v_m, the expected confidence (used for AUC and calibration metrics)."""
        if self.conf_logps is None:
            return None
        m = len(self.conf_logps)
        return sum(math.exp(lp) * i / (m - 1) for i, lp in enumerate(self.conf_logps))

    @property
    def conf_entropy(self) -> float | None:
        """Entropy of pi^c in nats."""
        if self.conf_logps is None:
            return None
        return -sum(math.exp(lp) * lp for lp in self.conf_logps if lp > -math.inf)


def build_prompt(objective: str, history: Sequence[tuple[str, Observation]], obs: Observation,
                 cfg: PolicyConfig) -> str:
    """Text of h_t: the objective, the last `history_window` steps (the command and, depending on
    `history_mode`, the whole observation that followed or only the game's feedback), the current observation
    and the numbered admissible commands. With a chat `prompt_format` this text (with its own last instruction)
    is the user turn of `chat_prompt`."""
    lines = [f"Objective: {objective}", ""]
    recent = list(history)[-cfg.history_window:] if cfg.history_window > 0 else []
    if recent:
        lines.append("Recent steps:")
        for cmd, after in recent:
            lines += [f"> {cmd}", after.feedback if cfg.history_mode == "feedback" else after.text, ""]
    lines += ["Current observation:", obs.text, "", "Options:"]
    lines += [f"{j + 1}. {c}" for j, c in enumerate(obs.commands)]
    lines.append("")
    chat = cfg.prompt_format != "plain"
    if cfg.max_reasoning_tokens <= 0:
        lines.append("Reply with the number of one option.")
    elif chat:
        lines.append(f'First analyse the situation in at most three sentences, then write "{cfg.answer_prefix}" '
                     "followed by the number of one option.")
    else:
        lines += [f'Think step by step, then write "{cfg.answer_prefix}" followed by the number of one option.',
                  cfg.reasoning_prefix]
    text = "\n".join(lines)
    return chat_prompt(text, think=cfg.prompt_format == "chat_think") if chat else text


THINK_CLOSE = "</think>"


def chat_prompt(content: str, think: bool) -> str:
    """One user turn and the start of the assistant turn, written out by hand with Qwen3's markers so that CPU
    tests can use the byte tokenizer (scripts/check_prompt_formats.py compares with the tokenizer's template).
    think = False: the assistant turn starts with an empty thinking block, exactly the template with
    enable_thinking=False. think = True: it starts with "<think>\n"; the template with enable_thinking=True stops
    after "<|im_start|>assistant\n" and leaves "<think>\n" to the model, here it is part of the prompt."""
    start = "<think>\n" if think else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{start}"


PLATT_FILE = "platt.json"


def chat_forecast_suffix(question: str) -> str:
    """What forecast_format "chat" appends after the option (which ends with EOS = <|im_end|>): a new user turn
    with the question and the start of an assistant turn with empty thinking, as Qwen's chat template renders it
    with enable_thinking=False (scripts/check_chat_forecast.py compares with the tokenizer's template)."""
    return f"\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LMPolicy:
    def __init__(self, model, tokenizer, cfg: PolicyConfig):
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("the tokenizer needs an EOS token to delimit option numbers")
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_id
        # Forced suffixes (no likelihood terms) that make the action context end with "Answer:".
        self.answer_ids = self.encode("\n" + cfg.answer_prefix)
        self.think_answer_ids = self.encode(f"\n{THINK_CLOSE}\n\n{cfg.answer_prefix}")   # closes an open thinking block
        self.confidence_prompt_ids = self.encode(cfg.confidence_prompt)
        # What follows the option before the forecast is read, and the two tokens that are read.
        if cfg.forecast_format == "chat":
            self.forecast_tail_ids = self.encode(chat_forecast_suffix(cfg.forecast_question))
            self.yes_id = single_token_id(tokenizer, cfg.chat_yes_text)
            self.no_id = single_token_id(tokenizer, cfg.chat_no_text)
        else:
            self.forecast_tail_ids = self.confidence_prompt_ids
            self.yes_id = single_token_id(tokenizer, cfg.yes_text)
            self.no_id = single_token_id(tokenizer, cfg.no_text)
        if self.yes_id == self.no_id:
            raise ValueError("yes and no map to the same token")
        # Platt scaling p = sigmoid(a s + b). FP32 like s; trained by the trainer in their own parameter group.
        self.platt_a = self.platt_b = None
        self.platt_initialized = False    # True once b was set from data (Trainer) or a, b were loaded
        if cfg.platt:
            device = next(model.parameters()).device
            self.platt_a = torch.nn.Parameter(torch.ones((), dtype=torch.float32, device=device))
            self.platt_b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
        # Verbal confidence (T0008): the prompt u and the M answers y^m = " {100 m / (M - 1)}<EOS>".
        self.verbal = cfg.confidence_mode == "verbal"
        self.verbal_prompt_ids = self.bin_ids = None
        if self.verbal:
            self.verbal_prompt_ids = self.encode(cfg.verbal_prompt)
            m = cfg.confidence_bins
            self.bin_ids = [self.encode(f" {100 * i / (m - 1):g}") + [self.eos_id] for i in range(m)]
            if not self.verbal_prompt_ids:
                raise ValueError("verbal_prompt must not be empty")
            if len({tuple(b) for b in self.bin_ids}) != m:
                raise ValueError("the confidence answers do not encode to distinct token sequences")
        # Recent transformers renamed past_key_values to cache (argument and output field).
        params = inspect.signature(model.base_model.forward).parameters
        self._cache_kw = "cache" if "cache" in params else "past_key_values"

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _tensor(self, ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(ids)], dtype=torch.long, device=self.device)

    # ---------------------------------------------------------------- forecast: s -> p

    def calibrated_logit(self, s: torch.Tensor) -> torch.Tensor:
        """a * s + b with Platt scaling, s itself without."""
        return s if self.platt_a is None else self.platt_a * s + self.platt_b

    def forecast_prob(self, s: torch.Tensor) -> torch.Tensor:
        """p = sigmoid(s), or sigmoid(a s + b) with Platt scaling."""
        return torch.sigmoid(self.calibrated_logit(s))

    def platt_parameters(self) -> list[torch.nn.Parameter]:
        return [] if self.platt_a is None else [self.platt_a, self.platt_b]

    @torch.no_grad()
    def init_platt_bias(self, win_rate: float, mean_logit: float) -> float:
        """Data-dependent start: b = logit(clip(win_rate, 0.05, 0.95)) - a * mean_logit (a = 1 at that time), so
        that the forecast at the mean logit is the (clipped) win rate."""
        w = min(max(win_rate, 0.05), 0.95)
        self.platt_b.fill_(math.log(w / (1.0 - w)) - float(self.platt_a) * mean_logit)
        self.platt_initialized = True
        return float(self.platt_b)

    def save_platt(self, directory) -> None:
        """Write platt.json ({"a": .., "b": ..}) into a checkpoint directory; nothing without Platt scaling."""
        if self.platt_a is not None:
            (Path(directory) / PLATT_FILE).write_text(json.dumps({"a": float(self.platt_a), "b": float(self.platt_b)}))

    @torch.no_grad()
    def load_platt(self, directory) -> bool:
        """Read a, b from platt.json in `directory` if it is there (the policy must have platt = True)."""
        path = Path(directory) / PLATT_FILE
        if self.platt_a is None or not path.is_file():
            return False
        values = json.loads(path.read_text())
        self.platt_a.fill_(float(values["a"]))
        self.platt_b.fill_(float(values["b"]))
        self.platt_initialized = True
        return True

    # ---------------------------------------------------------------- sampling (theta_old, no gradients)

    @torch.no_grad()
    def act(self, objective: str, history: Sequence[tuple[str, Observation]], obs: Observation,
            generator: torch.Generator) -> Decision:
        """One decision for one episode (the serial reference; `rollout.collect_batch` uses the three steps below
        with `sample_reasoning_batch` in place of `_sample_reasoning`)."""
        prompt_ids = self.encode_prompt(objective, history, obs)
        reasoning, logps = self._sample_reasoning(prompt_ids, generator)
        return self.decide(prompt_ids, reasoning, logps, obs, generator)

    def encode_prompt(self, objective: str, history: Sequence[tuple[str, Observation]], obs: Observation) -> list[int]:
        if not obs.commands:
            raise ValueError("no admissible commands")
        return self.encode(build_prompt(objective, history, obs, self.cfg))

    @torch.no_grad()
    def decide(self, prompt_ids: list[int], reasoning: list[int], logps: list[float], obs: Observation,
               generator: torch.Generator) -> Decision:
        """Score the options after the sampled reasoning, sample a_t ~ q and read p_t."""
        cfg = self.cfg
        context = self._action_context(prompt_ids, reasoning)
        options = [self.encode(f" {j + 1}") + [self.eos_id] for j in range(len(obs.commands))]
        decision = Decision(command="", action=-1, prompt_ids=prompt_ids, reasoning_ids=reasoning,
                            action_context_ids=context, option_ids=options, behavior_logps=[],
                            reasoning_end=self._finish(reasoning)[2])
        if self.verbal:                   # confidence answers after every option; the chosen one is used
            _, scores, s, conf = self.verbal_forward(self.model, decision, cfg.forecast, range(len(options)))
        else:
            _, scores, s = self.decision_forward(self.model, decision, forecast=cfg.forecast)
        log_q = torch.log_softmax(scores, dim=0)
        action = int(torch.multinomial(log_q.exp().to(generator.device), 1, generator=generator))
        decision.command, decision.action = obs.commands[action], action
        decision.behavior_logps = logps + [float(log_q[action])]
        if self.verbal:                   # m_t ~ pi^c(. | h_t, a_t), drawn after a_t with the episode's generator
            log_pc = torch.log_softmax(conf[action], dim=0)
            m = int(torch.multinomial(log_pc.exp().to(generator.device), 1, generator=generator))
            decision.conf_logps = [float(v) for v in log_pc]
            decision.conf_bin, decision.conf_behavior_logp = m, decision.conf_logps[m]
        if s is not None:
            decision.confidence_ids = self._forecast_sequence(context, options[action])
            decision.forecast_logits_options = [float(v) for v in s]
            decision.forecast_logit = decision.forecast_logits_options[action]
            decision.p_options = [float(v) for v in self.forecast_prob(s)]
            decision.p = decision.p_options[action]
        return decision

    def _forecast_sequence(self, context: list[int], option: list[int]) -> list[int]:
        """The full sequence after whose last token the forecast of `option` is read."""
        if self.cfg.forecast_format == "no_eos":
            return context + option[:-1] + self.forecast_tail_ids
        return context + option + self.forecast_tail_ids

    def _backbone(self, model, ids: torch.Tensor, cache=None, **kwargs):
        """Transformer body without the output layer. Returns (hidden states, KV cache)."""
        if cache is not None:
            kwargs[self._cache_kw] = cache
        out = model.base_model(input_ids=ids, use_cache=True, **kwargs)
        return out.last_hidden_state, getattr(out, self._cache_kw)

    @staticmethod
    def _head(model, hidden: torch.Tensor, rows: Sequence[int] | None = None) -> torch.Tensor:
        """FP32 output layer: F.linear(hidden.float(), W.float()), shape (len(hidden), vocab), or only the columns
        `rows` of the vocabulary. The backbone may run in bf16, but logits computed in bf16 are rounded to bf16
        (spacing 0.125 for values between 16 and 32), which is far coarser than the log-prob differences the loss
        works with (T0002, E0003). Casting the hidden states and W first keeps full precision; gradients still flow
        back to the bf16 parameters through the cast."""
        head = model.get_output_embeddings()
        weight, bias = head.weight, getattr(head, "bias", None)
        if rows is not None:
            index = torch.as_tensor(rows, device=weight.device)
            weight, bias = weight[index], None if bias is None else bias[index]
        return F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())

    @classmethod
    def _logprobs(cls, model, hidden: torch.Tensor) -> torch.Tensor:
        """FP32 output layer + log-softmax for the given hidden states, shape (rows, vocab). Sampling (serial and
        batched) and training all go through this one function. Only ever called on the few positions whose
        next-token distribution is needed: full-vocabulary logits for a whole context do not fit on a GPU (E0001)."""
        return torch.log_softmax(cls._head(model, hidden), dim=-1)

    def _sample_reasoning(self, prompt_ids: list[int], generator: torch.Generator) -> tuple[list[int], list[float]]:
        tokens, logps = [], []
        if self.cfg.max_reasoning_tokens <= 0:
            return tokens, logps
        hidden, cache = self._backbone(self.model, self._tensor(prompt_ids))
        for _ in range(self.cfg.max_reasoning_tokens):
            logprobs = self._logprobs(self.model, hidden[0, -1:])[0]        # last position only
            tok = int(torch.multinomial(logprobs.exp().to(generator.device), 1, generator=generator))
            tokens.append(tok)
            logps.append(float(logprobs[tok]))
            if tok == self.eos_id or self._answer_reached(tokens):
                break
            hidden, cache = self._backbone(self.model, self._tensor([tok]), cache)
        return tokens, logps

    @torch.no_grad()
    def sample_reasoning_batch(self, prompts: Sequence[list[int]], generators: Sequence[torch.Generator]
                               ) -> list[tuple[list[int], list[float]]]:
        """`_sample_reasoning` for several episodes at once: (tokens, log-probs) per prompt, in order.

        Row i is sampled from exactly pi_theta(. | prompt_i, its own earlier tokens) with its own generator:
          * prompts are LEFT-padded, so the last column is the last prompt token of every row; pads are masked out
            and position_ids = cumsum(mask) - 1 gives every real token the position it has in the unpadded prompt;
          * rows of a batch never attend to each other, so a row's logits depend on nothing but its own tokens;
          * each row stops on its own (EOS, `_answer_reached`, or the cap). A stopped row keeps being fed a pad
            token with attention mask 0 so that the batch stays rectangular; its outputs are discarded.
        KV cache: rows x (longest prompt + generated tokens) x layers x 2 x kv_heads x head_dim values."""
        b = len(prompts)
        tokens: list[list[int]] = [[] for _ in range(b)]
        logps: list[list[float]] = [[] for _ in range(b)]
        cap = self.cfg.max_reasoning_tokens
        if cap <= 0 or b == 0:
            return list(zip(tokens, logps))
        device, width = self.device, max(len(p) for p in prompts)
        ids = torch.full((b, width), self.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((b, width), dtype=torch.long, device=device)
        for i, prompt in enumerate(prompts):
            ids[i, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
            mask[i, width - len(prompt):] = 1
        hidden, cache = self._backbone(self.model, ids, attention_mask=mask,
                                       position_ids=(mask.cumsum(1) - 1).clamp(min=0))
        next_pos = mask.sum(1)                                   # position of the next real token of every row
        alive = list(range(b))
        for step in range(cap):
            logprobs = self._logprobs(self.model, hidden[alive, -1])            # last position only
            probs = logprobs.exp()
            drawn = torch.stack([torch.multinomial(probs[r].to(generators[i].device), 1, generator=generators[i])[0]
                                 for r, i in enumerate(alive)]).to(device)
            drawn_lp = logprobs.gather(1, drawn[:, None])[:, 0]
            still = []
            for i, tok, lp in zip(alive, drawn.tolist(), drawn_lp.tolist()):
                tokens[i].append(tok)
                logps[i].append(lp)
                if not (tok == self.eos_id or self._answer_reached(tokens[i])):
                    still.append(i)
            alive = still
            if not alive or step == cap - 1:
                break
            feed = torch.full((b, 1), self.pad_id, dtype=torch.long, device=device)
            keep = torch.zeros((b, 1), dtype=torch.long, device=device)
            rows = torch.tensor(alive, device=device)
            feed[rows, 0] = torch.tensor([tokens[i][-1] for i in alive], dtype=torch.long, device=device)
            keep[rows, 0] = 1
            mask = torch.cat([mask, keep], dim=1)
            hidden, cache = self._backbone(self.model, feed, cache, attention_mask=mask,
                                           position_ids=next_pos[:, None])
            next_pos = next_pos + keep[:, 0]
        return list(zip(tokens, logps))

    def _ends_with_answer(self, tokens: list[int]) -> bool:
        tail = self.tokenizer.decode(tokens[-16:])
        return tail.rstrip().endswith(self.cfg.answer_prefix)

    def _think_open(self, tokens: list[int]) -> bool:
        """prompt_format "chat_think" only: the thinking block opened by the prompt is not closed in `tokens`."""
        return self.cfg.prompt_format == "chat_think" and THINK_CLOSE not in self.tokenizer.decode(tokens)

    def _answer_reached(self, tokens: list[int]) -> bool:
        """The stop condition of sampling besides EOS and the cap: c_t ends with "Answer:", and with "chat_think"
        that "Answer:" stands after "</think>" (one written inside the thinking block does not stop c_t)."""
        return self._ends_with_answer(tokens) and not self._think_open(tokens)

    def _finish(self, reasoning: list[int]) -> tuple[list[int], list[int], str | None]:
        """How c_t ended and what follows it in the action context: (body, forced suffix, kind). The action context
        is prompt + body + suffix and always ends with "Answer:".

          kind            c_t                                              body      suffix
          "answer"        ends with "Answer:" (`_answer_reached`)          c         nothing
          "eos"           ends with a sampled EOS                          c[:-1]    forced, see below
          "cap"           max_reasoning_tokens reached, no open thinking   c         "\nAnswer:"
          "cap_in_think"  cap reached inside the thinking block            c         "\n</think>\n\nAnswer:"
          None            no reasoning (max_reasoning_tokens = 0)          empty     forced, see below

        The forced suffix is "\n</think>\n\nAnswer:" when the body leaves the thinking block of "chat_think" open
        and "\nAnswer:" otherwise (always, in the other two formats). Since the suffix is never empty unless
        body = c, len(prompt) + |c| <= len(context) holds in every case, and context[:P + |c| - 1] is the prompt
        followed by c[:-1]: the invariant `decision_forward` needs to score the reasoning tokens."""
        eos = bool(reasoning) and reasoning[-1] == self.eos_id
        body = reasoning[:-1] if eos else reasoning
        if body and not eos and self._answer_reached(body):
            return body, [], "answer"
        think_open = self._think_open(body)
        suffix = self.think_answer_ids if think_open else self.answer_ids
        kind = None if not reasoning else "eos" if eos else "cap_in_think" if think_open else "cap"
        return body, suffix, kind

    def _action_context(self, prompt_ids: list[int], reasoning: list[int]) -> list[int]:
        body, suffix, _ = self._finish(reasoning)
        return prompt_ids + body + suffix

    # ---------------------------------------------------------------- scoring (differentiable)

    def decision_forward(self, model, d: Decision, forecast: bool = False
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Everything one decision needs from `model`, from a single forward in which the shared prefix appears once.

        Returns (reasoning_logps (|c|,), option_scores l(a^j) (K,), s (K,) or None), all FP32. s[j] is the forecast
        logit l(yes) - l(no) read after option j; `forecast_prob(s[j])` is "the probability of winning if a^j is
        chosen", and `forecast_prob(s[d.action])` is p_t of eq. (5).

        With x = d.action_context_ids (length n), every sequence x + y^j shares the hidden states of its first n
        positions (causal attention). So the K suffixes r_j = (x[n-1]) + y^j (+ z, the confidence question, when
        `forecast`) are laid end to end after the prefix x[0:n-1] in ONE batch-1 sequence, with
          * a 4D attention mask: causal inside the prefix; a token of r_j sees the whole prefix and the earlier
            tokens of r_j, never another option's tokens;
          * position_ids that restart at n-1 for every r_j.
        Each token then has exactly the context and position it has in the separate sequence x + y^j + z, so the
        numbers and gradients are those of K full-context forwards, while keys/values exist once (a batch of K
        suffix rows on a KV cache would hold K copies of the prefix keys/values per layer: 22 GiB with gradients
        at n=2200, K=16). Appending z does not change l(a^j), again because attention is causal.
        forecast_format "chat" only changes z. With "no_eos" the question must follow " j" without the EOS, which
        r_j cannot provide (l(a^j) needs the EOS term), so r_j = (x[n-1]) + y^j and every option gets a second
        segment f_j = (x[n-1]) + y^j[:-1] + z after all the r_j, again seeing only the prefix and itself.
        The logits at position i predict token i + 1, and the output layer is applied only where needed."""
        x, c, options = d.action_context_ids, d.reasoning_ids, d.option_ids
        n, n_prompt, k = len(x), len(d.prompt_ids), len(options)
        # The reasoning tokens are predicted at positions P-1 .. P+|c|-2, which must lie inside the prefix x[0:n-1],
        # i.e. P + |c| <= n. Every way the reasoning can end satisfies it (`_finish`): ending with "Answer:" gives
        # n = P + |c|; a sampled EOS gives n = P + |c| - 1 + |suffix|; hitting the token cap gives
        # n = P + |c| + |suffix|, with |suffix| >= 1. In each case x[0:P+|c|-1] is the prompt followed by c[:-1].
        assert n >= 2 and n_prompt >= 1 and n_prompt + len(c) <= n, (n_prompt, len(c), n)
        assert x[:n_prompt + max(len(c) - 1, 0)] == d.prompt_ids + c[:max(len(c) - 1, 0)]

        tail = self.forecast_tail_ids if forecast else []
        separate = forecast and self.cfg.forecast_format == "no_eos"
        rows = [[x[n - 1]] + opt + ([] if separate else tail) for opt in options]
        if separate:                      # forecast segments f_j; rows[k + j] belongs to option j
            rows += [[x[n - 1]] + opt[:-1] + tail for opt in options]
        starts, total = [], n - 1                       # where each r_j begins in the folded sequence
        for r in rows:
            starts.append(total)
            total += len(r)
        ids = self._tensor(x[:n - 1] + [t for r in rows for t in r])
        position_ids = torch.cat([torch.arange(n - 1, device=self.device)]
                                 + [torch.arange(n - 1, n - 1 + len(r), device=self.device) for r in rows])[None]
        # segment 0 = prefix, j + 1 = r_j. Query q may attend to key s iff s <= q and s is in the prefix or in q's
        # own segment.
        segment = torch.cat([torch.zeros(n - 1, dtype=torch.long, device=self.device)]
                            + [torch.full((len(r),), j + 1, device=self.device) for j, r in enumerate(rows)])
        allowed = torch.ones(total, total, dtype=torch.bool, device=self.device).tril()
        allowed &= (segment[None, :] == 0) | (segment[None, :] == segment[:, None])
        dtype = next(model.parameters()).dtype
        mask = torch.zeros(total, total, dtype=dtype, device=self.device).masked_fill(~allowed, torch.finfo(dtype).min)
        hidden = model.base_model(input_ids=ids, attention_mask=mask[None, None], position_ids=position_ids,
                                  use_cache=False).last_hidden_state[0]

        # One output-layer call for the reasoning tokens and the option tokens (W.float() is built once).
        # l_j = sum_{i=1..m_j} log softmax(lm_head(h_{j,i-1}))[y^j_i]; h_{j,0} is the state at x[n-1].
        row_idx = torch.tensor([j for j, opt in enumerate(options) for _ in opt], device=self.device)
        pos_idx = torch.tensor(list(range(n_prompt - 1, n_prompt + len(c) - 1))
                               + [starts[j] + i for j, opt in enumerate(options) for i in range(len(opt))],
                               device=self.device)
        target = torch.tensor(c + [t for opt in options for t in opt], device=self.device)
        token_lp = self._logprobs(model, hidden[pos_idx]).gather(1, target[:, None])[:, 0]
        reasoning_logps, option_lp = token_lp[:len(c)], token_lp[len(c):]
        scores = torch.zeros(k, device=self.device, dtype=token_lp.dtype).index_add(0, row_idx, option_lp)

        s = None
        if forecast:                      # read at the last token of r_j (f_j); only the yes / no logits are needed
            first = k if separate else 0
            last = torch.tensor([starts[first + j] + len(rows[first + j]) - 1 for j in range(k)], device=self.device)
            yes_no = self._head(model, hidden[last], rows=[self.yes_id, self.no_id])
            s = yes_no[:, 0] - yes_no[:, 1]
        return reasoning_logps, scores, s

    def verbal_forward(self, model, d: Decision, forecast: bool, branches: Sequence[int]
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """`decision_forward` plus the verbal confidence (T0008), again from one forward over a folded tree.

        Returns (reasoning_logps, option_scores (K,), s (K,) or None, conf (len(branches), M)) with
        conf[b, m] = l^c(m | h_t, a^{branches[b]}) = sum_k log pi(y^m_k | x + y^j + u + y^m_<k), unnormalised
        (pi^c = softmax over m). Segments after the prefix x[0:n-1], each seeing the prefix, its ancestors and
        itself (causally), with positions continuing from its parent:
          r_j = (x[n-1]) + y^j                   every option (parent: prefix)
          f_j = forecast question               when `forecast` (parent r_j): s is read at its last token
          u_j = u                               for j in `branches` (parent r_j)
          y_{j,m} = y^m[:-1]                    every answer m (parent u_j); its first token is predicted at the
                                                last token of u_j, token k at token k - 1 of y_{j,m}
        so every token has exactly the context and position of the separate sequence x + y^j (+ u + y^m).
        Option scores and s equal those of `decision_forward` (up to floating point: the layout differs)."""
        x, c, options = d.action_context_ids, d.reasoning_ids, d.option_ids
        n, n_prompt, k = len(x), len(d.prompt_ids), len(options)
        assert n >= 2 and n_prompt >= 1 and n_prompt + len(c) <= n, (n_prompt, len(c), n)
        assert x[:n_prompt + max(len(c) - 1, 0)] == d.prompt_ids + c[:max(len(c) - 1, 0)]
        assert self.verbal, 'verbal_forward needs confidence_mode = "verbal"'
        if forecast and self.cfg.forecast_format == "no_eos":
            raise ValueError('the verbal forward supports forecast_format "after_eos" and "chat" only')
        branches = list(branches)
        segs: list[tuple[list[int], int]] = [([x[n - 1]] + opt, -1) for opt in options]   # (tokens, parent)
        tail_seg = []
        if forecast:
            for j in range(k):
                tail_seg.append(len(segs))
                segs.append((self.forecast_tail_ids, j))
        u_seg, bin_seg = [], []
        for j in branches:
            u_seg.append(len(segs))
            segs.append((self.verbal_prompt_ids, j))
            row = []
            for y in self.bin_ids:
                row.append(len(segs) if len(y) > 1 else None)
                if len(y) > 1:
                    segs.append((y[:-1], u_seg[-1]))
            bin_seg.append(row)
        starts, pos0, total = [], [], n - 1          # folded index and position of every segment's first token
        for tokens, parent in segs:
            starts.append(total)
            pos0.append(n - 1 if parent < 0 else pos0[parent] + len(segs[parent][0]))
            total += len(tokens)
        dev = self.device
        ids = self._tensor(x[:n - 1] + [t for tokens, _ in segs for t in tokens])
        position_ids = torch.cat([torch.arange(n - 1, device=dev)]
                                 + [torch.arange(p, p + len(tokens), device=dev)
                                    for (tokens, _), p in zip(segs, pos0)])[None]
        # ancestor-or-self table over segment ids (0 = prefix, s + 1 = segs[s])
        n_seg = len(segs) + 1
        anc = torch.zeros(n_seg, n_seg, dtype=torch.bool)
        anc[:, 0] = True
        for s_id, (_, parent) in enumerate(segs):
            row = anc[parent + 1].clone() if parent >= 0 else anc[0].clone()
            row[s_id + 1] = True
            anc[s_id + 1] = row
        anc = anc.to(dev)
        segment = torch.cat([torch.zeros(n - 1, dtype=torch.long, device=dev)]
                            + [torch.full((len(tokens),), s_id + 1, device=dev) for s_id, (tokens, _) in enumerate(segs)])
        allowed = torch.ones(total, total, dtype=torch.bool, device=dev).tril()
        allowed &= anc[segment[:, None], segment[None, :]]
        dtype = next(model.parameters()).dtype
        mask = torch.zeros(total, total, dtype=dtype, device=dev).masked_fill(~allowed, torch.finfo(dtype).min)
        hidden = model.base_model(input_ids=ids, attention_mask=mask[None, None], position_ids=position_ids,
                                  use_cache=False).last_hidden_state[0]

        positions = list(range(n_prompt - 1, n_prompt + len(c) - 1))
        targets = list(c)
        for j, opt in enumerate(options):
            positions += [starts[j] + i for i in range(len(opt))]
            targets += opt
        conf_index = []                               # (branch b, bin m) of every confidence token
        for b, j in enumerate(branches):
            last_u = starts[u_seg[b]] + len(self.verbal_prompt_ids) - 1
            for m, y in enumerate(self.bin_ids):
                positions += [last_u] + [starts[bin_seg[b][m]] + i for i in range(len(y) - 1)]
                targets += y
                conf_index += [b * len(self.bin_ids) + m] * len(y)
        token_lp = self._logprobs(model, hidden[torch.tensor(positions, device=dev)]).gather(
            1, torch.tensor(targets, device=dev)[:, None])[:, 0]
        n_opt = sum(len(o) for o in options)
        reasoning_logps = token_lp[:len(c)]
        row_idx = torch.tensor([j for j, opt in enumerate(options) for _ in opt], device=dev)
        scores = torch.zeros(k, device=dev, dtype=token_lp.dtype).index_add(0, row_idx, token_lp[len(c):len(c) + n_opt])
        conf = torch.zeros(len(branches) * len(self.bin_ids), device=dev, dtype=token_lp.dtype).index_add(
            0, torch.tensor(conf_index, device=dev, dtype=torch.long), token_lp[len(c) + n_opt:])
        conf = conf.view(len(branches), len(self.bin_ids))
        s = None
        if forecast:
            last = torch.tensor([starts[tail_seg[j]] + len(self.forecast_tail_ids) - 1 for j in range(k)], device=dev)
            yes_no = self._head(model, hidden[last], rows=[self.yes_id, self.no_id])
            s = yes_no[:, 0] - yes_no[:, 1]
        return reasoning_logps, scores, s, conf

    def score_with_confidence(self, d: Decision, model=None, forecast: bool = False
                              ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """`score` plus log pi^c (M,) after the chosen option (verbal mode): one `verbal_forward` with only the
        chosen option's confidence branch."""
        model = self.model if model is None else model
        reasoning_logps, scores, s, conf = self.verbal_forward(model, d, forecast, [d.action])
        log_q = torch.log_softmax(scores, dim=0)
        logps = torch.cat([reasoning_logps, log_q[d.action:d.action + 1]])
        return logps, (s[d.action] if s is not None else None), torch.log_softmax(conf[0], dim=0)

    def score(self, d: Decision, model=None, forecast: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Log-probabilities of the decision's |c_t| + 1 terms under `model` (default: the policy), plus the
        forecast logit s_t when `forecast` (p_t = `forecast_prob(s_t)`). One call to `decision_forward`; gradients
        flow when called outside torch.no_grad()."""
        model = self.model if model is None else model
        reasoning_logps, scores, s = self.decision_forward(model, d, forecast=forecast)
        log_q = torch.log_softmax(scores, dim=0)
        logps = torch.cat([reasoning_logps, log_q[d.action:d.action + 1]])
        return logps, (s[d.action] if s is not None else None)
