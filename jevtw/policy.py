"""Language-model policy, eqs. (4)-(5).

At step t the model sees a prompt built from h_t, samples reasoning tokens c_t (temperature 1, full vocabulary, so the
sampling law is exactly pi_theta), then picks a_t from q_theta = softmax over the K_t candidate commands, where the
logit l(a^j) of option j is the log-likelihood of the continuation " j<EOS>" after "Answer:". Afterwards it can read
the forecast p_t = sigmoid(l(yes) - l(no)) after a confidence question that is never written into the history.

All three come from one forward in `decision_forward`: the shared prefix (the action context) followed by the K
suffixes "option j + confidence question", which are kept apart by the attention mask.

Likelihood terms: every sampled reasoning token (its exact ids, including a sampled EOS or the tokens of "Answer:")
and the single categorical action. Prompts, observations and the forced "Answer:" have none. Token ids are stored
and reused verbatim, so scoring never re-tokenises generated text."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .config import PolicyConfig
from .envs import Observation
from .readout import single_token_id, yes_no_probability


@dataclass
class Decision:
    command: str
    action: int                           # index of a_t among the options
    prompt_ids: list[int]
    reasoning_ids: list[int]              # c_t
    action_context_ids: list[int]         # prompt + c_t (minus a final EOS) + forced "Answer:" when needed
    option_ids: list[list[int]]           # " j<EOS>" for every option j
    behavior_logps: list[float]           # log-probs at sampling time: |c_t| reasoning tokens, then log q(a_t)
    p: float | None = None                # forecast at sampling time (theta_old), used for b_i
    p_options: list[float] | None = None  # p_j for every option j: "win probability if a^j were chosen" (Q_win)
    confidence_ids: list[int] | None = None
    old_logps: list[float] | None = field(default=None, repr=False)   # filled by the trainer
    ref_logps: list[float] | None = field(default=None, repr=False)

    @property
    def n_terms(self) -> int:
        return len(self.reasoning_ids) + 1


def build_prompt(objective: str, history: Sequence[tuple[str, str]], obs: Observation, cfg: PolicyConfig) -> str:
    """Text of h_t: the objective, the last `history_window` (command, observation) pairs, the current observation
    and the numbered admissible commands."""
    lines = [f"Objective: {objective}", ""]
    recent = list(history)[-cfg.history_window:] if cfg.history_window > 0 else []
    if recent:
        lines.append("Recent steps:")
        for cmd, text in recent:
            lines += [f"> {cmd}", text, ""]
    lines += ["Current observation:", obs.text, "", "Options:"]
    lines += [f"{j + 1}. {c}" for j, c in enumerate(obs.commands)]
    lines.append("")
    if cfg.max_reasoning_tokens > 0:
        lines += [f'Think step by step, then write "{cfg.answer_prefix}" followed by the number of one option.',
                  cfg.reasoning_prefix]
    else:
        lines.append("Reply with the number of one option.")
    return "\n".join(lines)


class LMPolicy:
    def __init__(self, model, tokenizer, cfg: PolicyConfig):
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("the tokenizer needs an EOS token to delimit option numbers")
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_id
        self.answer_ids = self.encode("\n" + cfg.answer_prefix)
        self.confidence_prompt_ids = self.encode(cfg.confidence_prompt)
        self.yes_id = single_token_id(tokenizer, cfg.yes_text)
        self.no_id = single_token_id(tokenizer, cfg.no_text)
        if self.yes_id == self.no_id:
            raise ValueError("yes and no map to the same token")
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

    # ---------------------------------------------------------------- sampling (theta_old, no gradients)

    @torch.no_grad()
    def act(self, objective: str, history: Sequence[tuple[str, str]], obs: Observation,
            generator: torch.Generator) -> Decision:
        """One decision for one episode (the serial reference; `rollout.collect_batch` uses the three steps below
        with `sample_reasoning_batch` in place of `_sample_reasoning`)."""
        prompt_ids = self.encode_prompt(objective, history, obs)
        reasoning, logps = self._sample_reasoning(prompt_ids, generator)
        return self.decide(prompt_ids, reasoning, logps, obs, generator)

    def encode_prompt(self, objective: str, history: Sequence[tuple[str, str]], obs: Observation) -> list[int]:
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
                            action_context_ids=context, option_ids=options, behavior_logps=[])
        _, scores, p = self.decision_forward(self.model, decision, forecast=cfg.forecast)
        log_q = torch.log_softmax(scores, dim=0)
        action = int(torch.multinomial(log_q.exp().to(generator.device), 1, generator=generator))
        decision.command, decision.action = obs.commands[action], action
        decision.behavior_logps = logps + [float(log_q[action])]
        if p is not None:
            decision.confidence_ids = context + options[action] + self.confidence_prompt_ids
            decision.p_options = [float(v) for v in p]
            decision.p = decision.p_options[action]
        return decision

    def _backbone(self, model, ids: torch.Tensor, cache=None, **kwargs):
        """Transformer body without the output layer. Returns (hidden states, KV cache)."""
        if cache is not None:
            kwargs[self._cache_kw] = cache
        out = model.base_model(input_ids=ids, use_cache=True, **kwargs)
        return out.last_hidden_state, getattr(out, self._cache_kw)

    @staticmethod
    def _logprobs(model, hidden: torch.Tensor) -> torch.Tensor:
        """Output layer + FP32 log-softmax for the given hidden states, shape (rows, vocab). Only ever called on the
        few positions whose next-token distribution is needed: full-vocabulary logits for a whole context do not
        fit on a GPU (E0001)."""
        return torch.log_softmax(model.get_output_embeddings()(hidden).float(), dim=-1)

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
            if tok == self.eos_id or self._ends_with_answer(tokens):
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
          * each row stops on its own (EOS, ends with "Answer:", or the cap). A stopped row keeps being fed a pad
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
                if not (tok == self.eos_id or self._ends_with_answer(tokens[i])):
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

    def _action_context(self, prompt_ids: list[int], reasoning: list[int]) -> list[int]:
        body = reasoning[:-1] if reasoning and reasoning[-1] == self.eos_id else reasoning
        if body and body is reasoning and self._ends_with_answer(body):
            return prompt_ids + body
        return prompt_ids + body + self.answer_ids

    # ---------------------------------------------------------------- scoring (differentiable)

    def decision_forward(self, model, d: Decision, forecast: bool = False
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Everything one decision needs from `model`, from a single forward in which the shared prefix appears once.

        Returns (reasoning_logps (|c|,), option_scores l(a^j) (K,), p (K,) or None), all FP32. p[j] is the forecast
        read after option j, i.e. "the probability of winning if a^j is chosen"; p[d.action] is p_t of eq. (5).

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
        The logits at position i predict token i + 1, and the output layer is applied only where needed."""
        x, c, options = d.action_context_ids, d.reasoning_ids, d.option_ids
        n, n_prompt, k = len(x), len(d.prompt_ids), len(options)
        # The reasoning tokens are predicted at positions P-1 .. P+|c|-2, which must lie inside the prefix x[0:n-1],
        # i.e. P + |c| <= n. All three ways the reasoning can end satisfy it: ending with "Answer:" gives
        # n = P + |c|; a sampled EOS gives n = P + |c| - 1 + |answer|; hitting the token cap gives
        # n = P + |c| + |answer|. In each case x[0:P+|c|-1] is the prompt followed by c[:-1].
        assert n >= 2 and n_prompt >= 1 and n_prompt + len(c) <= n, (n_prompt, len(c), n)
        assert x[:n_prompt + max(len(c) - 1, 0)] == d.prompt_ids + c[:max(len(c) - 1, 0)]

        tail = self.confidence_prompt_ids if forecast else []
        rows = [[x[n - 1]] + opt + tail for opt in options]
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

        if c:
            logprobs = self._logprobs(model, hidden[n_prompt - 1:n_prompt + len(c) - 1])
            target = torch.tensor(c, device=self.device)
            reasoning_logps = logprobs.gather(1, target[:, None])[:, 0]
        else:
            reasoning_logps = torch.zeros(0, device=self.device)

        # l_j = sum_{i=1..m_j} log softmax(lm_head(h_{j,i-1}))[y^j_i]; h_{j,0} is the state at x[n-1].
        row_idx = torch.tensor([j for j, opt in enumerate(options) for _ in opt], device=self.device)
        pos_idx = torch.tensor([starts[j] + i for j, opt in enumerate(options) for i in range(len(opt))],
                               device=self.device)
        target = torch.tensor([t for opt in options for t in opt], device=self.device)
        token_lp = self._logprobs(model, hidden[pos_idx]).gather(1, target[:, None])[:, 0]
        scores = torch.zeros(k, device=self.device, dtype=token_lp.dtype).index_add(0, row_idx, token_lp)

        p = None
        if forecast:                      # read at the last token of every r_j
            last = torch.tensor([starts[j] + len(r) - 1 for j, r in enumerate(rows)], device=self.device)
            p = yes_no_probability(model.get_output_embeddings()(hidden[last]), self.yes_id, self.no_id)
        return reasoning_logps, scores, p

    def score(self, d: Decision, model=None, forecast: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Log-probabilities of the decision's |c_t| + 1 terms under `model` (default: the policy), plus p_t when
        `forecast`. One call to `decision_forward`; gradients flow when called outside torch.no_grad()."""
        model = self.model if model is None else model
        reasoning_logps, scores, p = self.decision_forward(model, d, forecast=forecast)
        log_q = torch.log_softmax(scores, dim=0)
        logps = torch.cat([reasoning_logps, log_q[d.action:d.action + 1]])
        return logps, (p[d.action] if p is not None else None)
