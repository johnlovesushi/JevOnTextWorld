"""Language-model policy, eqs. (4)-(5).

At step t the model sees a prompt built from h_t, samples reasoning tokens c_t (temperature 1, full vocabulary, so the
sampling law is exactly pi_theta), then picks a_t from q_theta = softmax over the K_t candidate commands, where the
logit l(a^j) of option j is the log-likelihood of the continuation " j<EOS>" after "Answer:". Afterwards it can read
the forecast p_t = sigmoid(l(yes) - l(no)) after a confidence question that is never written into the history.

Likelihood terms: every sampled reasoning token (its exact ids, including a sampled EOS or the tokens of "Answer:")
and the single categorical action. Prompts, observations and the forced "Answer:" have none. Token ids are stored
and reused verbatim, so scoring never re-tokenises generated text."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F

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
        self._cache_kw = "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"

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
        if not obs.commands:
            raise ValueError("no admissible commands")
        cfg = self.cfg
        prompt_ids = self.encode(build_prompt(objective, history, obs, cfg))
        reasoning, logps = self._sample_reasoning(prompt_ids, generator)
        context = self._action_context(prompt_ids, reasoning)
        options = [self.encode(f" {j + 1}") + [self.eos_id] for j in range(len(obs.commands))]
        log_q = torch.log_softmax(self._option_scores(self.model, context, options), dim=0)
        action = int(torch.multinomial(log_q.exp().cpu(), 1, generator=generator))
        decision = Decision(command=obs.commands[action], action=action, prompt_ids=prompt_ids,
                            reasoning_ids=reasoning, action_context_ids=context, option_ids=options,
                            behavior_logps=logps + [float(log_q[action])])
        if cfg.forecast:
            decision.confidence_ids = context + options[action] + self.confidence_prompt_ids
            decision.p = float(self._forecast(self.model, decision.confidence_ids))
        return decision

    def _sample_reasoning(self, prompt_ids: list[int], generator: torch.Generator) -> tuple[list[int], list[float]]:
        tokens, logps = [], []
        if self.cfg.max_reasoning_tokens <= 0:
            return tokens, logps
        out = self.model(input_ids=self._tensor(prompt_ids), use_cache=True)
        for _ in range(self.cfg.max_reasoning_tokens):
            logprobs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            tok = int(torch.multinomial(logprobs.exp().cpu(), 1, generator=generator))
            tokens.append(tok)
            logps.append(float(logprobs[tok]))
            if tok == self.eos_id or self._ends_with_answer(tokens):
                break
            out = self.model(input_ids=self._tensor([tok]), use_cache=True,
                             **{self._cache_kw: getattr(out, self._cache_kw)})
        return tokens, logps

    def _ends_with_answer(self, tokens: list[int]) -> bool:
        tail = self.tokenizer.decode(tokens[-16:])
        return tail.rstrip().endswith(self.cfg.answer_prefix)

    def _action_context(self, prompt_ids: list[int], reasoning: list[int]) -> list[int]:
        body = reasoning[:-1] if reasoning and reasoning[-1] == self.eos_id else reasoning
        if body and body is reasoning and self._ends_with_answer(body):
            return prompt_ids + body
        return prompt_ids + body + self.answer_ids

    # ---------------------------------------------------------------- scoring (differentiable)

    def _option_scores(self, model, context: list[int], options: list[list[int]]) -> torch.Tensor:
        """l(a^j) = sum of log-probs of option j's tokens after the context, for every option. Shape (K,), FP32."""
        scores = []
        n = len(context)
        for start in range(0, len(options), self.cfg.option_batch_size):
            chunk = options[start:start + self.cfg.option_batch_size]
            width = n + max(len(o) for o in chunk)
            ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for r, opt in enumerate(chunk):
                ids[r, :n + len(opt)] = torch.tensor(context + opt, device=self.device)
                mask[r, :n + len(opt)] = 1
            # Only positions n-1 .. n-2+|a^j| predict option tokens. Run the transformer body on the whole batch
            # but apply the output layer (and FP32 log-softmax) to those positions alone: full-vocabulary logits
            # for every context position of every row do not fit on a GPU with a real vocabulary
            # (16 rows x ~2200 tokens x 152k vocab x 4 bytes = 20 GiB for Qwen).
            hidden = model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
            rows = torch.tensor([r for r, opt in enumerate(chunk) for _ in opt], device=self.device)
            pos = torch.tensor([n - 1 + k for opt in chunk for k in range(len(opt))], device=self.device)
            target = torch.tensor([t for opt in chunk for t in opt], device=self.device)
            logprobs = torch.log_softmax(model.get_output_embeddings()(hidden[rows, pos]).float(), dim=-1)
            token_lp = logprobs.gather(1, target[:, None])[:, 0]
            sizes = [len(opt) for opt in chunk]
            scores.extend(part.sum() for part in token_lp.split(sizes))
        return torch.stack(scores)

    def _reasoning_logps(self, model, d: Decision) -> torch.Tensor:
        if not d.reasoning_ids:
            return torch.zeros(0, device=self.device)
        n = len(d.prompt_ids)
        logits = model(input_ids=self._tensor(d.prompt_ids + d.reasoning_ids)).logits[0, n - 1:-1].float()
        target = torch.tensor(d.reasoning_ids, device=self.device)
        return torch.log_softmax(logits, dim=-1).gather(1, target[:, None])[:, 0]

    def _forecast(self, model, confidence_ids: list[int]) -> torch.Tensor:
        logits = model(input_ids=self._tensor(confidence_ids)).logits[0, -1]
        return yes_no_probability(logits, self.yes_id, self.no_id)

    def score(self, d: Decision, model=None, forecast: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Log-probabilities of the decision's |c_t| + 1 terms under `model` (default: the policy), plus p_t when
        `forecast`. Gradients flow when called outside torch.no_grad()."""
        model = self.model if model is None else model
        log_q = torch.log_softmax(self._option_scores(model, d.action_context_ids, d.option_ids), dim=0)
        logps = torch.cat([self._reasoning_logps(model, d), log_q[d.action:d.action + 1]])
        p = None
        if forecast:
            if d.confidence_ids is None:
                d.confidence_ids = d.action_context_ids + d.option_ids[d.action] + self.confidence_prompt_ids
            p = self._forecast(model, d.confidence_ids)
        return logps, p
