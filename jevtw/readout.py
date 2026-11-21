"""Success forecast read from logits, eq. (5): p = sigmoid(l(yes) - l(no)).

This equals P(yes) after renormalising the next-token distribution over the two tokens {yes, no}:
e^{l_y} / (e^{l_y} + e^{l_n}) = 1 / (1 + e^{-(l_y - l_n)}). Only the difference matters."""
from __future__ import annotations

import torch


def yes_no_probability(logits: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    """logits: (..., vocab) next-token logits. Returns p with shape (...), differentiable, computed in FP32."""
    logits = logits.float()
    return torch.sigmoid(logits[..., yes_id] - logits[..., no_id])


def single_token_id(tokenizer, text: str) -> int:
    """The id of `text`, which must encode to exactly one token (e.g. " yes" for Qwen)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} encodes to {len(ids)} tokens {ids}; the readout needs exactly one")
    return ids[0]
