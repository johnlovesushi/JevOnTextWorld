"""Download-free stand-ins for CPU tests and smoke runs: a byte tokenizer and a tiny random causal LM."""
from __future__ import annotations

import torch


class ByteTokenizer:
    """UTF-8 bytes are tokens 0..255, EOS is 256. Implements the subset of the HF tokenizer API the policy uses."""

    eos_token_id = 256
    pad_token_id = 257
    vocab_size = 258

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def tiny_model(seed: int = 0, vocab_size: int = ByteTokenizer.vocab_size):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=8192,
                         pad_token_id=ByteTokenizer.pad_token_id, eos_token_id=ByteTokenizer.eos_token_id,
                         bos_token_id=None, tie_word_embeddings=False)
    model = LlamaForCausalLM(config).float()
    model.eval()                          # no dropout: sampling and training forwards agree
    return model
