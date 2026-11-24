"""Time and peak memory of one decision's scoring on a real model (T0001).

Builds a synthetic decision from random token ids with an action context of n tokens, K options and |c| reasoning
tokens, then measures (a) `decision_forward` without gradients, as in rollouts, and (b) `score()` + backward, as in
one training term. Prints one JSON line per setting.

    python scripts/bench_decision.py --model Qwen/Qwen3-1.7B --setting 2200,16,320 --setting 2600,40,320
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevtw.config import PolicyConfig  # noqa: E402
from jevtw.policy import Decision, LMPolicy  # noqa: E402


def synthetic_decision(policy: LMPolicy, n: int, k: int, c: int, seed: int = 0) -> Decision:
    """Reasoning that used its whole budget: context = prompt + c + forced "Answer:", so n = P + |c| + |answer|."""
    n_prompt = n - c - len(policy.answer_ids)
    if n_prompt < 1:
        raise ValueError(f"n={n} is too short for |c|={c}")
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 50000, (n_prompt + c,), generator=g).tolist()      # ordinary text tokens
    prompt, reasoning = ids[:n_prompt], ids[n_prompt:]
    options = [policy.encode(f" {j + 1}") + [policy.eos_id] for j in range(k)]
    d = Decision(command="", action=k // 2, prompt_ids=prompt, reasoning_ids=reasoning,
                 action_context_ids=prompt + reasoning + policy.answer_ids, option_ids=options, behavior_logps=[])
    assert len(d.action_context_ids) == n
    return d


def measure(fn, device: torch.device, repeats: int) -> dict:
    cuda = device.type == "cuda"
    times, peak = [], 0
    for i in range(repeats + 1):                  # the first call is a warm-up and is not counted
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize(device)
        if i > 0:
            times.append(time.perf_counter() - start)
            peak = max(peak, torch.cuda.max_memory_allocated(device) if cuda else 0)
    return {"seconds": round(min(times), 4), "seconds_all": [round(t, 4) for t in times],
            "peak_gib": round(peak / 2 ** 30, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--setting", action="append", default=None, metavar="N,K,C",
                    help="context length, number of options, reasoning tokens (repeatable)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-forecast", action="store_true", help="leave out the p_t readout (alpha = 0)")
    args = ap.parse_args()
    settings = [tuple(int(v) for v in s.split(",")) for s in (args.setting or ["2200,16,320", "2600,40,320"])]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype).to(device).eval()
    policy = LMPolicy(model, tokenizer, PolicyConfig())
    forecast = not args.no_forecast
    base = torch.cuda.memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
    print(json.dumps({"model": args.model, "device": str(device), "dtype": str(dtype), "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                      "model_gib": round(base, 3), "forecast": forecast, "repeats": args.repeats}), flush=True)

    for n, k, c in settings:
        d = synthetic_decision(policy, n, k, c)

        def no_grad():
            with torch.no_grad():
                policy.decision_forward(model, d, forecast=forecast)

        def with_grad():
            model.zero_grad(set_to_none=True)
            logps, s = policy.score(d, forecast=forecast)
            (logps.sum() + (policy.forecast_prob(s) if s is not None else 0.0)).backward()

        row = {"n": n, "K": k, "c": c}
        try:
            row["no_grad"] = measure(no_grad, device, args.repeats)
            row["score_backward"] = measure(with_grad, device, args.repeats)
        except torch.OutOfMemoryError as e:
            row["error"] = str(e)[:300]
        model.zero_grad(set_to_none=True)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
