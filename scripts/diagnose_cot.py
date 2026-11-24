"""E0007: on the walkthrough states of E0006, sample reasoning, then read the option distribution after it.

    python scripts/diagnose_cot.py --out runs/E0007/cot.jsonl --model Qwen/Qwen3-1.7B \
        --games L0=data/games_L0/valid L1=data/games_L1/valid

States: every state on a game's walkthrough (scripts/diagnose_history.py, `walk`) whose walkthrough action is
`prepare meal` or `eat meal`. Settings: history_mode x prompt_format (SETTINGS), window 8, no forecast. For every
state and setting, `--samples` reasonings c are sampled with `LMPolicy.sample_reasoning_batch`; after each one
`LMPolicy.decision_forward` gives the option distribution q(. | prompt, c) and an action is sampled from q with
the same generator (the three steps of `LMPolicy.act`). The generator of a sample is seeded from
sha256(game, step, history_mode, prompt_format, k), so a sample does not depend on the batch it was drawn in.
One JSON line per (level, game, step, setting, k).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_history import GAME_SUFFIXES, walk  # noqa: E402
from jevtw.config import PolicyConfig  # noqa: E402
from jevtw.policy import Decision, LMPolicy, build_prompt  # noqa: E402

# name -> PolicyConfig fields (issue #24: 2 histories x 2 reasoning formats)
SETTINGS = {
    "full_plain": {"history_mode": "full", "prompt_format": "plain", "max_reasoning_tokens": 320},
    "full_chat": {"history_mode": "full", "prompt_format": "chat_nothink", "max_reasoning_tokens": 128},
    "feedback_plain": {"history_mode": "feedback", "prompt_format": "plain", "max_reasoning_tokens": 320},
    "feedback_chat": {"history_mode": "feedback", "prompt_format": "chat_nothink", "max_reasoning_tokens": 128},
}
TYPES = ("prepare", "eat")
# Commands that process an ingredient. At the `prepare meal` and `eat meal` states every needed processing step
# is done, so each of these does it a second time, which loses the game.
REPROCESS = ("cook", "chop", "slice", "dice", "fry", "roast", "grill")
TOP = 5

_STATE = r"(?:fried|roasted|grilled|chopped|sliced|diced|cooked|prepared)"
_VERB = r"(?:fry|roast|grill|cook|chop|slice|dice)"
# "already fried", "has already been chopped", "is fried", "are now diced", ...
NOTICES_DONE = re.compile(rf"\balready\s+(?:been\s+)?{_STATE}\b|\b(?:is|are|was|were|been)\s+(?:already\s+|now\s+)?{_STATE}\b",
                          re.IGNORECASE)
# "need to fry", "should cook the ...", "next, chop ..." (within one sentence, at most 60 characters in between)
PLANS_PROCESS = re.compile(rf"\b(?:needs?\s+to|should|ha(?:ve|s)\s+to|must|let's|let\s+us|next)\b[^.\n]{{0,60}}?\b{_VERB}\b",
                           re.IGNORECASE)


def tags(reasoning: str) -> dict:
    """Rough keyword tags of a reasoning text (they do not see negations: "should not cook" plans_process)."""
    return {"notices_done": bool(NOTICES_DONE.search(reasoning)), "plans_process": bool(PLANS_PROCESS.search(reasoning))}


def is_reprocess(command: str) -> bool:
    return command.split()[0] in REPROCESS


def setting_config(name: str, cap=None, **overrides) -> PolicyConfig:
    """`cap` and `overrides` are for tests (a short reasoning; yes / no texts of another tokenizer)."""
    fields = {**overrides, **SETTINGS[name], "history_window": 8, "forecast": False}
    if cap is not None:
        fields["max_reasoning_tokens"] = cap
    return PolicyConfig(**fields)


def sample_seed(game: str, t: int, history_mode: str, prompt_format: str, k: int) -> int:
    digest = hashlib.sha256(f"{game}|{t}|{history_mode}|{prompt_format}|{k}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def states(directory, limit=None, types=TYPES):
    """(game name, t, objective, history, obs, a_star) for the walkthrough states of the given action types."""
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix in GAME_SUFFIXES)[:limit]
    for path in paths:
        for item in walk(path):
            if isinstance(item, dict):
                if item["problem"]:
                    raise RuntimeError(f"{path.name}: {item['problem']}")
                continue
            t, objective, history, obs, a_star = item
            if a_star.split()[0] in types:
                yield path.name, t, objective, history, obs, a_star


@torch.no_grad()
def decide(policy: LMPolicy, prompt_ids, reasoning, obs, a_star, generator) -> dict:
    """What LMPolicy.decide does after the reasoning (same context, same scores, same draw from `generator`),
    keeping the whole distribution."""
    body, _, end = policy._finish(reasoning)
    context = policy._action_context(prompt_ids, reasoning)
    options = [policy.encode(f" {j + 1}") + [policy.eos_id] for j in range(len(obs.commands))]
    d = Decision(command="", action=-1, prompt_ids=prompt_ids, reasoning_ids=reasoning, action_context_ids=context,
                 option_ids=options, behavior_logps=[], reasoning_end=end)
    _, scores, _ = policy.decision_forward(policy.model, d, forecast=False)
    log_q = torch.log_softmax(scores, dim=0)
    action = int(torch.multinomial(log_q.exp().to(generator.device), 1, generator=generator))
    q = log_q.exp().cpu().tolist()
    target = obs.commands.index(a_star)
    order = sorted(range(len(q)), key=lambda j: -q[j])
    text = policy.tokenizer.decode(reasoning)
    return {"reasoning": text, "reasoning_tokens": len(reasoning), "reasoning_end": end,
            "q_star": q[target], "logq_star": float(log_q[target]), "rank_star": order.index(target) + 1,
            "top": [{"command": obs.commands[j], "p": q[j]} for j in order[:TOP]],
            "q_reprocess": sum(p for c, p in zip(obs.commands, q) if is_reprocess(c)),
            "action": obs.commands[action], "action_is_star": action == target,
            "action_is_reprocess": is_reprocess(obs.commands[action]), **tags(text)}


def run(model, tokenizer, levels: dict, out, samples=5, limit=None, settings=tuple(SETTINGS), batch_states=8,
        cap=None, **overrides) -> dict:
    summary = {}
    todo = {level: list(states(directory, limit)) for level, directory in levels.items()}
    for name in settings:
        policy = LMPolicy(model, tokenizer, setting_config(name, cap, **overrides))
        cfg = policy.cfg
        for level, items in todo.items():
            start, n = time.time(), 0
            for b in range(0, len(items), batch_states):
                batch = items[b:b + batch_states]
                prompts = [policy.encode(build_prompt(objective, history, obs, cfg))
                           for _, _, objective, history, obs, _ in batch]
                seeds = [[sample_seed(game, t, cfg.history_mode, cfg.prompt_format, k) for k in range(samples)]
                         for game, t, *_ in batch]
                generators = [torch.Generator().manual_seed(s) for row in seeds for s in row]
                sampled = policy.sample_reasoning_batch([p for p in prompts for _ in range(samples)], generators)
                for i, (game, t, objective, history, obs, a_star) in enumerate(batch):
                    for k in range(samples):
                        reasoning, _ = sampled[i * samples + k]
                        row = decide(policy, prompts[i], reasoning, obs, a_star, generators[i * samples + k])
                        out.write(json.dumps({
                            "level": level, "game": game, "t": t, "type": a_star.split()[0], "a_star": a_star,
                            "K": len(obs.commands), "commands": obs.commands, "setting": name,
                            "history_mode": cfg.history_mode, "prompt_format": cfg.prompt_format,
                            "max_reasoning_tokens": cfg.max_reasoning_tokens, "k": k, "seed": seeds[i][k],
                            "prompt_tokens": len(prompts[i]), **row}) + "\n")
                        n += 1
                out.flush()
            summary[f"{name}|{level}"] = {"states": len(items), "rows": n, "seconds": round(time.time() - start, 1)}
            print(f"{name} {level}: {summary[f'{name}|{level}']}", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--games", nargs="+", default=["L0=data/games_L0/valid", "L1=data/games_L1/valid"],
                    help="LEVEL=directory pairs")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--limit", type=int, help="only the first N games of each directory")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--batch-states", type=int, default=8, help="states per sampling batch (x --samples rows)")
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    if out_path.exists():
        ap.error(f"{out_path} already exists; choose a new file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    levels = dict(pair.split("=", 1) for pair in args.games)
    with out_path.open("w") as out:
        run(model, tokenizer, levels, out, args.samples, args.limit, tuple(args.settings), args.batch_states)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
