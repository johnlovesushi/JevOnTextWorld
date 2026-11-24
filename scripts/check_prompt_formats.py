"""Checks for PolicyConfig.prompt_format (T0006).

1. Template (tokenizer only, no GPU):

    HF_HUB_OFFLINE=1 python scripts/check_prompt_formats.py --model Qwen/Qwen3-1.7B

   compares the hand-written chat prompts of jevtw.policy.chat_prompt with tokenizer.apply_chat_template(...,
   enable_thinking=False / True), and checks how the markers and the forced suffixes are tokenised. Exit status 0
   only if every check passes.

2. Sampling (GPU, through Slurm: scripts/slurm/check_prompt_formats.sbatch):

    python scripts/check_prompt_formats.py --sample --games data/games_L1/valid

   plays the first games of the directory with every prompt format until `--decisions` decisions were sampled and
   prints, per format, how the reasonings ended, their mean length and a few of them."""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevtw.config import PROMPT_FORMATS, PolicyConfig  # noqa: E402
from jevtw.policy import THINK_CLOSE, LMPolicy, chat_prompt  # noqa: E402


def check_template(tok) -> bool:
    ok = True

    def report(label: str, passed: bool) -> None:
        nonlocal ok
        print(f"{label} : {passed}")
        ok &= passed

    content = "USER-CONTENT line 1\nline 2"
    turns = [{"role": "user", "content": content}]
    print(f"eos_token = {tok.eos_token!r} (id {tok.eos_token_id})")
    for think in (False, True):
        ours = chat_prompt(content, think=think)
        official = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        print(f"\nenable_thinking={think}")
        print(f"  jevtw    : {ours!r}")
        print(f"  template : {official!r}")
        if think:
            # The template stops after "<|im_start|>assistant\n"; the model then writes "<think>\n" itself. T0006
            # specifies that the prompt already contains it.
            report('  jevtw == template + "<think>\\n"', ours == official + "<think>\n")
        else:
            report("  jevtw == template", ours == official)
        ids = tok(ours, add_special_tokens=False)["input_ids"]
        pieces = tok.convert_ids_to_tokens(ids)
        for marker in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
            report(f"  {marker!r} is one token wherever it occurs ({ours.count(marker)}x)",
                   len(tok(marker, add_special_tokens=False)["input_ids"]) == 1
                   and pieces.count(marker) == ours.count(marker))
        report("  decode(encode(prompt)) == prompt", tok.decode(ids) == ours)
        if not think:
            report("  token ids equal the template's", ids == tok(official, add_special_tokens=False)["input_ids"])
    report("\n<|im_end|> is the EOS token", tok.eos_token == "<|im_end|>")
    cfg = PolicyConfig()
    for suffix in ("\n" + cfg.answer_prefix, f"\n{THINK_CLOSE}\n\n{cfg.answer_prefix}"):
        ids = tok(suffix, add_special_tokens=False)["input_ids"]
        print(f"forced suffix {suffix!r}: ids {ids}, tokens {tok.convert_ids_to_tokens(ids)}")
        report("  decodes back to itself", tok.decode(ids) == suffix)
    print("\nALL CHECKS PASSED" if ok else "\nCHECK FAILED")
    return ok


def sample(args, tok) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from jevtw.rollout import REASONING_ENDS, run_episode
    from jevtw.train import textworld_games

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype).to(device).eval()
    games = textworld_games(args.games)
    print(f"model {args.model} on {device}; games {args.games} ({len(games)}); max_reasoning_tokens "
          f"{args.max_reasoning_tokens}; {args.decisions} decisions per format, {args.steps} per game, "
          f"sampling seed {args.seed}\n")
    for fmt in args.formats:
        cfg = PolicyConfig(prompt_format=fmt, max_reasoning_tokens=args.max_reasoning_tokens, forecast=False)
        policy = LMPolicy(model, tok, cfg)
        decisions = []
        for name, make_env, env_seed in games:
            env = make_env()
            try:
                episode = run_episode(policy, env, name, env_seed, args.seed, args.steps)
            finally:
                env.close()
            decisions += [(name, t, d) for t, d in enumerate(episode.decisions)]
            if len(decisions) >= args.decisions:
                break
        decisions = decisions[:args.decisions]
        ends = Counter(d.reasoning_end for _, _, d in decisions)
        lengths = [len(d.reasoning_ids) for _, _, d in decisions]
        print(f"## prompt_format = {fmt}\n")
        print(f"- decisions: {len(decisions)}")
        print("- endings: " + ", ".join(f"{k} {ends.get(k, 0)}/{len(decisions)}" for k in REASONING_ENDS))
        print(f"- reasoning tokens: mean {sum(lengths) / len(lengths):.1f}, min {min(lengths)}, max {max(lengths)}")
        print(f"- prompt tokens: mean {sum(len(d.prompt_ids) for _, _, d in decisions) / len(decisions):.0f}\n")
        for name, t, d in decisions[:args.show]:
            text = tok.decode(d.reasoning_ids)
            cut = text[:args.chars] + (" […]" if len(text) > args.chars else "")
            print(f"**{name}, step {t}** — {len(d.reasoning_ids)} tokens, ended by `{d.reasoning_end}`, chose "
                  f"option {d.action + 1} of {len(d.option_ids)} (`{d.command}`)\n")
            print("```\n" + cut.replace("```", "'''") + "\n```\n")


def main(argv=None) -> int:
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--sample", action="store_true", help="sample decisions with the model instead of the template check")
    ap.add_argument("--games", default="data/games_L1/valid")
    ap.add_argument("--formats", nargs="+", default=list(PROMPT_FORMATS), choices=PROMPT_FORMATS)
    ap.add_argument("--decisions", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2, help="decisions per game")
    ap.add_argument("--max-reasoning-tokens", type=int, default=PolicyConfig().max_reasoning_tokens)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--show", type=int, default=3, help="reasonings printed per format")
    ap.add_argument("--chars", type=int, default=300)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if args.sample:
        sample(args, tok)
        return 0
    return 0 if check_template(tok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
