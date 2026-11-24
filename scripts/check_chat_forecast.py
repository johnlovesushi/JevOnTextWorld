"""Check forecast_format = "chat" against a real tokenizer (T0005 b). Needs only the tokenizer, no GPU.

    HF_HUB_OFFLINE=1 python scripts/check_chat_forecast.py --model Qwen/Qwen3-1.7B

1. The suffix jevtw appends after the action equals, character for character, what the tokenizer's own chat template
   puts between the end of an assistant turn and the first answer token of the next one (enable_thinking=False).
2. chat_yes_text / chat_no_text are single tokens, and the special tokens of the suffix are parsed as special tokens.
Exit status 0 only if every check passes."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevtw.config import PolicyConfig  # noqa: E402
from jevtw.policy import chat_forecast_suffix  # noqa: E402


def main(argv=None) -> int:
    from transformers import AutoTokenizer
    defaults = PolicyConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--question", default=defaults.forecast_question)
    ap.add_argument("--yes", default=defaults.chat_yes_text)
    ap.add_argument("--no", default=defaults.chat_no_text)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    ok = True

    suffix = chat_forecast_suffix(args.question)
    first, answer = "FIRST-USER-TURN", "ASSISTANT-ANSWER"
    turns = [{"role": "user", "content": first}, {"role": "assistant", "content": answer},
             {"role": "user", "content": args.question}]
    text = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    after_answer = text[text.index(answer) + len(answer):]
    eos = tok.eos_token
    print(f"tokenizer            : {args.model}  ({type(tok).__name__}, eos_token = {eos!r})")
    print(f"jevtw suffix         : {suffix!r}")
    print(f"template after answer: {after_answer!r}")
    # jevtw's action ends with the EOS token, which is the template's end-of-turn marker
    same = after_answer == eos + suffix
    print(f"template == eos_token + suffix : {same}")
    ok &= same

    ids = tok(suffix, add_special_tokens=False)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    print(f"suffix tokens ({len(ids)}): {pieces}")
    for special in ("<|im_start|>", "<|im_end|>"):
        single = len(tok(special, add_special_tokens=False)["input_ids"]) == 1 and pieces.count(special) == \
            suffix.count(special)
        print(f"{special!r} parsed as one token everywhere : {single}")
        ok &= single
    same_ids = ids == tok(after_answer, add_special_tokens=False)["input_ids"][1:]
    print(f"token ids equal the template's : {same_ids}")
    ok &= same_ids

    for name, text_ in (("chat_yes_text", args.yes), ("chat_no_text", args.no)):
        one = tok(text_, add_special_tokens=False)["input_ids"]
        print(f"{name} = {text_!r}: ids {one}, tokens {tok.convert_ids_to_tokens(one)}, single token : {len(one) == 1}")
        ok &= len(one) == 1
    # the answer token in context: tokenising suffix + answer must give suffix ids + that one id
    for text_ in (args.yes, args.no):
        joined = tok(suffix + text_, add_special_tokens=False)["input_ids"]
        in_context = joined == ids + tok(text_, add_special_tokens=False)["input_ids"]
        print(f"tokenize(suffix + {text_!r}) == suffix ids + [{text_!r}] : {in_context}")
        ok &= in_context
    print("ALL CHECKS PASSED" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
