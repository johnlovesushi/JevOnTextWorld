"""E0012 diagnostic (run on one GPU from the repository root: python scripts/diag_verbal_layout.py): does verbal_forward give the same numbers with all K confidence branches (rollout, `decide`)
as with only the chosen one (training, `score_with_confidence`), and do both match unfolded full-sequence forwards?"""
import json, sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from jevtw.config import Config
from jevtw.policy import Decision, LMPolicy

cfg = Config.load("configs/experiments/E0012b.json")
dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
text = open("README.md").read()


def run(dtype, attn=None):
    kw = {"dtype": dtype}
    if attn:
        kw["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", **kw).to(dev).eval()
    pol = LMPolicy(model, tok, cfg.policy)
    out = {"attn": model.config._attn_implementation, "dtype": str(dtype)}
    for n_chars, k in ((3000, 6), (8000, 12)):
        prompt = pol.encode(text[:n_chars] + "\nAnswer:")
        options = [pol.encode(f" {j + 1}") + [pol.eos_id] for j in range(k)]
        d = Decision(command="", action=k // 2, prompt_ids=prompt, reasoning_ids=[], action_context_ids=prompt,
                     option_ids=options, behavior_logps=[])
        with torch.no_grad():
            _, sc_all, _, conf_all = pol.verbal_forward(model, d, False, range(k))
            _, sc_one, _, conf_one = pol.verbal_forward(model, d, False, [d.action])
            lq_all, lq_one = torch.log_softmax(sc_all, 0), torch.log_softmax(sc_one, 0)
            pc_all = torch.log_softmax(conf_all, -1)
            pc_one = torch.log_softmax(conf_one[0], -1)
            # unfolded: separate sequence x + y^j + u + y^m for every j, m
            ref_sc = torch.zeros(k, device=dev)
            ref_conf = torch.zeros(k, len(pol.bin_ids), device=dev)
            for j in range(k):
                for m, y in enumerate(pol.bin_ids):
                    seq = prompt + options[j] + pol.verbal_prompt_ids + y
                    h = model.base_model(input_ids=torch.tensor([seq], device=dev)).last_hidden_state[0]
                    lp = pol._logprobs(model, h[:-1]).gather(1, torch.tensor(seq[1:], device=dev)[:, None])[:, 0]
                    a0 = len(prompt) - 1
                    ref_sc[j] = lp[a0:a0 + len(options[j])].sum()
                    c0 = len(prompt) + len(options[j]) + len(pol.verbal_prompt_ids) - 1
                    ref_conf[j, m] = lp[c0:c0 + len(y)].sum()
            lq_ref, pc_ref = torch.log_softmax(ref_sc, 0), torch.log_softmax(ref_conf, -1)
        a = d.action
        out[f"n={len(prompt)},K={k}"] = {
            "logq_all_vs_one": float((lq_all - lq_one).abs().max()),
            "logq_all_vs_ref": float((lq_all - lq_ref).abs().max()),
            "logq_one_vs_ref": float((lq_one - lq_ref).abs().max()),
            "pc_all[a]_vs_one": float((pc_all[a] - pc_one).abs().max()),
            "pc_all_vs_ref(all j)": float((pc_all - pc_ref).abs().max()),
            "pc_one_vs_ref[a]": float((pc_one - pc_ref[a]).abs().max()),
            "pc_ref[a]": [round(float(v), 3) for v in pc_ref[a].exp()],
            "pc_all[a]": [round(float(v), 3) for v in pc_all[a].exp()],
            "pc_one": [round(float(v), 3) for v in pc_one.exp()],
        }
    del model
    torch.cuda.empty_cache()
    return out


for dtype, attn in ((torch.bfloat16, None), (torch.bfloat16, "eager"), (torch.float32, None)):
    print(json.dumps(run(dtype, attn)), flush=True)
