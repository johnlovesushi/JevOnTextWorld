"""Turn one run directory (config.json + metrics.jsonl from jevtw.train) into a fixed-format report.

  python scripts/summarize_run.py runs/E0001a --id E0001a --title "plain GRPO bring-up" \
      --log runs/slurm/E0001a.out --notes notes.md
  -> reports/E0001a.md (for people) and reports/E0001a.json (for comparing runs)

Every report has the same sections, so runs can be compared without reading logs. The script flags anomalies
automatically (non-finite loss, no learning signal, large sampling/training mismatch, heavy clipping, large updates
(kl_old), truncated reasoning,
incomplete run).
A run of `python -m jevtw.sft` is recognised by its training lines (they have `sft_loss` and no `passes`): the
training section is then one row per epoch (sft_loss, train_acc, ...), and the GRPO-only checks are skipped.
Standard library only."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 10                               # batches averaged at the start and end of training
TRAIN_KEYS = ("win_rate", "mean_return", "mean_bonus", "mixed_groups", "active_groups", "mean_length",
              "mean_forecast", "mean_forecast_logit", "cal_loss", "platt_a", "platt_b", "decisions", "reasoning_tokens", "rollout_s", "update_s",
              "gen_s", "score_s", "env_s", "prompt_s", "mean_reasoning_tokens", "reasoning_end_answer",
              "reasoning_end_eos", "reasoning_end_cap", "reasoning_end_cap_in_think", "mean_options",
              "behavior_old_mean_abs_diff", "lr", "mean_conf", "mean_conf_hat", "conf_entropy", "conf_brier",
              "mean_abs_conf_adv")
SAMPLING_KEYS = ("sec_per_decision", "gen_s", "score_s", "env_s", "prompt_s", "mean_reasoning_tokens",
                 "reasoning_end_answer", "reasoning_end_eos", "reasoning_end_cap", "reasoning_end_cap_in_think",
                 "mean_options")
PASS_KEYS = ("loss", "grad_norm", "clip_frac", "kl", "cal_loss", "kl_old", "tis_mean_w", "tis_capped_frac",
             "conf_loss", "conf_clip_frac", "conf_kl")
FORECAST_KEYS = ("forecast_auc", "forecast_auc_t0", "forecast_s_mean_win", "forecast_s_mean_lose",
                 "mean_forecast_logit")
CONF_KEYS = ("conf_brier", "conf_brier_const", "mean_conf_hat", "conf_auc", "conf_auc_t0", "conf_entropy",
             "conf_offset", "conf_logloss", "conf_ece", "conf_auc_within", "conf_auc_step_gain", "conf_saturation")
CALIB_P_KEYS = ("forecast_offset", "forecast_logloss", "forecast_ece", "forecast_auc_within", "forecast_auc_step_gain",
                "forecast_saturation")
LOGP_DIFF_WARN = 0.05
CLIP_WARN = 0.2
KL_OLD_WARN = 0.1                         # mean last-pass KL(pi_old || pi_theta) per likelihood term
CAP_WARN = 0.5                            # fraction of reasonings cut off at max_reasoning_tokens
SFT_KEYS = ("sft_loss", "train_acc", "decisions", "steps", "grad_norm", "max_grad_norm", "lr", "update_s")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut off by a killed job
                pass
    return rows


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None


def window_stats(rows: list[dict]) -> dict:
    out = {k: mean([r.get(k) for r in rows]) for k in TRAIN_KEYS}
    for k in PASS_KEYS:
        out[f"first_pass_{k}"] = mean([r["passes"][0].get(k) for r in rows if r.get("passes")])
        out[f"last_pass_{k}"] = mean([r["passes"][-1].get(k) for r in rows if r.get("passes")])
    decisions = sum(r.get("decisions") or 0 for r in rows)
    rollout = sum(r.get("rollout_s") or 0 for r in rows)
    update = sum(r.get("update_s") or 0 for r in rows)
    out["sec_per_decision"] = rollout / decisions if decisions else None                 # sampling only
    out["sec_per_decision_with_update"] = (rollout + update) / decisions if decisions else None
    out["batches"] = [rows[0]["batch"], rows[-1]["batch"]] if rows else None
    return out


def defaults() -> dict:
    import sys

    sys.path.insert(0, str(ROOT))
    from jevtw.config import Config

    return asdict(Config())


def non_default(config: dict) -> dict:
    try:
        base = defaults()
    except Exception:  # jevtw not importable: report everything
        return {s: config.get(s, {}) for s in ("policy", "grpo")}
    return {s: {k: v for k, v in config.get(s, {}).items() if base[s].get(k) != v} for s in ("policy", "grpo")}


def summarize(run_dir: Path, run_id: str, title: str) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    rows = read_jsonl(run_dir / "metrics.jsonl")
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    sft = any("sft_loss" in r for r in train)             # written by jevtw.sft: one training line per epoch
    planned = config.get("args", {}).get("epochs" if sft else "batches")
    anomalies = []

    for r in train:
        for i, p in enumerate(r.get("passes", [])):
            for k in ("loss", "grad_norm"):
                v = p.get(k)
                if v is None or not math.isfinite(v):
                    anomalies.append(f"batch {r['batch']} pass {i}: {k} = {v}")
    for r in train if sft else []:
        for k in ("sft_loss", "grad_norm"):
            v = r.get(k)
            if v is None or not math.isfinite(v):
                anomalies.append(f"epoch {r['batch']}: {k} = {v}")
    if planned and len(train) < planned:
        anomalies.append(f"incomplete: {len(train)} of {planned} {'epochs' if sft else 'batches'}")
    if sft:
        pass                              # no groups, no advantages
    elif train and all(r.get("active_groups", 0) == 0 for r in train):
        anomalies.append("no group ever had a non-zero advantage: no learning signal")
    elif train and config.get("grpo", {}).get("lam", 0) == 0 and all(r.get("mixed_groups", 0) == 0 for r in train):
        anomalies.append("no group ever mixed wins and losses (lam = 0): only KL / calibration gradients")
    diffs = [r.get("sampling_vs_old_max_abs_logp_diff") or 0.0 for r in train]
    if diffs and max(diffs) > LOGP_DIFF_WARN:
        anomalies.append(f"sampling vs training log-prob max diff {max(diffs):.3g} > {LOGP_DIFF_WARN} "
                         "(rollouts are off-policy by that much)")
    clip = mean([r["passes"][-1].get("clip_frac") for r in train if r.get("passes")])
    if clip is not None and clip > CLIP_WARN:
        anomalies.append(f"mean last-pass clip fraction {clip:.2f} > {CLIP_WARN}")
    kl_old = mean([r["passes"][-1].get("kl_old") for r in train if r.get("passes")])
    if kl_old is not None and kl_old > KL_OLD_WARN:
        anomalies.append(f"mean last-pass kl_old {kl_old:.3g} > {KL_OLD_WARN} (each batch moves the policy far from "
                         "the one that sampled it)")
    def capped(r):                        # truncated reasonings, inside or outside a thinking block
        if r.get("reasoning_end_cap") is None:
            return None
        return r["reasoning_end_cap"] + (r.get("reasoning_end_cap_in_think") or 0)

    cap = mean([capped(r) for r in train])
    if cap is not None and cap > CAP_WARN:
        anomalies.append(f"training: {cap:.2f} of the reasonings hit max_reasoning_tokens (> {CAP_WARN})")
    for r in val:
        if (capped(r) or 0) > CAP_WARN:
            anomalies.append(f"validation at batch {r['batch']}: {capped(r):.2f} of the reasonings hit "
                             f"max_reasoning_tokens (> {CAP_WARN})")
    if not val:
        anomalies.append("no validation evaluation recorded")

    peaks = [r.get("gpu_peak_gb") for r in train + val if r.get("gpu_peak_gb") is not None]
    return {
        "id": run_id, "title": title, "run_dir": str(run_dir), "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "meta": config.get("meta", {}), "args": config.get("args", {}),
        "config": {"policy": config.get("policy", {}), "grpo": config.get("grpo", {})},
        "non_default_config": non_default(config),
        "progress": {"batches_done": len(train), "batches_planned": planned,
                     "last_batch": train[-1]["batch"] if train else None},
        "val": [{k: r.get(k) for k in ("batch", "win_rate", "forecast_brier", "forecast_brier_const", "mean_forecast",
                                       "mean_length") + FORECAST_KEYS + CONF_KEYS + CALIB_P_KEYS + (
                                       "episodes", "eval_s") + SAMPLING_KEYS} for r in val],
        "cal_loss_kind": train[-1].get("cal_loss_kind") if train else None,
        "sft": [{"epoch": r["batch"], **{k: r.get(k) for k in SFT_KEYS}} for r in train] if sft else None,
        "sft_setup": config.get("sft"),
        "train_first": window_stats(train[:WINDOW]) if train and not sft else None,
        "train_last": window_stats(train[-WINDOW:]) if train and not sft else None,
        "gpu_peak_gb": max(peaks) if peaks else None,
        "max_sampling_vs_old_logp_diff": max(diffs) if diffs else None,
        "anomalies": anomalies,
    }


def fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def markdown(s: dict, log_tail: str | None, notes: str | None) -> str:
    m = s["meta"]
    lines = [f"# {s['id']}: {s['title']}", "",
             f"- commit: `{m.get('commit')}`{' (dirty)' if m.get('dirty') else ''} on `{m.get('branch')}`",
             f"- host: {m.get('host')}, device: {m.get('cuda_device') or 'CPU'}, "
             f"torch {m.get('torch')}, transformers {m.get('transformers')}",
             f"- started: {m.get('started')}; report generated: {s['generated']}",
             f"- run dir: `{s['run_dir']}`",
             f"- progress: {s['progress']['batches_done']} / {s['progress']['batches_planned']} "
             f"{'epochs (SFT)' if s.get('sft') else 'batches'}",
             f"- model: `{s['args'].get('model')}`, seed {s['args'].get('seed')}, "
             f"eval seeds {s['args'].get('eval_seeds')}",
             "", "## Configuration (non-default values)", "", "```json",
             json.dumps(s["non_default_config"], indent=1), "```", "",
             "<details><summary>Full configuration</summary>", "", "```json",
             json.dumps(s["config"], indent=1), "```", "", "</details>", "",
             "## Anomalies", ""]
    lines += [f"- {a}" for a in s["anomalies"]] or ["- none detected"]
    lines += ["", "## Validation", "",
              "| batch | win rate | forecast Brier | constant-forecast Brier w(1-w) | mean p_t | mean length | episodes | "
              "eval s |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {v['batch']} | {fmt(v['win_rate'])} | {fmt(v['forecast_brier'])} | {fmt(v.get('forecast_brier_const'))} | "
              f"{fmt(v['mean_forecast'])} | "
              f"{fmt(v['mean_length'])} | {fmt(v['episodes'])} | {fmt(v['eval_s'])} |" for v in s["val"]]
    if any(v.get(k) is not None for v in s["val"] for k in FORECAST_KEYS):
        lines += ["", "Forecast logit s = l(yes) - l(no) during validation (one sample per decision, label z of its "
                  "episode; AUC t0 uses the first decision of every episode only; the Brier columns above are over "
                  "decisions too, w being the decision-weighted win rate):", "",
                  "| batch | AUC | AUC t0 | mean s (won) | mean s (lost) | mean s |", "|---|---|---|---|---|---|"]
        lines += ["| " + " | ".join([str(v["batch"])] + [fmt(v.get(k)) for k in FORECAST_KEYS]) + " |"
                  for v in s["val"]]
    if any(v.get(k) is not None for v in s["val"] for k in CALIB_P_KEYS):
        lines += ["", "Calibration of p_t during validation (offset = |mean p - win rate|, log loss with p clipped to "
                  "[0.01, 0.99], ECE over 10 equal-width bins, AUC within games, AUC gain over the step index t, "
                  "fraction of p <= 0.005 or >= 0.995):", "",
                  "| batch | offset | log loss | ECE | AUC within | AUC gain over t | saturation |",
                  "|---|---|---|---|---|---|---|"]
        lines += ["| " + " | ".join([str(v["batch"])] + [fmt(v.get(k)) for k in CALIB_P_KEYS]) + " |"
                  for v in s["val"]]
    if any(v.get(k) is not None for v in s["val"] for k in CONF_KEYS):
        lines += ["", "Verbal confidence during validation (c-hat = sum_m pi^c(m) v_m per decision; Brier and AUC of "
                  "c-hat against z; entropy of pi^c in nats; the last six columns as for p_t):", "",
                  "| batch | Brier | const Brier | mean c-hat | AUC | AUC t0 | entropy | offset | log loss | ECE | "
                  "AUC within | AUC gain over t | saturation |", "|" + "---|" * 13]
        lines += ["| " + " | ".join([str(v["batch"])] + [fmt(v.get(k)) for k in CONF_KEYS]) + " |"
                  for v in s["val"]]
    if any(v.get("sec_per_decision") is not None for v in s["val"]):
        lines += ["", "Sampling during validation (seconds are totals for the evaluation; the reasoning columns are "
                  "the mean token count and the fractions ending with the answer prefix / EOS / the token cap):", "",
                  "| batch | s / decision | gen s | score s | env s | prompt s | reasoning tokens | answer | eos | "
                  "cap | mean options |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += ["| " + " | ".join([str(v["batch"])] + [fmt(v.get(k)) for k in SAMPLING_KEYS]) + " |"
                  for v in s["val"]]
    first, last = s["train_first"], s["train_last"]
    if s.get("sft"):
        setup = s.get("sft_setup") or {}
        lines += ["", "## Training (SFT, one row per epoch)", "",
                  f"- {fmt(setup.get('states'))} expert states from {fmt(setup.get('games'))} games; "
                  f"{fmt(setup.get('steps_per_epoch'))} optimizer steps per epoch, {fmt(setup.get('total_steps'))} in "
                  f"all, linear warmup over the first {fmt(setup.get('warmup_steps'))}",
                  f"- lr {fmt(s['args'].get('lr'))}, {fmt(s['args'].get('batch_decisions'))} states per step, "
                  f"weight decay {fmt(s['args'].get('weight_decay'))}; train data `{s['args'].get('train_data')}`",
                  "- sft_loss = mean -log q(a*) and train_acc = share of states whose most likely option is a*, "
                  "both over the states as they were visited during the epoch (the model changes meanwhile)", "",
                  "| epoch | " + " | ".join(SFT_KEYS) + " |", "|---|" + "---|" * len(SFT_KEYS)]
        lines += ["| " + " | ".join([str(r["epoch"])] + [fmt(r.get(k)) for k in SFT_KEYS]) + " |" for r in s["sft"]]
        lines += ["", f"- peak GPU memory: {fmt(s['gpu_peak_gb'])} GiB"]
    elif not first and s["gpu_peak_gb"] is not None:      # evaluation only: the peak of the evaluations
        lines += ["", f"- peak GPU memory (evaluation): {fmt(s['gpu_peak_gb'])} GiB"]
    if first:
        lines += ["", f"## Training (mean over the first and last {WINDOW} batches)", "",
                  f"| metric | first {first['batches']} | last {last['batches']} |", "|---|---|---|"]
        for k in first:
            if k != "batches":
                lines.append(f"| {k} | {fmt(first[k])} | {fmt(last[k])} |")
        lines += ["", f"- peak GPU memory: {fmt(s['gpu_peak_gb'])} GiB",
                  f"- max sampling vs training log-prob diff: {fmt(s['max_sampling_vs_old_logp_diff'])}",
                  f"- cal_loss is the {s.get('cal_loss_kind') or 'brier'} loss of the rollout forecasts (logged also "
                  "when alpha = 0); first/last_pass_cal_loss are the values inside the update (0 when alpha = 0)"]
    if log_tail is not None:
        lines += ["", "## Log tail", "", "```", log_tail, "```"]
    lines += ["", "## Executor notes", "",
              notes.strip() if notes else "_(executor: deviations from the issue, problems met, your reading of the "
                                          "results, open questions)_", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--id", required=True, help="experiment id, e.g. E0001a")
    ap.add_argument("--title", default="")
    ap.add_argument("--log", help="job log; its last --tail lines go into the report")
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--notes", help="markdown file with the executor's notes")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    summary = summarize(Path(args.run_dir), args.id, args.title)
    log_tail = None
    if args.log:
        log_tail = "\n".join(Path(args.log).read_text(errors="replace").splitlines()[-args.tail:])
    notes = Path(args.notes).read_text() if args.notes else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.id}.json").write_text(json.dumps(summary, indent=1))
    (out / f"{args.id}.md").write_text(markdown(summary, log_tail, notes))
    print(f"wrote {out / (args.id + '.md')} and {out / (args.id + '.json')}")
    for a in summary["anomalies"]:
        print(f"ANOMALY: {a}")


if __name__ == "__main__":
    main()
