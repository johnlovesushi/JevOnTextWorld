"""E0008: calibration tables from the forecasts of a run (`jevtw.train --save-forecasts`).

    python scripts/forecast_calibration.py runs/E0008c [--out reports/E0008c_calibration.json] [--markdown]

Reads runs/<ID>/forecasts/batchNNNN.jsonl, runs/<ID>/config.json (is Platt scaling on?) and runs/<ID>/metrics.jsonl
(win rate and the training rows). Every decision is one sample with the label z of its episode, the same samples
as `forecast_brier` (jevtw/diagnostics.py). Per evaluation:
  log_loss        -mean[z log p + (1 - z) log(1 - p)]. Without Platt scaling p = sigmoid(s) and the terms are
                  computed from s with log-sigmoid, which stays finite. With Platt scaling the stored p is used,
                  clipped to [EPS, 1 - EPS]; `p_exact_01` counts the decisions whose stored p is exactly 0 or 1.
  log_loss_const  -[w log w + (1 - w) log(1 - w)], w = decision-weighted win rate (always forecasting w)
  saturated       share of decisions with p < 0.007 or p > 0.993
  by_step         t = 0 .. 5 and t >= 6: n, AUC of s, Brier score
  reliability     10 equal-width bins of p: count, mean p, actual win rate (printed for the first and last eval)
  platt_fit       with Platt scaling: a, b of logit(p) = a s + b, least squares over the decisions whose p is
                  not within 1e-4 of 0 or 1 (None if fewer than two different s are left). The training rows of
                  metrics.jsonl log a, b at the start of a batch; the fit reads them off the evaluation itself.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevtw.diagnostics import auc  # noqa: E402

EPS = 1e-7                 # float32 keeps 1 - p only down to 6e-8
SAT_LOW, SAT_HIGH = 0.007, 0.993
STEP_BINS = 6              # t = 0 .. 5, then one row for t >= 6
BINS = 10


def log_sigmoid(x: float) -> float:
    return -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))


def samples(records: list[dict]) -> list[tuple]:
    """(p, s, z, t) of every decision."""
    return [(st["p"], st["s"], r["z"], st["t"]) for r in records for st in r["steps"] if st["p"] is not None]


def log_loss(rows: list[tuple], platt: bool) -> float:
    total = 0.0
    for p, s, z, _ in rows:
        if platt:
            p = min(max(p, EPS), 1 - EPS)
            total -= math.log(p) if z else math.log(1 - p)
        else:
            total -= log_sigmoid(s) if z else log_sigmoid(-s)
    return total / len(rows)


def const_log_loss(w: float) -> float:
    return 0.0 if w in (0.0, 1.0) else -(w * math.log(w) + (1 - w) * math.log(1 - w))


def by_step(rows: list[tuple]) -> list[dict]:
    out = []
    for b in range(STEP_BINS + 1):
        part = [r for r in rows if (r[3] == b if b < STEP_BINS else r[3] >= STEP_BINS)]
        out.append({"t": str(b) if b < STEP_BINS else f">={STEP_BINS}", "n": len(part),
                    "wins": sum(z for _, _, z, _ in part),
                    "auc": auc([s for _, s, _, _ in part], [z for _, _, z, _ in part]) if part else None,
                    "brier": sum((p - z) ** 2 for p, _, z, _ in part) / len(part) if part else None})
    return out


def reliability(rows: list[tuple]) -> list[dict]:
    out = []
    for b in range(BINS):
        part = [(p, z) for p, _, z, _ in rows if min(int(p * BINS), BINS - 1) == b]
        out.append({"bin": f"{b / BINS:.1f}-{(b + 1) / BINS:.1f}", "n": len(part),
                    "mean_p": sum(p for p, _ in part) / len(part) if part else None,
                    "win_rate": sum(z for _, z in part) / len(part) if part else None})
    return out


def platt_fit(rows: list[tuple]):
    pts = [(s, math.log(p / (1 - p))) for p, s, _, _ in rows if 1e-4 < p < 1 - 1e-4]
    if len({s for s, _ in pts}) < 2:
        return None
    n = len(pts)
    ms, ml = sum(s for s, _ in pts) / n, sum(v for _, v in pts) / n
    var = sum((s - ms) ** 2 for s, _ in pts)
    a = sum((s - ms) * (v - ml) for s, v in pts) / var
    return {"a": a, "b": ml - a * ms, "n": n}


def evaluate(records: list[dict], platt: bool) -> dict:
    rows = samples(records)
    n = len(rows)
    w = sum(z for _, _, z, _ in rows) / n
    first = [(s, z) for _, s, z, t in rows if t == 0]
    return {"episodes": len(records), "wins": sum(r["z"] for r in records),
            "win_rate": sum(r["z"] for r in records) / len(records), "decisions": n, "w": w,
            "mean_p": sum(p for p, _, _, _ in rows) / n, "mean_s": sum(s for _, s, _, _ in rows) / n,
            "brier": sum((p - z) ** 2 for p, _, z, _ in rows) / n, "brier_const": w * (1 - w),
            "log_loss": log_loss(rows, platt), "log_loss_const": const_log_loss(w),
            "auc": auc([s for _, s, _, _ in rows], [z for _, _, z, _ in rows]),
            "auc_t0": auc([s for s, _ in first], [z for _, z in first]),
            "saturated": sum(1 for p, _, _, _ in rows if p < SAT_LOW or p > SAT_HIGH) / n,
            "p_exact_01": sum(1 for p, _, _, _ in rows if p in (0.0, 1.0)),
            "platt_fit": platt_fit(rows) if platt else None,
            "by_step": by_step(rows), "reliability": reliability(rows)}


def analyze(run_dir) -> dict:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    platt = bool(config.get("policy", {}).get("platt", False))
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    train = {m["batch"]: m for m in metrics if m.get("split") != "val"}
    val = {m["batch"]: m for m in metrics if m.get("split") == "val"}
    evals = []
    for path in sorted((run_dir / "forecasts").glob("batch*.jsonl")):
        batch = int(path.stem[len("batch"):])
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        row = {"batch": batch, **evaluate(records, platt)}
        last = (train.get(batch) or {}).get("passes") or []
        if last:                                   # the training batch that ended right before this evaluation
            row.update(kl=last[-1].get("kl"), kl_old=last[-1].get("kl_old"), clip_frac=last[-1].get("clip_frac"))
        if batch in val:                           # cross-check with what the trainer logged
            row["logged"] = {k: val[batch].get(k) for k in ("win_rate", "forecast_brier", "forecast_auc",
                                                            "forecast_auc_t0", "mean_forecast")}
        evals.append(row)
    passes = [m["passes"][-1] for m in train.values() if m.get("passes")]
    kls = [(m["batch"], m["passes"][-1].get("kl")) for m in train.values() if m.get("passes")]
    finite = all(math.isfinite(p.get(k)) for m in train.values() for p in m.get("passes", [])
                 for k in ("loss", "grad_norm") if p.get(k) is not None)
    training = {"batches": len(train), "passes_per_batch": max((len(m.get("passes", [])) for m in train.values()), default=0),
                "mean_mixed_groups": (sum(m["mixed_groups"] for m in train.values()) / len(train)) if train else None,
                "max_kl_last_pass": max((k for _, k in kls if k is not None), default=None),
                "max_kl_batch": max(kls, key=lambda x: -1 if x[1] is None else x[1])[0] if kls else None,
                "all_finite": finite, "last_pass_rows": len(passes),
                "platt_logged": {b: [m.get("platt_a"), m.get("platt_b")] for b, m in sorted(train.items())
                                 if "platt_a" in m}}
    return {"run": run_dir.name, "platt": platt, "cal_loss": config.get("grpo", {}).get("cal_loss"),
            "cal_grad": config.get("grpo", {}).get("cal_grad"), "alpha": config.get("grpo", {}).get("alpha"),
            "evals": evals, "training": training}


def _f(x, digits=3):
    return "–" if x is None else f"{x:.{digits}f}"


def markdown(result: dict) -> str:
    evals, platt = result["evals"], result["platt"]
    head = ["batch", "wins", "win rate", "mean p", "mean s", "Brier", "Brier const", "log loss", "log loss const",
            "AUC", "AUC t0", "saturated", "kl", "kl_old", "clip_frac"]
    if platt:
        head += ["Platt a", "Platt b", "p = 0 or 1"]
    lines = [f"**{result['run']}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in evals:
        cells = [str(e["batch"]), f"{e['wins']}/{e['episodes']}", _f(e["win_rate"], 4), _f(e["mean_p"]),
                 _f(e["mean_s"], 2), _f(e["brier"]), _f(e["brier_const"]), _f(e["log_loss"]), _f(e["log_loss_const"]),
                 _f(e["auc"]), _f(e["auc_t0"]), _f(e["saturated"]), _f(e.get("kl"), 4), _f(e.get("kl_old"), 4),
                 _f(e.get("clip_frac"))]
        if platt:
            fit = e["platt_fit"] or {}
            cells += [_f(fit.get("a")), _f(fit.get("b"), 2), str(e["p_exact_01"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "By step (n, AUC of s, Brier):", "", "| batch | " + " | ".join(f"t {r['t']}" for r in evals[0]["by_step"]) + " |",
              "|" + "---|" * (len(evals[0]["by_step"]) + 1)]
    for e in evals:
        lines.append(f"| {e['batch']} | " + " | ".join(f"{r['n']}, {_f(r['auc'], 2)}, {_f(r['brier'])}"
                                                       for r in e["by_step"]) + " |")
    for e in ([evals[0], evals[-1]] if len(evals) > 1 else evals):
        lines += ["", f"Reliability, batch {e['batch']}:", "", "| p bin | n | mean p | win rate |", "|---|---|---|---|"]
        lines += [f"| {r['bin']} | {r['n']} | {_f(r['mean_p'])} | {_f(r['win_rate'])} |" for r in e["reliability"]]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", help="write the result as JSON")
    ap.add_argument("--markdown", action="store_true", help="print the tables as markdown")
    args = ap.parse_args(argv)
    result = analyze(args.run_dir)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    print(markdown(result) if args.markdown else json.dumps(result["training"], indent=1))


if __name__ == "__main__":
    main()
