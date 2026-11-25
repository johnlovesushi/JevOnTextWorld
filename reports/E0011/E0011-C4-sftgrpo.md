# E0011-C4-sftgrpo: sftgrpo model on the C4 validation games, evaluation only

- commit: `33dd3b19596296fd878cca533e949eaa687c222a` on `exp/E0011-difficulty-ladder`
- host: kn029, device: NVIDIA L40S, torch 2.14.1, transformers 5.22.2
- started: 2026-11-25T10:43:06-0500; report generated: 2026-11-25T10:53:50-0500
- run dir: `runs/E0011-C4-sftgrpo`
- progress: 0 / 0 batches
- model: `runs/E0010e-s17/checkpoints/batch0010`, seed 17, eval seeds [17, 23]

## Configuration (non-default values)

```json
{
 "policy": {
  "max_reasoning_tokens": 0
 },
 "grpo": {
  "max_actions": 102
 }
}
```

<details><summary>Full configuration</summary>

```json
{
 "policy": {
  "max_reasoning_tokens": 0,
  "reasoning_prefix": "Reasoning:",
  "answer_prefix": "Answer:",
  "prompt_format": "plain",
  "history_window": 8,
  "history_mode": "full",
  "confidence_prompt": "\nQuestion: will you complete the objective? Answer yes or no:",
  "yes_text": " yes",
  "no_text": " no",
  "forecast": true,
  "forecast_format": "after_eos",
  "forecast_question": "Will you complete the objective? Answer yes or no.",
  "chat_yes_text": "Yes",
  "chat_no_text": "No",
  "platt": false,
  "option_batch_size": 16
 },
 "grpo": {
  "games_per_batch": 4,
  "group_size": 8,
  "max_actions": 102,
  "lam": 0.0,
  "alpha": 0.0,
  "beta": 0.0,
  "clip_eps": 0.2,
  "adv_delta": 1e-08,
  "normalization": "token",
  "optimization_passes": 2,
  "learning_rate": 2e-06,
  "weight_decay": 0.0,
  "grad_clip": 1.0,
  "parallel_episodes": 32,
  "tis_cap": null,
  "warmup_batches": 0,
  "cal_loss": "brier",
  "cal_grad": "full",
  "platt_lr": 0.01
 }
}
```

</details>

## Anomalies

- none detected

## Validation

| batch | win rate | forecast Brier | constant-forecast Brier w(1-w) | mean p_t | mean length | episodes | eval s |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 | 31.43 | 80 | 152.6 |

Forecast logit s = l(yes) - l(no) during validation (one sample per decision, label z of its episode; AUC t0 uses the first decision of every episode only; the Brier columns above are over decisions too, w being the decision-weighted win rate):

| batch | AUC | AUC t0 | mean s (won) | mean s (lost) | mean s |
|---|---|---|---|---|---|
| 0 | – | – | – | 11.95 | 11.95 |

Sampling during validation (seconds are totals for the evaluation; the reasoning columns are the mean token count and the fractions ending with the answer prefix / EOS / the token cap):

| batch | s / decision | gen s | score s | env s | prompt s | reasoning tokens | answer | eos | cap | mean options |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.06069 | 0.006 | 124.5 | 22.41 | 5.66 | 0 | – | – | – | – | 5.379 |

- peak GPU memory (evaluation): 4.48 GiB

## Log tail

```
job 8033815 (E0011-C4-sftgrpo) on kn029, started 2026-11-25T10:42:16-05:00
commit 33dd3b19596296fd878cca533e949eaa687c222a on exp/E0011-difficulty-ladder
name, memory.total [MiB], driver_version
NVIDIA L40S, 46068 MiB, 580.178.04
torch 2.14.1 cuda 13.2 available True

Loading weights:   0%|          | 0/310 [00:00<?, ?it/s]
Loading weights:  29%|██▊       | 89/310 [00:00<00:00, 803.56it/s]
Loading weights:  55%|█████▍    | 170/310 [00:00<00:00, 758.07it/s]
Loading weights:  79%|███████▉  | 246/310 [00:00<00:00, 697.20it/s]
Loading weights: 100%|██████████| 310/310 [00:00<00:00, 752.42it/s]
{"batch": 0, "split": "val", "eval_s": 152.61, "episodes": 80, "win_rate": 0.0, "mean_length": 31.425, "gpu_peak_gb": 4.48, "forecast_brier": 0.999957525788047, "forecast_brier_const": 0.0, "mean_forecast": 0.9999787617317721, "mean_forecast_logit": 11.94587710174949, "forecast_auc": null, "forecast_auc_t0": null, "forecast_s_mean_win": null, "forecast_s_mean_lose": 11.94587710174949, "decisions": 2514, "sec_per_decision": 0.060687049002332544, "gen_s": 0.006, "score_s": 124.493, "env_s": 22.407, "prompt_s": 5.66, "mean_reasoning_tokens": 0.0, "mean_options": 5.37867939538584, "reasoning_end_answer": null, "reasoning_end_eos": null, "reasoning_end_cap": null, "reasoning_end_cap_in_think": null}
finished 2026-11-25T10:45:40-05:00, wall time 197 s
```

## Executor notes

- Evaluation only (`--batches 0`), part of E0011 part B (issue #30): the E0010 SFT -> GRPO model `runs/E0010e-s17/checkpoints/batch0010`, trained on C1 or not trained, on `data/games_C4/valid`, H = 102, evaluation seeds 17 and 23 (80 episodes), no reasoning, `full` history. No training, no test set.
- Different from the issue: H was decided by the user (C2 38, C3 30, C4 102); the training seed of the E0010 model (17) is my choice, stated on the issue before the jobs.
- Problems: none; the job ran once and completed.
- Interpretation, failure types and decisions needed: see `reports/E0011.md`, part B.
