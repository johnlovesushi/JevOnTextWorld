# JevOnTextWorld

把 TextWorld 多轮游戏当作部分可观测马尔可夫决策过程（POMDP）来训练 LLM 智能体：
逐局（trajectory-level）GRPO，外加可选的“成功概率校准”目标。

- 数学定义：[`docs/textworld_pomdp_math_note.md`](docs/textworld_pomdp_math_note.md)（原始 PDF：[`docs/textworld_pomdp_math_note.pdf`](docs/textworld_pomdp_math_note.pdf)）
- 公式与代码的对应关系、实现选择、注意事项：[`docs/implementation.md`](docs/implementation.md)

## 方法一览

每一步：模型先写推理 $c_t$，再从 $K_t$ 个可执行命令里选动作 $a_t$，
然后读出“这局能赢吗”的概率 $p_t=\sigma(\ell(\text{yes})-\ell(\text{no}))$。
每局游戏从同一初始状态跑 $G$ 条完整轨迹，终局只有一个奖励 $z\in\{0,1\}$。

$$R_i = z_i + \lambda b_i,\quad b_i = 1-\tfrac{1}{T_i}\textstyle\sum_t (p^i_t-z^i)^2,\qquad
\mathcal L = \mathcal L_\text{GRPO} + \alpha\,\mathcal L_\text{cal} + \beta\,\mathcal L_\text{KL}$$

$\lambda=\alpha=0$ 就是普通多步 GRPO。

## 目录

```
jevtw/
  config.py      PolicyConfig / GRPOConfig（λ、α、β、归一化方式等）
  envs.py        Observation、KeyChestEnv（测试用小 POMDP）、TextWorldEnv
  policy.py      LMPolicy：采样 c_t、选 a_t、读 p_t；可求导的打分
  readout.py     (5) 式
  rewards.py     (6)–(7) 式
  losses.py      (8)–(9) 式与 KL；token / episode 两种归一化
  rollout.py     Episode、collect_batch（多条轨迹同步推进）、run_episode（逐条执行的参考实现）
  trainer.py     一次 GRPO 更新
  train.py       训练循环与评测（含 --smoke CPU 冒烟）
  sft.py         在专家状态上做监督微调（动作交叉熵；含 --smoke），输出的 checkpoint 可作为 RL 的起点
  testing.py     字节分词器 + 随机小模型（测试用，不需要下载）
configs/         grpo.json（λ=α=0）、grpo_calibrated.json（λ=0.3, α=0.1）、grpo_feedback.json（grpo.json + history_mode=feedback）
  experiments/   每个实验一份配置：<ID>.json
scripts/         make_games.py：用 tw-make 生成 cooking 游戏；make_levels.sh：按 docs/data.md 重新生成 C0–C4 / L0 / L1 并核对
                 data_fingerprint.py：写出 / 核对游戏目录的逐局指纹（docs/fingerprints/）
                 summarize_run.py：把一次运行整理成固定格式的报告
                 build_sft_data.py：沿通关路线生成 SFT 数据（CPU）
                 final_eval.py：在测试集上用固定种子 101/102/103 做最终评测；aggregate_final.py：合并多个训练种子的结果
                 prompt_tokens.py：按通关路线统计两种 history_mode 下提示词的 token 数（CPU）
                 check_prompt_formats.py：核对对话格式提示词和官方模板（CPU）；加 --sample 用模型采样若干决策（GPU）
reports/         实验报告 <ID>.md / <ID>.json
tests/           CPU 单元测试
CLAUDE.md        给服务器端 agent 的工作说明
```

## 数据

难度 C0–C4（以及早期的 L0 / L1）的定义、生成命令、train / valid / test 的用法、训练和评测种子、每个实验用了哪份数据：
见 [`docs/data.md`](docs/data.md)。

## 协作方式

规划端（claude.ai 会话）在 GitHub issue 里写任务，服务器端 agent 按 `CLAUDE.md` 执行：
每个任务开一个分支，跑完用 `scripts/summarize_run.py` 生成报告，提交到 `reports/`，再开 PR 给规划端审阅。

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest -q tests                       # CPU 测试
python -m jevtw.train --smoke --out runs/smoke  # CPU 冒烟：随机小模型 + KeyChest，3 个 batch

# 生成游戏并训练（GPU）
python scripts/make_games.py --out data/games --train 300 --valid 40 --test 80
# 简单难度：--skills 选择游戏需要的技能（open / cook / cut，默认全有，不带值表示都不要），--name 是文件名前缀
python scripts/make_games.py --out data/games_L0 --name L0 --recipe 1 --take 1 --go 1 --skills
python -m jevtw.train --config configs/grpo_calibrated.json --model Qwen/Qwen3-1.7B \
    --train-games data/games/train --val-games data/games/valid --out runs/grpo_cal
```

`runs/<name>/metrics.jsonl` 每个 batch 记录一行：胜率、平均回报、$b_i$、有胜有负的组数、
有非零优势的组数、平均 $p_t$、每一遍的损失、梯度范数、裁剪比例、KL、校准损失。
评测行另外记录 `forecast_brier`，也就是 $p_t$ 相对最终结果 $z$ 的 Brier 分数；以及 `forecast_brier_const` $= w(1-w)$
（$w$ 是这次评测按决策加权的胜率），即“永远预测 $w$”的 Brier 分数，作为参照。
评测行还有预测 logit $s$ 的诊断：`forecast_auc`、`forecast_auc_t0`、`forecast_s_mean_win`、`forecast_s_mean_lose`；
加 `--save-forecasts` 会把每次评测中每一步的 $s$、$p$ 写到 `runs/<name>/forecasts/batchNNNN.jsonl`。
加 `--save-reasonings` 会把每次评测中每个决策的推理文本写到 `runs/<name>/reasonings/batchNNNN.jsonl`；`scripts/sample_reasonings.py` 从中随机抽几段。
只评测一个 checkpoint：`python -m jevtw.train --model runs/<name>/checkpoints/batchNNNN --val-games … --batches 0 --save-forecasts --out runs/<新目录>`。
监督微调：`python scripts/build_sft_data.py --games data/games_<难度>/train --out data/sft/<名字>.jsonl`，然后 `python -m jevtw.sft --config … --train-data data/sft/<名字>.jsonl [更多文件…] --val-games … --out runs/<name>`（多个文件时合并训练，按难度分别记录损失和准确率；`build_sft_data.py` 默认跳过多余的 `open`，见 T0010）（每个 epoch 评测一次并保存 `checkpoints/epochNN`）；从它开始 RL：`python -m jevtw.train --model runs/<name>/checkpoints/epochNN --tokenizer Qwen/Qwen3-1.7B …`。
最终评测（只在 issue 要求时）：`python scripts/final_eval.py --config … --model … --games data/games_<难度>/test --out runs/<ID>/final_test`，多个训练种子用 `scripts/aggregate_final.py` 合并。
提示词格式（`prompt_format`：`plain` / `chat_nothink` / `chat_think`）、预测的读法（`forecast_format`）、交叉熵校准损失（`cal_loss`）和 Platt 缩放（`platt`）见 `docs/implementation.md`。

> 当前状态：CPU 测试和冒烟测试都通过了；还没有在 GPU 上用真实模型训练过。
