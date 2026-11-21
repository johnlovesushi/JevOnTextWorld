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
  rollout.py     Episode、run_episode、collect_group
  trainer.py     一次 GRPO 更新
  train.py       训练循环与评测（含 --smoke CPU 冒烟）
  testing.py     字节分词器 + 随机小模型（测试用，不需要下载）
configs/         grpo.json（λ=α=0）、grpo_calibrated.json（λ=0.3, α=0.1）
  experiments/   每个实验一份配置：<ID>.json
scripts/         make_games.py：用 tw-make 生成 cooking 游戏
                 summarize_run.py：把一次运行整理成固定格式的报告
reports/         实验报告 <ID>.md / <ID>.json
tests/           CPU 单元测试
CLAUDE.md        给服务器端 agent 的工作说明
```

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
python -m jevtw.train --config configs/grpo_calibrated.json --model Qwen/Qwen3-1.7B \
    --train-games data/games/train --val-games data/games/valid --out runs/grpo_cal
```

`runs/<name>/metrics.jsonl` 每个 batch 记录一行：胜率、平均回报、$b_i$、有胜有负的组数、
有非零优势的组数、平均 $p_t$、每一遍的损失、梯度范数、裁剪比例、KL、校准损失。
评测行另外记录 `forecast_brier`，也就是 $p_t$ 相对最终结果 $z$ 的 Brier 分数。

> 当前状态：CPU 测试和冒烟测试都通过了；还没有在 GPU 上用真实模型训练过。
