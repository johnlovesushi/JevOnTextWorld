# 实现说明：笔记公式 → 代码

本文件说明 [`textworld_pomdp_math_note.md`](textworld_pomdp_math_note.md) 里每个公式在代码中的位置，
以及笔记没有规定、需要由实现来选择的细节。

## 1. 公式对照

| 笔记 | 含义 | 代码 |
|---|---|---|
| (1)–(2) | POMDP：隐藏状态 $s_t$、观测 $o_t$、历史 $h_t$、可执行命令 $\mathcal{A}_g(s_t)$ | `jevtw/envs.py`：`Observation`、`KeyChestEnv`（测试用的小 POMDP）、`TextWorldEnv` |
| (3) | 终局奖励 $z\in\{0,1\}$，只有赢才算 1 | `rollout.Episode.z`；输掉或走满 $H$ 步都是 0 |
| (4) 上 | 推理 token $c_t\sim\pi^c_\theta$ | `policy.LMPolicy._sample_reasoning`（温度 1、全词表采样，保证采样分布就是 $\pi_\theta$） |
| (4) 下 | 动作 $a_t\sim q_\theta=\mathrm{softmax}(\ell_\theta(a^j))$ | `policy.LMPolicy._option_scores`：$\ell(a^j)$ = `Answer:` 之后续写 ` j<EOS>` 的对数似然 |
| (5) | $p_t=\sigma(\ell(\text{yes})-\ell(\text{no}))$ | `readout.yes_no_probability`，在 `policy.LMPolicy._forecast` 中读取 |
| 第 2 节末 | 同一局、同一初始状态下的 $G$ 条轨迹 | `rollout.collect_group`：同一个 `env_seed`，每条轨迹的 `sample_seed` 不同 |
| (6) | $b_i$、$R_i=z_i+\lambda b_i$ | `rewards.brier_bonus`、`rewards.episode_return`；在 `trainer.assign_returns` 中调用 |
| (7) | 组内优势 $\widehat A_i$ | `rewards.group_advantages` |
| (8) | 裁剪的 GRPO 损失，除以 $N=\sum(\lvert c\rvert+1)$ | `losses.clipped_surrogate`、`losses.policy_weights(..., "token")` |
| (9) | Brier 损失 $\mathcal L_\text{cal}$，总损失 $\mathcal L$，按局平均 | `losses.calibration_weights`、`losses.decision_loss`、`trainer.Trainer.step` |
| 命题 1 | $\mathcal L_\text{cal}$ 的最优解是 $P(z=1\mid h,c,a)$ | 测试 `test_proposition_1_brier_minimizer_is_success_probability` |

把 $\lambda=\alpha=0$（`configs/grpo.json`）就回到普通多步 GRPO；
`configs/grpo_calibrated.json` 打开校准（$\lambda=0.3,\ \alpha=0.1$）。

## 2. 笔记没有规定、由实现选择的细节

**似然项。** 每一步有 $\lvert c_t\rvert+1$ 个似然项：每个采样出的推理 token（保留原始 token id，
包括采样出的 EOS 或 `Answer:` 的 token），加上一个分类动作。提示词、观测和被强制补上的 `Answer:`
都没有似然项。打分时直接复用存下的 token id，不会对生成的文本重新分词。

**$p_t$ 的读取。** 先在 动作上下文 + 所选选项 ` j<EOS>` 之后接上一句确认问题（默认
`Question: will you complete the objective? Answer yes or no:`），再读下一个 token 的 yes/no logit。
这句问题不写进历史，也不影响后续决策。` yes` 和 ` no` 必须各自是单个 token，否则构造策略时直接报错。

**$\lambda$ 项只用于奖励，$\alpha$ 项可以求梯度。** $b_i$ 用的是采样时（$\theta_\text{old}$）的 $p_t$，
所以 $R_i$ 和 $\widehat A_i$ 都与梯度无关。$\alpha\mathcal L_\text{cal}$ 则在训练时重新读取 $p_\theta$，
梯度会直接流到 logit 上。

**旧策略概率。** 优化前，每个决策都在 $\theta_\text{old}$ 下用和训练完全相同的前向重新打分一次，
所以第一遍的比率严格等于 1。采样时（带 KV cache）记录的概率只用于诊断，
记录在 `sampling_vs_old_max_abs_logp_diff` 里。

**归一化（`grpo.normalization`）。**

- `token`（默认，即笔记的 (8) 式）：每局内部除以 $N$。步数多、推理长的轨迹，权重更大。
- `episode`：先在每条轨迹内部平均，再在 $G$ 条轨迹之间平均。每条轨迹权重相同，这是 rlcd-bench `baselines` 分支的做法。

两种方式最后都在各局之间取平均。$\mathcal L_\text{KL}$ 使用与 $\mathcal L_\text{GRPO}$ 相同的权重，
采用 k3 估计 $r-\log r-1$，其中 $r=\pi_\text{ref}/\pi_\theta$。

**按决策累加梯度。** 所有权重在优化前就已固定，所以可以逐个决策地反向传播，显存有上界，
结果和一次性计算整个损失完全相同（测试 `test_accumulated_loss_equals_eq8_plus_eq9`）。

**标准差。** (7) 式中的 sd 使用总体标准差（ddof = 0）。如果组内所有回报都相同，优势全为 0。

**历史截断。** 提示词只显示最近 `history_window` 步；菜谱一旦读过，会作为 notes 一直保留。
这是对 $h_t$ 的近似：notes 只是历史的函数，不包含任何特权信息。

**TextWorld 观测。** 默认去掉 look / inventory / examine X（cookbook 除外）这些纯信息命令，
因为观测里已经显示了房间和背包。默认也删除 “Your score has just gone up” 这类提示，
让 $z$ 成为唯一的成功信号。两者都可以在 `TextWorldEnv` 中关闭。TextWorld 的游戏是确定性的，
所以 `env_seed` 不会改变 $s_0$；一组轨迹之间只有策略采样不同。

## 3. 需要注意的地方

1. **$\lambda>0$ 时，全输的组也会产生满量级的优势。** 笔记保证了“任何一局赢都比任何一局输好”，
   但 (7) 式会除以组内标准差。如果一组 8 局全输，$R_i=\lambda b_i$ 之间只有很小的差别，
   除以同样很小的 sd 之后，优势仍然是 $\pm1$ 的量级。这时梯度完全由校准信号决定，
   和胜负信号的权重一样大。CPU 冒烟训练里就能看到 `mixed_groups=0` 但 `active_groups=2`。
   如果不想要这种效果，可以调大 `adv_delta`，或者只在有胜有负的组里加 $\lambda$ 项（尚未实现）。
2. **$\lambda$ 项奖励的是“诚实的悲观”。** 在输掉的轨迹之间，它奖励那些早就预测自己会输的轨迹。
   这是恰当评分规则本来的要求，但它和 $\alpha$ 项的直接监督会不会互相冲突，需要做实验验证。
3. **$p_t$ 的目标会随策略移动**（命题 1）：$p^*$ 是 $\theta_\text{old}$ 下的胜率，策略变强以后，
   旧的 $p_t$ 会系统性地偏悲观。
4. **效率。** 打分时，每个决策要对 $K_t$ 个选项各做一次带完整上下文的前向（按 `option_batch_size` 分批；
   输出层和 FP32 log-softmax 只作用在选项 token 所在的位置上，否则全词表 logits 会占满显存），
   此外还有一次推理前向和一次 $p_t$ 前向；轨迹也是逐条串行采样的。在 GPU 上正式训练之前，
   值得改成共享前缀的 KV cache，并让多局并行采样。

## 4. 测试

`python -m pytest -q tests`，全部在 CPU 上运行。模型是随机初始化的小 Llama，配字节级分词器，
不需要下载任何东西。如果安装了 TextWorld，会额外生成一局真实的 cooking 游戏来测试。

| 文件 | 内容 |
|---|---|
| `tests/test_math.py` | (5) 式等于两类 softmax、平移不变性、Brier 对 logit 的梯度；(6) 式的手算例子、任何赢都大于任何输、$\lambda$ 的取值范围；(7) 式；(8)–(9) 式两种归一化的权重，以及逐决策累加的结果等于直接按公式计算；裁剪的 5 种情况；KL；命题 1；配置校验 |
| `tests/test_envs.py` | KeyChest 的通关路线和部分可观测性；TextWorld 适配器按通关路线能赢、过滤掉的命令不在通关路线上、分数提示已删除、菜谱进入 notes；小模型能在真实 TextWorld 游戏里完成采样 |
| `tests/test_policy_trainer.py` | 提示词的历史窗口；KV cache 采样的概率等于完整前向打分的概率；动作分布等于对选项的 softmax；无推理模式；同一组共享初始状态且可复现；第一遍的损失等于公式值（两种归一化）；参数确实被更新；$\lambda=\alpha=\beta=0$ 且全组结果相同时不更新；KL 的检查 |
