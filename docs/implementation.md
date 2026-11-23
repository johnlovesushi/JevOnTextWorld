# 实现说明：笔记公式 → 代码

本文件说明 [`textworld_pomdp_math_note.md`](textworld_pomdp_math_note.md) 里每个公式在代码中的位置，
以及笔记没有规定、需要由实现来选择的细节。

## 1. 公式对照

| 笔记 | 含义 | 代码 |
|---|---|---|
| (1)–(2) | POMDP：隐藏状态 $s_t$、观测 $o_t$、历史 $h_t$、可执行命令 $\mathcal{A}_g(s_t)$ | `jevtw/envs.py`：`Observation`、`KeyChestEnv`（测试用的小 POMDP）、`TextWorldEnv` |
| (3) | 终局奖励 $z\in\{0,1\}$，只有赢才算 1 | `rollout.Episode.z`；输掉或走满 $H$ 步都是 0 |
| (4) 上 | 推理 token $c_t\sim\pi^c_\theta$ | `policy.LMPolicy._sample_reasoning`（温度 1、全词表采样，保证采样分布就是 $\pi_\theta$） |
| (4) 下 | 动作 $a_t\sim q_\theta=\mathrm{softmax}(\ell_\theta(a^j))$ | `policy.LMPolicy.decision_forward`：$\ell(a^j)$ = `Answer:` 之后续写 ` j<EOS>` 的对数似然 |
| (5) | $p_t=\sigma(\ell(\text{yes})-\ell(\text{no}))$ | `readout.yes_no_probability`，在 `policy.LMPolicy.decision_forward` 中读取 |
| 第 2 节末 | 同一局、同一初始状态下的 $G$ 条轨迹 | `train.py` 里同一局的 $G$ 个任务共用 `env_seed`，`sample_seed` 各不相同，由 `rollout.collect_batch` 采样 |
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

**共享前缀的打分（`policy.LMPolicy.decision_forward`，T0001）。** 记动作上下文为 $x$（长度 $n$），选项为 $y^j$
（长度 $m_j$），确认问题为 $z$。因果注意力下，所有 $x\oplus y^j$ 的前 $n$ 个位置的隐藏状态都相同，
所以每个决策只做**一次** batch=1 的前向，同时得到推理 token 的概率、所有 $\ell(a^j)$ 和所有 $p_j$。
序列是 $x_{0:n-1}\oplus r_1\oplus\dots\oplus r_K$，其中 $r_j=(x_{n-1})\oplus y^j\oplus z$（不需要 $p$ 时不接 $z$）：

1. **4D 注意力掩码**：前缀内部是普通的因果掩码；$r_j$ 里的 token 只能看到前缀和 $r_j$ 中自己之前的位置，
   看不到其他选项的 token。
2. **`position_ids`**：前缀是 $0,\dots,n-2$，每个 $r_j$ 都从 $n-1$ 重新开始。
3. **读取**：$c_t$ 的 log-prob 在位置 $P-1,\dots,P+\lvert c\rvert-2$（$P$ 是提示词长度）上读，推理的三种结束方式都保证
   $P+\lvert c\rvert\le n$，即这些位置都在前缀里；
   $\ell_j=\sum_{i=1}^{m_j}\log\mathrm{softmax}(\text{lm\_head}(h_{j,i-1}))[y^j_i]$，$h_{j,0}$ 是 $r_j$ 第 0 个位置（$x_{n-1}$）的隐藏状态；
   $p_j$ 在 $r_j$ 的最后一个 token 上读。后面接 $z$ 不改变 $\ell_j$。

这样每个 token 的上下文和位置都与单独前向 $x\oplus y^j\oplus z$ 时完全相同。没有采用“前缀 KV cache 扩成 $K$ 行的后缀批”：
那样注意力仍要为每一行拼一份长度 $n$ 的 key/value，带梯度时显存是 $O(K\cdot n\cdot\text{层数})$
（$n=2200,K=16$ 时 22 GiB）。

输出层和 FP32 log-softmax 只作用在被读取的位置上（全词表 logits 铺满整个上下文会占满显存，见 E0001）。
数值和梯度都与“每个量各做一次完整上下文前向”的朴素实现一致，朴素实现保留在
`tests/test_policy_trainer.py` 里作为参考。`act()` 和 `score()` 各调用它一次。

**`p_options`。** $p_j$ 是“选了 $a^j$ 之后能赢的概率”，即命题 1 里的 $Q_\text{win}(h,c,a^j)$。
采样时所有 $K$ 个 $p_j$ 都存进 `Decision.p_options`，$p_t$ = `Decision.p` = $p_{a_t}$；
训练时 `score()` 只返回 $p_{a_t}$。`PolicyConfig.option_batch_size` 自 T0001 起不再使用，只为兼容旧配置而保留。

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
4. **效率。** 已完成：
   - T0001：每个决策的打分只对共享前缀前向一次，所有选项的后缀接在同一条序列里（见第 2 节），
     采样 $c_t$ 时的 prefill 也只在最后一个位置过输出层。带梯度时的显存相当于一条长度
     $n+\sum_j(1+m_j+\lvert z\rvert)$ 的序列（`scripts/bench_decision.py` 可以测）。
   - T0002：`rollout.collect_batch` 让最多 `grpo.parallel_episodes` 条轨迹同步推进，推理 $c_t$ 批量生成
     （`LMPolicy.sample_reasoning_batch`），训练和评测都用它（`scripts/bench_rollout.py` 可以测）。见下面一段。

   剩下的：选项打分（`decision_forward`）和参数更新仍是逐个决策进行的。

**批量采样（T0002）。** 每个同步步骤：为所有进行中的轨迹构造 prompt → 批量生成推理 → 每条轨迹调用一次
`decision_forward` 打分、采样动作、读 $p_t$ → 依次 `env.step`。有轨迹结束就由排队中的下一条补上。
采样分布仍然严格是 $\pi_\theta$：

- prompt **左侧补齐**，补齐位置被 `attention_mask` 屏蔽，`position_ids = cumsum(mask) - 1`，
  所以每个真实 token 的上下文和位置与不补齐时相同；batch 内各行之间本来就互不可见。
- 每行各自判断停止（EOS、以 `Answer:` 结尾、达到 `max_reasoning_tokens`）。已停止的行继续喂 pad，
  但 mask 为 0，输出被丢弃。
- 每条轨迹有自己的环境实例和自己的 `torch.Generator`（建在模型所在设备上，种子是 `sample_seed`），
  逐行 `torch.multinomial`，温度 1，全词表。所以一条轨迹采到什么，不取决于旁边跑的是哪些轨迹
  （批量计算带来的浮点误差除外：在 bf16 下，同一个种子在不同 `parallel` 下可能走出不同的轨迹，
  但每条都是 $\pi_\theta$ 的合法样本）。
- `run_episode` 是逐条执行的参考实现，它用 CPU 上的 generator。在 CPU 上，`parallel=1` 的
  `collect_batch` 与它逐 token 相同；在 GPU 上两者的随机数流不同（设备端 generator）。

**批量采样的显存。** 生成推理时的 KV cache 约为 并行数 × 序列长度 × 每 token 约 114 KB
（Qwen3-1.7B：28 层 × key 和 value 两份 × 8 个 KV 头 × head_dim 128 × bf16 的 2 字节 = 114,688 字节）。
序列长度是 batch 里最长的 prompt 加上已生成的 token 数。例如 32 条 × 2,400 token ≈ 8.2 GiB，
64 条就是它的两倍，所以带推理时 `parallel_episodes` 不能随意调大；无推理（`max_reasoning_tokens=0`）时
不生成、不建 cache，这一项为 0。

**采样指标。** 训练行和评测行都记录：`gen_s` / `score_s` / `env_s` / `prompt_s`（生成推理、打分、
环境、构造 prompt 各自的秒数；`env_s` 包含创建和重置环境）、`sec_per_decision`（采样墙钟时间 ÷ 决策数，
不含参数更新）、`mean_reasoning_tokens`、`reasoning_end_answer` / `reasoning_end_eos` / `reasoning_end_cap`
（推理以 `Answer:` 结束、以 EOS 结束、达到上限被截断的比例）和 `mean_options`（$\bar K$）。
截断比例超过 0.5 时 `scripts/summarize_run.py` 会报警。

## 4. 测试

`python -m pytest -q tests`，全部在 CPU 上运行。模型是随机初始化的小 Llama，配字节级分词器，
不需要下载任何东西。如果安装了 TextWorld，会额外生成一局真实的 cooking 游戏来测试。

| 文件 | 内容 |
|---|---|
| `tests/test_math.py` | (5) 式等于两类 softmax、平移不变性、Brier 对 logit 的梯度；(6) 式的手算例子、任何赢都大于任何输、$\lambda$ 的取值范围；(7) 式；(8)–(9) 式两种归一化的权重，以及逐决策累加的结果等于直接按公式计算；裁剪的 5 种情况；KL；命题 1；配置校验 |
| `tests/test_envs.py` | KeyChest 的通关路线和部分可观测性；TextWorld 适配器按通关路线能赢、过滤掉的命令不在通关路线上、分数提示已删除、菜谱进入 notes；小模型能在真实 TextWorld 游戏里完成采样 |
| `tests/test_batched_rollout.py` | 批量生成推理（prompt 长度各不相同、三种结束方式都出现）记录的概率等于 `score()`；同一行单独采样和放在 batch 里采样结果相同；`parallel=1` 的 `collect_batch` 与 `run_episode` 完全相同；同样的种子可复现；任务多于并行数时输出顺序和内容正确、同时打开的环境不超过并行数、出错时环境被关闭；无推理模式；TextWorld |
| `tests/test_policy_trainer.py` | 提示词的历史窗口；KV cache 采样的概率等于打分的概率；共享前缀打分与朴素参考实现的数值、梯度一致（推理的四种结束方式，一位数和两位数选项）；嵌入层和输出层处理的 token 数等于“一次前缀 + 各选项后缀”；动作分布等于对选项的 softmax，`p_options` 等于逐选项读出的 $p_j$；无推理模式；同一组共享初始状态且可复现；第一遍的损失等于公式值（两种归一化）；参数确实被更新；$\lambda=\alpha=\beta=0$ 且全组结果相同时不更新；KL 的检查 |
