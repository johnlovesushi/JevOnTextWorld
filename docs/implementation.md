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
| (5) | $s_t=\ell(\text{yes})-\ell(\text{no})$，$p_t=\sigma(s_t)$（开 Platt 时 $\sigma(a s_t+b)$，见补遗 A.4） | `policy.LMPolicy.decision_forward` 返回 $s$，`LMPolicy.forecast_prob` 把 $s$ 变成 $p$（yes / no 两个 logit 由 FP32 输出层 `LMPolicy._head` 算出）；`readout.yes_no_probability` 是同一公式的独立实现，测试里用作参考 |
| 第 2 节末 | 同一局、同一初始状态下的 $G$ 条轨迹 | `train.py` 里同一局的 $G$ 个任务共用 `env_seed`，`sample_seed` 各不相同，由 `rollout.collect_batch` 采样 |
| (6) | $b_i$、$R_i=z_i+\lambda b_i$ | `rewards.brier_bonus`、`rewards.episode_return`；在 `trainer.assign_returns` 中调用 |
| (7) | 组内优势 $\widehat A_i$ | `rewards.group_advantages` |
| (8) | 裁剪的 GRPO 损失，除以 $N=\sum(\lvert c\rvert+1)$ | `losses.clipped_surrogate`、`losses.policy_weights(..., "token")` |
| (9) | Brier 损失 $\mathcal L_\text{cal}$，总损失 $\mathcal L$，按局平均 | `losses.calibration_weights`、`losses.decision_loss`、`trainer.Trainer.step` |
| (A1) | 交叉熵校准损失（`cal_loss="bce"`） | `losses.bce`、`losses.calibration_loss` |
| (A4) | Platt 缩放 $p=\sigma(a s+b)$ | `LMPolicy.platt_a / platt_b`、`calibrated_logit`、`init_platt_bias`；`Trainer._init_platt` |
| 命题 1 | $\mathcal L_\text{cal}$ 的最优解是 $P(z=1\mid h,c,a)$（交叉熵下同样成立，补遗 A.3） | 测试 `test_proposition_1_brier_minimizer_is_success_probability` |

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

**预测 logit $s$ 和概率 $p$（T0005）。** 记 $s=\ell(\text{yes})-\ell(\text{no})$。`decision_forward` 返回
`(推理 log-prob, ℓ_j, s_j)`，`score()` 返回 `(log-prob, s_{a_t})`，都不再直接返回 $p$；$s\to p$ 只在
`LMPolicy.forecast_prob` 一处做（$p=\sigma(s)$，开 Platt 时 $\sigma(as+b)$）。`Decision` 同时存
`forecast_logit` / `forecast_logits_options`（$s$）和 `p` / `p_options`（经过 Platt 之后的 $p$）。
存 $s$ 是因为 $p$ 饱和以后（E0003d 里 $p\approx1$）在 FP32 下已经分不出大小，AUC 之类的诊断必须在 $s$ 上算。

**预测的三种读法（`PolicyConfig.forecast_format`，T0005）。** $\ell_j$ 的定义三种都一样（含 EOS 项），都只做一次前向。

| 取值 | 读 $s_j$ 的序列 | 读哪两个 token |
|---|---|---|
| `"after_eos"`（默认） | $x\oplus y^j\oplus z$，$y^j$ = ` j<EOS>`，$z$ = `confidence_prompt` | `yes_text` / `no_text` |
| `"no_eos"` | $x\oplus y^j_{:-1}\oplus z$：问题接在选项编号后面，中间没有 EOS | `yes_text` / `no_text` |
| `"chat"` | $x\oplus y^j\oplus u$，$u$ = `\n<|im_start|>user\n{forecast_question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` | `chat_yes_text` / `chat_no_text`（默认 `Yes` / `No`） |

- `after_eos` 和 `chat` 沿用上面的折叠方式，只是接在选项后面的那一段不同（`LMPolicy.forecast_tail_ids`）。
- `no_eos` 下 $\ell_j$ 需要 EOS 而预测序列不能有 EOS，两者不再互为前缀，所以每个选项有两段：
  打分段 $(x_{n-1})\oplus y^j$ 和预测段 $(x_{n-1})\oplus y^j_{:-1}\oplus z$，共 $2K$ 段。预测段只能看到前缀和它自己。
- `chat` 的 $u$ 由 `policy.chat_forecast_suffix` 生成。Qwen3 的 EOS 就是 `<|im_end|>`，所以选项结尾的 EOS 正好是
  上一轮 assistant 的结束符。`scripts/check_chat_forecast.py` 用真实分词器核对：$u$ 与
  `apply_chat_template(..., enable_thinking=False)` 的输出逐字符相同，`Yes` / `No` 各是一个 token。
- 采样、old、ref、训练用的都是 `LMPolicy._forecast_sequence` 定义的同一个序列；`Decision.confidence_ids` 存的就是它。
- 三种格式之间 $\ell_j$ 的算术完全相同，但各行 padding 的长度不同，BLAS 的舍入可能不同：CPU 小模型上实测相差
  0 或 1 个 FP32 ulp（$|\ell_j|\approx17$ 时 $1.9\times10^{-6}$），不是逐位相同。默认配置（`after_eos`）走的是和改动前
  同一条代码路径，与改动前的结果逐位相同（`tests/snapshots/t0005_default.json`）。

**校准损失（`GRPOConfig.cal_loss`，T0005）。** `"brier"`（默认）是 (9) 式；`"bce"` 是
`binary_cross_entropy_with_logits(a·s+b, z)`（补遗 (A1)），直接在 logit 上算，不经过 $p$。两者的权重相同
（`calibration_weights`）。$\lambda$ 项的 $b_i$ 不随 `cal_loss` 改变，始终是 Platt 之后的 $p$ 的 Brier。
训练行的 `cal_loss` 是当前所用的损失在**采样时**的预测上的值（权重同上；$\alpha=0$ 时也记录，但不进损失），
`cal_loss_kind` 写明是哪一种；`passes[k].cal_loss` 是第 $k$ 遍训练前向上的值（$\alpha=0$ 时校准项不计算，记为 0）。

**Platt 缩放（`PolicyConfig.platt`，T0005）。** $p=\sigma(as+b)$，$a$、$b$ 是 `LMPolicy` 持有的两个 FP32 标量
`nn.Parameter`，初值 $a=1,b=0$，不属于语言模型。

- **数据初始化**（`Trainer._init_platt`，只在第一次更新前做一次；从带 `platt.json` 的 checkpoint 加载时不做）：
  $b=\operatorname{logit}(\operatorname{clip}(\bar z,0.05,0.95))-\bar s$，$\bar z$ 是第一批按决策加权的胜率，
  $\bar s$ 是第一批所有决策的 $s_{a_t}$ 的平均；$a$ 保持 1。初始化之后，第一批已经存下的 `Decision.p / p_options`
  会用新的 $b$ 从存下的 $s$ 重算，所以第一批的 $b_i$（$\lambda$ 项）用的是初始化之后的 $p$。
- **优化器**：$a,b$ 单独一个参数组，学习率 `GRPOConfig.platt_lr`（默认 $10^{-2}$），没有权重衰减。
  学习率预热只作用于语言模型那一组；梯度裁剪和 `grad_norm` 也只算语言模型的参数。
- **`GRPOConfig.cal_grad`**：`"full"`（默认）校准项的梯度同时流向语言模型和 $a,b$；`"platt_only"` 把 $s$ detach，
  校准项只训练 $a,b$，语言模型只收到 GRPO 和 KL 的梯度（要求 `platt=True`，否则构造 `Trainer` 时报错）。
- 训练行记录 `platt_a`、`platt_b`（更新前的值）。

**预测诊断（`jevtw/diagnostics.py`，T0005）。** 评测行新增：

| 字段 | 含义 |
|---|---|
| `forecast_auc` | 以每个决策为样本、$s_{a_t}$ 为分数、所在局的 $z$ 为标签的 AUC（Mann–Whitney，同分按 1/2 计）；全赢或全输时为 `null` |
| `forecast_auc_t0` | 只用每局第一步的 AUC |
| `forecast_s_mean_win` / `forecast_s_mean_lose` | 赢 / 输的局里所有决策的平均 $s$ |
| `mean_forecast_logit` | 所有决策的平均 $s$（训练行也有） |
| `forecast_brier_const` | $w(1-w)$，$w$ 现在是**按决策加权**的胜率（赢的局里的决策数 / 总决策数），和 `forecast_brier` 的样本一致；T0005 之前是按局的胜率 |

`jevtw.train --save-forecasts` 在每次评测时写 `runs/<ID>/forecasts/batchNNNN.jsonl`，每局一行：
`game, env_seed, sample_seed, z, length, steps`，每步 `{t, command, s, p, s_options}`。
`--save-reasonings`（E0004）在每次评测时写 `runs/<ID>/reasonings/batchNNNN.jsonl`，每个决策一行：
`game, env_seed, sample_seed, z, length, t, command, reasoning_end, reasoning_tokens, reasoning`（推理的文本）。
评测行里的 `gpu_peak_gb` 是这次评测期间的峰值显存（CPU 上为 `null`）。

**Checkpoint 和 `--tokenizer`（T0005）。** `train.save_checkpoint` 除了模型，还保存分词器和（开 Platt 时）
`platt.json`（`{"a":…, "b":…}`）。`--tokenizer` 指定分词器，默认与 `--model` 相同。`--model` 指向的目录里有
`platt.json` 时会加载它，并且即使配置里 `platt=false` 也会打开 Platt（打印一行提示），这样评测 checkpoint 时
$p$ 和训练时一致。`--batches 0` 只做一次评测，这时不需要 `--train-games`。`--smoke` 现在也保存 checkpoint，
并且 `--model` 是目录时从该目录加载小模型（端到端测试用）。

**$\lambda$ 项只用于奖励，$\alpha$ 项可以求梯度。** $b_i$ 用的是采样时（$\theta_\text{old}$）的 $p_t$，
所以 $R_i$ 和 $\widehat A_i$ 都与梯度无关。$\alpha\mathcal L_\text{cal}$ 则在训练时重新读取 $p_\theta$，
梯度会直接流到 logit 上。

**FP32 输出层（T0003）。** 主干可以用 bf16，但输出层一律按
`F.linear(h.float(), W.float())` 计算再做 log-softmax（`LMPolicy._head` / `_logprobs`）。
原来是先用 bf16 的 lm_head 算出 logits 再转 FP32，logits 已经被舍入到 bf16
（值在 16 到 32 之间时相邻两个可表示值差 0.125），这个误差比损失里用到的 log-prob 差还大。
采样（逐条和批量）和训练都走 `_logprobs` 这一个函数，梯度照常经过类型转换流回 bf16 的权重。
读 $p$ 时只需要 yes / no 两个 logit，所以只取 $W$ 的这两行转 FP32（数学上等于整层 FP32 之后取两列）。
`decision_forward` 里推理 token 和选项 token 合成一次输出层调用，$W$ 的 FP32 副本每个决策只建一次。

**旧策略概率和序列一致（T0003）。** 优化前，每个决策都在 $\theta_\text{old}$ 下用和训练完全相同的前向重新打分一次，
所以第一遍的比率严格等于 1。计算 old、ref 和训练这三处调用 `score()` 时都传 `forecast=policy.cfg.forecast`，
也就是和采样时 `decide()` 完全相同的序列（采样时读了 $p$，每个选项后面就都接着确认问题）；
只有 $\alpha>0$ 时才把 $p$ 放进损失。不带推理时，采样时的 $\log q(a)$ 和重算的 old 因此是同一个函数在同一个输入上的结果
（FP32 CPU 上逐位相同）。带推理时，推理 token 的采样概率来自 KV cache 的增量前向，和整段前向仍有浮点误差。
采样时记录的概率（`Decision.behavior_logps`）用于诊断：`sampling_vs_old_max_abs_logp_diff`（最大值）和
`behavior_old_mean_abs_diff`（全部似然项上 $\lvert\text{old}-\text{behavior}\rvert$ 的平均），以及下面的 TIS。

**截断重要性采样（`grpo.tis_cap`，T0003，默认关闭）。** 开启时，每个似然项 $u$ 的裁剪 surrogate 乘上

$$w_u=\min\big(\exp(\text{old}_u-\text{behavior}_u),\ C\big),\qquad C=\texttt{tis\_cap}$$

（`losses.tis_weights`）。$w_u$ 是常数，不参与求导；只乘在 policy 项上，不乘 KL 项和校准项。
它修正的是“样本实际按 behavior 的概率采出，而损失把它们当作按 old 采出”这一偏差。
每一遍记录 `tis_mean_w`（$w_u$ 的平均）和 `tis_capped_frac`（未截断的比值超过 $C$ 的比例）。

**学习率 warmup（`grpo.warmup_batches`，T0003，默认 0）。** 第 $b$ 个 batch（从 1 开始）的学习率是
$\text{lr}\cdot\min(1,b/\texttt{warmup\_batches})$，同一个 batch 的各遍相同；`Trainer.batches_done` 记录已完成的 batch 数，
实际用的值记录在 `lr` 里。需要它是因为 Adam 第一步的更新量约等于 $\text{lr}\cdot\mathrm{sign}(g)$，与梯度大小无关。
`warmup_batches=0` 时不碰优化器的学习率。

**更新幅度 `kl_old`（T0003）。** 每一遍记录 $\frac1{N}\sum_u(r_u-\log r_u-1)$，$r_u=\pi_\theta(u)/\pi_{\theta_\text{old}}(u)$，
在本 batch 所有参与更新的似然项上取平均。样本（近似）来自 $\pi_{\theta_\text{old}}$，所以它估计的是
$\mathrm{KL}(\pi_{\theta_\text{old}}\Vert\pi_\theta)$；第一遍严格为 0。最后一遍的平均值超过 0.1 时 `scripts/summarize_run.py` 会报警。

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
`history_mode` 决定过去每一步在命令之后显示什么：`"full"`（默认，旧实验的行为）显示这一步之后的整份观测，
房间描述、背包和 notes 在每一步都重复；`"feedback"` 只显示游戏对这条命令的回应（`Observation.feedback`：
不含房间描述、背包和 notes；移动之后只保留新房间的标题行，如 `-= Kitchen =-`）。当前观测总是完整给出一次。
两种模式下 $h_t$ 包含的信息在窗口内基本相同（`"feedback"` 少了窗口内更早房间的描述），所以这也是对 $h_t$ 的截断近似。
参考配置 `configs/grpo_feedback.json` 和 `grpo.json` 只差这一项。

**提示词格式和推理的结束方式（`PolicyConfig.prompt_format`，T0006）。** 默认 `"plain"` 是续写式：提示词以
`Think step by step, then write "Answer:" …` 和一行 `Reasoning:` 结尾。两种对话格式把同样的内容放进 Qwen3
对话模板的 user 消息（`policy.chat_prompt`，手写字符串），最后的说明换成
`First analyse the situation in at most three sentences, then write "Answer:" followed by the number of one option.`，
没有 `Reasoning:` 一行：

| `prompt_format` | assistant 一侧的开头（属于提示词，没有似然项） | $c_t$ |
|---|---|---|
| `"chat_nothink"` | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | assistant 写出的文字 |
| `"chat_think"` | `<\|im_start\|>assistant\n<think>\n` | 思考块的内容、`</think>` 和它后面的文字 |

`chat_nothink` 的字符串和 `tokenizer.apply_chat_template(..., enable_thinking=False)` 逐字相同。
`enable_thinking=True` 时官方模板停在 `<|im_start|>assistant\n`，由模型自己写出 `<think>\n`；这里按 T0006 的规定把
`<think>\n` 放进提示词，所以 `chat_think` = 官方模板 + `<think>\n`（`scripts/check_prompt_formats.py` 核对这两点）。
`max_reasoning_tokens = 0` 时 user 消息以 `Reply with the number of one option.` 结尾，和 `plain` 相同。

采样 $c_t$ 在三种情况下停止：采样到 EOS（Qwen3 的 EOS 是 `<|im_end|>`）、$c_t$ 以 `Answer:` 结尾、达到
`max_reasoning_tokens`。`chat_think` 下 `Answer:` 必须出现在 `</think>` 之后才算数，思考块里面写的 `Answer:` 不会停止。
动作上下文 $x$ = 提示词 $\oplus$ body $\oplus$ 后缀，总是以 `Answer:` 结尾；这些都由 `LMPolicy._finish` 一个函数决定：

| 结束方式（`Decision.reasoning_end`） | body | 强制补上的后缀（没有似然项） |
|---|---|---|
| `"answer"`：以 `Answer:` 结尾 | $c_t$ | 无 |
| `"eos"`：采样到 EOS | $c_t$ 去掉 EOS | 思考块未闭合时 `\n</think>\n\nAnswer:`，否则 `\nAnswer:` |
| `"cap"`：达到上限，思考块已闭合（或格式没有思考块） | $c_t$ | `\nAnswer:` |
| `"cap_in_think"`：达到上限时思考块未闭合（只有 `chat_think`） | $c_t$ | `\n</think>\n\nAnswer:` |
| 没有推理（`max_reasoning_tokens = 0`） | 空 | 同 `"eos"` 一行的规则 |

“思考块未闭合”指 `chat_think` 下 body 解码后的文本里没有 `</think>`。记 $P$ 为提示词长度、$n=\lvert x\rvert$：`"answer"` 时
$n=P+\lvert c\rvert$，其余情况后缀非空，所以 $P+\lvert c\rvert\le n$ 总成立，并且 $x_{0:P+\lvert c\rvert-1}$ 等于提示词加上 $c$ 去掉最后一个
token。这正是 `decision_forward` 在前缀里给推理 token 打分所需要的不变式（T0001），所以打分代码不用改。
对话格式配上不是 `"chat"` 的 `forecast_format` 时，`Config` 会给出警告（确认问题会以纯文本接在 assistant 的回答后面）。

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
- 每行各自判断停止（EOS、`LMPolicy._answer_reached`、达到 `max_reasoning_tokens`）。已停止的行继续喂 pad，
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
不含参数更新）、`mean_reasoning_tokens`、`reasoning_end_answer` / `reasoning_end_eos` / `reasoning_end_cap` /
`reasoning_end_cap_in_think`（推理以 `Answer:` 结束、以 EOS 结束、达到上限被截断、在思考块里被截断的比例，
四项互不重叠，和为 1）和 `mean_options`（$\bar K$）。
截断比例（后两项之和）超过 0.5 时 `scripts/summarize_run.py` 会报警。

## 4. 测试

`python -m pytest -q tests`，全部在 CPU 上运行。模型是随机初始化的小 Llama，配字节级分词器，
不需要下载任何东西。如果安装了 TextWorld，会额外生成一局真实的 cooking 游戏来测试。

| 文件 | 内容 |
|---|---|
| `tests/test_math.py` | (5) 式等于两类 softmax、平移不变性、Brier 对 logit 的梯度；(6) 式的手算例子、任何赢都大于任何输、$\lambda$ 的取值范围；(7) 式；(8)–(9) 式两种归一化的权重，以及逐决策累加的结果等于直接按公式计算；裁剪的 5 种情况；KL；命题 1；配置校验 |
| `tests/test_envs.py` | KeyChest 的通关路线和部分可观测性；TextWorld 适配器按通关路线能赢、过滤掉的命令不在通关路线上、分数提示已删除、菜谱进入 notes；小模型能在真实 TextWorld 游戏里完成采样 |
| `tests/test_batched_rollout.py` | 批量生成推理（prompt 长度各不相同、三种结束方式都出现）记录的概率等于 `score()`；同一行单独采样和放在 batch 里采样结果相同；`parallel=1` 的 `collect_batch` 与 `run_episode` 完全相同；同样的种子可复现；任务多于并行数时输出顺序和内容正确、同时打开的环境不超过并行数、出错时环境被关闭；无推理模式；TextWorld |
| `tests/test_t0003.py` | bf16 小模型上 `_logprobs` 逐位等于 FP32 输出层的结果，梯度能流回 bf16 权重；不带推理时重算的 old 逐位等于采样时的概率，old / ref / 训练三处的 `forecast` 都等于 `policy.cfg.forecast`；$\alpha=0$ 时 $p$ 不进损失；TIS 关闭时损失不变、开启时等于手算值、权重不带梯度且不影响 KL 和校准项；warmup 每个 batch 的学习率、同一 batch 各遍相同、不开时不碰优化器；`kl_old` 第一遍为 0、更新后大于 0 |
| `tests/test_t0004.py` | `history_mode="full"` 的提示词和改动前逐字相同（KeyChest 固定轨迹的快照、TextWorld 上按旧格式重建）；`"feedback"` 模式下房间描述、背包、notes 各出现一次；移动的反馈只保留房间标题；两种模式下 `collect_batch(parallel=1)` 等于 `run_episode`；不传 `--wandb-project` 时不 import wandb，传了之后 `init` / `log` / `finish` 的参数（step、`val/` 前缀、`pass1/`、`pass_last/`），online 初始化失败时改用 offline 并继续训练 |
| `tests/test_t0006.py` | `plain` 的提示词和改动前逐字相同；两种对话格式的提示词快照；每种格式、每种结束方式（`Answer:`、EOS、思考块内外截断、没有推理，共 17 种情形）下的结束类型、后缀和不变式，`decision_forward` 等于朴素参考（1e-4），梯度一致；采样（逐条和 batch）时记录的概率等于打分时的概率，四种结束方式都由采样走到；`collect_batch(parallel=1)` 等于 `run_episode`；`reasoning_end_cap_in_think` 指标；`Config` 的警告 |
| `tests/test_t0005.py` | 默认配置下 $\ell_j$、$p$、两遍的损失和更新后的参数与改动前的快照逐位相同；三种 `forecast_format` 的序列构造、$\ell_j$ 一致（1e-5）、$s$ 和梯度等于“每个选项一条完整序列”的朴素参考（1e-4）；BCE 的数值和对 logit 的梯度 $p-z$，Brier 的梯度在 $p\to1$ 时消失；`platt_only` 下语言模型的更新与没有校准项时逐位相同；Platt 的数据初始化公式和截断、参数组学习率、预热不作用于 Platt 组、`platt.json` 存取；AUC 等于两两比较的暴力计数；`forecast_brier_const` 按决策加权；forecasts 记录的字段；端到端：`--smoke` 训练并保存 checkpoint，再用 `--model <目录> --batches 0 --save-forecasts` 评测 |
| `tests/test_policy_trainer.py` | 提示词的历史窗口；KV cache 采样的概率等于打分的概率；共享前缀打分与朴素参考实现的数值、梯度一致（推理的四种结束方式，一位数和两位数选项）；嵌入层和输出层处理的 token 数等于“一次前缀 + 各选项后缀”；动作分布等于对选项的 softmax，`p_options` 等于逐选项读出的 $p_j$；无推理模式；同一组共享初始状态且可复现；第一遍的损失等于公式值（两种归一化）；参数确实被更新；$\lambda=\alpha=\beta=0$ 且全组结果相同时不更新；KL 的检查 |
