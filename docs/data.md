# 数据设定：游戏难度（C0–C4）、划分和种子

这份文档说明本仓库实验用到的全部数据：每个难度的游戏怎么生成、train / valid / test 怎么划分、每个实验用了哪一份、
训练和评测各用什么种子。照着做就能拿到和我们相同的游戏，并按相同方式使用。

`data/` 不进 git（约 140–180 MB / 难度）。仓库里有的是：

| 文件 | 内容 |
|---|---|
| `scripts/make_games.py` | 用 `tw-make tw-cooking` 生成一个难度的 train / valid / test |
| `scripts/make_levels.sh` | 按下面表格里的命令重新生成 C0–C4、L0、L1，并逐局核对指纹 |
| `scripts/data_fingerprint.py` | 写出 / 核对一个游戏目录的指纹 |
| `docs/fingerprints/games_<难度>.jsonl` | 我们这份游戏的逐局指纹：split、文件名、recipe / take / go / skills、`seed`、`recipe_seed`、TextWorld `uuid`、通关路线、`.z8` 的 sha256 |
| `docs/fingerprints/sft_sha256.txt` | SFT 数据文件的 sha256 |

## 1. 难度

全部是 TextWorld cooking 游戏（textworld 1.7.0）。参数含义：`recipe` = 菜谱里的食材数，`take` = 需要去找的食材数
（脚本里取 `min(take, recipe)`），`go` = 房间数，skills = 游戏可能要求的技能（`open` 开门 / 容器，`cook` 烹饪，`cut` 切）。
H 是一局最多的动作数（`grpo.max_actions`），实验配置里按难度固定。步数 = 通关路线长度，不含开头的 `inventory`。

| 难度 | recipe | take | go（房间） | skills | 通关步数（train / valid / test） | H | 首次使用 |
|---|---|---|---|---|---|---|---|
| **C0** | 1 | 1 | 1 | cook cut | 8 / 8 / 8 | 30 | E0009 |
| **C1** | 1 | 1 | 1 | open cook cut | 8–9（均值 8.5 / 8.6 / 8.5） | 30 | E0009 |
| **C2** | 2, 3 | = recipe | 1 | open cook cut | 13–21（17.3 / 17.4 / 16.6） | 38 | E0011 |
| **C3** | 1 | 1 | 6 | open cook cut | 8–13（10.1 / 10.1 / 9.9） | 30 | E0011 |
| **C4** | 2, 3 | = recipe | 6, 9 | open cook cut | 13–56（22.9 / 23.1 / 23.8） | 102 | E0011 |
| L0 | 1 | 1 | 1 | 无 | 4 | 30 | E0003 |
| L1 | 1 | 1 | 1 | open cut | 7–8 | 30 | E0003 |

- C0 → C1 加 `open`；C1 → C2 加食材数；C1 → C3 加房间；C4 = 多食材 + 多房间。C0–C4 是现在的难度阶梯。
- L0 / L1 是更早的难度（E0003–E0008）。L0 不开 `cook` 技能，但菜谱仍写着烹饪指令，食材拿到时已经做好；
  再烹饪一次会烧焦判输（E0003 报告）。E0009 起改用没有这个陷阱的 C0 / C1。
- C2 / C4 的 recipe × 房间分布（每局从列表里均匀抽）：

  | 集合 | 分布 |
  |---|---|
  | C2 train / valid / test | r2: 86 / 19 / 22，r3: 114 / 21 / 18 |
  | C4 train | r2g6 45，r2g9 41，r3g6 69，r3g9 45 |
  | C4 valid | r2g6 10，r2g9 9，r3g6 6，r3g9 15 |
  | C4 test | r2g6 8，r2g9 14，r3g6 8，r3g9 10 |

- 所有游戏都用 `scripts/check_games.py` 检查过：通关路线覆盖菜谱的每条指令，且在 `TextWorldEnv` 里能通关
  （E0009 报告 A 节、E0011 报告 Part A）。
- 更早的 `data/games`（300 / 40 / 80 局，recipe 1–3、take 1–3、go 1/6/9、全部技能）只在 E0001 用过，这里不再列出。

## 2. 生成命令和种子

每个难度 **200 / 40 / 40** 局（train / valid / test），生成种子 **0**（`make_games.py` 默认值）：

```bash
python scripts/make_games.py --out data/games_C0 --train 200 --valid 40 --test 40 --recipe 1 --take 1 --go 1 --skills cook cut
python scripts/make_games.py --out data/games_C1 --train 200 --valid 40 --test 40 --recipe 1 --take 1 --go 1 --skills open cook cut
python scripts/make_games.py --out data/games_C2 --train 200 --valid 40 --test 40 --recipe 2 3 --take 3 --go 1 --skills open cook cut
python scripts/make_games.py --out data/games_C3 --train 200 --valid 40 --test 40 --recipe 1 --take 1 --go 6 --skills open cook cut
python scripts/make_games.py --out data/games_C4 --train 200 --valid 40 --test 40 --recipe 2 3 --take 3 --go 6 9 --skills open cook cut
python scripts/make_games.py --out data/games_L0 --train 200 --valid 40 --test 40 --recipe 1 --take 1 --go 1 --skills
python scripts/make_games.py --out data/games_L1 --train 200 --valid 40 --test 40 --recipe 1 --take 1 --go 1 --skills open cut
```

或者一次全做并核对：`bash scripts/make_levels.sh [难度 …]`。

种子是怎么来的：

- `make_games.py` 用 `random.Random(0)` 依次生成 train、valid、test，每局抽 recipe / take / go，再抽一对
  `seed`、`recipe_seed` 传给 `tw-make`（`--split train|valid|test`）。每局的这两个数记在 `data/games_<难度>/manifest.json`，
  也在 `docs/fingerprints/games_<难度>.jsonl` 里。
- **划分不重叠**：`tw-make --split` 让 train / valid / test 用 TextWorld 互不相交的食材集合。每个难度 280 局的 `uuid` 互不相同。
- **文件名在三个划分里重复**：编号 k 在每个划分里都从 0000 开始，`train/cook_r1_t1_g1_0000.z8` 和
  `valid/cook_r1_t1_g1_0000.z8` 是两局不同的游戏。引用一局游戏时要带上划分。
- **C0、C1、C3、L0、L1 共用同一串种子**（它们的参数列表都只有一个值，随机数消耗相同）：第 k 局的 (seed, recipe_seed)
  在这五个难度里相同，所以食材和菜谱基本相同，只是技能 / 房间不同。例如 C1 只在 109 / 200 局 train（24 / 40 valid，
  20 / 40 test）比 C0 多一步 `open`。跨这几个难度比较时要记住它们不是独立的样本。C2、C4 抽的种子不同（和 C0 只有约 10 局重合）。

### 复现时的注意事项

- `tw-make` **不是逐字节可复现的**：同一条命令重新生成，游戏的 `uuid`、物品、菜谱、通关路线都相同，但 `.json` / `.z8`
  里一些集合的顺序不同（Python 的 hash 随机化），所以 sha256 不同。我们核对过一局（C3 valid 0000）。
- 因此 `scripts/data_fingerprint.py verify` 只要求 `uuid` 和通关路线一致；字节完全一致的局数只报告，不要求。
  拿到我们原始文件的拷贝时，sha256 会全部一致（280 / 280）。
- 想要逐字节相同的游戏，就直接拷贝我们的 `data/games_<难度>`，不要重新生成。

## 3. 每个划分怎么用

| 划分 | 用途 | 种子 |
|---|---|---|
| `train`（200 局） | GRPO 训练时抽游戏；SFT 数据沿 train 的通关路线生成 | 训练种子 `--seed`：**17、23、29**（每组实验跑三个，E0012 起部分只跑 17） |
| `valid`（40 局） | 训练中的评测、选 checkpoint、调参、预运行；跨难度（OOD）评测 | 采样种子 `--eval-seeds` **17 23** → 每次 80 局（E0003a/b 用 17 23 29；E0001 用 17） |
| `test`（40 局） | **只用于最终评测**，只通过 `scripts/final_eval.py` | 固定采样种子 **101 102 103** → 120 局 |

具体含义（`jevtw/train.py`、`scripts/final_eval.py`）：

- TextWorld 游戏的环境种子永远是 0（`textworld_games` 返回的 env_seed），同一局游戏每次的初始状态相同；
  评测时每个 (游戏, 采样种子) 跑一局，差别只来自策略采样。
- GRPO 训练：`random.Random(--seed)` 每个 batch 从 train 里有放回地抽 `games_per_batch`（默认 4）局，
  每局一组 `group_size`（默认 8；E0014 草稿里是 16）个采样种子，也由同一个 rng 抽。`torch.manual_seed(--seed)`。
- 训练种子 17 / 23 / 29 和验证采样种子 17 / 23 数值相同，但作用不同（一个决定训练里抽哪些游戏和初始化随机数，
  一个是评测时的采样种子），互不影响。
- 测试集不能用来选任何东西。checkpoint 在 valid 上选好以后，才在 test 上跑 `final_eval.py`；多个训练种子用
  `scripts/aggregate_final.py` 合并。
- 跨难度评测（OOD）用别的难度的 **valid**，不用 test：`final_eval.py --games data/games_C2/valid --seeds 17 23`，
  H 用那个难度自己的值（C2 38，C4 102）。

## 4. SFT 数据

`scripts/build_sft_data.py` 沿 train 每局的通关路线，每一步生成一条 (状态, 正确命令) 记录。T0010 起默认跳过多余的
`open X`（X 已经开着），旧行为用 `--keep-redundant-open`。

```bash
python scripts/build_sft_data.py --games data/games_C3/train --out data/sft/T0010/after/C3_train.jsonl
```

| 文件 | 来源 | 局数 | 状态数 | 跳过的多余 open | 用在 |
|---|---|---|---|---|---|
| `data/sft/C1_train.jsonl` | C1 train（T0010 之前） | 200 | 1709 | — | E0010b |
| `data/sft/T0010/after/C1_train.jsonl` | C1 train | 200 | 1709 | 0 | E0013b（和上一行逐字节相同） |
| `data/sft/T0010/after/C2_train.jsonl` | C2 train | 200 | 3354 | 111 | — |
| `data/sft/T0010/after/C3_train.jsonl` | C3 train | 200 | 2024 | 0 | E0013a、E0013b |
| `data/sft/T0010/after/C4_train.jsonl` | C4 train | 200 | 4315 | 274 | — |

sha256 见 `docs/fingerprints/sft_sha256.txt`。SFT 每个 epoch 在 valid 上评测（`--eval-seeds 17 23`），选 epoch 也只看 valid。

## 5. 每个实验用了哪些数据

“训练”列是 train 划分（GRPO）或 SFT 文件；“起点”是初始模型。所有 valid 评测的采样种子是 17 23，所有 test 评测是 101 102 103。
数字来自 `reports/` 里各次运行记录的命令行参数。

| 实验 | 起点 | 训练 | 训练种子 | 训练中评测 | 最终评测（test） | 其他评测 |
|---|---|---|---|---|---|---|
| E0001a/b | Qwen3-1.7B | `games/train`，2 batch（冒烟） | 17 | `games/valid` 前 10 局，种子 17 | — | — |
| E0003a/b | Qwen3-1.7B | 无（难度探测） | — | L0 / L1 valid，种子 17 23 29 | — | — |
| E0003c/d | Qwen3-1.7B | L0 train，100 batch | 17 | L0 valid | — | — |
| E0004a–h | Qwen3-1.7B | 无（只评测） | — | L1 valid | — | — |
| E0005a–f / g | Qwen3-1.7B | 无（只评测） | — | L0 valid / L1 valid | — | — |
| E0005h–k | Qwen3-1.7B | L0 train，40 batch | 17 | L0 valid | — | — |
| E0006、E0007 | Qwen3-1.7B | 无（诊断） | — | L0 / L1 valid | — | — |
| E0008a–e | Qwen3-1.7B | L0 train，40 batch | 17 | L0 valid | — | — |
| E0009a–d / e–h | Qwen3-1.7B | 无（只评测） | — | C0 valid / C1 valid | — | — |
| E0010a | Qwen3-1.7B | 无（基线） | — | — | C1 test | — |
| E0010b | Qwen3-1.7B | SFT `sft/C1_train.jsonl`，5 epoch | 17 23 29 | C1 valid | C1 test | — |
| E0010c/d/e | Qwen3-1.7B | C1 train，60 batch | 17 23 29 | C1 valid | C1 test | — |
| E0011 | base、E0010b-s17 epoch01、E0010c-s17 batch0020、E0010e-s17 batch0010 | 无（只评测） | — | — | — | C2 / C3 / C4 valid |
| E0012a–e | E0010b-s17 epoch01 | C3 train，60 batch | 17 23 29（d、e 有重跑 s17r / s29r） | C3 valid | C3 test | C2 valid（H 38）、C4 valid（H 102） |
| E0013a | Qwen3-1.7B | SFT `T0010/after/C3_train.jsonl` | 17 23 29 | C3 valid | C3 test | C2、C4 valid |
| E0013b | Qwen3-1.7B | SFT C1 + C3（`T0010/after`） | 17 23 29 | C3 valid | C3 test | C2、C4 valid |
| E0013c | E0013a-s17 epoch02 | C3 train，60 batch | 17 | C3 valid | C3 test | C2、C4 valid |
| E0013r | E0010b-s17（参照） | 无 | — | — | C3 test | C2、C4 valid |
| E0014（草稿） | 同 E0012 A | C2、C3、C4 各自的 train，G = 16，120 batch | 17 | 各自 valid | — | 另外两个难度的 valid |
| E0015（配置已写） | — | g0–g6 × C2 / C3 / C4 | — | — | — | — |

（E0014、E0015 还没有报告，以 issue 为准。）

## 6. 拿到数据后自检

```bash
python scripts/data_fingerprint.py verify data/games_C3 docs/fingerprints/games_C3.jsonl
# 280 games (expected 280), 0 problems, 280 byte-identical .z8     ← 我们文件的拷贝
# 280 games (expected 280), 0 problems, 0 byte-identical .z8       ← 重新生成的（正常）
sha256sum -c docs/fingerprints/sft_sha256.txt   # 在 data/sft 目录下运行
```

只要有 problems（缺局、多局、uuid 或通关路线不同），这份数据就和我们的不一样，不要用来对比我们的数字。
