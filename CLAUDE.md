# 给服务器端 agent 的工作说明

本仓库由两个 Claude 协作开发，用户 johnlovesushi 有最终决定权。

- **规划端**：claude.ai 里的会话。负责设计方法、写任务、审代码、分析结果。它**只能看到 GitHub**，
  看不到服务器、Slurm、W&B，也看不到你的终端。没有推送到 GitHub 的东西，规划端都看不到。
- **执行端**：你，运行在 GPU 服务器上。负责按任务实现、运行实验，并把结果推回 GitHub。

用户在终端里直接给你的指令，优先级高于 issue。

## 工作流程

1. **领任务**：任务都是 GitHub issue，标题格式为 `[E0001] ...`（实验）或 `[T0001] ...`（代码任务）。
   只做 issue 里写的事。
2. **开始前**：在 issue 下留言，用一两句话说明计划。
3. **开分支**：从最新的 `main` 切出 `exp/E0001-<简称>` 或 `task/T0001-<简称>`。
   不要直接推送 `main`，也不要 force-push。
4. **先跑测试**：`python -m pytest -q tests`。测试失败就停下来，在 issue 里报告。
5. **使用实验配置**：每个实验对应 `configs/experiments/<ID>.json`，通常规划端已经写好。
   不要改 `configs/grpo*.json` 或 `jevtw/config.py` 里的默认值来跑实验。
   **预运行（pilot）默认 `--batches 40 --eval-every 5`，除非 issue 另有规定。**
6. **偏离任务时先说明**：任何和 issue 不一致的地方（改超参数、改命令、跳过步骤、修 bug），
   都要先在 issue 里留言说明原因。如果这个改动会影响结论，就等用户或规划端回复后再继续。
   纯环境问题（路径、模块名等）可以直接处理，但要在报告里写明。
7. **修 bug**：可以修 `jevtw/` 里妨碍任务完成的 bug，规则如下：
   - 单独提交一个 commit，在 commit message 里写清楚问题和修法；
   - 同时加一个能复现这个 bug 的 CPU 测试；
   - 如果改了公式相关的代码，要同步更新 `docs/implementation.md`。
8. **长时间运行的任务**：提交作业后，在 issue 里留言 Slurm job id 和预计完成时间。
   中途失败也要回报，附上日志末尾。同一个作业最多重跑一次，之后就要先报告再说。
9. **回报结果**：
   ```bash
   python scripts/summarize_run.py runs/<ID> --id <ID> --title "<一句话>" --log <作业日志> --notes notes.md
   ```
   然后提交 `reports/<ID>.md` 和 `reports/<ID>.json`。`notes.md` 的内容会进入报告的 Executor notes
   一节，要写这几项：
   - 和 issue 不一致的地方
   - 遇到的问题
   - 你对结果的解读
   - 需要决定的问题
10. **开 PR**：目标分支是 `main`，正文第一行写 `Closes #<issue 编号>`，然后列出：
    - 做了什么
    - 报告文件的位置
    - 需要规划端或用户决定的问题
11. **测试集**：`data/games*/test` 只在 issue 明确要求“最终评测”时使用，并且只通过
    `scripts/final_eval.py`（固定种子 101、102、103）。选 checkpoint、调超参数、预运行一律用验证集。

## 不要做的事

- 不要提交 `data/`、`runs/`、checkpoint、模型权重，或超过 1 MB 的日志。只提交代码和报告。
- 不要提交任何密钥、token 或 W&B key。
- 不要启动 issue 里没有的实验。如果有想法，写在 Executor notes 或 issue 留言里。
- 不要删除或覆盖已有的 `runs/` 目录。每次运行都用新目录。
- 不要挑着报数字。所有跑过的种子和配置都要报告，包括失败的。

## 服务器环境

执行端在 E0001 中填写（2026-11-21，集群 killarney，登录节点 `klogin03`）。

- **仓库路径**：`/scratch/$USER/JevOnTextWorld`
- **模块**：`source /etc/profile; module load python/3.12 arrow cuda`
  （得到 Python 3.12.4、arrow 18.1.0、cuda 12.2；torch 自带到 CUDA 13.2 库的路径，不依赖 `cuda` 模块的版本）
- **venv**：仓库下的 `.venv/`（已在 `.gitignore` 里）。建法：
  ```bash
  virtualenv --no-download .venv && source .venv/bin/activate
  pip install --no-index torch transformers pydantic spacy pytest   # 先装集群 wheelhouse 里的预编译包
  pip install -r requirements.txt                                   # 再装 textworld 等剩下的包
  ```
  必须先跑 `--no-index` 那一步：集群的 pip 不接受 PyPI 上的 manylinux wheel，直接
  `pip install -r requirements.txt` 会去从源码编译 `pydantic-core`（需要 Rust），然后失败。
- **版本**：torch 2.14.1（CUDA 13.2）、transformers 5.22.2、textworld 1.7.0、jericho 3.3.1
- **`HF_HOME`**：`/scratch/$USER/hf_cache`。`Qwen/Qwen3-1.7B` 已下载
  （revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`）。
- **W&B（可选）**：`pip install --no-index wandb` 安装（不在 `requirements.txt` 里）。由**用户自己**在服务器上运行
  `wandb login`；key 永远不进仓库，执行端也不经手。训练时加 `--wandb-project <项目>` 才启用
  （`--wandb-run-name` 默认是输出目录名，`--wandb-mode {online,offline}` 默认 online，online 初始化失败会自动改用
  offline，之后可以用 `wandb sync` 上传）。作业会继承提交时 shell 里的 `WANDB_*` 环境变量（Slurm 默认行为），
  不要在脚本里写 key。规划端看不到 W&B，`reports/` 照常生成。
- **网络**：计算节点可以联网，但作业一律用离线模式（`HF_HUB_OFFLINE=1`）运行，以保证可复现；
  模型、pip 包、游戏数据都要先在登录节点准备好。
- **Slurm**：分区 `gpubase_l40s_b3`（每节点 4 张 L40S、64 核、约 500 GB 内存，时限 24 小时），
  账户 `aip-rjliao`。GPU 任务一律用 Slurm 提交，不在登录节点上跑。
- **游戏数据**：`data/games/{train,valid,test}`（300 / 40 / 80 局，`scripts/make_games.py` 默认种子 0），
  清单在 `data/games/manifest.json`。不进 git。
- **提交作业**（在仓库根目录执行）：
  ```bash
  mkdir -p runs/slurm
  sbatch --job-name=<ID> scripts/slurm/train.sbatch configs/experiments/<ID>.json runs/<ID> \
      --model Qwen/Qwen3-1.7B --train-games data/games/train --val-games data/games/valid [其余 jevtw.train 参数]
  ```
  日志在 `runs/slurm/<作业名>-<jobid>.out`。脚本默认单卡、8 核、48 GB、3 小时；要改就在 `sbatch` 命令行上
  加 `--time` 等参数。输出目录已存在时脚本会直接退出，不会覆盖。
- **登录节点上的注意事项**：
  - 跑 CPU 测试时设 `OMP_NUM_THREADS=4`，否则 torch 会按整台机器的核数开线程，非常慢。
  - `gh` 装在 `~/.local/bin`；它偶尔会因为创建线程失败而崩溃，前面加 `GOMAXPROCS=2` 即可。

## 代码约定

- 目录结构见 `README.md`；公式和代码的对应关系见 `docs/implementation.md`。改算法前先读这两个文件。
- 测试必须能在 CPU 上运行。随机小模型用 `jevtw/testing.py`。
- 编号规则：
  - 实验编号 `E0001`、`E0002`…；同一个实验有多个分组时加后缀，如 `E0001a`、`E0001b`；
  - 代码任务编号 `T0001`…；
  - 报告文件名就是实验编号。
