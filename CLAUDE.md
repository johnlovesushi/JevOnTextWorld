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

## 不要做的事

- 不要提交 `data/`、`runs/`、checkpoint、模型权重，或超过 1 MB 的日志。只提交代码和报告。
- 不要提交任何密钥、token 或 W&B key。
- 不要启动 issue 里没有的实验。如果有想法，写在 Executor notes 或 issue 留言里。
- 不要删除或覆盖已有的 `runs/` 目录。每次运行都用新目录。
- 不要挑着报数字。所有跑过的种子和配置都要报告，包括失败的。

## 服务器环境

（由执行端在 E0001 中填写：仓库路径、venv、`module load`、`HF_HOME`、Slurm 分区/账户/GPU 型号、
游戏数据路径、提交作业的命令。）

参考 rlcd-bench 当时的设置，但以服务器实际情况为准：
- L40S，分区 `gpubase_l40s_b3`，账户 `aip-rjliao`
- `source /etc/profile; module load python/3.12 arrow cuda`
- `HF_HOME=/scratch/$USER/hf_cache`

## 代码约定

- 目录结构见 `README.md`；公式和代码的对应关系见 `docs/implementation.md`。改算法前先读这两个文件。
- 测试必须能在 CPU 上运行。随机小模型用 `jevtw/testing.py`。
- 编号规则：
  - 实验编号 `E0001`、`E0002`…；同一个实验有多个分组时加后缀，如 `E0001a`、`E0001b`；
  - 代码任务编号 `T0001`…；
  - 报告文件名就是实验编号。
