# JevOnTextWorld

把 TextWorld 多轮游戏当作部分可观测马尔可夫决策过程（POMDP）来训练 LLM 智能体：
逐局（trajectory-level）GRPO，外加可选的“成功概率校准”目标。

数学定义见 [`docs/textworld_pomdp_math_note.md`](docs/textworld_pomdp_math_note.md)
（原始 PDF：[`docs/textworld_pomdp_math_note.pdf`](docs/textworld_pomdp_math_note.pdf)）。
