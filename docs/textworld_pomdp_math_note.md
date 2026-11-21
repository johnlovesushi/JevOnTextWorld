# TextWorld as a Partially Observed Markov Decision Process

*A Mathematical Comparison with LLM Problem Solving* — November 21, 2026

Markdown transcription of `textworld_pomdp_math_note.pdf`. Equation numbers match
the PDF. How each equation maps to code is in [`implementation.md`](implementation.md).

## Abstract

We formalize multi-turn TextWorld play as episodic control in a partially observed
environment, then specify trajectory-level group-relative policy optimization (GRPO)
and an optional success-probability calibration objective. We contrast this with
standard LLM mathematical reasoning, which is a contextual bandit at the
whole-response level, despite admitting an equivalent token-prefix Markov decision
process. The essential distinction is the presence of *external state transitions
and observations* between TextWorld decisions.

## 1 TextWorld: environment and reward

For a sampled game and seed $g \sim \mathcal{D}_{\mathrm{TW}}$, let

$$
\mathcal{M}_g = (\mathcal{S}_g, \mathcal{A}_g, P_g, \mathcal{O}_g, \Omega_g, r_g, \rho_{0,g}, H) \tag{1}
$$

be a finite-horizon partially observable Markov decision process (POMDP). Its hidden
state $s_t \in \mathcal{S}_g$ includes the agent's location, inventory, object states
and quest progress. Conditional on $g$, assume deterministic game dynamics and
observations:

$$
\begin{aligned}
s_0 &\sim \rho_{0,g}, & s_{t+1} &= f_g(s_t, a_t), & o_t &= O_g(s_t),\\
h_t &= (o_0, a_0, \dots, a_{t-1}, o_t), & a_t &\in \mathcal{A}_g(s_t).
\end{aligned} \tag{2}
$$

The model conditions on the observable history $h_t$, not necessarily on $s_t$. An
admissible-command list, when provided to the agent, is part of its observation;
otherwise the agent must produce a valid command without this privileged
information. Deterministic transitions do *not* make the environment static.

Let $T \le H$ be the number of actions before termination. In the terminal-reward
setting discussed here,

$$
z = \mathbf{1}\{\text{quest completed}\} \in \{0,1\}, \qquad
r_t = \begin{cases} 0, & t < T-1,\\ z, & t = T-1. \end{cases} \tag{3}
$$

Thus one episode has *one* outcome $z$, and its discounted return from decision $t$
is $\gamma^{T-1-t} z$. The binary outcome is supplied by the environment, not
predicted by the model.

## 2 Language-model policy and rollouts

At environment step $t$, a language model first samples reasoning tokens $c_t$, then
selects an action $a_t$ from $K_t$ available commands:

$$
\begin{aligned}
c_t &\sim \pi^c_\theta(\cdot \mid h_t), &
\pi^c_\theta(c_t \mid h_t) &= \prod_{k=1}^{|c_t|} \pi_\theta(c_{t,k} \mid h_t, c_{t,<k}),\\
a_t &\sim q_\theta(\cdot \mid h_t, c_t), &
q_\theta(a^j \mid h_t, c_t) &= \frac{\exp \ell_\theta(a^j)}{\sum_{m=1}^{K_t} \exp \ell_\theta(a^m)}.
\end{aligned} \tag{4}
$$

The second distribution assumes a finite candidate-action interface. The same network
may additionally predict *future episode success*, after choosing $a_t$ but before
seeing the next observation:

$$
p_t = p_\theta(h_t, c_t, a_t) = \sigma\big(\ell_\theta(\text{yes}) - \ell_\theta(\text{no})\big) \in [0,1]. \tag{5}
$$

Here $p_t$ is a forecast, not a reward or a sampled environment action. For instance,
$p_t = 0.8$ means the model assigns 80% probability to eventual success; the actual
$z$ is observed only at the end. For each game $g$, collect $G$ independent complete
trajectories $\tau^1, \dots, \tau^G$ from the same initial state with a fixed rollout
policy $\theta_{\text{old}}$. Each $\tau^i$ contains $(h^i_t, c^i_t, a^i_t, p^i_t)_{t=0}^{T_i-1}$
and the terminal outcome $z^i$.

## 3 Group-relative optimization and calibration

A proposed RLCD-*like* extension adds a proper-scoring-rule component to the terminal
success reward. Define

$$
b_i = 1 - \frac{1}{T_i} \sum_{t=0}^{T_i-1} (p^i_t - z^i)^2, \qquad
R_i = z^i + \lambda b_i, \qquad 0 \le \lambda < 1, \tag{6}
$$

where $b_i \in [0,1]$. Consequently, any win has larger return than any loss.
Group-relative advantages compare only rollouts of the *same game and seed*:

$$
\widehat{A}_i = \frac{R_i - \overline{R}}{\operatorname{sd}(R_1, \dots, R_G) + \delta}, \qquad
\overline{R} = G^{-1} \textstyle\sum_{j=1}^{G} R_j. \tag{7}
$$

Each reasoning token and the final categorical action of a trajectory inherits its
episode advantage. Let $u_{i,t,k}$ enumerate these sampled decisions, $\Pi_\theta$
denote their appropriate conditional law ($\pi^c_\theta$ or $q_\theta$), and
$\rho_{i,t,k} = \Pi_\theta(u_{i,t,k} \mid \cdot) / \Pi_{\theta_{\text{old}}}(u_{i,t,k} \mid \cdot)$.
The clipped policy loss is

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{N} \sum_{i,t,k}
\min\Big\{ \rho_{i,t,k} \widehat{A}_i,\ \operatorname{clip}(\rho_{i,t,k}, 1-\varepsilon, 1+\varepsilon) \widehat{A}_i \Big\}, \tag{8}
$$

where $N = \sum_{i,t} (|c^i_t| + 1)$ and $R_i, \widehat{A}_i$ are detached. When $p_t$ is
read from model logits, one may also train it directly using the Brier loss

$$
\mathcal{L}_{\text{cal}} = \frac{\sum_{i,t} \big(p_\theta(h^i_t, c^i_t, a^i_t) - z^i\big)^2}{\sum_i T_i}, \qquad
\mathcal{L} = \mathcal{L}_{\text{GRPO}} + \alpha \mathcal{L}_{\text{cal}} + \beta \mathcal{L}_{\text{KL}}, \tag{9}
$$

with sampled contexts held fixed; average these losses over games. The
reference-policy KL is optional.

**Proposition 1** (Interpretation of $p_t$). *For a fixed rollout policy
$\theta_{\text{old}}$, the population minimizer of $\mathcal{L}_{\text{cal}}$ satisfies*

$$
p^*(h, c, a) = \mathbb{P}_{\theta_{\text{old}}}(z = 1 \mid h, c, a) = Q^{\theta_{\text{old}}}_{\text{win}}(h, c, a). \tag{10}
$$

*Proof.* Conditionally on $(h, c, a)$, squared error is minimized by
$\mathbb{E}[z \mid h, c, a]$; for binary $z$, this equals its conditional success
probability. $\square$

Thus calibration yields a history-conditioned *undiscounted* success critic, whose
target changes with the policy. Ordinary multi-step GRPO requires no forecast: set
$\lambda = \alpha = 0$. If confidence is emitted as text rather than read from logits,
its tokens join the sampled policy and calibration can be scored through $R_i$ instead.

## 4 Difference from LLM mathematical reasoning

For a fixed math problem $x \sim \mathcal{D}_{\text{math}}$ and full completion
$y \sim \pi_\theta(\cdot \mid x)$, a terminal verifier provides
$z_{\text{math}}(x, y) \in \{0,1\}$. The respective objectives are

$$
\begin{aligned}
J_{\text{math}}(\theta) &= \mathbb{E}_{x \sim \mathcal{D}_{\text{math}},\, y \sim \pi_\theta(\cdot \mid x)}\big[z_{\text{math}}(x, y)\big],\\
J_{\text{TW}}(\theta) &= \mathbb{E}_{g \sim \mathcal{D}_{\text{TW}},\, \tau \sim (\pi_\theta, P_g)}\big[z(\tau)\big].
\end{aligned} \tag{11}
$$

At the *whole-response* level, standard math solving is a contextual bandit: one
problem, one completion, one score. It also admits a token-level MDP with
$m_k = (x, y_{<k})$, action $y_k$ and deterministic concatenation
$m_{k+1} = (x, y_{\le k})$. However, this updates the *generated text prefix*, not an
external world. In TextWorld, $a_t$ changes $s_t$, then $o_{t+1}$ supplies new
information unavailable before the action. Consequently, TextWorld requires
closed-loop control, exploration under partial observability and long-horizon credit
assignment; static math solving does not require external state tracking.
Tool-augmented or interactively verified math solving can itself become a multi-step
control problem.

**Scope.** The confidence-augmented objective is a formulation discussed in the
source conversation, not a claim about any proprietary RLCD training recipe.
State-matching approaches to stepwise credit assignment must use the full underlying
game state (and remaining horizon), not merely identical observation strings.
