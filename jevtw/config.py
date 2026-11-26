"""Configuration. Defaults reproduce plain multi-step GRPO (lam = alpha = 0, note section 3)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
import warnings
from pathlib import Path

NORMALIZATIONS = ("token", "episode")
HISTORY_MODES = ("full", "feedback")
FORECAST_FORMATS = ("after_eos", "no_eos", "chat")
PROMPT_FORMATS = ("plain", "chat_nothink", "chat_think")
CAL_LOSSES = ("brier", "bce")
CAL_GRADS = ("full", "platt_only")
CONFIDENCE_MODES = ("logit", "verbal")
CONF_ADVANTAGES = ("shared", "decoupled")
ADV_NORMS = ("std", "mean")


@dataclass
class PolicyConfig:
    """How the language model sees a step and how c_t, a_t and p_t are produced (eqs. 4-5)."""

    max_reasoning_tokens: int = 320      # |c_t| cap; 0 = choose directly after the prompt
    reasoning_prefix: str = "Reasoning:"
    answer_prefix: str = "Answer:"        # sampling c_t stops once the reasoning ends with this, or at EOS
    # How the prompt is laid out and where the reasoning c_t is written (T0006):
    #   "plain"         a text to continue; with reasoning it ends with the line `reasoning_prefix`;
    #   "chat_nothink"  one user turn in Qwen's chat template and an assistant turn that starts with an empty
    #                   thinking block ("<think>\n\n</think>\n\n"); c_t is what the assistant writes;
    #   "chat_think"    the same user turn, the assistant turn starts with "<think>\n"; c_t includes the thinking
    #                   block, and "Answer:" ends c_t only after "</think>".
    prompt_format: str = "plain"
    history_window: int = 8               # past (command, observation) pairs shown; h_t is truncated to this
    # What a past step shows after its command: "full" = the whole observation that followed (room, inventory and
    # notes again at every step), "feedback" = only what the game answered (Observation.feedback). The current
    # observation is always shown in full.
    history_mode: str = "full"
    # p_t = sigmoid(l(yes) - l(no)) read after this question; it is never written into the history.
    confidence_prompt: str = "\nQuestion: will you complete the objective? Answer yes or no:"
    yes_text: str = " yes"                # must encode to exactly one token
    no_text: str = " no"
    forecast: bool = True                 # compute p_t during rollouts (needed when lam > 0, logged otherwise)
    # Where the forecast logit s = l(yes) - l(no) is read (T0005). x = action context, y^j = " j<EOS>":
    #   "after_eos"  after x + y^j + confidence_prompt (the question follows the option's EOS);
    #   "no_eos"     after x + " j" + confidence_prompt (no EOS before the question);
    #   "chat"       after x + y^j + "\n<|im_start|>user\n{forecast_question}<|im_end|>\n<|im_start|>assistant\n
    #                <think>\n\n</think>\n\n", reading chat_yes_text / chat_no_text (a new chat turn, Qwen template
    #                with thinking off).
    # The option scores l(a^j) are the same in all three.
    forecast_format: str = "after_eos"
    forecast_question: str = "Will you complete the objective? Answer yes or no."
    chat_yes_text: str = "Yes"            # must encode to exactly one token (checked only for "chat")
    chat_no_text: str = "No"
    # Platt scaling: p = sigmoid(a * s + b) with two extra parameters owned by LMPolicy (a = 1, b = 0 at first).
    platt: bool = False
    # Verbal confidence (T0008). "logit": only the yes / no forecast above. "verbal": after the chosen option
    # x_t + " j<EOS>" the prompt `verbal_prompt` u is appended and the policy samples one of `confidence_bins` M
    # answers y^m = " {100 m / (M - 1)}<EOS>" from pi^c = softmax over their summed log-likelihoods (constrained
    # sampling); c_t = v_m = m / (M - 1) is part of the policy (it has its own likelihood term), never enters the
    # history. `forecast` (the yes / no logit) can stay on next to it, e.g. for the zero-shot comparison.
    confidence_mode: str = "logit"
    confidence_bins: int = 11
    verbal_prompt: str = "\nConfidence (0-100) that you will complete the objective:"
    # Ignored since T0001 (all options are scored in one suffix batch on a shared prefix); kept so that old
    # configuration files still load.
    option_batch_size: int = 16

    def __post_init__(self):
        if self.history_mode not in HISTORY_MODES:
            raise ValueError(f"history_mode must be one of {HISTORY_MODES}, got {self.history_mode!r}")
        if self.prompt_format not in PROMPT_FORMATS:
            raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}, got {self.prompt_format!r}")
        if self.forecast_format not in FORECAST_FORMATS:
            raise ValueError(f"forecast_format must be one of {FORECAST_FORMATS}, got {self.forecast_format!r}")
        if self.confidence_mode not in CONFIDENCE_MODES:
            raise ValueError(f"confidence_mode must be one of {CONFIDENCE_MODES}, got {self.confidence_mode!r}")
        if self.confidence_bins < 2 or self.confidence_bins > 101:
            raise ValueError("confidence_bins must be between 2 and 101")
        if self.confidence_mode == "verbal" and self.forecast and self.forecast_format == "no_eos":
            raise ValueError('confidence_mode = "verbal" with the yes / no forecast needs forecast_format '
                             '"after_eos" or "chat"')


@dataclass
class GRPOConfig:
    """Trajectory-level GRPO plus calibration (eqs. 6-9)."""

    games_per_batch: int = 4
    group_size: int = 8                   # G, rollouts per game from the same initial state
    max_actions: int = 50                 # H
    lam: float = 0.0                      # lambda in R_i = z_i + lambda * b_i, 0 <= lambda < 1
    alpha: float = 0.0                    # weight of the Brier loss L_cal
    beta: float = 0.0                     # weight of the reference-policy KL (optional)
    clip_eps: float = 0.2                 # epsilon in eq. (8)
    adv_delta: float = 1e-8               # delta in eq. (7)
    # "token": divide by N = sum_{i,t}(|c_t^i| + 1) within each game, as in eq. (8).
    # "episode": average the terms within each episode, then over the G episodes (rlcd-bench baselines).
    normalization: str = "token"
    optimization_passes: int = 2
    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    parallel_episodes: int = 32           # episodes advanced in lockstep while sampling (training and evaluation)
    # Truncated importance sampling: every policy term is weighted by min(exp(old - behavior), tis_cap), where
    # behavior is the log-prob recorded while sampling and old the one recomputed by the training forward.
    # None = off.
    tis_cap: float | None = None
    # Learning rate of batch b (from 1) is learning_rate * min(1, b / warmup_batches); 0 = no warmup.
    warmup_batches: int = 0
    # Calibration loss of the alpha term: "brier" = (p - z)^2, "bce" = -z log p - (1 - z) log(1 - p). The lambda
    # bonus b_i always uses the Brier score.
    cal_loss: str = "brier"
    # "full": the calibration loss trains the language model (and a, b with Platt scaling); "platt_only": s is
    # detached, so it trains only a and b.
    cal_grad: str = "full"
    platt_lr: float = 1e-2                # learning rate of a and b (own parameter group, no weight decay)
    # Verbal confidence (T0008, PolicyConfig.confidence_mode = "verbal"). The confidence term
    # -conf_coef * psi(rho^c, A^c) of every decision is added with the decision's action weight (N counts only the
    # action terms); the KL covers it too (not scaled by conf_coef).
    #   "shared"     A^c = the episode's advantage (from R_i = z_i + lam b_i, b_i from the sampled c_t): RLCR;
    #   "decoupled"  A^c_t = S(c_t, z) - sum_m pi^c_old(m) S(v_m, z) per decision, S(c, z) = 1 - (c - z)^2.
    conf_advantage: str = "shared"
    conf_coef: float = 1.0                # kappa
    # Advantages within a game: "std" = (R - mean) / (sd + delta) (eq. 7), "mean" = R - mean.
    adv_norm: str = "std"

    def __post_init__(self):
        if not 0.0 <= self.lam < 1.0:
            raise ValueError(f"lam must satisfy 0 <= lam < 1 (eq. 6), got {self.lam}")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {self.normalization!r}")
        if self.parallel_episodes < 1:
            raise ValueError("parallel_episodes must be at least 1")
        if self.tis_cap is not None and self.tis_cap <= 0:
            raise ValueError("tis_cap must be positive (or None to turn truncated importance sampling off)")
        if self.cal_loss not in CAL_LOSSES:
            raise ValueError(f"cal_loss must be one of {CAL_LOSSES}, got {self.cal_loss!r}")
        if self.cal_grad not in CAL_GRADS:
            raise ValueError(f"cal_grad must be one of {CAL_GRADS}, got {self.cal_grad!r}")
        if self.platt_lr <= 0:
            raise ValueError("platt_lr must be positive")
        if self.warmup_batches < 0:
            raise ValueError("warmup_batches must be non-negative")
        if self.conf_advantage not in CONF_ADVANTAGES:
            raise ValueError(f"conf_advantage must be one of {CONF_ADVANTAGES}, got {self.conf_advantage!r}")
        if self.adv_norm not in ADV_NORMS:
            raise ValueError(f"adv_norm must be one of {ADV_NORMS}, got {self.adv_norm!r}")
        if self.conf_coef < 0:
            raise ValueError("conf_coef must be non-negative")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for group-relative advantages")


@dataclass
class Config:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    def __post_init__(self):
        if self.policy.prompt_format != "plain" and self.policy.forecast_format != "chat":
            warnings.warn(f"prompt_format = {self.policy.prompt_format!r} is a chat format but forecast_format = "
                          f"{self.policy.forecast_format!r}: the confidence question is appended as plain text "
                          'inside the assistant turn (forecast_format = "chat" asks it in a new user turn)',
                          stacklevel=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        def build(kind, values):
            known = {f.name for f in fields(kind)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"unknown {kind.__name__} keys: {sorted(unknown)}")
            return kind(**values)

        return cls(policy=build(PolicyConfig, data.get("policy", {})), grpo=build(GRPOConfig, data.get("grpo", {})))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)
