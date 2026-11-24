"""T0004: history_mode ("full" / "feedback") and optional Weights & Biases logging."""
import json
import sys
import types

import pytest

from jevtw.config import Config, PolicyConfig
from jevtw.envs import KeyChestEnv, Observation, TextWorldEnv
from jevtw.policy import build_prompt
from jevtw.rollout import collect_batch, run_episode
from jevtw import train

# build_prompt of the code before T0004 on a fixed KeyChest trajectory (seed 0: the key is in the box).
KEYCHEST_COMMANDS = ["open chest", "open box", "take key"]
KEYCHEST_FULL = (
    'Objective: Find the key and open the chest.\n\nRecent steps:\n> open chest\nThe chest is locked.\n'
    'You are carrying: nothing.\n\n> open box\nYou open the box. Inside you see the key.\nYou are carrying: nothing.'
    '\n\n> take key\nYou take the key.\nYou are carrying: the key.\n\nCurrent observation:\nYou take the key.\n'
    'You are carrying: the key.\n\nOptions:\n1. open drawer\n2. open chest\n3. smash chest\n4. wait\n\n'
    'Think step by step, then write "Answer:" followed by the number of one option.\nReasoning:')
KEYCHEST_FEEDBACK = (
    'Objective: Find the key and open the chest.\n\nRecent steps:\n> open chest\nThe chest is locked.\n\n'
    '> open box\nYou open the box. Inside you see the key.\n\n> take key\nYou take the key.\n\n'
    'Current observation:\nYou take the key.\nYou are carrying: the key.\n\nOptions:\n1. open drawer\n2. open chest\n'
    '3. smash chest\n4. wait\n\nThink step by step, then write "Answer:" followed by the number of one option.\n'
    'Reasoning:')


def keychest_trajectory():
    env = KeyChestEnv()
    obs = env.reset(0)
    assert obs.feedback == ""
    history = []
    for cmd in KEYCHEST_COMMANDS:
        obs = env.step(cmd)
        history.append((cmd, obs))
    return env, history, obs


def test_history_mode_is_validated_and_defaults_to_full():
    assert PolicyConfig().history_mode == "full"
    with pytest.raises(ValueError, match="history_mode"):
        PolicyConfig(history_mode="short")
    with pytest.raises(ValueError, match="history_mode"):
        Config.from_dict({"policy": {"history_mode": "none"}})
    reference, feedback = Config.load("configs/grpo.json"), Config.load("configs/grpo_feedback.json")
    assert feedback.policy.history_mode == "feedback"
    feedback.policy.history_mode = "full"
    assert feedback.to_dict() == reference.to_dict()


def test_keychest_full_mode_prompt_is_unchanged_and_feedback_mode_shows_inventory_once():
    env, history, obs = keychest_trajectory()
    assert build_prompt(env.objective, history, obs, PolicyConfig()) == KEYCHEST_FULL
    text = build_prompt(env.objective, history, obs, PolicyConfig(history_mode="feedback"))
    assert text == KEYCHEST_FEEDBACK
    assert text.count("You are carrying") == 1 and KEYCHEST_FULL.count("You are carrying") == 4
    assert [o.feedback for _, o in history] == ["The chest is locked.", "You open the box. Inside you see the key.",
                                                "You take the key."]


def old_full_prompt(objective, history, obs, cfg):
    """The prompt as the code before T0004 built it: every past step shows the whole observation."""
    lines = [f"Objective: {objective}", "", "Recent steps:"]
    for cmd, after in history[-cfg.history_window:]:
        lines += [f"> {cmd}", after.text, ""]
    lines += ["Current observation:", obs.text, "", "Options:"]
    lines += [f"{j + 1}. {c}" for j, c in enumerate(obs.commands)]
    return "\n".join(lines + ["", "Reply with the number of one option."])


def test_textworld_feedback_mode(cooking_game):
    import textworld

    raw = textworld.start(str(cooking_game), request_infos=textworld.EnvInfos(extras=["walkthrough"]))
    walkthrough = raw.reset()["extra.walkthrough"]
    raw.close()

    full, short = PolicyConfig(max_reasoning_tokens=0), PolicyConfig(max_reasoning_tokens=0, history_mode="feedback")
    env = TextWorldEnv(cooking_game)
    obs = env.reset(0)
    assert obs.feedback == ""
    history, read = [], False
    for cmd in walkthrough:
        if not env._keep(cmd):
            continue
        obs = env.step(cmd)
        history.append((cmd, obs))
        assert "score has just gone" not in obs.feedback
        assert "Inventory:" not in obs.feedback and "Notes:" not in obs.feedback
        assert "-=" not in obs.feedback and obs.text.startswith(obs.feedback)
        if cmd == "examine cookbook":
            read = True
            assert obs.feedback == "You read the recipe (see notes)."
        if obs.done:
            break
        assert build_prompt(env.objective, history, obs, full) == old_full_prompt(env.objective, history, obs, full)
        text = build_prompt(env.objective, history, obs, short)
        steps, current = text.split("Current observation:\n")
        assert current.startswith(obs.text)
        assert text.count("Inventory:") == 1 and text.count("Notes:") == ("Notes:" in obs.text)
        assert text.count("Recipe #") == ("Notes:" in obs.text)
        # The room description (everything after its title line) is in the current observation only.
        room = obs.text.split("=-", 1)[1].split("Inventory:")[0].strip()
        assert len(room) > 40 and text.count(room) == 1 and room not in steps
        assert len(text) < len(old_full_prompt(env.objective, history, obs, full))
    env.close()
    assert read and obs.won


class FakeGameState(dict):
    def __init__(self, feedback, **infos):
        super().__init__(won=False, lost=False, inventory="You are carrying nothing.", admissible_commands=["go west"],
                         **infos)
        self.feedback = feedback


def test_textworld_feedback_of_a_move_is_the_room_title():
    """The single-room test game has no move, so the game state is written by hand here."""
    env = TextWorldEnv.__new__(TextWorldEnv)
    env.filter_info_commands = env.strip_score_text = True
    env._notes = ""
    room = "-= Kitchen =-\nYou find yourself in a kitchen. You see a fridge.\n\nThere is an exit to the west."
    prompt = "\n\n> -= Kitchen =-0/3"

    moved = env._observe(FakeGameState(f"\n\n{room}{prompt}", description=room), done=False)
    assert moved.feedback == "-= Kitchen =-"
    assert moved.text.startswith(room) and moved.text.count("-= Kitchen =-") == 1 and "Inventory:" in moved.text

    scored = env._observe(FakeGameState(f"You open the fridge.\n\nYour score has just gone up by one point.{prompt}",
                                        description=room), done=False)
    assert scored.feedback == "You open the fridge."
    assert scored.text.startswith(f"You open the fridge.\n\n{room}")

    first = env._observe(FakeGameState(f"\n\nWelcome!\n\n{room}{prompt}", description=room), done=False,
                         initial=True)
    assert first.feedback == "" and "Welcome!" in first.text


@pytest.mark.parametrize("mode", ["full", "feedback"])
def test_collect_batch_parallel_one_equals_run_episode(tokenizer, policy_cfg, mode):
    from jevtw.policy import LMPolicy
    from jevtw.testing import tiny_model

    policy_cfg.history_mode = mode
    policy = LMPolicy(tiny_model(seed=0), tokenizer, policy_cfg)
    jobs = [(f"keychest-{s}", KeyChestEnv, s, 100 + s) for s in range(4)]
    batch = collect_batch(policy, jobs, max_actions=5, parallel=1)
    for (game, make_env, env_seed, sample_seed), ep in zip(jobs, batch):
        ref = run_episode(policy, make_env(), game, env_seed, sample_seed, 5)
        assert [d.command for d in ep.decisions] == [d.command for d in ref.decisions]
        assert [d.prompt_ids for d in ep.decisions] == [d.prompt_ids for d in ref.decisions]
        assert [d.reasoning_ids for d in ep.decisions] == [d.reasoning_ids for d in ref.decisions]
        assert [d.behavior_logps for d in ep.decisions] == [d.behavior_logps for d in ref.decisions]
    carried = [tokenizer.decode(d.prompt_ids).count("You are carrying") for ep in batch for d in ep.decisions]
    assert max(carried) > 1 if mode == "full" else set(carried) == {1}


# ---------------------------------------------------------------------------------------------- W&B


class FakeRun:
    def __init__(self):
        self.logged, self.finished = [], 0

    def log(self, data, step=None):
        self.logged.append((step, dict(data)))

    def finish(self):
        self.finished += 1


def fake_wandb(monkeypatch, fail_online=False):
    module = types.ModuleType("wandb")
    module.inits, module.run = [], FakeRun()

    def init(**kwargs):
        module.inits.append(kwargs)
        if fail_online and kwargs.get("mode") == "online":
            raise TimeoutError("no network")
        return module.run

    module.init = init
    module.Settings = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def test_without_wandb_project_wandb_is_not_imported(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    train.main(["--smoke", "--out", str(tmp_path / "run"), "--batches", "1"])
    assert "wandb" not in sys.modules


def test_wandb_logging(tmp_path, monkeypatch):
    wandb = fake_wandb(monkeypatch)
    out = tmp_path / "myrun"
    train.main(["--smoke", "--out", str(out), "--wandb-project", "proj"])

    (init,) = wandb.inits
    assert init["project"] == "proj" and init["name"] == "myrun" and init["mode"] == "online"
    assert init["settings"] == {"init_timeout": train.WANDB_INIT_TIMEOUT}
    config = json.loads((out / "config.json").read_text())
    assert init["config"] == config and "commit" in init["config"]["meta"]

    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(wandb.run.logged) == len(rows) and wandb.run.finished == 1
    steps = [step for step, _ in wandb.run.logged]
    assert steps == [r["batch"] for r in rows] and steps == sorted(steps)        # W&B needs non-decreasing steps
    for row, (_, data) in zip(rows, wandb.run.logged):
        assert not {"batch", "split", "passes"} & set(data)
        if row["split"] == "val":
            assert all(k.startswith("val/") for k in data)
            assert data["val/win_rate"] == row["win_rate"]
        else:
            assert data["win_rate"] == row["win_rate"] and data["rollout_s"] == row["rollout_s"]
            assert data["pass1/loss"] == row["passes"][0]["loss"]
            assert data["pass_last/kl_old"] == row["passes"][-1]["kl_old"]
            assert data["pass1/kl_old"] == 0.0
            assert not any(k.startswith("val/") for k in data)
    assert {r["split"] for r in rows} == {"train", "val"}


def test_wandb_run_name_and_offline_mode(tmp_path, monkeypatch):
    wandb = fake_wandb(monkeypatch)
    train.main(["--smoke", "--out", str(tmp_path / "run"), "--batches", "1", "--wandb-project", "proj",
                "--wandb-run-name", "named", "--wandb-mode", "offline"])
    (init,) = wandb.inits
    assert init["name"] == "named" and init["mode"] == "offline"


def test_wandb_online_failure_falls_back_to_offline(tmp_path, monkeypatch, capsys):
    wandb = fake_wandb(monkeypatch, fail_online=True)
    out = tmp_path / "run"
    train.main(["--smoke", "--out", str(out), "--batches", "2", "--wandb-project", "proj"])
    assert [i["mode"] for i in wandb.inits] == ["online", "offline"]
    assert "wandb online init failed" in capsys.readouterr().out
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert sum(r["split"] == "train" for r in rows) == 2                        # training went on
    assert len(wandb.run.logged) == len(rows) and wandb.run.finished == 1


def test_wandb_record_flattens_passes():
    record = {"batch": 3, "split": "train", "win_rate": 0.5,
              "passes": [{"loss": 1.0, "kl_old": 0.0}, {"loss": 0.5, "kl_old": 0.1}, {"loss": 0.2, "kl_old": 0.3}]}
    assert train.wandb_record(record) == {"win_rate": 0.5, "pass1/loss": 1.0, "pass1/kl_old": 0.0,
                                          "pass_last/loss": 0.2, "pass_last/kl_old": 0.3}
    assert train.wandb_record({"batch": 0, "split": "val", "win_rate": 0.25}) == {"val/win_rate": 0.25}
    assert isinstance(Observation("x", []).feedback, str)
