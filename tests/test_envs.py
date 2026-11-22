import pytest

from jevtw.envs import KeyChestEnv, TextWorldEnv


def test_keychest_walkthrough_and_partial_observability():
    env = KeyChestEnv()
    for seed in range(6):
        obs = env.reset(seed)
        assert "key" not in obs.text.replace("Find the key", "")       # where the key is stays hidden
        assert "take key" not in obs.commands
        key_in = env._state["key_in"]
        obs = env.step("open chest")
        assert "locked" in obs.text and not obs.done
        obs = env.step(f"open {key_in}")
        assert "the key" in obs.text and "take key" in obs.commands
        env.step("take key")
        obs = env.step("open chest")
        assert obs.done and obs.won and obs.commands == []


def test_keychest_same_seed_same_initial_state_and_loss():
    env = KeyChestEnv()
    env.reset(3)
    first = env._state["key_in"]
    env.reset(3)
    assert env._state["key_in"] == first
    obs = env.step("smash chest")
    assert obs.done and not obs.won
    with pytest.raises(RuntimeError):
        env.step("wait")
    env.reset(0)
    with pytest.raises(ValueError):
        env.step("take key")


def test_textworld_walkthrough(cooking_game):
    import textworld

    infos = textworld.EnvInfos(extras=["walkthrough"])
    raw = textworld.start(str(cooking_game), request_infos=infos)
    walkthrough = raw.reset()["extra.walkthrough"]
    raw.close()

    env = TextWorldEnv(cooking_game)
    obs = env.reset(0)
    assert "cookbook" in env.objective and "Inventory:" in obs.text
    assert "look" not in obs.commands and "inventory" not in obs.commands
    saw_notes = False
    for cmd in walkthrough:
        if not env._keep(cmd):            # pure information commands (e.g. "inventory") are not offered
            continue
        assert not obs.done
        assert cmd in obs.commands, f"walkthrough command {cmd!r} filtered out"
        obs = env.step(cmd)
        assert "score has just gone" not in obs.text
        saw_notes |= "Notes:" in obs.text
    assert obs.done and obs.won and saw_notes
    env.close()


def test_policy_plays_textworld(cooking_game, policy):
    from jevtw.rollout import collect_group

    group = collect_group(policy, lambda: TextWorldEnv(cooking_game), cooking_game.name, 0, [0, 1], max_actions=3)
    assert len(group) == 2 and all(1 <= ep.length <= 3 for ep in group)
    assert all(d.p is not None and d.command for ep in group for d in ep.decisions)
