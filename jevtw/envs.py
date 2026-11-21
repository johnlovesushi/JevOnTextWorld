"""Environments as POMDPs (eqs. 1-3): the agent sees observations o_t and admissible commands, never s_t.

Every environment implements
    reset(seed) -> Observation          s_0 ~ rho_0 for this seed, returns o_0
    step(command) -> Observation        s_{t+1} = f(s_t, a_t), returns o_{t+1}; sets done / won
A group of G rollouts resets the same game with the same seed, so all of them start from the same initial state;
they differ only through the policy's sampling."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Observation:
    text: str
    commands: list[str]                   # admissible commands A_g(s_t), shown to the agent as numbered options
    done: bool = False
    won: bool = False


@dataclass
class KeyChestEnv:
    """A tiny deterministic POMDP for tests: a key is hidden in the drawer or the box (chosen by the seed), and the
    quest is to open the locked chest. Where the key is stays unobserved until a container is opened."""

    objective: str = "Find the key and open the chest."
    _state: dict = field(default_factory=dict)

    def reset(self, seed: int) -> Observation:
        self._state = {"key_in": random.Random(seed).choice(["drawer", "box"]), "open": set(), "has_key": False,
                       "done": False, "won": False}
        return self._observe("You are in a small room with a drawer, a box and a locked chest.")

    def step(self, command: str) -> Observation:
        s = self._state
        if s["done"]:
            raise RuntimeError("episode is over")
        if command not in self._commands():
            raise ValueError(f"inadmissible command {command!r}")
        if command.startswith("open ") and command != "open chest":
            target = command.split()[1]
            s["open"].add(target)
            inside = "the key" if s["key_in"] == target and not s["has_key"] else "nothing"
            text = f"You open the {target}. Inside you see {inside}."
        elif command == "take key":
            s["has_key"] = True
            text = "You take the key."
        elif command == "open chest":
            if s["has_key"]:
                s["done"] = s["won"] = True
                text = "You unlock the chest with the key. You win!"
            else:
                text = "The chest is locked."
        elif command == "smash chest":
            s["done"] = True
            text = "You break the chest and everything inside. You lose."
        else:
            text = "Time passes."
        return self._observe(text)

    def _commands(self) -> list[str]:
        s = self._state
        cmds = [f"open {c}" for c in ("drawer", "box") if c not in s["open"]]
        if s["key_in"] in s["open"] and not s["has_key"]:
            cmds.append("take key")
        return cmds + ["open chest", "smash chest", "wait"]

    def _observe(self, text: str) -> Observation:
        s = self._state
        inventory = "the key" if s["has_key"] else "nothing"
        commands = [] if s["done"] else self._commands()
        return Observation(text=f"{text}\nYou are carrying: {inventory}.", commands=commands, done=s["done"],
                           won=s["won"])


def _clean_feedback(feedback: str) -> str:
    text = feedback.rsplit("\n>", 1)[0] if "\n>" in feedback else feedback   # drop the "> -= Room =-0/1" prompt line
    return re.sub(r"\n\s*\n(\s*\n)+", "\n\n", text).strip()


_SCORE = re.compile(r"\s*Your score has just gone (?:up|down) by \w+ points?\.\s*", re.I)
_INFO_ONLY = ("look", "inventory")


class TextWorldEnv:
    """A compiled TextWorld game (.z8 / .ulx). Game dynamics are deterministic, so `seed` does not change s_0.

    filter_info_commands  drop look / inventory / examine X (except the cookbook): the observation already shows the
                          room and the inventory
    strip_score_text      remove "Your score has just gone up by one point." so that z is the only success signal
    Once the recipe has been read it is kept as notes in later observations (it is needed until the end)."""

    def __init__(self, game_path: str | Path, filter_info_commands: bool = True, strip_score_text: bool = True):
        import textworld  # optional dependency

        self.path = str(game_path)
        self.filter_info_commands = filter_info_commands
        self.strip_score_text = strip_score_text
        infos = textworld.EnvInfos(description=True, inventory=True, admissible_commands=True, objective=True,
                                   won=True, lost=True)
        self._env = textworld.start(self.path, request_infos=infos)
        self.objective = ""
        self._notes = ""

    def reset(self, seed: int = 0) -> Observation:
        gs = self._env.reset()
        self.objective = gs["objective"]
        self._notes = ""
        return self._observe(gs, done=False)

    def step(self, command: str) -> Observation:
        gs, _, done = self._env.step(command)
        return self._observe(gs, done=done or gs["won"] or gs["lost"])

    def close(self):
        self._env.close()

    def _keep(self, cmd: str) -> bool:
        if not self.filter_info_commands:
            return True
        return not (cmd in _INFO_ONLY or (cmd.startswith("examine ") and cmd != "examine cookbook"))

    def _observe(self, gs, done: bool) -> Observation:
        text = _clean_feedback(gs.feedback)
        if self.strip_score_text:
            text = _SCORE.sub("\n", text).strip()
        if "Recipe #" in text:
            self._notes, text = text, "You read the recipe (see notes)."
        if "-=" not in text:                    # the feedback is not a room description: add the room
            text = f"{text}\n\n{_clean_feedback(gs['description'])}".strip()
        parts = [text, f"Inventory: {gs['inventory'].strip()}"]
        if self._notes:
            parts.append(f"Notes:\n{self._notes}")
        commands = [] if done else [c for c in gs["admissible_commands"] if self._keep(c)]
        return Observation(text="\n\n".join(parts), commands=commands, done=done, won=bool(gs["won"]))
