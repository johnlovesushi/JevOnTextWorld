import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevtw.config import PolicyConfig  # noqa: E402
from jevtw.policy import LMPolicy  # noqa: E402
from jevtw.testing import ByteTokenizer, tiny_model  # noqa: E402


@pytest.fixture
def tokenizer():
    return ByteTokenizer()


@pytest.fixture
def policy_cfg():
    # " yes" is four byte tokens, so the byte tokenizer reads single-letter answers.
    return PolicyConfig(max_reasoning_tokens=6, history_window=3, yes_text="Y", no_text="N",
                        confidence_prompt="\nWill you win? Y or N:", option_batch_size=4)


@pytest.fixture
def policy(tokenizer, policy_cfg):
    return LMPolicy(tiny_model(seed=0), tokenizer, policy_cfg)


@pytest.fixture(scope="session")
def cooking_game(tmp_path_factory):
    pytest.importorskip("textworld")
    if shutil.which("tw-make") is None:
        pytest.skip("tw-make not available")
    path = tmp_path_factory.mktemp("tw") / "cook.z8"
    proc = subprocess.run(["tw-make", "tw-cooking", "--recipe", "1", "--take", "1", "--go", "1", "--split", "train",
                           "--seed", "1", "--output", str(path), "-f"], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.skip(f"tw-make failed: {proc.stderr[-300:]}")
    return path
