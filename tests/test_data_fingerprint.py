"""scripts/data_fingerprint.py: write and verify on a fake game directory (no tw-make)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_fingerprint", ROOT / "scripts/data_fingerprint.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def fake_games(root: Path, walk=("look", "eat meal"), z8=b"game") -> None:
    manifest = []
    for split, k in (("train", 0), ("valid", 1)):
        (root / split).mkdir(parents=True)
        name = "cook_r1_t1_g1_0000"                                # same name in both splits, as make_games.py does
        (root / split / f"{name}.z8").write_bytes(z8 + bytes([k]))
        (root / split / f"{name}.json").write_text(json.dumps(
            {"metadata": {"uuid": f"tw-cooking-{k}", "walkthrough": list(walk)}}))
        manifest.append({"split": split, "path": f"x/{split}/{name}.z8", "recipe": 1, "take": 1, "go": 1,
                         "skills": ["cut"], "seed": 10 + k, "recipe_seed": 20 + k})
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_write_then_verify_same(tmp_path):
    fake_games(tmp_path / "g")
    out = tmp_path / "fp.jsonl"
    assert fp.main(["write", str(tmp_path / "g"), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["split"], r["seed"], r["recipe_seed"]) for r in rows] == [("train", 10, 20), ("valid", 11, 21)]
    assert fp.main(["verify", str(tmp_path / "g"), str(out)]) == 0


def test_verify_tolerates_bytes_but_not_walkthrough(tmp_path):
    fake_games(tmp_path / "ours")
    out = tmp_path / "fp.jsonl"
    fp.main(["write", str(tmp_path / "ours"), "--out", str(out)])
    expected = [json.loads(line) for line in out.read_text().splitlines()]

    fake_games(tmp_path / "regen", z8=b"other bytes")            # same game, different bytes: accepted
    r = fp.compare(fp.fingerprint(str(tmp_path / "regen")), expected)
    assert r["problems"] == [] and r["sha256_identical"] == 0

    fake_games(tmp_path / "bad", walk=("look",))                   # different walkthrough: rejected
    assert fp.main(["verify", str(tmp_path / "bad"), str(out)]) == 1
