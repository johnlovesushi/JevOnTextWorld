"""Fingerprints of a game directory made by scripts/make_games.py, so that another copy can be checked against ours.

    python scripts/data_fingerprint.py write  data/games_C3 --out docs/fingerprints/games_C3.jsonl
    python scripts/data_fingerprint.py verify data/games_C3 docs/fingerprints/games_C3.jsonl

One line per game, from the game's .json (TextWorld's metadata) and the manifest written by make_games.py:
split, file, recipe, take, go, skills, seed, recipe_seed, uuid, walkthrough, sha256 (of the .z8).

tw-make is not byte-reproducible: a game regenerated with the same command line has the same uuid, objects, recipe and
walkthrough, but some sets in its .json and .z8 come out in a different order (Python hash randomisation), so the
sha256 differs. `verify` therefore fails only on a missing / extra game or a different uuid or walkthrough; sha256
matches are counted and reported, not required (they all match on a copy of our files).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

SPLITS = ("train", "valid", "test")
MANIFEST_KEYS = ("recipe", "take", "go", "skills", "seed", "recipe_seed")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(directory: str) -> list[dict]:
    root = Path(directory)
    # file names repeat across splits (the index k restarts in each split), so the key is (split, name)
    manifest = {(g["split"], Path(g["path"]).name): g for g in json.loads((root / "manifest.json").read_text())}
    rows = []
    for split in SPLITS:
        for z8 in sorted((root / split).glob("*.z8")):
            meta = json.loads(z8.with_suffix(".json").read_text())["metadata"]
            m = manifest.get((split, z8.name), {})
            rows.append({"split": split, "file": z8.name, **{k: m.get(k) for k in MANIFEST_KEYS},
                         "uuid": meta["uuid"], "walkthrough": meta["walkthrough"], "sha256": sha256(z8)})
    return rows


def compare(ours: list[dict], theirs: list[dict]) -> dict:
    """Problems (missing / extra games, uuid or walkthrough differences) and the number of identical .z8 files."""
    a = {(r["split"], r["file"]): r for r in ours}
    b = {(r["split"], r["file"]): r for r in theirs}
    problems = [f"missing {s}/{f}" for s, f in sorted(set(b) - set(a))]
    problems += [f"extra {s}/{f}" for s, f in sorted(set(a) - set(b))]
    for key in sorted(set(a) & set(b)):
        for field in ("uuid", "walkthrough"):
            if a[key][field] != b[key][field]:
                problems.append(f"{key[0]}/{key[1]}: {field} differs")
    same = sum(a[k]["sha256"] == b[k]["sha256"] for k in set(a) & set(b))
    return {"games": len(a), "expected": len(b), "problems": problems, "sha256_identical": same}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("games")
    w.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("games")
    v.add_argument("fingerprint")
    args = ap.parse_args(argv)

    if args.cmd == "write":
        rows = fingerprint(args.games)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{len(rows)} games -> {args.out}")
        return 0
    expected = [json.loads(line) for line in Path(args.fingerprint).read_text().splitlines() if line]
    result = compare(fingerprint(args.games), expected)
    for p in result["problems"][:50]:
        print(p)
    print(f"{result['games']} games (expected {result['expected']}), {len(result['problems'])} problems, "
          f"{result['sha256_identical']} byte-identical .z8")
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
