"""Generate TextWorld cooking games with tw-make, one directory per split.

TextWorld's cooking food splits (train / valid / test) keep ingredients disjoint across splits. Difficulty per game is
drawn from --recipe / --take / --go choices with a seeded RNG, so the whole set is reproducible.

  python scripts/make_games.py --out data/games --train 300 --valid 40 --test 80
  python scripts/make_games.py --out data/games_L0 --name L0 --recipe 1 --take 1 --go 1 --skills     # no extra skills
"""
import argparse
import json
import random
import subprocess
from pathlib import Path


SKILLS = ("open", "cook", "cut")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/games")
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--valid", type=int, default=40)
    ap.add_argument("--test", type=int, default=80)
    ap.add_argument("--recipe", type=int, nargs="+", default=[1, 2, 3], help="ingredients in the recipe")
    ap.add_argument("--take", type=int, nargs="+", default=[1, 2, 3], help="ingredients to find")
    ap.add_argument("--go", type=int, nargs="+", default=[1, 6, 9], help="number of rooms")
    ap.add_argument("--skills", nargs="*", choices=SKILLS, default=list(SKILLS),
                    help="skills a game may require (doors to open, cooking, cutting); pass no value for none")
    ap.add_argument("--name", default="cook", help="file name prefix")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    skills = [s for s in SKILLS if s in args.skills]           # fixed order, whatever the command line order

    rng = random.Random(args.seed)
    manifest = []
    for split, n in (("train", args.train), ("valid", args.valid), ("test", args.test)):
        directory = Path(args.out) / split
        directory.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            recipe = rng.choice(args.recipe)
            take = min(rng.choice(args.take), recipe)
            go = rng.choice(args.go)
            seed, recipe_seed = rng.randrange(2 ** 31), rng.randrange(2 ** 31)
            path = directory / f"{args.name}_r{recipe}_t{take}_g{go}_{k:04d}.z8"
            subprocess.run(["tw-make", "tw-cooking", "--recipe", str(recipe), "--take", str(take), "--go", str(go),
                            *(f"--{s}" for s in skills), "--split", split, "--seed", str(seed),
                            "--recipe-seed", str(recipe_seed), "--output", str(path), "-f"],
                           check=True, capture_output=True)
            manifest.append({"split": split, "path": str(path), "recipe": recipe, "take": take, "go": go,
                             "skills": skills, "seed": seed, "recipe_seed": recipe_seed})
        print(f"{split}: {n} games in {directory}")
    (Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
