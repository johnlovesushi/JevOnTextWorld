#!/usr/bin/env bash
# Regenerate the game sets of docs/data.md (C0-C4, L0, L1) with the exact commands we used, then check each one
# against the committed fingerprints. CPU only; each level takes 45-90 min (they can run in parallel).
#
#   bash scripts/make_levels.sh            # all levels
#   bash scripts/make_levels.sh C3 C4      # some levels
#
# Refuses to overwrite an existing data/games_<level>. tw-make is not byte-reproducible (see
# scripts/data_fingerprint.py): a correct copy has 0 problems but usually 0 byte-identical .z8 files.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON=(--train 200 --valid 40 --test 40 --seed 0)
declare -A ARGS=(
  [L0]="--recipe 1 --take 1 --go 1 --skills"
  [L1]="--recipe 1 --take 1 --go 1 --skills open cut"
  [C0]="--recipe 1 --take 1 --go 1 --skills cook cut"
  [C1]="--recipe 1 --take 1 --go 1 --skills open cook cut"
  [C2]="--recipe 2 3 --take 3 --go 1 --skills open cook cut"
  [C3]="--recipe 1 --take 1 --go 6 --skills open cook cut"
  [C4]="--recipe 2 3 --take 3 --go 6 9 --skills open cook cut"
)

levels=("$@")
[ ${#levels[@]} -eq 0 ] && levels=(C0 C1 C2 C3 C4 L0 L1)
for level in "${levels[@]}"; do
  [ -n "${ARGS[$level]+x}" ] || { echo "unknown level $level"; exit 2; }
  out="data/games_$level"
  [ -e "$out" ] && { echo "$out exists, not overwriting"; exit 1; }
  # shellcheck disable=SC2086
  python scripts/make_games.py --out "$out" "${COMMON[@]}" ${ARGS[$level]}
  python scripts/data_fingerprint.py verify "$out" "docs/fingerprints/games_$level.jsonl"
done
