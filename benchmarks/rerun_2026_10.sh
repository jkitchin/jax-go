#!/usr/bin/env bash
# Re-run of the README's MINLPLib benchmarks after PRs #76/#77 (October 2026).
# For each set: jaxglobal cold (fresh, empty compile cache), jaxglobal warm (same cache, primed by the cold run), and SCIP once.
# All at a 60 s limit with 2 concurrent jobs. Results go to results/rerun_<set>_{cold,warm,scip}.csv; the 1-min load is recorded per row.
set -e
cd "$(dirname "$0")"
CACHE_ROOT=${CACHE_ROOT:-/tmp/jaxglobal-rerun-cache}
names() { grep -v '^#' "sets/$1.txt" | awk 'NF { print (NF > 1 ? $2 : $1) }' | paste -sd, -; }
for s in ${SETS:-global50 start50 int50 heldout}; do
  n=$(names "$s")
  rm -rf "$CACHE_ROOT/$s"; mkdir -p "$CACHE_ROOT/$s"
  echo "== $s cold";  JAXGLOBAL_CACHE_DIR="$CACHE_ROOT/$s" python run.py --solvers jaxglobal --names "$n" --max-vars 200 --time-limit 60 --jobs 2 --tag "rerun_${s}_cold"
  echo "== $s warm";  JAXGLOBAL_CACHE_DIR="$CACHE_ROOT/$s" python run.py --solvers jaxglobal --names "$n" --max-vars 200 --time-limit 60 --jobs 2 --tag "rerun_${s}_warm"
  echo "== $s scip";  python run.py --solvers scip --names "$n" --max-vars 200 --time-limit 60 --jobs 2 --tag "rerun_${s}_scip"
done
