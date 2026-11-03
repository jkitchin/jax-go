#!/usr/bin/env bash
# Re-run of the ROADMAP's stage-5 scale measurement (results/int328_stage5.csv, 2026-10-17) on current code:
# all 328 MINLPLib integer instances with at most 100 variables (sets/int100.txt), 60 s limit, 2 jobs.
# jaxglobal cold (empty compile cache) and SCIP (pyscipopt; skipped in the stage-5 run).
# Results: results/scale_int100_{cold,scip}.csv, with the 1-min load per row. Resumable (run.py skips done rows).
# Needs: python benchmarks/minlplib/fetch.py --integer --max-vars 100
set -e
cd "$(dirname "$0")"
CACHE=${CACHE:-/tmp/jaxglobal-scale-cache}
n=$(grep -v '^#' sets/int100.txt | awk 'NF { print $1 }' | paste -sd, -)
[ -s results/scale_int100_cold.csv ] || { rm -rf "$CACHE"; mkdir -p "$CACHE"; }
echo "== cold"; JAXGLOBAL_CACHE_DIR="$CACHE" python run.py --solvers jaxglobal --names "$n" --max-vars 100 --time-limit 60 --jobs 2 --tag scale_int100_cold
echo "== scip"; python run.py --solvers scip --names "$n" --max-vars 100 --time-limit 60 --jobs 2 --tag scale_int100_scip
