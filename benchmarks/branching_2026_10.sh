#!/usr/bin/env bash
# Re-measure the branching-rule comparison quoted in the README ("Choosing branch_var") on current main.
# For each rule (width, pseudocost, strong):
#   1. the 21 built-in problems (builtin.py: cold and warm solve in a fresh process, no disk cache)
#   2. univariate_sum_10/20, nn_tanh_5x8x2, qcqp_10_5 (synthetic.py)
#   3. MINLPLib start50 and int50, warm (run.py, 60 s, 2 jobs, on the compile caches primed by rerun_2026_10.sh)
# Outputs: results/branch_<rule>_{builtin,synthetic}.json and results/branch_<rule>_{start50,int50}.csv.
set -e
cd "$(dirname "$0")"
CACHE_ROOT=${CACHE_ROOT:-/tmp/jaxglobal-rerun-cache}
SYN=univariate_sum_10,univariate_sum_20,nn_tanh_5x8x2,qcqp_10_5
names() { grep -v '^#' "sets/$1.txt" | awk 'NF { print (NF > 1 ? $2 : $1) }' | paste -sd, -; }
for rule in ${RULES:-width pseudocost strong}; do
  opts="{\"branch_var\": \"$rule\"}"
  [ -f "results/branch_${rule}_builtin.json" ] || { echo "== $rule builtin"; python builtin.py --opts "$opts" --out "results/branch_${rule}_builtin.json"; }
  [ -f "results/branch_${rule}_synthetic.json" ] || { echo "== $rule synthetic"; python synthetic.py --only "$SYN" --opts "$opts" --timeout 600 --out "results/branch_${rule}_synthetic.json"; }
  for s in start50 int50; do
    echo "== $rule $s"
    JAXGLOBAL_OPTS="$opts" JAXGLOBAL_CACHE_DIR="$CACHE_ROOT/$s" python run.py --solvers jaxglobal --names "$(names $s)" --time-limit 60 --jobs 2 --tag "branch_${rule}_$s"
  done
done
echo "BRANCHING DONE"
