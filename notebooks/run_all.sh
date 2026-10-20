#!/usr/bin/env bash
# Re-run every notebook in order. 07 runs last: it summarizes the results/NN.json files written by 08-13.
set -e
cd "$(dirname "$0")"
rm -f results/[0-9][0-9].json
for src in src/[0-9][0-9]-*.py; do
  n=$(basename "$src" .py)
  [ "$n" = 07-pyomo-overview ] && continue
  echo "== $n"
  ./run_nb.sh "$n"
done
echo "== 07-pyomo-overview"
./run_nb.sh 07-pyomo-overview
