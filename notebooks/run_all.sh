#!/usr/bin/env bash
# Re-run every notebook in order (00 last, since it reads results/*.json).
set -e
cd "$(dirname "$0")"
rm -f results/*.json
for n in 01-basics 02-control-flow 03-ode-models 04-neural-network-surrogates 05-things-pyomo-cannot-express 00-index; do
  echo "== $n"
  ./run_nb.sh "$n"
done
