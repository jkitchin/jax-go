#!/bin/bash
# usage: ./run_nb.sh 01-relaxations   (any directory)
# Converts src/<name>.py (jupytext percent format) to <name>.ipynb and executes it in place.
set -e
cd "$(dirname "$0")"
n=$1
jupytext --to ipynb src/$n.py -o $n.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 $n.ipynb
