#!/bin/bash
# usage: ./run_nb.sh 01-basics   (from the notebooks/ directory)
set -e
n=$1
jupytext --to ipynb src/$n.py -o $n.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 $n.ipynb
