#!/usr/bin/env bash
# Models B and C (seeds 0, 1), then model-D candidates (seed 0, compared on validation only).
# Runs one at a time so epoch timings are not shared with another job.
set -e
PY=${PY:-python}
run() { name=$1; shift; echo "== $name"; $PY -u -m q2.train "$@" > "logs/$name.log" 2>&1; tail -1 "logs/$name.log"; }
run B_s0 --model B --seed 0
run B_s1 --model B --seed 1
run C_s0 --model C --seed 0
run C_s1 --model C --seed 1
run Dcand_dil2_s0  --model D --seed 0 --dilation 2 --tag cand_dil2
run Dcand_dil3_s0  --model D --seed 0 --dilation 3 --tag cand_dil3
run Dcand_patch8_s0 --model D --seed 0 --patch 8   --tag cand_patch8
