#!/bin/bash
# All runs reported in RESULTS.md, strictly sequential (two CUDA processes at once crashed on the test machine).
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="bash $REPO/probing/scripts/run_probe.sh"
while read m s h; do
  $RUN $m $s $h < /dev/null || echo "FAILED $m s$s h$h"
done <<JOBS
l4 42 512
l4 43 512
l4 44 512
l2 42 512
l2 43 512
l2 44 512
l3 42 0
l3 42 128
l3 42 512
l3 42 2048
l4 42 128
l4 42 2048
JOBS
# random-encoder baseline: untrained l3 architecture, official MLP512 probe
OUT=$REPO/probing/results/F_random/l3_s42_h512 $RUN l3 42 512 +random_init=true
echo SWEEP_DONE
