#!/bin/bash
# One cached probing run: <model l2|l3|l4> <model seed> <probe hidden_dim> [extra hydra overrides...]
# Probe seed stays 42 like the official callback. Retries up to 3 times (see "Notes" in RESULTS.md).
set -u
: "${HJEPA_HOME:?set HJEPA_HOME to the directory holding the H-JEPA data and checkpoints}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
m=$1; s=$2; h=$3; shift 3
OUT=${OUT:-$REPO/probing/results/E/${m}_s${s}_h${h}}
mkdir -p "$(dirname "$OUT")"
CK=$HJEPA_HOME/ckpts/fourroom/fourroom_hjepa_$m/seed$s/fourroom_hjepa_${m}_object.ckpt
cd "$REPO/h_jepa"
for try in 1 2 3; do
  [ -f "$OUT/metrics.yaml" ] && exit 0
  echo "$(date +%T) start $m s$s h$h try$try"
  python main_probing_cached.py --config-name fourroom policy="$CK" +output_dir="$OUT" +probe_hidden_dim="$h" "$@" > "$OUT.try$try.log" 2>&1
  rc=$?
  echo "$(date +%T) end $m s$s h$h try$try rc=$rc"
  [ $rc -eq 0 ] && [ -f "$OUT/metrics.yaml" ] && exit 0
done
exit 1
