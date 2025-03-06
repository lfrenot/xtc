#!/usr/bin/env bash
set -euo pipefail

outdir="${1-.}"

NOSHOW="${NOSHOW:-}"
STRATEGIES="${STRATEGIES:-" \
          "tile3d " \
          "tile4d tile4dvr tile4dv " \
          "tile7d tile7dv tile7dvr " \
          "tile8d tile8dv tile8dvr" \
          "}"

opts=""
[ -z "$NOSHOW" ] || opts="$opts --no-show"



for s in $STRATEGIES; do
    # match file = results.backend.strategy.trials.csv -> file:backend:X:peak
    args="$(ls "$outdir"/results.*.$s.*.csv 2>/dev/null | sed 's|/\(\([^.]*\)\.\([^.]*\)\.\([^.]*\)\.\([^.]*\)\.csv\)|/\1:\3:X:peak|' || true)"
    [ -n "$args" ] || continue
    (set -x && loop-display $opts --title "Distributions for strategy $s" --output "$outdir/results.$s.png" $args)
done
