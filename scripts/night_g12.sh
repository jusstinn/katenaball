#!/usr/bin/env bash
# The registered G1 gate and G2 screen in one unattended run (docs/decision-log.md, 2026-11-09,
# "G1 and G2 (pre-registration)"). External matches only. Resumable: every stage skips the folds it has.
#
#   scripts/night_g12.sh                 # G1 gate (28 folds), G2 screen (8 folds), then G2's gate if a variant passes
#   WAIT_AC=1 scripts/night_g12.sh       # start now, begin when the charger is connected
#   G2_GATE=0 scripts/night_g12.sh       # stop after the G2 screen
#
# The cache defaults to the main checkout's (work/final, built by `make model`); override with KATENA_WORK and
# KATENA_EXTERNAL. It refuses to start if the registration or the code is not committed, on battery, or in
# Low Power Mode: ALLOW_UNREGISTERED=1 and ALLOW_BATTERY=1 override, for rehearsals only.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v caffeinate >/dev/null && [ -z "${_NIGHT_G12_AWAKE:-}" ]; then
  _NIGHT_G12_AWAKE=1 exec caffeinate -i -s "$0" "$@"        # no idle sleep while this runs; the lid must stay open
fi

MAIN="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
export KATENA_EXTERNAL="${KATENA_EXTERNAL:-$MAIN/external_data}"
export KATENA_WORK="${KATENA_WORK:-$MAIN/work/final}"
LOG="$KATENA_WORK/night_g12.log"
fail() { echo "night_g12: $*" >&2; exit 2; }

[ -d "$KATENA_EXTERNAL/canonical" ] || fail "no corpus at $KATENA_EXTERNAL (set KATENA_EXTERNAL)"
[ -d "$KATENA_WORK/pool_d2j" ] && [ -d "$KATENA_WORK/touch" ] || fail "no final cache at $KATENA_WORK (run make model, or set KATENA_WORK)"
FREE_GB=$(df -g "$KATENA_WORK" | awk 'NR==2 {print $4}')
[ "$FREE_GB" -ge 5 ] || fail "only ${FREE_GB} GB free under $KATENA_WORK"

if [ "${ALLOW_UNREGISTERED:-0}" != 1 ]; then
  grep -q "G1 and G2 (pre-registration)" docs/decision-log.md || fail "the registration is not in docs/decision-log.md"
  git ls-files --error-unmatch docs/decision-log.md scripts/run_g12.py scripts/night_g12.sh >/dev/null 2>&1 \
    || fail "commit the registration and the scripts first: it is the registration's timestamp"
  git diff --quiet HEAD -- docs/decision-log.md scripts/run_g12.py scripts/night_g12.sh src \
    || fail "the registration or the code has uncommitted changes; commit first"
fi

if [ "$(uname)" = Darwin ] && [ "${ALLOW_BATTERY:-0}" != 1 ]; then
  if [ "${WAIT_AC:-0}" = 1 ]; then
    until pmset -g batt | grep -q "AC Power"; do sleep 60; done
  fi
  pmset -g batt | grep -q "AC Power" || fail "on battery: connect the charger (WAIT_AC=1 waits for it)"
  [ "$(pmset -g | awk '/lowpowermode/ {print $2}')" != 1 ] \
    || fail "Low Power Mode is on: turn it off in System Settings > Battery (it made the last long run 10x slower)"
fi

run() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}
G12=(uv run python scripts/run_g12.py)

echo "=== night_g12 start $(date '+%F %T'); HEAD $(git rev-parse --short HEAD); cache $KATENA_WORK ===" | tee -a "$LOG"
run "${G12[@]}" g1-gbdt g1-tcn g1-score g1-gate
run "${G12[@]}" g2-train g2-report --folds 8
PICK="$("${G12[@]}" g2-pick)"
if [ -n "$PICK" ] && [ "${G2_GATE:-1}" = 1 ]; then
  run "${G12[@]}" g2-train g2-report --folds all --variants "$PICK"
else
  echo "[$(date '+%F %T')] no G2 gate (pick: '${PICK:-none}', G2_GATE=${G2_GATE:-1})" | tee -a "$LOG"
fi
echo "=== night_g12 done $(date '+%F %T') ===" | tee -a "$LOG"
