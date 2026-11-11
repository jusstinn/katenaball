#!/usr/bin/env bash
# The registered S1 gate and S2 screen in one unattended run (docs/decision-log.md, 2026-11-10,
# "S1 and S2 (pre-registration)"). External matches only. Resumable: every stage skips the folds it has.
#
#   scripts/night_s12.sh                 # S1 gate (28 folds), S2 screen (8 folds), then S2's gate if the screen passes
#   WAIT_AC=1 scripts/night_s12.sh       # start now, begin when the charger is connected
#   S2_GATE=0 scripts/night_s12.sh       # stop after the S2 screen
#
# Needs the cache of the Nov 9 night run (scripts/night_g12.sh) under KATENA_WORK: the incumbent is read from it.
# It refuses to start if the registration or the code is not committed, on battery, or in Low Power Mode:
# ALLOW_UNREGISTERED=1 and ALLOW_BATTERY=1 override, for rehearsals only.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v caffeinate >/dev/null && [ -z "${_NIGHT_S12_AWAKE:-}" ]; then
  _NIGHT_S12_AWAKE=1 exec caffeinate -i -s "$0" "$@"        # no idle sleep while this runs; the lid must stay open
fi

MAIN="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
export KATENA_EXTERNAL="${KATENA_EXTERNAL:-$MAIN/external_data}"
export KATENA_WORK="${KATENA_WORK:-$MAIN/work/final}"
LOG="$KATENA_WORK/night_s12.log"
fail() { echo "night_s12: $*" >&2; exit 2; }

[ -d "$KATENA_EXTERNAL/canonical" ] || fail "no corpus at $KATENA_EXTERNAL (set KATENA_EXTERNAL)"
[ -d "$KATENA_WORK/pool_d2j" ] && [ -d "$KATENA_WORK/touch" ] || fail "no final cache at $KATENA_WORK (run make model, or set KATENA_WORK)"
for v in g1_gbdt1200 g2_tcn4800t g1_gbdt150 g1_tcn; do
  [ "$(find "$KATENA_WORK/oof/$v" -name '*.npz' 2>/dev/null | wc -l)" -ge 28 ] \
    || fail "the incumbent's cache $v is incomplete under $KATENA_WORK/oof (run scripts/night_g12.sh first)"
done
FREE_GB=$(df -g "$KATENA_WORK" | awk 'NR==2 {print $4}')
[ "$FREE_GB" -ge 5 ] || fail "only ${FREE_GB} GB free under $KATENA_WORK"

if [ "${ALLOW_UNREGISTERED:-0}" != 1 ]; then
  grep -q "S1 and S2 (pre-registration)" docs/decision-log.md || fail "the registration is not in docs/decision-log.md"
  git ls-files --error-unmatch docs/decision-log.md scripts/run_s12.py scripts/night_s12.sh >/dev/null 2>&1 \
    || fail "commit the registration and the scripts first: it is the registration's timestamp"
  git diff --quiet HEAD -- docs/decision-log.md scripts/run_s12.py scripts/run_g12.py scripts/night_s12.sh src \
    || fail "the registration or the code has uncommitted changes; commit first"
fi

if [ "$(uname)" = Darwin ] && [ "${ALLOW_BATTERY:-0}" != 1 ]; then
  if [ "${WAIT_AC:-0}" = 1 ]; then
    until pmset -g batt | grep -q "AC Power"; do sleep 60; done
  fi
  pmset -g batt | grep -q "AC Power" || fail "on battery: connect the charger (WAIT_AC=1 waits for it)"
  [ "$(pmset -g | awk '/lowpowermode/ {print $2}')" != 1 ] \
    || fail "Low Power Mode is on: turn it off in System Settings > Battery (it made a long run 10x slower)"
fi

run() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}
S12=(uv run python scripts/run_s12.py)

echo "=== night_s12 start $(date '+%F %T'); HEAD $(git rev-parse --short HEAD); cache $KATENA_WORK ===" | tee -a "$LOG"
run "${S12[@]}" s1-gbdt s1-gate
run "${S12[@]}" s2-train s2-report --folds 8
PICK="$("${S12[@]}" s2-pick)"
if [ -n "$PICK" ] && [ "${S2_GATE:-1}" = 1 ]; then
  run "${S12[@]}" s2-train s2-report --folds all
else
  echo "[$(date '+%F %T')] no S2 gate (screen: '${PICK:-did not pass}', S2_GATE=${S2_GATE:-1})" | tee -a "$LOG"
fi
echo "=== night_s12 done $(date '+%F %T') ===" | tee -a "$LOG"
