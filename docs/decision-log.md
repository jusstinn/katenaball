# Decision log

One entry per model, feature, or decoder decision: what changed, the external (leave-one-match-out)
score, and keep or drop. Every look at the public 5-minute window is logged here too, with its purpose.

| Date | Who | Change | External evidence | Decision |
|---|---|---|---|---|
| 2026-11-07 | David + Claude | Provisional `Match` (`src/katena/match.py`) and synthetic fixture (`src/katena/synth.py`) so D1-D3 can start before J1 lands. Coordinates are centred metres. | n/a | Keep until J1 lands, then reconcile with J1 |
