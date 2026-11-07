# Decision log

One entry per model, feature, or decoder decision: what changed, the external (leave-one-match-out)
score, and keep or drop. Every look at the public 5-minute window is logged here too, with its purpose.

| Date | Who | Change | External evidence | Decision |
|---|---|---|---|---|
| 2026-11-07 | David + Claude | Provisional `Match` (`src/katena/match.py`) and synthetic fixture (`src/katena/synth.py`) so D1-D3 can start before J1 lands. Coordinates are centred metres. | n/a | Keep until J1 lands, then reconcile with J1 |
| 2026-11-07 | David + Claude (D3) | Feature store v1 (`src/katena/feats/`, `featurize` / `swap_teams`). 191 float32 columns per graded frame: ball state and gap-aware kinematics 21, players-vs-ball 20 (observed, short-gap interpolated, and margin held over gaps), pressing 4, ball-free team shape 38, team-vs-team 6, stillness 4, centred temporal context 98 (0.5/2/6 s means, min/max, 1/4 s steps). Attack-normalised so home always attacks +x; home/away pairs with a name-driven swap for H/A augmentation. NaN = unobserved, no imputation (LightGBM routes NaN); flags, timers and counts never NaN. Package is `feats`, not `features`, because the firewall deny rule on `features/` matches at any depth. Full-length synthetic match (199k frames, 46 slots): 5.7 s, 0.7 GB peak. | Synthetic smoke only (`scripts/feature_smoke.py`, 5 matches leave-one-out, ATD-like stressors, 41% bursty ball loss): all features acc 0.967 / macroF1 0.966 (ball-missing frames 0.925 / 0.922); ball-free only 0.838 / 0.836 (ball-missing 0.837 / 0.837). Synthetic is easy, so this shows signal, not value. | Keep as the D4 v1 input. Ablate groups and windows on external leave-one-match-out once D1 lands |
