# David: blind brainstorm (do this first, about 45 minutes)

We want your independent thinking before you see our plan, so we can catch what we missed. **Do not open
`docs/PLAN.md` until you finish this.**

## Start your Claude Code with this

Open Claude Code in the repo root and paste:

> Read `CLAUDE.md` and follow its data firewall strictly. Then read `docs/upstream/CHALLENGE.md`,
> `docs/upstream/DATA.md`, `docs/upstream/EVALUATION.md`, `docs/upstream/RULES.md`, and
> `evaluation/possession_metrics.py`. Do NOT open `docs/PLAN.md` or anything in `docs/handoff/` other than this
> file. Spawn independent expert subagents (sports analytics, computer vision, machine learning, and one skeptic)
> to brainstorm how to maximise full-match `spellF1` in 10 days. Have them answer the questions in
> `docs/handoff/DAVID_BRAINSTORM.md`. Then write the synthesis to `docs/brainstorm/david.md`, under 800 words.

## Questions to answer

1. What single insight about `spellF1` would change how you build this? Test it with the scorer, not by reasoning.
2. What does "possession" mean in the public ground truth, compared with the written definition? Which cases look
   surprising? Use this only to understand the concept, since rules forbid tuning on it.
3. What signals tell you which team has the ball when the ball itself is missing?
4. How would you find dead-ball stretches with no ball position?
5. What could most teams be missing? Which approach is underrated, and which popular one is a trap?
6. There's no leaderboard. How would you know a change is real?
7. Which external data would you use, and how well does its idea of possession match ours?
8. What's one idea that sounds a bit crazy but might work? Give a one-day test that would kill it.

## Format of `docs/brainstorm/david.md`

- Top 5 ideas, ranked. For each: expected gain, cost in days, how to test it cheaply, and what would make you drop it.
- 3 things you would refuse to build.
- Your biggest worry.

## Then

Commit `docs/brainstorm/david.md` on a branch, tell Justin, and open `docs/handoff/DAVID_START_HERE.md`.
Your disagreements with our plan are the most useful part, so don't soften them.
