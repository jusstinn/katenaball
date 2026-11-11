# katenaball: More than a Hack 2027, Phase 1

Label every frame of one football match H/A/D. Ranked only on `spellF1` against a private ground
truth. Rules and metric: `docs/upstream/`. The scorer in `evaluation/` is the specification.

## Data firewall (competition terms; breaking this can disqualify the team)

The terms forbid making the videos, datasets, or "any extracts thereof" available to a hosted LLM.
Claude Code is a hosted LLM. So:

- **Never read, print, or display anything from `data/`**: no `cat`, `head`, `Read`, notebook output,
  frame images, sample rows, or crops. This includes the ATD, homography, tagging, and videos.
- **Never print frame-level values derived from the match**: per-frame features, tracks, or predicted
  spell lists. Scripts run on the match may log only aggregates: counts, rates, durations, runtimes,
  and shapes.
- **Write and debug against synthetic fixtures or external matches.** The format is documented in
  `docs/upstream/DATA.md` and `starter/`, so there is no need to look at the real file.
- **A human checks the real data** with local tools. If a real-data bug needs diagnosing, ask the human
  to describe the symptom. Do not ask them to paste rows.
- Never send match data to a hosted service: no hosted LLMs, vision APIs, Supabase, Vercel, or public
  buckets. Whether a rented GPU is acceptable is an open question for the organisers.
- `evaluation/gt/*_public*` and the docs are public, so reading them is fine.

## Competition rules that shape the code

- **The public 5-minute GT is a pipeline check only.** Do not train on it, tune on it, or choose
  between two versions with it. Model selection comes from leave-one-match-out on external matches.
- **No hand labels or hand corrections on this match.** No rules or constants tied to specific frames
  of it.
- **Train, fit and tune on external data only** (organisers' clarification of 2026-11-03, which overrides the
  documentation; `docs/decision-log.md`, 2026-11-10). No pseudo-labels on this match, and nothing fitted or
  calibrated on its tracking, its video or its ground truth. The tracking, the Smart Tagging and the video
  may be inputs at inference.
- **Nobody checks our output against the video.** Automatic checks may use only the provided data.
- **Byte-identical reproduction.** One command must regenerate the exact submission from the data
  bundle plus this repo: pin dependencies, fix seeds, run final inference deterministically, and record
  training data in the technical document.
- **Run `python evaluation/validate_submission.py <file>` on the exact file before submitting.**
- **Never commit** match data, frame-level exports, or submission CSVs. `.gitignore` covers these paths.
