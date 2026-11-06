# katenaball

Frame-level ball possession for **More than a Hack 2027, Phase 1**: label every frame of one football match
`H`, `A`, or `D`, ranked on `spellF1`. The pipeline is under construction.

- Rules, data, and metric: [`docs/upstream/`](docs/upstream/)
- Setup for contributors: [`CLAUDE.md`](CLAUDE.md) has the data rules everyone must follow

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Put the match data in `data/` (see [`data/README.md`](data/README.md)). It is gated and must never be committed.

## Attribution

`evaluation/`, `starter/`, `games.py`, and `docs/upstream/` come from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027), under the MIT Licence
in [`LICENSE-upstream`](LICENSE-upstream). The match data is not covered by that licence.
