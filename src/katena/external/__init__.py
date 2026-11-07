"""External open data (other matches) converted to the shared `Match` format.

Nothing in this package reads the competition match. Sources:

- IDSSE / DFL open data (Bassek et al. 2025, CC BY 4.0): `idsse.py`
- Metrica Sports open sample Game 3 (FIFA/EPTS): `epts.py` (generic loader), `metrica_open.py`
- Label harmonisation towards "controlled" possession: `harmonise.py`
- Download, convert, cache and report: `corpus.py`
"""
from .common import load_match_npz, save_match_npz
from .harmonise import DEFAULT_HARMONISE, harmonise

__all__ = ["harmonise", "DEFAULT_HARMONISE", "save_match_npz", "load_match_npz"]
