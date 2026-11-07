"""Build or summarise the external corpus (D1). Never touches data/ or the competition match.

    python scripts/external_corpus.py build                 # download + convert + cache
    python scripts/external_corpus.py report                # spell statistics, markdown
    python scripts/external_corpus.py report --min-control-s 3

See src/katena/external/corpus.py and docs/external_data.md.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "evaluation")]

from katena.external.corpus import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
