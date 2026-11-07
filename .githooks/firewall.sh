#!/bin/sh
# Shared by .githooks/pre-commit and .githooks/pre-push: find a Python and run the data firewall
# (scripts/firewall_check.py) with the arguments given. The checker needs only the standard library.
#
# Enable the hooks once per clone (it is a per-repository setting, never global):
#     git config core.hooksPath .githooks

root=$(git rev-parse --show-toplevel) || exit 2

for candidate in "$root/.venv/Scripts/python.exe" "$root/.venv/bin/python" python3 python; do
    # Running it, not just finding it, also skips the Windows "python3" alias that only opens the Store.
    if "$candidate" -c "import sys; sys.exit(sys.version_info < (3, 8))" >/dev/null 2>&1; then
        exec "$candidate" "$root/scripts/firewall_check.py" --repo "$root" "$@"
    fi
done

echo "data firewall: no Python 3.8+ found (looked for .venv, python3, python)." >&2
echo "Run 'uv sync' in this clone, or install Python, then retry." >&2
exit 2
