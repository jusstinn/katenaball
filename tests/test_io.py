import pytest

from katena.io import load_target_match


def test_missing_bundle_raises_cleanly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_target_match(data_dir=str(tmp_path))
