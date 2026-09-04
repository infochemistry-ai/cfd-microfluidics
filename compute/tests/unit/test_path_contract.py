from __future__ import annotations

from pathlib import Path

import microfluidics.path_contract as pc


def test_all_root_constants_live_under_results_and_use_forward_slashes() -> None:
    root_constants = {
        name: value
        for name, value in vars(pc).items()
        if name.endswith("_REL") and isinstance(value, Path)
    }
    assert root_constants, "Expected path contract constants to be defined."
    for name, value in root_constants.items():
        assert not value.is_absolute(), name
        if name == "RESULTS_ROOT_REL":
            assert value == Path("results"), name
        else:
            assert value.as_posix().startswith("results/"), name
        assert "\\" not in value.as_posix(), name


def test_create_timestamped_run_dir_preserves_naming_contract(
    tmp_path: Path, monkeypatch
):
    class _FixedDatetime:
        @classmethod
        def now(cls):
            from datetime import datetime

            return datetime(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(pc, "datetime", _FixedDatetime)

    first = pc.create_timestamped_run_dir(tmp_path, "TJ")
    second = pc.create_timestamped_run_dir(tmp_path, "TJ")

    assert first.name == "20260102_030405_TJ"
    assert second.name == "20260102_030405_TJ_1"
    assert first.is_dir()
    assert second.is_dir()


def test_normalize_user_path_unix_style_forward_slashes() -> None:
    value = pc.normalize_user_path(r"results\gmsh_import_runs\case")
    if pc.os.name != "nt":
        assert value == Path("results/gmsh_import_runs/case")
        assert "\\" not in str(value)
    else:
        assert value == Path(r"results\gmsh_import_runs\case")
