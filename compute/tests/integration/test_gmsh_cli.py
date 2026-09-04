from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from microfluidics.gmsh import (
    GmshCli,
    GmshCommandError,
    resolve_gmsh_executable,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_MSH = PROJECT_ROOT / "data" / "meshes" / "gmsh" / "t_junction.msh"


def _create_fake_gmsh(tmp_path: Path) -> tuple[Path, Path]:
    helper_path = tmp_path / "fake_gmsh_runner.py"
    helper_path.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import shutil",
                "import sys",
                "from pathlib import Path",
                "",
                "args = sys.argv[1:]",
                "log_path = os.environ.get('FAKE_GMSH_LOG', '').strip()",
                "if log_path:",
                "    Path(log_path).write_text(json.dumps(args), encoding='utf-8')",
                "if '-version' in args or '--version' in args:",
                "    print('4.13.1-fake')",
                "    raise SystemExit(0)",
                "if '--fail' in args:",
                "    print('forced failure', file=sys.stderr)",
                "    raise SystemExit(17)",
                "try:",
                "    output_path = Path(args[args.index('-o') + 1])",
                "except (ValueError, IndexError):",
                "    print('missing -o output', file=sys.stderr)",
                "    raise SystemExit(2)",
                "fixture = Path(os.environ['FAKE_GMSH_SOURCE_MSH'])",
                "output_path.parent.mkdir(parents=True, exist_ok=True)",
                "shutil.copyfile(fixture, output_path)",
                "print(f'generated {output_path}')",
            ]
        ),
        encoding="utf-8",
    )

    if os.name == "nt":
        wrapper_path = tmp_path / "gmsh.cmd"
        wrapper_path.write_text(
            f'@echo off\r\n"{sys.executable}" "{helper_path}" %*\r\n',
            encoding="utf-8",
        )
    else:
        wrapper_path = tmp_path / "gmsh"
        wrapper_path.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{helper_path}" "$@"\n',
            encoding="utf-8",
        )
        wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC)

    return wrapper_path.resolve(), (tmp_path / "gmsh-args.json").resolve()


def test_resolve_gmsh_executable_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_gmsh, _ = _create_fake_gmsh(tmp_path)
    monkeypatch.setenv("GMSH_EXECUTABLE", str(fake_gmsh))

    resolved = resolve_gmsh_executable()

    assert resolved == fake_gmsh


def test_gmsh_cli_reports_external_version(tmp_path: Path) -> None:
    fake_gmsh, log_path = _create_fake_gmsh(tmp_path)
    client = GmshCli(
        fake_gmsh,
        startup_args=("-noenv",),
        env_overrides={
            "FAKE_GMSH_SOURCE_MSH": str(FIXTURE_MSH),
            "FAKE_GMSH_LOG": str(log_path),
        },
    )

    version = client.version()

    assert version == "4.13.1-fake"
    assert json.loads(log_path.read_text(encoding="utf-8")) == ["-noenv", "-version"]


def test_gmsh_cli_can_replace_inherited_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_environment: dict[str, str] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_environment.update(kwargs["env"])  # type: ignore[arg-type]
        return subprocess.CompletedProcess(args[0], 0, stdout="4.15.2\n", stderr="")

    monkeypatch.setenv("LEAKED_USER_SETTING", "must-not-reach-gmsh")
    monkeypatch.setattr(subprocess, "run", fake_run)
    client = GmshCli(sys.executable, environment={"CONTROLLED_SETTING": "yes"})

    assert client.version() == "4.15.2"
    assert captured_environment == {"CONTROLLED_SETTING": "yes"}


def test_generate_mesh_from_geo_source_runs_external_cli(tmp_path: Path) -> None:
    fake_gmsh, log_path = _create_fake_gmsh(tmp_path)
    client = GmshCli(
        fake_gmsh,
        startup_args=("-noenv",),
        env_overrides={
            "FAKE_GMSH_SOURCE_MSH": str(FIXTURE_MSH),
            "FAKE_GMSH_LOG": str(log_path),
        },
    )

    generation = client.generate_mesh_from_geo_source(
        "Point(1) = {0, 0, 0, 1.0};",
        tmp_path / "generated",
        stem="unit_mesh",
        additional_args=["-clscale", "0.5"],
    )

    recorded_args = json.loads(log_path.read_text(encoding="utf-8"))
    assert generation.geo_path.exists()
    assert generation.msh_path.exists()
    assert generation.command_result.returncode == 0
    assert recorded_args[0] == "-noenv"
    assert recorded_args[1].endswith("unit_mesh.geo")
    assert "-3" in recorded_args
    assert "-format" in recorded_args
    assert "msh4" in recorded_args
    assert "-o" in recorded_args
    assert "-clscale" in recorded_args


def test_import_generated_tetra_mesh_from_geo_source_returns_mesh(
    tmp_path: Path,
) -> None:
    fake_gmsh, log_path = _create_fake_gmsh(tmp_path)
    client = GmshCli(
        fake_gmsh,
        env_overrides={
            "FAKE_GMSH_SOURCE_MSH": str(FIXTURE_MSH),
            "FAKE_GMSH_LOG": str(log_path),
        },
    )

    generation, mesh = client.import_generated_tetra_mesh_from_geo_source(
        "Point(1) = {0, 0, 0, 1.0};",
        tmp_path / "imported",
        stem="import_mesh",
    )

    assert generation.msh_path.exists()
    assert mesh.source_path == generation.msh_path
    assert mesh.tetrahedra.shape[1] == 4
    assert mesh.points.shape[1] == 3


def test_gmsh_cli_raises_on_non_zero_exit(tmp_path: Path) -> None:
    fake_gmsh, log_path = _create_fake_gmsh(tmp_path)
    client = GmshCli(
        fake_gmsh,
        env_overrides={
            "FAKE_GMSH_SOURCE_MSH": str(FIXTURE_MSH),
            "FAKE_GMSH_LOG": str(log_path),
        },
    )
    geo_path = tmp_path / "broken.geo"
    geo_path.write_text("Point(1) = {0, 0, 0, 1.0};", encoding="utf-8")

    with pytest.raises(GmshCommandError) as exc_info:
        client.generate_mesh(
            geo_path,
            tmp_path / "broken.msh",
            additional_args=["--fail"],
        )

    assert exc_info.value.returncode == 17
