"""External Gmsh CLI integration that keeps Gmsh as a separate process.

This module intentionally works only with an external ``gmsh`` executable via
standard CLI arguments and file-based inputs/outputs. It does not import the
Gmsh Python SDK or link against Gmsh libraries.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from microfluidics.gmsh.gmsh_mesh_import import import_gmsh_tetra_mesh
from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh

_GMSH_ENV_KEYS = ("GMSH_EXECUTABLE", "GMSH_PATH", "GMSH_BIN")


class GmshCliError(RuntimeError):
    """Base error for external Gmsh CLI failures."""


class GmshExecutableNotFoundError(FileNotFoundError):
    """Raised when the external Gmsh executable cannot be resolved."""


class GmshCommandError(GmshCliError):
    """Raised when an external Gmsh command exits with a non-zero status."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        rendered = " ".join(command)
        message = (
            f"Gmsh command failed with exit code {returncode}: {rendered}\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )
        super().__init__(message)
        self.command = tuple(str(part) for part in command)
        self.returncode = int(returncode)
        self.stdout = str(stdout)
        self.stderr = str(stderr)


@dataclass(frozen=True)
class GmshCommandResult:
    """Result of an external Gmsh CLI invocation."""

    executable: Path
    command: tuple[str, ...]
    working_directory: Path
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class GmshMeshGenerationResult:
    """Artifacts produced by a Gmsh mesh generation run."""

    geo_path: Path
    msh_path: Path
    dimension: int
    mesh_format: str
    command_result: GmshCommandResult


def resolve_gmsh_executable(
    executable: str | Path | None = None,
    *,
    search_paths: Sequence[str | Path] = (),
) -> Path:
    """Resolve an external Gmsh executable without importing Gmsh."""

    candidates: list[str] = []
    if executable is not None and str(executable).strip():
        candidates.append(str(executable).strip())
    for key in _GMSH_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            candidates.append(value)
    candidates.extend(str(path).strip() for path in search_paths if str(path).strip())
    candidates.extend(("gmsh", "gmsh.exe"))

    seen: set[str] = set()
    for raw_candidate in candidates:
        if raw_candidate in seen:
            continue
        seen.add(raw_candidate)
        candidate_path = _resolve_candidate_path(raw_candidate)
        if candidate_path is not None:
            return candidate_path

    searched = ", ".join(candidates)
    raise GmshExecutableNotFoundError(
        "Could not resolve an external Gmsh executable. "
        f"Searched candidates: {searched or '[none]'}. "
        "Install Gmsh separately and pass its path explicitly or through "
        "GMSH_EXECUTABLE/GMSH_PATH/GMSH_BIN."
    )


def _resolve_candidate_path(raw_candidate: str) -> Path | None:
    candidate = Path(raw_candidate)
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
        return None

    which = shutil.which(raw_candidate)
    if which is None:
        return None
    return Path(which).resolve()


def _normalize_cwd(value: str | Path | None) -> Path:
    if value is None:
        return Path.cwd().resolve()
    return Path(value).expanduser().resolve()


def isolated_gmsh_environment(directory: str | Path) -> dict[str, str]:
    """Build a replacement environment that excludes user Gmsh configuration."""

    isolated_directory = Path(directory).expanduser().resolve()
    isolated_directory.mkdir(parents=True, exist_ok=True)
    for file_name in (".gmshrc", ".gmsh-options"):
        (isolated_directory / file_name).write_text("", encoding="utf-8")
    isolated_home = str(isolated_directory)
    environment = {
        "GMSH_HOME": isolated_home,
        "HOME": isolated_home,
        "TMPDIR": isolated_home,
        "TMP": isolated_home,
        "TEMP": isolated_home,
        "LANG": "C",
        "LC_ALL": "C",
    }
    if os.name == "nt":
        for name in ("SystemRoot", "WINDIR", "COMSPEC", "PATHEXT"):
            if name in os.environ:
                environment[name] = os.environ[name]
    return environment


class GmshCli:
    """Thin client for an external Gmsh executable."""

    def __init__(
        self,
        executable: str | Path | None = None,
        *,
        working_directory: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
        env_overrides: Mapping[str, str] | None = None,
        startup_args: Sequence[str | Path] = (),
        search_paths: Sequence[str | Path] = (),
    ) -> None:
        self.executable = resolve_gmsh_executable(
            executable,
            search_paths=search_paths,
        )
        self.working_directory = _normalize_cwd(working_directory)
        self.environment = None if environment is None else dict(environment)
        self.env_overrides = dict(env_overrides or {})
        self.startup_args = tuple(str(arg) for arg in startup_args)

    def version(self) -> str:
        """Return the version string reported by the external Gmsh executable."""

        result = self.run(["-version"])
        for stream in (result.stdout, result.stderr):
            for line in stream.splitlines():
                cleaned = line.strip()
                if cleaned:
                    return cleaned
        return ""

    def run(
        self,
        args: Sequence[str | Path],
        *,
        cwd: str | Path | None = None,
        timeout_seconds: float | None = None,
        check: bool = True,
    ) -> GmshCommandResult:
        """Execute the external Gmsh CLI."""

        run_cwd = _normalize_cwd(cwd) if cwd is not None else self.working_directory
        command = [
            str(self.executable),
            *self.startup_args,
            *(str(arg) for arg in args),
        ]
        env = os.environ.copy() if self.environment is None else self.environment.copy()
        env.update(self.env_overrides)
        completed = subprocess.run(
            command,
            cwd=run_cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if check and completed.returncode != 0:
            raise GmshCommandError(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        return GmshCommandResult(
            executable=self.executable,
            command=tuple(command),
            working_directory=run_cwd,
            returncode=int(completed.returncode),
            stdout=str(completed.stdout),
            stderr=str(completed.stderr),
        )

    def generate_mesh(
        self,
        geo_path: str | Path,
        msh_path: str | Path,
        *,
        dimension: int = 3,
        mesh_format: str = "msh4",
        optimize: bool = False,
        binary: bool = False,
        additional_args: Sequence[str | Path] = (),
        cwd: str | Path | None = None,
        timeout_seconds: float | None = None,
    ) -> GmshMeshGenerationResult:
        """Generate a mesh from a .geo file using the external Gmsh CLI."""

        if dimension not in (1, 2, 3):
            raise ValueError(f"Unsupported Gmsh dimension: {dimension}.")

        resolved_geo = Path(geo_path).expanduser().resolve()
        if not resolved_geo.is_file():
            raise FileNotFoundError(f"Gmsh .geo file not found: {resolved_geo}")

        resolved_msh = Path(msh_path).expanduser().resolve()
        resolved_msh.parent.mkdir(parents=True, exist_ok=True)

        args: list[str | Path] = [
            resolved_geo,
            f"-{dimension}",
            "-format",
            mesh_format,
            "-o",
            resolved_msh,
        ]
        if optimize:
            args.append("-optimize")
        if binary:
            args.append("-bin")
        args.extend(additional_args)

        command_result = self.run(
            args,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            check=True,
        )
        if not resolved_msh.exists():
            raise GmshCliError(
                "Gmsh completed without producing the expected mesh file: "
                f"{resolved_msh}"
            )
        return GmshMeshGenerationResult(
            geo_path=resolved_geo,
            msh_path=resolved_msh,
            dimension=dimension,
            mesh_format=mesh_format,
            command_result=command_result,
        )

    def generate_mesh_from_geo_source(
        self,
        geo_source: str,
        output_dir: str | Path,
        *,
        stem: str = "mesh",
        dimension: int = 3,
        mesh_format: str = "msh4",
        optimize: bool = False,
        binary: bool = False,
        additional_args: Sequence[str | Path] = (),
        timeout_seconds: float | None = None,
    ) -> GmshMeshGenerationResult:
        """Write a .geo source string and generate the corresponding .msh file."""

        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = stem.strip() or "mesh"
        geo_path = resolved_output_dir / f"{safe_stem}.geo"
        msh_path = resolved_output_dir / f"{safe_stem}.msh"
        geo_path.write_text(str(geo_source), encoding="utf-8")
        return self.generate_mesh(
            geo_path,
            msh_path,
            dimension=dimension,
            mesh_format=mesh_format,
            optimize=optimize,
            binary=binary,
            additional_args=additional_args,
            cwd=resolved_output_dir,
            timeout_seconds=timeout_seconds,
        )

    def import_generated_tetra_mesh(
        self,
        geo_path: str | Path,
        msh_path: str | Path,
        *,
        dimension: int = 3,
        mesh_format: str = "msh4",
        optimize: bool = False,
        binary: bool = False,
        additional_args: Sequence[str | Path] = (),
        timeout_seconds: float | None = None,
    ) -> tuple[GmshMeshGenerationResult, ImportedTetraMesh]:
        """Generate a .msh file and import it into the internal tetra mesh contract."""

        generation = self.generate_mesh(
            geo_path,
            msh_path,
            dimension=dimension,
            mesh_format=mesh_format,
            optimize=optimize,
            binary=binary,
            additional_args=additional_args,
            timeout_seconds=timeout_seconds,
        )
        mesh = import_gmsh_tetra_mesh(generation.msh_path)
        return generation, mesh

    def import_generated_tetra_mesh_from_geo_source(
        self,
        geo_source: str,
        output_dir: str | Path,
        *,
        stem: str = "mesh",
        dimension: int = 3,
        mesh_format: str = "msh4",
        optimize: bool = False,
        binary: bool = False,
        additional_args: Sequence[str | Path] = (),
        timeout_seconds: float | None = None,
    ) -> tuple[GmshMeshGenerationResult, ImportedTetraMesh]:
        """Write a .geo string, generate a .msh file, and import it."""

        generation = self.generate_mesh_from_geo_source(
            geo_source,
            output_dir,
            stem=stem,
            dimension=dimension,
            mesh_format=mesh_format,
            optimize=optimize,
            binary=binary,
            additional_args=additional_args,
            timeout_seconds=timeout_seconds,
        )
        mesh = import_gmsh_tetra_mesh(generation.msh_path)
        return generation, mesh


__all__ = [
    "GmshCli",
    "GmshCliError",
    "GmshCommandError",
    "GmshCommandResult",
    "GmshExecutableNotFoundError",
    "GmshMeshGenerationResult",
    "resolve_gmsh_executable",
]
