"""Local mesh discovery for the standalone MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microfluidics_contracts import RuntimeSettings

DEFAULT_LOCAL_MESH_ROOT = Path("data") / "meshes"


@dataclass(frozen=True)
class MeshRef:
    key: str
    size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "size": self.size}


class LocalMeshStore:
    """Meshes live on disk and must remain inside the project root."""

    def __init__(self, *, project_root: Path, mesh_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.mesh_root = Path(mesh_root).resolve()

    def list_meshes(self, limit: int) -> list[MeshRef]:
        if limit <= 0 or not self.mesh_root.is_dir():
            return []
        found: list[MeshRef] = []
        for path in sorted(self.mesh_root.rglob("*.msh")):
            try:
                resolved = path.resolve()
                if not resolved.is_relative_to(self.project_root):
                    continue
                found.append(
                    MeshRef(
                        key=path.relative_to(self.project_root).as_posix(),
                        size=resolved.stat().st_size,
                    )
                )
            except (OSError, RuntimeError, ValueError):
                continue
            if len(found) >= limit:
                break
        return found

    def register(self, path: str) -> MeshRef:
        raw = str(path).strip()
        if not raw:
            raise ValueError("Mesh path must not be empty.")
        try:
            candidate = (self.project_root / raw).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("Mesh path is invalid.") from exc
        if not candidate.is_relative_to(self.project_root):
            raise ValueError("Mesh path must stay inside the project root.")
        if candidate.suffix.lower() != ".msh":
            raise ValueError("Mesh path must point at a '.msh' file.")
        if not candidate.is_file():
            raise ValueError(f"Mesh file not found: {path!r}.")
        return MeshRef(
            key=candidate.relative_to(self.project_root).as_posix(),
            size=candidate.stat().st_size,
        )


def build_mesh_store(
    settings: RuntimeSettings,
    project_root: Path,
) -> LocalMeshStore:
    _ = settings
    return LocalMeshStore(
        project_root=project_root,
        mesh_root=Path(project_root) / DEFAULT_LOCAL_MESH_ROOT,
    )
