"""Integration tests for the standalone chemistry CLI boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "chemistry" / "run_chemistry_evaluation.py"
MECHANISM = REPO_ROOT / "data" / "examples" / "chemistry" / "exothermic_ab.yaml"


def test_cli_writes_versioned_standalone_evaluation(tmp_path: Path) -> None:
    output = tmp_path / "chemistry" / "evaluation.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mechanism",
            str(MECHANISM),
            "--temperature-k",
            "350",
            "--concentration",
            "A=100",
            "--concentration",
            "B=50",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["contract_version"] == "chemistry_evaluation_v1"
    assert payload["mechanism_provenance"]["version"] == "1.0"
    assert len(payload["mechanism_provenance"]["sha256"]) == 64
    assert payload["forward_reaction_rates_mol_per_m3_s"]["R1"] > 0.0
    assert payload["reverse_reaction_rates_mol_per_m3_s"]["R1"] == 0.0
    assert payload["species_creation_rates_mol_per_m3_s"]["C"] > 0.0
    assert payload["species_destruction_rates_mol_per_m3_s"]["A"] > 0.0
    assert payload["reaction_heat_release_w_per_m3"]["R1"] > 0.0
    assert payload["heat_release_w_per_m3"] > 0.0
    assert payload["diagnostics"]["max_abs_mass_balance_residual_kg_per_m3_s"] == 0.0
    assert json.loads(completed.stdout) == payload


def test_cli_rejects_kinetic_only_heat_claim() -> None:
    mechanism = (
        REPO_ROOT
        / "data"
        / "examples"
        / "chemistry"
        / "esterification_kinetics.yaml"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mechanism",
            str(mechanism),
            "--temperature-k",
            "343.15",
            "--concentration",
            "C2H4O2=10",
            "--concentration",
            "C2H5OH=10",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires dH" in completed.stderr


def test_cli_requires_explicit_flag_for_kinetic_only_result() -> None:
    mechanism = (
        REPO_ROOT
        / "data"
        / "examples"
        / "chemistry"
        / "esterification_kinetics.yaml"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mechanism",
            str(mechanism),
            "--temperature-k",
            "343.15",
            "--concentration",
            "C2H4O2=10",
            "--concentration",
            "C2H5OH=10",
            "--allow-missing-enthalpy",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["heat_release_w_per_m3"] is None
    assert payload["reaction_heat_release_w_per_m3"] is None
    assert payload["diagnostics"]["missing_enthalpy_reactions"] == ["r1"]
