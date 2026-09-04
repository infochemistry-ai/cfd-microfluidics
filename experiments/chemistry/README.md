# Standalone chemistry experiments

These entrypoints exercise chemistry without importing or invoking the Gmsh,
flow, transport or thermal stages.

From the repository root:

```powershell
uv run --no-sync python experiments/chemistry/run_chemistry_evaluation.py `
  --mechanism data/examples/chemistry/exothermic_ab.yaml `
  --temperature-k 350 `
  --concentration A=100 `
  --concentration B=50
```

The command prints a versioned JSON payload containing forward, reverse and net
reaction rates; species creation, destruction and net source rates; reaction and
total heat release in `W/m^3`; and a deterministic mechanism SHA-256. Use
`--output` to persist the same payload. Kinetic-only mechanisms require the explicit
`--allow-missing-enthalpy` diagnostic flag and return `null` heat release.
