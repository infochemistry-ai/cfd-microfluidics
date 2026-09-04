# Compute Test Architecture

Compute tests are organized by the responsibility and cost of the guarantee they
provide. New tests must be placed in exactly one primary zone; pytest applies the
matching marker automatically from the directory name.

## Zones

- `unit/`: fast isolated contracts, path and repository policies, mesh
  primitives, and small numerical operators. No subprocesses or external
  services.
- `integration/`: boundaries between modules, CLI entrypoints, filesystems,
  runner wiring, and stage integration.
- `regression/`: numerical and behavioral contracts whose values or acceptance
  gates must remain stable across solver changes.
- `system/`: complete multi-stage pipelines and platform wrappers. These tests
  may start subprocesses, consume tracked mesh fixtures, or be environment-gated.
- `performance/`: benchmark harness and compile/performance contracts. These are
  not correctness substitutes and are normally scheduled or run manually.

## Placement Decision

1. If the test validates one function or a tiny synthetic numerical primitive,
   use `unit/`.
2. If it validates communication across a module, process, CLI, or tool boundary,
   use `integration/`.
3. If it protects a numerical result, solver invariant, or historical behavior,
   use `regression/`.
4. If it exercises a user-visible workflow across multiple stages, use `system/`.
5. If its primary assertion concerns runtime, compilation, or benchmark behavior,
   use `performance/`.

Do not put `test_*.py` directly under `compute/tests/`. The collection hook and
layout policy test reject unclassified modules.

## Local Commands

```bash
pytest -q compute/tests/unit
pytest -q compute/tests/integration
pytest -q compute/tests/regression
pytest -q -m "not slow" compute/tests
```

Run `system/` and `performance/` explicitly when their environment and runtime
cost are appropriate. Platform requirements remain test-level conditions, such
as `RUN_POWERSHELL_SYSTEM_TESTS=1` on Windows for the PowerShell wrapper and
`RUN_PIPELINE_MESH_REGRESSION=1` for the full reference-mesh pipeline matrix.
