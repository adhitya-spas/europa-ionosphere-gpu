Physics Consistency Guardrails

This repository enforces a single source of truth for physical constants and units.

Rules
- Use `physics_constants.py` for C_LIGHT, C_IONO, R_EUROPA. Do not hard-code these elsewhere.
- Use meters (m) for geometry and altitudes. Convert to km only for plotting.
- TEC ↔ Δt formulas must follow:
  - forward: delta_t = (C_IONO / C_LIGHT) * STEC * (1/f1**2 - 1/f2**2)
  - inverse: TEC = (C_LIGHT * delta_t / C_IONO) * (f1**2 * f2**2) / (f2**2 - f1**2)
- Range error from TEC residual at frequency f: range_error = (C_IONO / f**2) * abs(TEC_residual)

Quick checks
- Run the quick sanity script:

```bash
python -m tools.iono_consistency_check
```

- Run unit tests:

```bash
pytest -q
```

If you add new modules that need constants, import them from `physics_constants`.
