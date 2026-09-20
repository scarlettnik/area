# Quality report

## What was reproduced

The reported production failure was traced to the final independent validation phase: a finalist could be materialized/refined and then be rejected after GeoJSON serialization by `validate_routing.py` (`134: insufficient horizontal utility clearance`). The old orchestrator treated that as a fatal exception.

## Hardened behavior

1. Every candidate intended for export is serialized to GeoJSON and independently validated.
2. If refinement makes a candidate invalid, the untouched finalist is tested next.
3. An invalid finalist is discarded instead of aborting the optimization.
4. Final `best_variant.geojson` and `variants_ranked.geojson` are validated again after final rank mutation.
5. Search uses a configurable 1 cm safety margin around special linear utilities to reduce round-trip boundary/tolerance disagreements; the final validator still uses the exact normative model.
6. ALNS simulated annealing never moves to lower coverage; it may only cross a score barrier at equal coverage.
7. `retie` and `hotspot` destroy operators permit larger structural moves instead of only local consumer reattachment.
8. Multi-seed portfolio reduces dependence on one random trajectory.

## Verification performed in this sandbox

- Modified/new Python files compile successfully.
- `test_hardening.py`: 3/3 tests pass.
- Technical requirements were cross-checked against the repository's extracted current technical appendix and participant clarifications.

## Verification not claimed

The sandbox does not contain a mountable copy of the current corrected GeoJSON repository input, and outbound Git clone/download is unavailable in this runtime. Therefore a full corrected-dataset benchmark was not executed here. The archive deliberately does **not** claim a numerically proven global minimum. `run_portfolio.py` is included to obtain the strongest best-found validated result on the user's local corrected dataset.
