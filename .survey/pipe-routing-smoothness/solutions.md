# Solutions

## Solutions

1. Heading-aware least-cost routing: include direction in the graph state and add cost when direction changes.
2. Bend-aware ranking: add monetary or normalized score penalties for each final bend.
3. Visibility/corridor graph: build candidate straight segments around obstacle corners and route over fewer edges.
4. Post-simplification only: simplify finished lines; useful, but weaker because it cannot change the selected corridor.

## Frequency Ranking

Most applicable now: heading-aware Dijkstra/A* plus bend-aware ranking. It fits the current stdlib-only prototype and does not require Shapely/GeoPandas.

Next iteration: sparse visibility graph or corridor graph over obstacle-expanded free space. This should further reduce bends and look closer to manual engineering alignments.

## Categories

- Grid search with bend penalties: robust and easy to port to Java.
- Multi-objective pipe routing: length, bend count, clearance, construction energy/cost.
- GIS least-cost corridor: useful for large areas and corridor screening.
- Line simplification: useful final cleanup, not sufficient as the core router.

## Curated Sources

- MDPI Buildings 2025 chiller plant routing paper: notes that traditional A* length-only cost neglects turning, and excessive turns increase resistance and construction cost.
- MDPI Materials 2022 distance-field pipe-routing paper: uses a feasible free-space grid and adds a turn penalty so straight paths are preferred over elbows.
- Springer Operations Research Forum 2023 automatic pipe-routing survey: lists bend count, minimum straight segments, Dijkstra/A* variants, obstacle enlargement, and pipe-shape constraints as common APR techniques.
- Bathyl 2026 GIS infrastructure route screening article: describes weighted least-cost corridors over constraints, with the cost surface and weights carrying the route rationale.

## Key Insight

The important change is to penalize bends inside the path search, not only after generating the route. Post-smoothing can remove vertices, but bend-aware search changes which corridor is selected.

