# Routing report

Input profile:
- features: 144
- connection points: 17
- tie candidates: 97
- forbidden obstacles: 88
- objective: minimum calculated monetary cost among feasible candidates
- optional routing turn bias = 0.0 m
- bend penalty in ranking = 0.0 rub per bend
- corridor mode: rotated orthogonal city-block grid, axis = -21.909 degrees
- reconstruction propagation: skipped, dataset has no heat_network flow_tph/upstream_object_id

Ranked variants:
1. cost_cheapest: S=12.877982, cost=263,904,938.62 rub, length=1829.548 m, connected=17/17, bends=14, tie-ins=2, unconnected=[]
2. cost_demand: S=17.822129, cost=465,694,727.88 rub, length=1594.226 m, connected=15/17, bends=13, tie-ins=2, unconnected=['4', '12']

Best result:
- cost_cheapest worked best on this case because it connects 17 connection points with the lowest calculated cost among the evaluated topologies. S is reported separately, not used to rank cost-first results.
- File: routing_results_retry/best_variant.geojson

Routing model:
- EPSG:4326 input is projected to EPSG:32637 by an internal UTM conversion.
- OKS polygons are blocked with a 5 m clearance; water with 1 m; railway with 2 m.
- Heading-aware A* compares the cost of a new tie-in with joining an existing branch.
- Search prices include pipe diameter, upstream flow increases, tie-ins and chambers.
- Orthogonal corridor mode disables diagonal cuts and aligns routes to the dominant OKS block axis.
- Leaf branches and complete downstream subtrees are reattached only when full network cost improves.
- Every grid edge is checked against continuous obstacle geometry, not only occupied grid cells.
- Actual connection coordinates are preserved; only short service leads may exit their own building.
- Equal-cost alternatives prefer fewer bends; the default monetary bend surcharge is zero.
- Dominant facade orientation is estimated robustly; final L-shaped shortcuts remove stair steps without increasing cost or crossing other pipes.
- Continuous same-diameter length is tracked across chambers; excessive demand is rejected.
- Branching cells are emitted as new heat chambers; pipe tie-ins also get a new chamber.
- Depth and special crossings are not enabled in this 2D prototype.
- This is a discrete local optimization, not a proof of a global minimum. Grid resolution affects the result.
- Building clearance is a fixed 5 m centerline buffer in this model; full diameter-dependent envelopes are not modeled.
