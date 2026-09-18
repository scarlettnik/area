"""Audit exported GeoJSON, independently of the optimizer's in-memory topology."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import heat_route_builder as h


def validate_result(input_path, result_path):
    terminals, _candidates, obstacles, _points, _meta = h.read_input(input_path)
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    features = data["features"]
    summary = next(f["properties"] for f in features if f["properties"]["object_type"] == "variant_summary")
    nodes = {t.id: t.point for t in terminals}
    nodes.update({f["properties"]["id"]: h.lonlat_to_utm37(*f["geometry"]["coordinates"][:2])
                  for f in features if f["geometry"] and f["geometry"]["type"] == "Point"})
    roots = {f["properties"]["id"] for f in features if f["properties"]["object_type"] == "tie_in"}
    input_ids = {t.id for t in terminals}
    flow_balance = defaultdict(float)
    parent = {}
    degree = defaultdict(int)
    lines = []
    line_cost, total_length = 0.0, 0.0
    for feature in features:
        props = feature["properties"]
        if props["object_type"] != "heat_network":
            continue
        a, b = props["start_node_id"], props["end_node_id"]
        coordinates = [h.lonlat_to_utm37(*p[:2]) for p in feature["geometry"]["coordinates"]]
        assert a in nodes and b in nodes, ("dangling node", a, b)
        assert h.dist(coordinates[0], nodes[a]) < 0.02, ("start mismatch", props["id"])
        assert h.dist(coordinates[-1], nodes[b]) < 0.02, ("end mismatch", props["id"])
        assert b not in parent, ("multiple upstream pipes", b)
        parent[b] = a
        degree[a] += 1
        degree[b] += 1
        flow_balance[a] -= props["flow_tph"]
        flow_balance[b] += props["flow_tph"]
        assert h.CAPACITY[props["diameter"]] >= props["flow_tph"] - 0.001
        length = h.path_length(coordinates)
        assert math.isclose(length, props["length"], abs_tol=0.02), ("length mismatch", props["id"])
        assert math.isclose(length * h.NEW_COST[props["diameter"]], props["cost"], abs_tol=3000)
        line_cost += props["cost"]
        total_length += length
        # Check every line against every obstacle. Only the last service lead
        # into its own input building may enter the buffer/footprint.
        for obstacle in obstacles:
            ends = [nodes[node] for node in (a, b) if node in input_ids and obstacle.contains_or_near(nodes[node])
                    and obstacle.kind in {"oks", "oks_existing", "oks_future"}]
            for p, q in zip(coordinates, coordinates[1:]):
                if not obstacle.blocks_segment(p, q):
                    continue
                assert ends, ("obstacle crossing", props["id"], obstacle.id)
                samples = max(1, math.ceil(h.dist(p, q) / 0.25))
                for i in range(samples + 1):
                    sample = p[0] + (q[0] - p[0]) * i / samples, p[1] + (q[1] - p[1]) * i / samples
                    if obstacle.contains_or_near(sample):
                        allowed = min(min(h.ring_distance(end, ring) for ring in obstacle.rings)
                                      + obstacle.clearance + 20 for end in ends)
                        assert min(h.dist(sample, end) for end in ends) <= allowed, ("long building crossing", props["id"])
        lines.append((props, coordinates))
    assert max(degree.values(), default=0) <= 4
    for i, (first, first_points) in enumerate(lines):
        for second, second_points in lines[i + 1:]:
            assert not h.polylines_conflict(first_points, second_points), (
                "unmodeled crossing or duplicated pipe", first["id"], second["id"])
    missing = set(summary["unconnected_oks_ids"])
    for terminal in terminals:
        if terminal.id in missing:
            assert terminal.id not in parent
            continue
        assert math.isclose(flow_balance[terminal.id], terminal.flow_tph, abs_tol=0.002), ("consumer flow", terminal.id)
        seen = set()
        node = terminal.id
        while node not in roots:
            assert node not in seen, ("cycle", node)
            seen.add(node)
            assert node in parent, ("disconnected consumer", terminal.id)
            node = parent[node]
    for node, balance in flow_balance.items():
        if node not in roots and node not in input_ids:
            assert abs(balance) < 0.003, ("junction flow imbalance", node, balance)
    incoming_pipe = {props["end_node_id"]: props for props, _coordinates in lines}
    for props, _coordinates in lines:
        continuous = props["length"]
        upstream = incoming_pipe.get(props["start_node_id"])
        while upstream and upstream["diameter"] == props["diameter"]:
            continuous += upstream["length"]
            upstream = incoming_pipe.get(upstream["start_node_id"])
        assert continuous <= h.MAX_LENGTH[props["diameter"]] + 0.01, ("continuous diameter length", props["id"])
    assert math.isclose(line_cost, summary["construction_cost"], abs_tol=0.2)
    assert math.isclose(total_length, summary["length"], abs_tol=0.1)
    assert len(terminals) - len(missing) == summary["connected_oks_count"]
    components = ("construction_cost", "chamber_construction_cost", "tie_in_cost", "reconstruction_cost",
                  "chamber_reconstruction_cost", "bend_penalty_cost", "unconnected_penalty")
    assert math.isclose(sum(summary[k] for k in components), summary["calculated_cost"], abs_tol=0.1)
    return {"valid": True, "connected": summary["connected_oks_count"], "terminals": len(terminals),
            "pipes": len(lines), "max_node_degree": max(degree.values(), default=0),
            "geometry_checks": "obstacles, pipe crossings/overlaps, shared nodes; own-building leads checked separately",
            "cost_rub": summary["calculated_cost"], "length_m": summary["length"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="!!!_Датасет.geojson")
    parser.add_argument("--result", default="routing_results/best_variant.geojson")
    parser.add_argument("--output", help="Optional JSON validation report path")
    args = parser.parse_args()
    report = json.dumps(validate_result(args.input, args.result), ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    print(report)
