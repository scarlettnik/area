"""Audit exported GeoJSON, independently of the optimizer's in-memory topology."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import heat_route_builder as h
import routing_depth as d


def validate_result(input_path, result_path):
    terminals, _candidates, obstacles, _points, _meta = h.read_input(input_path)
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    features = data["features"]
    summary = next(f["properties"] for f in features if f["properties"]["object_type"] == "variant_summary")
    nodes = {t.id: t.point for t in terminals}
    for adjustment in summary.get("connection_adjustments", []):
        tid = adjustment["terminal_id"]
        assert tid in nodes, ("unknown adjusted terminal", tid)
        original = h.lonlat_to_utm37(*adjustment["original_coordinates"])
        assert h.dist(nodes[tid], original) < .02, ("adjustment original mismatch", tid)
        nodes[tid] = h.lonlat_to_utm37(*adjustment["connection_coordinates"])
    nodes.update({f["properties"]["id"]: h.lonlat_to_utm37(*f["geometry"]["coordinates"][:2])
                  for f in features if f["geometry"] and f["geometry"]["type"] == "Point"})
    roots = {f["properties"]["id"] for f in features if f["properties"]["object_type"] == "tie_in"}
    input_ids = {t.id for t in terminals}
    flow_balance = defaultdict(float)
    parent = {}
    degree = defaultdict(int)
    lines = []
    depths = {}
    building_leads = set()
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
        if summary.get("depth_routing"):
            start, end = props["depth_start"], props["depth_end"]
            assert all(isinstance(v, (int, float)) and math.isfinite(v) and v >= .7 - 1e-6 for v in (start, end))
            assert length > 1e-6, ("empty pipe", props["id"])
            assert abs(start - end) <= .1 * length + 2e-5, ("excessive depth slope", props["id"])
            assert not (min(start, end) < 3 - 1e-6 and max(start, end) > 3 + 1e-6), ("missing depth=3 node", props["id"])
            for node, value in ((a, start), (b, end)):
                if node in depths:
                    assert abs(depths[node] - value) < 2e-5, ("depth discontinuity", node)
                depths[node] = value
            factor = (d.depth_factor(start) + d.depth_factor(end)) / 2
            assert math.isclose(factor, props["depth_factor"], abs_tol=1e-7)
            special = props["special_factor"]
            ids = props["crossing_object_ids"]
            objects = {o.id: o for o in _meta["crossing_objects"]}
            assert len(ids) <= 1 and all(v in objects for v in ids), ("unknown/overlapping crossing", ids)
            assert props["laying_method"] == ("special" if ids else "base")
            assert special == (objects[ids[0]].factor if ids else 1)
            expected_cost = length * h.NEW_COST[props["diameter"]] * factor * special
        else:
            expected_cost = length * h.NEW_COST[props["diameter"]]
        assert math.isclose(expected_cost, props["cost"], abs_tol=30), ("pipe cost", props["id"])
        line_cost += props["cost"]
        total_length += length
        # Check every line against every obstacle. Only the last service lead
        # into its own input building may enter the buffer/footprint.
        for source_obstacle in obstacles:
            required = d.clearance(source_obstacle.kind, props["diameter"])
            obstacle = h.Obstacle(source_obstacle.id, source_obstacle.kind, source_obstacle.rings, required,
                                  h.expand_bbox(h.bbox([p for r in source_obstacle.rings for p in r]), required),
                                  source_obstacle.holes)
            ends = [nodes[node] for node in (a, b) if node in input_ids and obstacle.contains_or_near(nodes[node])
                    and obstacle.kind in d.BUILDINGS and summary.get("building_entry_allowed", False)]
            if not any(obstacle.blocks_segment(p, q) for p, q in zip(coordinates, coordinates[1:])):
                continue
            assert len(ends) == 1, ("obstacle crossing", props["id"], obstacle.id)
            building_leads.add(props["id"])
            endpoint = ends[0]
            lead = coordinates if h.dist(coordinates[0], endpoint) < .02 else coordinates[::-1]
            allowed = min(h.ring_distance(endpoint, ring) for ring in obstacle.rings) + required + 20
            outside = False
            for p, q in zip(lead, lead[1:]):
                if outside:
                    assert not obstacle.blocks_segment(p, q), ("building re-entry", props["id"])
                samples = max(1, math.ceil(h.dist(p, q) / 0.25))
                for i in range(samples + 1):
                    sample = p[0] + (q[0] - p[0]) * i / samples, p[1] + (q[1] - p[1]) * i / samples
                    if obstacle.contains_or_near(sample):
                        assert not outside, ("building re-entry", props["id"])
                        assert h.dist(sample, endpoint) <= allowed, ("long building crossing", props["id"])
                    else:
                        outside = True
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
        assert degree[terminal.id] == 1, ("consumer used as transit", terminal.id)
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
    assert summary["bend_penalty_cost"] == 0, "Unofficial turn bias included in official cost"
    assert math.isclose(summary["score"], .7 * summary["calculated_cost"] / 25_000_000
                        + .3 * summary["length"] / 100, abs_tol=2.1e-6)  # Millimetre length rounding.
    if summary.get("depth_routing"):
        assert all(abs(depths[n] - 3) < 1e-6 for n in roots | (input_ids - missing)), "Endpoint depth differs from 3 m"
        validate_crossing_profiles(lines, roots, input_ids, nodes, _meta["crossing_objects"])
    return {"valid": True, "connected": summary["connected_oks_count"], "terminals": len(terminals),
            "pipes": len(lines), "max_node_degree": max(degree.values(), default=0),
            "building_service_leads": len(building_leads),
            "depth_checks": bool(summary.get("depth_routing")),
            "geometry_checks": "DN envelopes, obstacles, no building transit, no pipe crossings/overlaps, exact shared nodes",
            "cost_rub": summary["calculated_cost"], "length_m": summary["length"]}


def validate_crossing_profiles(lines, roots, terminals, nodes, objects):
    """Reassemble exported pieces and check crossing windows without solving depths again."""
    outgoing = defaultdict(list)
    incoming = {}
    for line in lines:
        props, _ = line
        outgoing[props["start_node_id"]].append(line)
        incoming[props["end_node_id"]] = line
    visited = set()
    for first in lines:
        props, _ = first
        start = props["start_node_id"]
        upstream = incoming.get(start)
        if upstream and len(outgoing[start]) == 1 and upstream[0]["diameter"] == props["diameter"]:
            continue
        chain, current = [], first
        while current[0]["id"] not in visited:
            visited.add(current[0]["id"])
            chain.append(current)
            end = current[0]["end_node_id"]
            if end in roots or end in terminals or len(outgoing[end]) != 1:
                break
            nxt = outgoing[end][0]
            if nxt[0]["diameter"] != current[0]["diameter"]:
                break
            current = nxt
        points = list(chain[0][1])
        for _, p in chain[1:]:
            points.extend(p[1:])
        segment = h.Segment(points, start, end, props["flow_tph"], props["diameter"],
                            h.path_length(points), start, end, 0)
        events = d.segment_passages(0, segment, objects, [nodes[r] for r in roots])
        station = 0.
        plateau_depth = {}
        for part, coords in chain:
            length = h.path_length(coords)
            finish = station + length
            middle = (station + finish) / 2
            active = [e for e in events if e.start - .002 <= middle <= e.end + .002]
            assert sorted(part["crossing_object_ids"]) == sorted(e.obstacle.id for e in active), ("special window mismatch", part["id"])
            for event in events:
                for boundary in (event.start, event.end):
                    assert not station + .002 < boundary < finish - .002, ("missing special boundary node", part["id"])
            for event in active:
                top = part["depth_start"]
                assert abs(top - part["depth_end"]) < 1e-5, ("sloping crossing plateau", part["id"])
                assert ((event.above is not None and top <= event.above + 1e-5)
                        or top >= event.below - 1e-5), ("vertical clearance", part["id"])
                key = event.obstacle.id, event.start
                if key in plateau_depth:
                    assert abs(top - plateau_depth[key]) < 1e-5
                plateau_depth[key] = top
            station = finish
    assert len(visited) == len(lines), "Unvisited/cyclic profile pieces"


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
