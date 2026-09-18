#!/usr/bin/env python3
"""Run the sparse visibility + topology optimizer and validate every export."""
import argparse
import json
from pathlib import Path
import time

import heat_route_builder as h
from routing_depth import clearance, transformed_objects
from routing_search import Search, structure
from routing_visibility import VisibilityGraph
from validate_routing import validate_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="!!!_Датасет.geojson")
    parser.add_argument("--output-dir", default="experiments/visibility-beam")
    parser.add_argument("--search-seconds", type=float, default=180.)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--routes", type=int, default=3)
    parser.add_argument("--turn-penalty-m", type=float, default=8.)
    args = parser.parse_args()
    if args.search_seconds <= 0 or args.beam_width < 1 or args.routes < 1:
        parser.error("Search budget, width and route count must be positive")
    started = time.monotonic()
    terminals, candidates, obstacles, all_points, meta = h.read_input(args.input)
    crossing = meta.pop("crossing_objects")
    existing = meta.pop("existing_network")
    diameter = h.select_diameter(sum(t.flow_tph for t in terminals))
    for o in obstacles:
        o.clearance = clearance(o.kind, diameter)
        o.bbox = h.expand_bbox(h.bbox([p for r in o.rings for p in r]), o.clearance)
    origin = tuple(sum(t.point[k] for t in terminals) / len(terminals) for k in (0, 1))
    transform = h.CoordinateTransform(origin, h.estimate_city_axis(obstacles))
    terminals, candidates, obstacles, _ = h.apply_coordinate_transform(terminals, candidates, obstacles, all_points, transform)
    print("Preparing visibility graph", flush=True)
    graph = VisibilityGraph(terminals, candidates, obstacles, transform)
    graph.crossing_objects = transformed_objects(crossing, transform.forward)
    graph.existing_network = existing
    prepared = time.monotonic() - started
    print(f"Graph: {len(graph.extra_points)} nodes, {sum(map(len, graph.graph.values())) // 2} edges in {prepared:.2f}s", flush=True)
    search = Search(terminals, candidates, graph, args.search_seconds, args.beam_width, args.routes, args.turn_penalty_m)
    finalists = search.run()
    diagnostics = search.report()
    selected, families = [], set()
    for finalist in finalists:
        tree = h.refine_geometry(finalist.tree, terminals, graph)
        family = structure(tree, terminals)
        if family in families:
            continue
        families.add(family)
        variant = h.materialize_variant(f"visibility_{len(selected) + 1}", "visibility + beam", tree, terminals, graph, meta)
        variant.summary["optimality"] = "Best found by bounded visibility topology search and exact engineering evaluation; global optimality is not proven"
        selected.append(variant)
        if len(selected) == 3:
            break
    selected.sort(key=h.variant_key)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for rank, variant in enumerate(selected, 1):
        variant.summary["rank"] = rank
        path = output / f"{variant.id}.geojson"
        h.write_geojson(str(path), variant.features)
        reports.append(dict(variant_id=variant.id, **validate_result(args.input, path)))
    h.write_geojson(str(output / "best_variant.geojson"), selected[0].features)
    h.write_geojson(str(output / "variants_ranked.geojson"), [f for v in selected for f in v.features])
    meta["terminal_building_access"] = "Original coordinates; straight own-building entry only"
    meta.update(algorithm="visibility-beam", preparation_seconds=round(prepared, 3),
                total_seconds=round(time.monotonic() - started, 3), existing_network_status=existing.status(),
                graph_nodes=len(graph.extra_points), graph_edges=sum(map(len, graph.graph.values())) // 2,
                search_budget_seconds=args.search_seconds, beam_width=args.beam_width, route_alternatives=args.routes)
    (output / "summary.json").write_text(json.dumps(dict(meta=meta, variants=[v.summary for v in selected]), ensure_ascii=False, indent=2))
    (output / "search.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    (output / "validation.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    h.write_gis_friendly_layers(str(output), selected)
    from routing_preview import write_preview
    write_preview(args.input, str(output / "best_variant.geojson"), str(output / "preview.svg"))
    print(json.dumps([v.summary for v in selected], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
