#!/usr/bin/env python3
"""Check every OKS independently on one corrected-model input.

This is a diagnostic, not a global optimum certificate: it proves that each
consumer has at least one independently valid route under the same graph,
engineering evaluator, GeoJSON round trip, and independent validator used by
`optimize_network.py`.
"""
import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import heat_route_builder as h
from routing_depth import clearance, transformed_objects
from routing_search import Search
from routing_visibility import VisibilityGraph
from validate_routing import validate_result


def check(input_path, neighbors=1, seconds=120.0):
    terminals, candidates, obstacles, all_points, meta = h.read_input(input_path)
    crossing = meta.pop('crossing_objects')
    diameter = min(h.select_diameter(t.flow_tph) for t in terminals)
    for obstacle in obstacles:
        obstacle.clearance = clearance(obstacle.kind, diameter)
        obstacle.bbox = h.expand_bbox(h.bbox([p for ring in obstacle.rings for p in ring]), obstacle.clearance)
    origin = tuple(sum(t.point[k] for t in terminals) / len(terminals) for k in (0, 1))
    transform = h.CoordinateTransform(origin, h.estimate_city_axis(obstacles))
    terminals, candidates, obstacles, _ = h.apply_coordinate_transform(
        terminals, candidates, obstacles, all_points, transform)
    graph = VisibilityGraph(terminals, candidates, obstacles, transform, neighbors=neighbors,
                            dn_aware=True, crossing_objects=transformed_objects(crossing, transform.forward))
    graph.depth_mode = False
    search = Search(terminals, candidates, graph, seconds=seconds, beam_width=1, routes=1,
                    turn_m=8., seed_portfolio=False)
    search.started = time.monotonic()
    search.deadline = search.started + seconds
    graph.deadline = search.deadline
    search.evaluate(h.BuiltTree([], set(), set(), {t.id for t in terminals}))
    roots = {r.cell: r for r in candidates if r.cell is not None}
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for terminal in terminals:
            paths = search.independent_paths(terminal)
            valid = False
            error = None
            length = None
            if paths:
                path = paths[0]
                length = sum(graph.graph[a][b] for a, b in zip(path, path[1:]))
                tree = h.BuiltTree([roots[path[0]]], {h.edge_key(a, b) for a, b in zip(path, path[1:])},
                                   {terminal.id}, {t.id for t in terminals if t.id != terminal.id})
                try:
                    variant = h.materialize_variant('reachability', 'independent feasibility check',
                                                    tree, terminals, graph, meta)
                    variant.summary['rank'] = 1
                    output = Path(directory) / f'{terminal.id}.geojson'
                    h.write_geojson(str(output), variant.features)
                    validate_result(input_path, output)
                    valid = True
                except Exception as exc:  # report exact validator/evaluator reason
                    error = f'{type(exc).__name__}: {exc}'
            rows.append({'input_id': terminal.input_id, 'ports': graph.entry_diagnostics[terminal.id]['ports'],
                         'independently_reachable': valid, 'path_length_m': length, 'error': error})
    return {'terminal_count': len(terminals),
            'individually_reachable_count': sum(row['independently_reachable'] for row in rows),
            'all_individually_reachable': all(row['independently_reachable'] for row in rows),
            'entries': rows,
            'note': ('Independent reachability does not prove that all routes can be combined into one globally '
                     'optimal non-crossing network; it is a regression check for false unreachable classifications.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='Датасет скорректированный.geojson')
    parser.add_argument('--neighbors', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.neighbors < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('neighbors and seconds must be positive')
    report = check(args.input, args.neighbors, args.seconds)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
