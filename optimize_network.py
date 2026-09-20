#!/usr/bin/env python3
"""Optimize and independently validate heat networks under the corrected LCT model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import heat_route_builder as h
from routing_depth import clearance, transformed_objects
from routing_search import Search, structure
from routing_alns import AdaptiveSearch
from routing_visibility import VisibilityGraph
from validate_routing import validate_result


def solve(args, mode):
    started = time.monotonic()
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))
                   if not p.name.startswith('test_')}
    terminals, candidates, obstacles, all_points, meta = h.read_input(args.input)
    if not terminals or not candidates:
        raise ValueError('Input must contain connection points and an existing network')
    crossing = meta.pop('crossing_objects')
    diameter = h.select_diameter(min(sum(t.flow_tph for t in terminals), h.CAPACITY[1400])) if args.conservative_dn else min(h.select_diameter(t.flow_tph) for t in terminals)
    for obstacle in obstacles:
        obstacle.clearance = clearance(obstacle.kind, diameter)
        obstacle.bbox = h.expand_bbox(h.bbox([p for ring in obstacle.rings for p in ring]), obstacle.clearance)
    origin = tuple(sum(t.point[k] for t in terminals) / len(terminals) for k in (0, 1))
    transform = h.CoordinateTransform(origin, h.estimate_city_axis(obstacles))
    terminals, candidates, obstacles, _ = h.apply_coordinate_transform(terminals, candidates, obstacles, all_points, transform)
    print(f'Preparing {mode} visibility graph', flush=True)
    graph = VisibilityGraph(terminals, candidates, obstacles, transform, neighbors=args.neighbors,
                            dn_aware=not args.conservative_dn,
                            crossing_objects=transformed_objects(crossing, transform.forward))
    graph.depth_mode = mode == 'depth'
    prepared = time.monotonic() - started
    edge_count = sum(map(len, graph.graph.values())) // 2
    print(f'Graph: {len(graph.extra_points)} nodes, {edge_count} edges in {prepared:.2f}s', flush=True)
    for entry in graph.entry_diagnostics.values():
        if entry['obstruction_witnesses']:
            print(f"Input {entry['input_id']}: closest straight entry blocked by its own building", flush=True)
    kwargs = dict(seconds=args.search_seconds, beam_width=args.beam_width, routes=args.routes,
                  turn_m=args.turn_penalty_m, seed_portfolio=not args.no_seed_portfolio)
    from routing_seed import load_seed_trees
    kwargs['initial_trees'] = [tree for path in args.warm_start
                              for tree in load_seed_trees(args.input, path, terminals, candidates, graph)]
    search_type = AdaptiveSearch if args.solver == 'alns' else Search
    if args.solver == 'alns': kwargs.update(seed=args.seed, iterations=args.iterations)
    search = search_type(terminals, candidates, graph, **kwargs)
    finalists = search.run()
    diagnostics = search.report()
    diagnostics['building_entries'] = list(graph.entry_diagnostics.values())
    output = Path(args.output_dir) / mode
    output.mkdir(parents=True, exist_ok=True)
    search.best.summary['rank'] = 1
    h.write_geojson(str(output / 'search_best.geojson'), search.best.features)
    (output / 'search.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + '\n')
    graph.deadline = math.inf
    best_coverage = max(len(v.tree.connected_terminal_ids) for v in finalists)
    finalists = [v for v in finalists if len(v.tree.connected_terminal_ids) == best_coverage]
    selected, families = [], set()
    refinement_deadline = time.monotonic() + args.refine_seconds
    print(f'Refining {min(len(finalists), args.refine_candidates)} finalists; coverage {best_coverage}/{len(terminals)}', flush=True)
    for index, finalist in enumerate(finalists[:args.refine_candidates]):
        tree = finalist.tree if args.no_refine else h.refine_geometry(finalist.tree, terminals, graph, deadline=refinement_deadline)
        trial = h.materialize_variant('trial', 'refined', tree, terminals, graph, meta)
        if h.search_key(trial) > h.search_key(finalist): tree = finalist.tree
        family = structure(tree, terminals)
        if family in families: continue
        families.add(family)
        label = f"{'conservative' if args.conservative_dn else 'DN-aware'} visibility / {args.solver}"
        variant = h.materialize_variant(f'{mode}_v{index + 1}', label, tree, terminals, graph, meta)
        variant.summary['unconnected_explanations'] = [entry for tid, entry in graph.entry_diagnostics.items()
                                                      if tid in tree.unconnected_terminal_ids]
        selected.append(variant)
    selected.sort(key=h.variant_key)
    selected = selected[:args.max_variants]
    reports = []
    for rank, variant in enumerate(selected, 1):
        variant.summary['rank'] = rank
        path = output / f'{variant.id}.geojson'
        h.write_geojson(str(path), variant.features)
        report = validate_result(args.input, path)
        reports.append(report)
    h.write_geojson(str(output / 'best_variant.geojson'), selected[0].features)
    h.write_geojson(str(output / 'variants_ranked.geojson'), [f for v in selected for f in v.features])
    validate_result(args.input, output / 'variants_ranked.geojson')
    meta.update(algorithm=args.solver, mode=mode, input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        python_version=platform.python_version(), preparation_seconds=round(prepared, 3),
        total_seconds=round(time.monotonic() - started, 3), graph_nodes=len(graph.extra_points), graph_edges=edge_count,
        arguments=vars(args), source_sha256=code_hashes,
        warm_start_sha256={str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.warm_start},
        search_budget_scope='Search only; preparation/refinement/validation reported separately')
    (output / 'summary.json').write_text(json.dumps(dict(meta=meta, variants=[v.summary for v in selected]), ensure_ascii=False, indent=2) + '\n')
    (output / 'search.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + '\n')
    (output / 'validation.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    h.write_gis_friendly_layers(str(output), selected)
    h.write_report(str(output / 'report.md'), selected, meta, str(output))
    from routing_preview import write_preview
    write_preview(args.input, str(output / 'best_variant.geojson'), str(output / 'preview.svg'))
    for variant in selected:
        s = variant.summary
        print(f"{mode} rank {s['rank']}: {s['connected_oks_count']}/{len(terminals)}, score={s['score']:.9f}, cost={s['calculated_cost']:.2f}, length={s['new_network_length']:.3f}", flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='Датасет скорректированный.geojson')
    parser.add_argument('--output-dir', default='routing_results_corrected')
    parser.add_argument('--mode', choices=['2d', 'depth', 'both'], default='2d')
    parser.add_argument('--solver', choices=['alns', 'beam'], default='alns')
    parser.add_argument('--search-seconds', type=float, default=180.)
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--routes', type=int, default=3)
    parser.add_argument('--neighbors', type=int, default=2)
    parser.add_argument('--turn-penalty-m', type=float, default=8.)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--warm-start', action='append', default=[], metavar='GEOJSON',
                        help='Validate and retain a previous result; may be repeated')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--max-variants', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--refine-candidates', type=int, default=6)
    parser.add_argument('--refine-seconds', type=float, default=60.)
    parser.add_argument('--no-refine', action='store_true')
    parser.add_argument('--no-seed-portfolio', action='store_true')
    parser.add_argument('--conservative-dn', action='store_true', help='Ablation: buffer the graph for total demand')
    args = parser.parse_args()
    if not math.isfinite(args.search_seconds) or args.search_seconds <= 0:
        parser.error('Search budget must be finite and positive')
    if not math.isfinite(args.refine_seconds) or args.refine_seconds < 0:
        parser.error('Refinement budget must be finite and nonnegative')
    if min(args.beam_width, args.routes, args.neighbors, args.iterations, args.refine_candidates) < 1:
        parser.error('Counts must be positive')
    if not math.isfinite(args.turn_penalty_m) or args.turn_penalty_m < 0:
        parser.error('Turn bias must be finite and nonnegative')
    from audit_input import audit_input
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'input_audit.json').write_text(json.dumps(audit_input(args.input), ensure_ascii=False, indent=2) + '\n')
    for mode in ('2d', 'depth') if args.mode == 'both' else (args.mode,): solve(args, mode)


if __name__ == '__main__':
    main()
