"""Import validated exports as incumbents for another search or depth evaluation."""
import json
from pathlib import Path

import heat_route_builder as h
from validate_routing import incidence, validate_result


def load_seed_trees(input_path, result_path, terminals, candidates, graph):
    # Never trust a previous score, reference, flow, or geometry without auditing
    # it against this input. The current evaluator recomputes the imported trees.
    validate_result(input_path, result_path)
    source = json.loads(Path(input_path).read_text())
    result = json.loads(Path(result_path).read_text())
    existing = [(f['properties'], h.project_geom_coords(f['geometry'])) for f in source['features']
                if f['properties']['object_type'] == 'heat_network']
    ts = {t.id: t for t in terminals}
    original_chambers = {h.input_key(r.input_id if r.input_id is not None else r.existing_object_id): r
                         for r in candidates if r.is_existing_chamber}
    groups = {}
    for feature in result['features']:
        groups.setdefault(str(feature['properties']['variant_id']), []).append(feature)
    trees = []
    for variant_id, features in groups.items():
        nodes = {tid: t.cell for tid, t in ts.items()}
        nodes.update({key: r.cell for key, r in original_chambers.items()})
        roots, edges = {}, set()
        summary = next(f['properties'] for f in features if f['properties']['object_type'] == 'variant_summary')
        for f in features:
            p = f['properties']
            if p['object_type'] not in {'heat_chamber', 'technical_node'}:
                continue
            utm = h.project_geom_coords(f['geometry'])
            point = graph.coord_transform.forward(utm)
            cell = graph.node(point)
            nodes[h.input_key(p['id'])] = cell
            old_degree, old_diameter = incidence(utm, existing)
            if p['object_type'] == 'heat_chamber' and old_degree:
                matching = next((r for r in candidates if not r.is_existing_chamber and h.dist(r.point, point) < .02), None)
                if matching is None:
                    owner = next(ep for ep, line in existing if any(h.point_segment_distance(utm, a, b) < .02
                                                                 for a, b in zip(line, line[1:])))
                    matching = h.TieCandidate(f'seed_{variant_id}_{p["id"]}', point, h.input_key(owner['id']),
                                              'heat_network', old_diameter, False, cell, old_degree, owner['id'])
                    matching.nearby_chambers = tuple((r.existing_object_id, r.existing_degree) for r in original_chambers.values()
                                                    if h.dist(r.point, point) <= 10.)
                    candidates.append(matching)
                nodes[h.input_key(p['id'])] = matching.cell
                roots[matching.cell] = matching
        for f in features:
            p = f['properties']
            if p['object_type'] != 'heat_network':
                continue
            ends = [h.input_key(p[key]) for key in ('start_node_id', 'end_node_id')]
            points = h.project_geom_coords(f['geometry'])
            cells = [nodes[ends[0]], *[graph.node(graph.coord_transform.forward(pt)) for pt in points[1:-1]], nodes[ends[1]]]
            for a, b in zip(cells, cells[1:]):
                if a != b:
                    graph.connect(a, b)
                    edges.add(h.edge_key(a, b))
            for end in ends:
                if end in original_chambers:
                    root = original_chambers[end]
                    roots[root.cell] = root
        for cell in roots:
            graph.tie_cells.add(cell)
            graph.access_nodes.add(cell)
        missing = {h.input_key(tid) for tid in summary['unconnected_oks_ids']}
        trees.append(h.BuiltTree(list(roots.values()), edges, set(ts) - missing, missing))
    # Extra routes can improve prior answers to shortest-path queries.
    graph.route_cache.clear()
    return trees
