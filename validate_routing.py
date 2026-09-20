"""Independently reconstruct and audit corrected-model GeoJSON exports."""
import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path

from shapely import LineString, Point

import heat_route_builder as h
import routing_depth as d
from routing_constraints import entry_departure_allowed


def require(condition, *message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, tolerance, label):
    require(isinstance(actual, (float, int)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=tolerance), label, actual, expected)




def _outer_boundary_distance(geometry, point):
    """Distance to the exterior shell of the polygon component containing point."""
    parts = list(getattr(geometry, 'geoms', [geometry]))
    containing = [part for part in parts if getattr(part, 'geom_type', None) == 'Polygon' and part.covers(point)]
    parts = containing or [part for part in parts if getattr(part, 'geom_type', None) == 'Polygon']
    return min((part.exterior.distance(point) for part in parts), default=math.inf)

def incidence(point, existing):
    touching = [(p, line) for p, line in existing if LineString(line).distance(Point(point)) < .02]
    degree = sum(1 if min(h.dist(point, line[0]), h.dist(point, line[-1])) < .02 else 2 for _, line in touching)
    diameter = max((p['diameter'] for p, _ in touching), default=0)
    return degree, diameter


def validate_result(input_path, result_path):
    source = json.loads(Path(input_path).read_text(encoding='utf-8'))
    data = json.loads(Path(result_path).read_text(encoding='utf-8'))
    require(data.get('type') == 'FeatureCollection', 'Expected FeatureCollection')
    groups = defaultdict(list)
    ids = set()
    for f in data['features']:
        p = f['properties']
        require('id' in p and 'variant_id' in p, 'Missing output identifier')
        require(h.input_key(p['id']) not in ids, 'Duplicate output identifier', p['id'])
        ids.add(h.input_key(p['id']))
        groups[str(p['variant_id'])].append(f)
    require(1 <= len(groups) <= 3, 'Expected one to three variants')
    reports = [_validate_variant(source, fs) for fs in groups.values()]
    summaries = [next(f['properties'] for f in fs if f['properties']['object_type'] == 'variant_summary') for fs in groups.values()]
    ranked = sorted(summaries, key=lambda s: s['rank'])
    require(len({s['rank'] for s in ranked}) == len(ranked), 'Duplicate ranks')
    require(all(a['score'] <= b['score'] + 1e-9 for a, b in zip(ranked, ranked[1:])), 'Rank contradicts score')
    require(len({s['mode'] for s in summaries}) == 1, 'Mixed calculation modes')
    return reports[0] if len(reports) == 1 else {'valid': True, 'variants': reports}


def _validate_variant(source, features):
    input_features = {h.input_key(f['properties']['id']): f for f in source['features']}
    require(len(input_features) == len(source['features']), 'Ambiguous input IDs')
    terminals = {key: f for key, f in input_features.items() if f['properties']['object_type'] == 'oks_connection_point'}
    chambers = {key: f for key, f in input_features.items() if f['properties']['object_type'] == 'heat_chamber'}
    existing = [(f['properties'], h.project_geom_coords(f['geometry'])) for f in source['features']
                if f['properties']['object_type'] == 'heat_network']
    summaries = [f['properties'] for f in features if f['properties']['object_type'] == 'variant_summary']
    require(len(summaries) == 1, 'One summary required')
    summary = summaries[0]
    depth_mode = summary.get('mode') == 'depth'
    require(summary.get('mode') in {'2d', 'depth'}, 'Unknown mode')
    require(not summary.get('connection_adjustments'), 'Input endpoints moved')
    nodes = {key: h.project_geom_coords(f['geometry']) for key, f in (terminals | chambers).items()}
    types = {key: f['properties']['object_type'] for key, f in (terminals | chambers).items()}
    output_nodes, lines, adjacency = {}, [], defaultdict(list)
    for f in features:
        p = f['properties']; kind = p['object_type']; key = h.input_key(p['id'])
        require(kind in {'heat_network', 'heat_chamber', 'technical_node', 'variant_summary'}, 'Unexpected output type', kind)
        if kind == 'variant_summary':
            require(f['geometry'] is None, 'Summary geometry must be null')
        elif kind in {'heat_chamber', 'technical_node'}:
            require(key not in input_features, 'Output ID shadows input', key)
            require(f['geometry']['type'] == 'Point', 'Node must be a Point')
            nodes[key] = h.project_geom_coords(f['geometry']); types[key] = kind; output_nodes[key] = p
        else:
            require(f['geometry']['type'] == 'LineString', 'Pipe must be LineString')
            points = h.project_geom_coords(f['geometry'])
            require(len(points) >= 2 and all(math.isfinite(v) for xy in points for v in xy), 'Invalid coordinates')
            lines.append((p, points))
    for i, (p, points) in enumerate(lines):
        a, b = h.input_key(p['start_node_id']), h.input_key(p['end_node_id'])
        require(a in nodes and b in nodes and a != b, 'Dangling/identical endpoints', p['id'])
        for key, point in ((a, points[0]), (b, points[-1])):
            require(h.dist(nodes[key], point) < .02, 'Endpoint mismatch', p['id'], key)
            if key in input_features:
                raw = input_features[key]['properties']['id']
                value = p['start_node_id'] if key == a else p['end_node_id']
                require(type(raw) is type(value) and raw == value, 'Input reference type changed', key)
        adjacency[a].append((b, i)); adjacency[b].append((a, i))
    roots = {key for key in adjacency if key in chambers}
    for key, p in output_nodes.items():
        require(key in adjacency, 'Isolated output node', key)
        degree, _ = incidence(nodes[key], existing)
        if degree:
            require(p['object_type'] == 'heat_chamber', 'Tie must be a chamber', key)
            roots.add(key)
    for key, neighbors in adjacency.items():
        old_degree, old_diameter = incidence(nodes[key], existing) if key in roots else (0, 0)
        require(len(neighbors) + old_degree <= 4, 'Chamber incidence exceeds four', key)
        if key in chambers:
            require(old_degree > 0, 'Existing chamber disconnected from existing network', key)
        if key in output_nodes:
            p = output_nodes[key]
            if p['object_type'] == 'technical_node':
                require(len(neighbors) == 2, 'Technical node cannot branch', key)
            else:
                require(key in roots or len(neighbors) >= 3, 'Unnecessary chamber at ordinary bend', key)
                diameter = max([old_diameter] + [lines[i][0]['diameter'] for _, i in neighbors])
                require(p['diameter'] == diameter, 'Chamber diameter', key)
                close(p['cost'], h.chamber_cost(diameter), .01, 'Chamber cost')
                if key in roots:
                    for cid in chambers:
                        old, _ = incidence(nodes[cid], existing)
                        require(h.dist(nodes[key], nodes[cid]) > 10 + .001
                                or old + len(adjacency[cid]) + len(neighbors) > 4, 'Eligible existing chamber within 10 m', key, cid)
    parent, parent_edge, ordered = {}, {}, []
    queue = deque()
    for root in sorted(roots):
        parent[root] = None; queue.append(root)
    while queue:
        node = queue.popleft(); ordered.append(node)
        for nxt, index in adjacency[node]:
            if nxt == parent[node]:
                continue
            require(nxt not in parent, 'Cycle or multiple ties in component', node, nxt)
            parent[nxt] = node; parent_edge[nxt] = index; queue.append(nxt)
    require(set(parent) == set(adjacency), 'Disconnected network component')
    missing_raw = summary['unconnected_oks_ids']
    missing = {h.input_key(v) for v in missing_raw}
    require(len(missing) == len(missing_raw) and missing <= terminals.keys(), 'Unknown/duplicate missing target')
    for value in missing_raw:
        raw = terminals[h.input_key(value)]['properties']['id']
        require(type(value) is type(raw) and value == raw, 'Missing-target ID type changed')
    flows = defaultdict(float)
    for key, feature in terminals.items():
        if key in missing:
            require(key not in adjacency, 'Missing target is connected', key)
        else:
            require(key in parent and len(adjacency[key]) == 1, 'Disconnected/transit target', key)
            flows[key] = feature['properties']['flow_tph']
    oriented = {}
    for node in reversed(ordered):
        upstream = parent[node]
        if upstream is None:
            continue
        p, points = lines[parent_edge[node]]
        close(p['flow_tph'], flows[node], 1e-6, 'Subtree flow')
        flows[upstream] += flows[node]
        require(p['diameter'] in h.CAPACITY and h.CAPACITY[p['diameter']] + 1e-9 >= p['flow_tph'], 'Undersized pipe')
        props = dict(p)
        if h.input_key(p['start_node_id']) != upstream:
            points = points[::-1]
            props['depth_start'], props['depth_end'] = p['depth_end'], p['depth_start']
        props['start_node_id'], props['end_node_id'] = upstream, node
        oriented[parent_edge[node]] = props, points
    # Reassemble constant-flow runs across all degree-two technical cuts.
    outgoing = defaultdict(list)
    for index, (p, points) in oriented.items(): outgoing[p['start_node_id']].append(index)
    runs, visited = [], set()
    for node in ordered:
        if node not in roots and len(adjacency[node]) == 2:
            continue
        for index in outgoing[node]:
            chain, points, current = [], [], index
            while current not in visited:
                visited.add(current)
                p, xy = oriented[current]; chain.append(p); points.extend(xy if not points else xy[1:])
                end = p['end_node_id']
                if len(adjacency[end]) != 2 or end in roots or end in terminals:
                    break
                current = outgoing[end][0]
            require(len({p['diameter'] for p in chain}) == 1, 'Diameter changes at constant flow')
            runs.append((node, end, chain, points))
    require(len(visited) == len(lines), 'Unvisited pipes')
    # Recompute the least permissible diameter bottom-up from the exported geometry.
    downstream = defaultdict(list)
    position = {node: i for i, node in enumerate(ordered)}
    for start, end, chain, points in sorted(runs, key=lambda run: position[run[0]], reverse=True):
        length = h.path_length(points); children = downstream[end]
        minimum = None
        for diameter in h.CAPACITY:
            if diameter < max((d for d, _ in children), default=50) or h.CAPACITY[diameter] + 1e-9 < chain[0]['flow_tph']:
                continue
            continuous = length + max((n for d, n in children if d == diameter), default=0.)
            if continuous <= h.MAX_LENGTH[diameter] + .002:
                minimum = diameter; downstream[start].append((diameter, continuous)); break
        require(chain[0]['diameter'] == minimum, 'Non-minimum flow/length diameter', chain[0]['id'], minimum)
    # Original restriction geometry; nearest facade is checked on reassembled paths.
    from shapely.geometry import shape
    from shapely.ops import transform
    obstacles = []
    for f in source['features']:
        p = f['properties']
        if p['object_type'] == 'restriction' and p['restriction_type'] not in d.SPECIAL:
            geometry = transform(lambda x, y, z=None: h.lonlat_to_utm37(x, y), shape(f['geometry']))
            obstacles.append((p, geometry))
    building_entries = 0
    for start, end, chain, points in runs:
        line = LineString(points)
        require(line.is_simple, 'Self-intersecting route')
        _check_turns(points)
        if start not in roots:
            previous = oriented[parent_edge[start]][1]
            _check_turns([previous[-2], points[0], points[1]])
        diameter = chain[0]['diameter']
        for p, geometry in obstacles:
            required = d.clearance(p['restriction_type'], diameter)
            if geometry.distance(line) >= required - .002:
                continue
            require(end in terminals and p['restriction_type'] in d.BUILDINGS and geometry.covers(Point(nodes[end])),
                    'Forbidden obstacle envelope', p['id'], chain[0]['id'])
            lead = h.remove_collinear(points[::-1]); entry = LineString(lead[:2])
            boundary_distance = _outer_boundary_distance(geometry, Point(nodes[end]))
            inside = entry.intersection(geometry)
            require(abs(inside.length - boundary_distance) < .005, 'Not the nearest straight building entry', p['id'])
            # The own-building setback is waived on the final straight entry.
            # Its first exterior turn may sit inside that nominal setback, but
            # the preceding leg must leave the envelope once and the remaining
            # route must obey normal clearance again.
            require(LineString(lead[1:]).intersection(geometry).length < .005, 'Building transit/re-entry')
            if len(lead) > 2:
                require(entry_departure_allowed(lead[1], lead[2], geometry, required),
                        'Invalid departure from building entry setback', p['id'])
                require(len(lead) <= 3 or geometry.distance(LineString(lead[2:])) >= required - .002,
                        'Building transit/re-entry', p['id'])
            building_entries += 1
    for i, (p, points) in enumerate(lines):
        for q, other in lines[i + 1:]:
            require(not h.polylines_conflict(points, other), 'Pipe crossing/overlap', p['id'], q['id'])
    objects = d.read_crossing_objects(source['features'], h.project_geom_coords)
    validate_crossing_profiles(list(oriented.values()), roots, set(terminals), nodes, objects, depth_mode)
    line_cost = length_total = 0.
    depths = {}
    for p, points in oriented.values():
        length = h.path_length(points); require(length > 1e-6, 'Zero-length pipe')
        close(p['length'], length, .002, 'Pipe length')
        a, b = p['depth_start'], p['depth_end']
        if depth_mode:
            require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= .7 - 1e-6 for v in (a, b)), 'Invalid depth')
            require(abs(a - b) <= .1 * length + 2e-5, 'Depth slope')
            require(not (min(a, b) < 3 - 1e-6 and max(a, b) > 3 + 1e-6), 'Missing depth=3 cut')
            for node, value in ((p['start_node_id'], a), (p['end_node_id'], b)):
                if node in depths: close(value, depths[node], 2e-5, 'Depth discontinuity')
                depths[node] = value
            factor = (d.depth_factor(a) + d.depth_factor(b)) / 2
        else:
            require(a is None and b is None, '2D depths must be null'); factor = 1.
        close(p.get('depth_factor', factor), factor, 1e-7, 'Depth coefficient')
        cost = length * h.NEW_COST[p['diameter']] * factor * p['special_factor']
        close(p['cost'], cost, 30, 'Pipe cost')
        line_cost += p['cost']; length_total += length
    if depth_mode:
        for node in roots | (set(terminals) - missing): close(depths[node], 3., 1e-5, 'Endpoint depth')
    chamber_total = sum(p['cost'] for p in output_nodes.values() if p['object_type'] == 'heat_chamber')
    tie_count = sum(len(adjacency[node]) for node in roots if node in chambers)
    construction = line_cost + chamber_total + tie_count * h.TIE_IN_COST
    penalty = sum(h.penalty_unconnected(terminals[t]['properties']['flow_tph']) for t in missing)
    close(summary['chamber_construction_cost'], chamber_total, .1, 'Chamber subtotal')
    require(summary['existing_chamber_tie_in_count'] == tie_count, 'Existing chamber tie count')
    close(summary['existing_chamber_tie_in_cost'], tie_count * h.TIE_IN_COST, .01, 'Tie subtotal')
    close(summary['construction_cost'], construction, .2, 'Construction subtotal')
    close(summary['unconnected_penalty'], penalty, .01, 'Penalty subtotal')
    close(summary['calculated_cost'], construction + penalty, .2, 'Total cost')
    close(summary['new_network_length'], length_total, .02, 'Total length')
    close(summary['score'], .7 * summary['calculated_cost'] / 25e6 + .003 * summary['new_network_length'], 1e-8, 'Score')
    require(summary['connected_oks_count'] == len(terminals) - len(missing), 'Connected count')
    return {'valid': True, 'variant_id': summary['variant_id'], 'mode': summary['mode'],
            'connected': len(terminals) - len(missing), 'terminals': len(terminals), 'pipes': len(lines),
            'max_node_degree': max((len(v) for v in adjacency.values()), default=0),
            'building_service_leads': building_entries, 'depth_checks': depth_mode,
            'exact_input_endpoints': True, 'nearest_building_entries': True,
            'score': summary['score'], 'cost_rub': summary['calculated_cost'], 'length_m': summary['new_network_length']}


def _check_turns(points):
    for a, b, c in zip(points, points[1:], points[2:]):
        u = b[0] - a[0], b[1] - a[1]; v = c[0] - b[0], c[1] - b[1]
        norm = math.hypot(*u) * math.hypot(*v)
        require(norm > 1e-12 and sum(x * y for x, y in zip(u, v)) / norm >= -2e-5, 'Turn exceeds 90 degrees')


def validate_crossing_profiles(lines, roots, terminals, nodes, objects, depth_mode=True):
    """Check exported crossing windows and coefficients without solving depths."""
    outgoing, incoming = defaultdict(list), {}
    for line in lines:
        p, _ = line; outgoing[p['start_node_id']].append(line); incoming[p['end_node_id']] = line
    visited = set()
    for first in lines:
        props, _ = first; start = props['start_node_id']; upstream = incoming.get(start)
        if upstream and len(outgoing[start]) == 1 and upstream[0]['diameter'] == props['diameter']:
            continue
        chain, current = [], first
        while current[0]['id'] not in visited:
            visited.add(current[0]['id']); chain.append(current); end = current[0]['end_node_id']
            if end in roots or end in terminals or len(outgoing[end]) != 1: break
            nxt = outgoing[end][0]
            if nxt[0]['diameter'] != current[0]['diameter']: break
            current = nxt
        points = list(chain[0][1])
        for _, xy in chain[1:]: points.extend(xy[1:])
        segment = h.Segment(points, start, end, props['flow_tph'], props['diameter'], h.path_length(points), start, end, 0)
        events = d.segment_passages(0, segment, objects, [nodes[r] for r in roots])
        station, plateau_depth = 0., {}
        for part, coords in chain:
            finish = station + h.path_length(coords); middle = (station + finish) / 2
            active = [e for e in events if e.start - .002 <= middle <= e.end + .002]
            ids = sorted({e.obstacle.id for e in active})
            require(sorted(part['crossing_object_ids']) == ids, 'Special window mismatch', part['id'])
            require(part['laying_method'] == ('special' if ids else 'base'), 'Laying method')
            close(part['special_factor'], max((e.obstacle.factor for e in active), default=1.), 1e-10, 'Special coefficient')
            for event in events:
                for boundary in (event.start, event.end):
                    require(not station + .002 < boundary < finish - .002, 'Missing special boundary node', part['id'])
            if depth_mode:
                for event in active:
                    top = part['depth_start']; close(top, part['depth_end'], 1e-5, 'Sloping crossing plateau')
                    require((event.above is not None and top <= event.above + 1e-5) or top >= event.below - 1e-5, 'Vertical clearance')
                    key = event.obstacle.id, event.start
                    if key in plateau_depth: close(top, plateau_depth[key], 1e-5, 'Plateau depth')
                    plateau_depth[key] = top
            station = finish
    require(len(visited) == len(lines), 'Unvisited crossing profile pieces')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='Датасет скорректированный.geojson')
    parser.add_argument('--result', default='routing_results_corrected/2d/best_variant.geojson')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = json.dumps(validate_result(args.input, args.result), ensure_ascii=False, indent=2)
    if args.output: Path(args.output).write_text(report + '\n', encoding='utf-8')
    print(report)
