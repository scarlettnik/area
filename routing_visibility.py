"""Sparse, direction-aware visibility routing with immutable physical endpoints.

The graph samples buffered obstacle corners, straight building exits and candidate
shared junctions. Edges are checked against the original (unsimplified) geometry.
The simplification affects candidate generation only, never admissibility.
"""
from collections import defaultdict
import heapq
import math

import numpy as np
from shapely import LineString, Point, Polygon, STRtree, make_valid, union_all, linestrings, distance

import heat_route_builder as h
from routing_constraints import validate_path, nearest_exits, nearest_entry_obstruction
from routing_depth import BUILDINGS, clearance


def simplify(points, tolerance=.8):
    if len(points) <= 2:
        return list(points)
    distance, index = max((h.point_segment_distance(p, points[0], points[-1]), i)
                          for i, p in enumerate(points[1:-1], 1))
    if distance <= tolerance:
        return [points[0], points[-1]]
    return simplify(points[:index + 1], tolerance)[:-1] + simplify(points[index:], tolerance)


class VisibilityGraph(h.RoutingGrid):
    def _build_blocked(self):
        pass  # All queries use indexed exact geometry; no rasterization.

    def __init__(self, terminals, candidates, obstacles, transform=h.IDENTITY_TRANSFORM, neighbors=2,
                 dn_aware=True, crossing_objects=()):
        points = [t.point for t in terminals] + [r.point for r in candidates]
        self.dn_aware = dn_aware
        self.minimum_diameter = min((h.select_diameter(t.flow_tph) for t in terminals), default=50)
        self.envelope_diameter = h.select_diameter(min(sum(t.flow_tph for t in terminals), h.CAPACITY[1400]))
        self._shapes = [o.geometry for o in obstacles]
        self._clearances = [o.clearance for o in obstacles]
        self._spatial = STRtree([g.buffer(clearance(o.kind, 1400) + .02) for g, o in zip(self._shapes, obstacles)])
        super().__init__(obstacles, points, step=5, margin=50,
                         coord_transform=transform, allow_building_leads=True)
        self.crossing_objects = list(crossing_objects)
        self.blocked = []
        self.graph = defaultdict(dict)
        self._coordinate_nodes = {}
        self._legal = {}
        self.edge_max_diameter = {}
        self.deadline = math.inf
        self.barrier_key = frozenset()
        self._barrier_cache = {}
        self._barrier_edge_cache = {}
        self.route_cache = {}
        self.cache_hits = 0
        self.searches = 0
        self.terminal_points = [t.point for t in terminals]
        self._terminal_ports = {}
        for t in terminals:
            t.cell = self.node(t.point)
            self.terminal_cells.add(t.cell)
            self.access_nodes.add(t.cell)
        for r in candidates:
            r.cell = None if self.is_blocked_point(r.point) else self.node(r.point)
            if r.cell is not None:
                self.access_nodes.add(r.cell)
        self.tie_cells = {r.cell for r in candidates if r.cell is not None}
        bounds = h.expand_bbox(h.bbox(points), 70)
        for obstacle in obstacles:
            if obstacle.linear:
                # Buffered line corners provide detours around linear no-go restrictions.
                rings = [list(p.exterior.coords) for p in
                         getattr(obstacle.geometry.buffer(obstacle.clearance + .2), "geoms",
                                 [obstacle.geometry.buffer(obstacle.clearance + .2)])]
            else:
                rings = obstacle.rings + obstacle.holes
            for ring in rings:
                ring = simplify(ring + ([] if ring[0] == ring[-1] else [ring[0]]))[:-1]
                if len(ring) < 3:
                    continue
                area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:] + ring[:1]))
                sign = 1 if area > 0 else -1
                for i, b in enumerate(ring):
                    a, c = ring[i - 1], ring[(i + 1) % len(ring)]
                    ab, bc = h.dist(a, b), h.dist(b, c)
                    if min(ab, bc) < .1:
                        continue
                    n1 = sign * (b[1] - a[1]) / ab, -sign * (b[0] - a[0]) / ab
                    n2 = sign * (c[1] - b[1]) / bc, -sign * (c[0] - b[0]) / bc
                    denominator = 1 + n1[0] * n2[0] + n1[1] * n2[1]
                    if denominator < .1:
                        continue
                    offsets = {obstacle.clearance + .15}
                    if dn_aware:
                        offsets.add(clearance(obstacle.kind, self.envelope_diameter) + .25)
                    for gap in sorted(offsets):
                        for side in (1, -1):
                            offset = side * gap / denominator
                            p = b[0] + (n1[0] + n2[0]) * offset, b[1] + (n1[1] + n2[1]) * offset
                            if h.in_bbox(p, bounds) and not self.is_blocked_point(p):
                                self.node(p)
        self.entry_diagnostics = {}
        for t in terminals:
            owners = [o for o in obstacles if o.kind in BUILDINGS and o.geometry.covers(Point(t.point))]
            diameter = h.select_diameter(t.flow_tph)
            angles = [] if owners else [i * math.pi / 12 for i in range(24)]
            for o in owners:
                for foot in nearest_exits(o, t.point):
                    if h.dist(foot, t.point) > 1e-7:
                        angles.append(math.atan2(foot[1] - t.point[1], foot[0] - t.point[0]))
                    else:
                        angles.extend(i * math.pi / 12 for i in range(24))
            reach = max((math.hypot(o.geometry.bounds[2] - o.geometry.bounds[0],
                                   o.geometry.bounds[3] - o.geometry.bounds[1]) + 50 for o in owners), default=30.)
            ports, rejected = set(), set()
            for angle in sorted(set(angles)):
                end = t.point[0] + reach * math.cos(angle), t.point[1] + reach * math.sin(angle)
                ray = LineString([t.point, end])
                nearby = self._spatial.query(ray)
                blocked_shape = union_all([self._shapes[i].buffer(clearance(obstacles[i].kind, diameter) + .04)
                                           for i in nearby])
                free = ray.difference(blocked_shape)
                intervals = list(getattr(free, "geoms", [free]))
                intervals = sorted((g for g in intervals if g.geom_type == "LineString" and g.length > .05),
                                   key=lambda g: ray.project(Point(g.coords[0])))
                for interval in intervals:
                    start = min(ray.project(Point(interval.coords[0])), ray.project(Point(interval.coords[-1])))
                    for offset in (.03, 5., 20.):
                        if offset >= interval.length:
                            continue
                        point = ray.interpolate(start + offset)
                        q = point.x, point.y
                        try:
                            validate_path([t.point, q], diameter, obstacles, [t.point], True)
                        except ValueError as error:
                            rejected.add(str(error))
                            continue
                        ports.add(self.node(q))
            self._terminal_ports[t.cell] = ports
            self.entry_diagnostics[t.id] = {"input_id": t.input_id if t.input_id is not None else t.id,
                "ports": len(ports), "owners": [o.id for o in owners],
                "status": "available" if ports else "no_legal_nearest_entry" if owners else "no_visible_port",
                "rejections": sorted(rejected),
                "obstruction_witnesses": nearest_entry_obstruction(t, owners, diameter) if not ports else []}
        # Pairwise shared-junction hypotheses; their economic merit is decided
        # later by full flow/DN/chamber evaluation.
        for i, a in enumerate(terminals):
            for b in terminals[i + 1:]:
                if h.dist(a.point, b.point) < 220:
                    for t in (.25, .5, .75):
                        p = tuple(a.point[k] * (1 - t) + b.point[k] * t for k in (0, 1))
                        if not self.is_blocked_point(p):
                            self.node(p)
        # Ports outside utility envelopes allow one straight crossing with its
        # full special window, followed by a turn outside the setback.
        for obj in self.crossing_objects:
            lines = obj.lines or [ring for polygon in obj.polygons for ring in polygon]
            gap = obj.horizontal_gap + (h.ENVELOPE[self.envelope_diameter][0] + obj.width) / 2 + 1.
            for line in lines:
                for a, b in zip(line, line[1:]):
                    length = h.dist(a, b)
                    if length < .1:
                        continue
                    count = max(1, math.ceil(length / 50.))
                    for i in range(count + 1):
                        center = tuple(a[k] + (b[k] - a[k]) * i / count for k in (0, 1))
                        for side in (-1, 1):
                            q = center[0] + side * gap * (b[1] - a[1]) / length, center[1] - side * gap * (b[0] - a[0]) / length
                            if h.in_bbox(q, bounds) and not self.is_blocked_point(q):
                                self.node(q)
        regular = [(n, p) for n, p in self.extra_points.items() if n not in self.terminal_cells]
        pairs = set()
        positions = np.array([p for _, p in regular])
        for node, p in regular:
            delta = positions - p
            distances = np.hypot(delta[:, 0], delta[:, 1])
            sectors = ((np.arctan2(delta[:, 1], delta[:, 0]) + math.pi) * 8 / math.pi).astype(int) % 16
            attempted, accepted = [0] * 16, [[] for _ in range(16)]
            root_accepted = [0] * 16
            for index in np.argsort(distances, kind="stable"):
                other, q = regular[index]
                sector = sectors[index]
                is_root = other in self.tie_cells
                if (other == node or attempted[sector] >= 40
                        or (root_accepted[sector] >= neighbors if is_root else len(accepted[sector]) >= neighbors)):
                    continue
                attempted[sector] += 1
                if self.line_clear(p, q):
                    pairs.add(h.edge_key(node, other))
                    if is_root:
                        root_accepted[sector] += 1
                    elif not any(h.dist(q, old) < 2. for old in accepted[sector]):
                        accepted[sector].append(q)
        for a, b in sorted(pairs):
            self.connect(a, b)
        for terminal, ports in self._terminal_ports.items():
            for port in sorted(ports):
                self.connect(terminal, port)
            # Outside a building, the terminal itself is a visibility point.
            if not ports:
                p = self.extra_points[terminal]
                for node, q in regular:
                    if self.line_clear(p, q):
                        self.connect(terminal, node)
        self._index_edge_diameters()

    def _index_edge_diameters(self):
        """Vectorized exact distances amortize feasibility across every route search."""
        edges = sorted({h.edge_key(a, b) for a in self.graph for b in self.graph[a]
                        if a not in self.terminal_cells and b not in self.terminal_cells})
        if not edges:
            return
        lines = linestrings([[self.extra_points[a], self.extra_points[b]] for a, b in edges])
        rows, obstacles = self._spatial.query(lines)
        values = distance(lines[rows], np.asarray(self._shapes, dtype=object)[obstacles])
        diameters = list(h.CAPACITY)
        maximum = np.full(len(edges), len(diameters) - 1, dtype=int)
        pair_maximum = np.full(len(rows), -1, dtype=int)
        for index, diameter in enumerate(diameters):
            required = np.array([clearance(o.kind, diameter) for o in self.obstacles])
            pair_maximum[values >= required[obstacles] - 1e-6] = index
        np.minimum.at(maximum, rows, pair_maximum)
        self.edge_max_diameter.update((edge, diameters[index] if index >= 0 else 0)
                                      for edge, index in zip(edges, maximum))

    def line_clear(self, a, b, endpoint_relief=0.):
        line = LineString([a, b])
        return not any(self._shapes[i].distance(line) <= self._clearances[i] + 1e-8 for i in self._spatial.query(line))

    def is_blocked_point(self, p):
        point = Point(p)
        return any(self._shapes[i].distance(point) <= self._clearances[i] + 1e-8 for i in self._spatial.query(point))

    def node(self, p):
        key = round(p[0], 6), round(p[1], 6)
        if key not in self._coordinate_nodes:
            node = (-2, len(self.extra_points))
            self.extra_points[node] = tuple(p)
            self._coordinate_nodes[key] = node
        return self._coordinate_nodes[key]

    def connect(self, a, b):
        if a != b:
            self.graph[a][b] = self.graph[b][a] = h.dist(self.extra_points[a], self.extra_points[b])

    def set_barriers(self, edges=()):
        self.barrier_key = frozenset(edges)
        if not edges:
            return
        if self.barrier_key not in self._barrier_cache:
            if len(self._barrier_cache) >= 128:
                self._barrier_cache.clear(); self._barrier_edge_cache.clear()
            paths = [self.edge_points(a, b) for a, b in sorted(edges)]
            self._barrier_cache[self.barrier_key] = STRtree([LineString(p) for p in paths]), paths
        self._barrier_index, self._barrier_paths = self._barrier_cache[self.barrier_key]

    def conflicts_with_network(self, a, b):
        if not self.barrier_key:
            return False
        key = self.barrier_key, h.edge_key(a, b)
        if key not in self._barrier_edge_cache:
            path = self.edge_points(a, b)
            nearby = self._barrier_index.query(LineString(path))
            conflict = any(h.polylines_conflict(path, self._barrier_paths[i]) for i in nearby)
            if len(self._barrier_edge_cache) >= 100_000:
                self._barrier_edge_cache.clear()
            self._barrier_edge_cache[key] = conflict
        return self._barrier_edge_cache[key]

    def legal(self, a, b, diameter):
        key = h.edge_key(a, b), diameter
        if key not in self._legal:
            if a not in self.terminal_cells and b not in self.terminal_cells:
                edge = h.edge_key(a, b)
                if edge not in self.edge_max_diameter:
                    line = LineString([self.extra_points[a], self.extra_points[b]])
                    nearby = [(self.obstacles[i].kind, self._shapes[i].distance(line)) for i in self._spatial.query(line)]
                    self.edge_max_diameter[edge] = max((d for d in h.CAPACITY
                        if all(distance >= clearance(kind, d) - 1e-6 for kind, distance in nearby)), default=0)
                self._legal[key] = diameter <= self.edge_max_diameter[edge]
                return self._legal[key]
            try:
                validate_path([self.extra_points[a], self.extra_points[b]], diameter,
                              self.obstacles, self.terminal_points, True)
                self._legal[key] = True
            except ValueError:
                self._legal[key] = False
        return self._legal[key]

    def least_cost_path(self, sources, target, blocked, rub_per_m, bend_cost_rub=0., banned=frozenset()):
        import time
        if target is None or not sources or not self.graph[target]:
            return []
        key = (tuple(sorted(sources.items())), target, frozenset(blocked), rub_per_m, bend_cost_rub, frozenset(banned), self.barrier_key)
        if key in self.route_cache:
            self.cache_hits += 1
            return list(self.route_cache[key])
        self.searches += 1
        diameter = min(h.NEW_COST, key=lambda d: abs(h.NEW_COST[d] - rub_per_m))
        best, previous, queue = {}, {}, []
        for source, price in sorted(sources.items()):
            state = source, None
            best[state] = price
            previous[state] = None
            heapq.heappush(queue, (price + h.dist(self.extra_points[source], self.extra_points[target]) * (rub_per_m + h.SCORE_LENGTH_RUB_PER_M),
                                   price, source, (-3, -3)))
        found = []
        while queue:
            if time.monotonic() >= self.deadline:
                return []  # Never cache an interrupted search as unreachable.
            _, price, node, prior = heapq.heappop(queue)
            prior = None if prior == (-3, -3) else prior
            state = node, prior
            if price > best.get(state, math.inf) + 1e-6:
                continue
            if node == target:
                while state is not None:
                    found.append(state[0])
                    state = previous[state]
                found.reverse()
                break
            for nxt, length in self.graph[node].items():
                if nxt == prior or h.edge_key(node, nxt) in banned or nxt in blocked:
                    continue
                if nxt in self.access_nodes and nxt != target:
                    continue
                if self.conflicts_with_network(node, nxt):
                    continue
                if not self.legal(node, nxt, diameter):
                    continue
                if prior is not None:
                    a, b, c = self.extra_points[prior], self.extra_points[node], self.extra_points[nxt]
                    if sum((b[k] - a[k]) * (c[k] - b[k]) for k in (0, 1)) < -1e-7:
                        continue
                weighted = self.crossing_weight(node, nxt, diameter, length)
                if not math.isfinite(weighted):
                    continue
                turn = prior is not None and h.polyline_bend_count([self.extra_points[prior], self.extra_points[node], self.extra_points[nxt]], 5.)
                value = price + weighted * rub_per_m + length * h.SCORE_LENGTH_RUB_PER_M + bend_cost_rub * bool(turn)
                next_state = nxt, node
                if value >= best.get(next_state, math.inf) - 1e-6:
                    continue
                best[next_state], previous[next_state] = value, (node, prior)
                estimate = h.dist(self.extra_points[nxt], self.extra_points[target]) * (rub_per_m + h.SCORE_LENGTH_RUB_PER_M)
                heapq.heappush(queue, (value + estimate, value, nxt, node))
        if len(self.route_cache) > 5000:
            self.route_cache.clear()
        if found:
            route_length = sum(self.graph[a][b] for a, b in zip(found, found[1:]))
            if route_length > h.MAX_LENGTH[diameter] + 1e-7:
                try:
                    larger = h.select_diameter(h.CAPACITY[diameter], route_length)
                except ValueError:
                    found = []
                else:
                    found = self.least_cost_path(sources, target, blocked, h.NEW_COST[larger],
                        bend_cost_rub * h.NEW_COST[larger] / rub_per_m, banned)
        if time.monotonic() < self.deadline:
            self.route_cache[key] = tuple(found)
        return found

    def alternatives(self, sources, target, diameter, count=3, turn_m=8.):
        """Diversified paths; remove individual edges of each accepted route."""
        paths, seen = [], set()
        pending = [frozenset()]
        while pending and len(paths) < count:
            banned = pending.pop(0)
            path = self.least_cost_path(sources, target, set(), h.NEW_COST[diameter],
                                        turn_m * h.NEW_COST[diameter], banned)
            if not path or tuple(path) in seen:
                continue
            seen.add(tuple(path))
            paths.append(path)
            edges = [h.edge_key(a, b) for a, b in zip(path, path[1:])]
            pending.extend(frozenset([edge]) for edge in edges[::max(1, len(edges) // 3)])
        return paths

    def crossing_weight(self, a_cell, b_cell, diameter, length):
        if not self.crossing_objects:
            return length
        key = h.edge_key(a_cell, b_cell), diameter
        if key in self._crossing_edge_cache:
            return self._crossing_edge_cache[key]
        if getattr(self, '_crossing_index_source', None) != id(self.crossing_objects):
            self._crossing_spatial = STRtree([obj.geometry.buffer(obj.horizontal_gap + (obj.width + h.ENVELOPE[1400][0]) / 2 + 3)
                                             for obj in self.crossing_objects])
            self._crossing_index_source = id(self.crossing_objects)
        from routing_depth import segment_passages
        points = self.edge_points(a_cell, b_cell)
        nearby = self._crossing_spatial.query(LineString(points))
        objects = [self.crossing_objects[i] for i in nearby]
        segment = h.Segment(points, a_cell, b_cell, 0, diameter, length, '', '', 0)
        ties = [self.cell_to_point(c) for c in (a_cell, b_cell) if c in self.tie_cells]
        try:
            events = segment_passages(0, segment, objects, ties)
            cuts = sorted({0., length, *(v for e in events for v in (e.start, e.end))})
            value = sum((end - start) * max((e.obstacle.factor for e in events if e.start <= (start + end) / 2 <= e.end), default=1.)
                        for start, end in zip(cuts, cuts[1:]))
        except ValueError:
            value = math.inf
        self._crossing_edge_cache[key] = value
        return value
