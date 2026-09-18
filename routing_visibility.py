"""Sparse, direction-aware visibility routing with immutable physical endpoints.

The graph samples buffered obstacle corners, straight building exits and candidate
shared junctions. Edges are checked against the original (unsimplified) geometry.
The simplification affects candidate generation only, never admissibility.
"""
from collections import defaultdict
import heapq
import math

import numpy as np
from shapely import LineString, Point, Polygon, STRtree, make_valid, union_all

import heat_route_builder as h
from routing_constraints import validate_path
from routing_depth import BUILDINGS


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

    def __init__(self, terminals, candidates, obstacles, transform=h.IDENTITY_TRANSFORM, neighbors=2):
        points = [t.point for t in terminals] + [r.point for r in candidates]
        self._shapes = [union_all([make_valid(Polygon(r)) for r in o.rings]).difference(
            union_all([make_valid(Polygon(r)) for r in o.holes])) for o in obstacles]
        self._clearances = [o.clearance for o in obstacles]
        self._spatial = STRtree([g.buffer(c + .02) for g, c in zip(self._shapes, self._clearances)])
        super().__init__(obstacles, points, step=5, margin=50,
                         coord_transform=transform, allow_building_leads=True)
        self.blocked = []
        self.graph = defaultdict(dict)
        self._coordinate_nodes = {}
        self._legal = {}
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
        bounds = h.expand_bbox(h.bbox(points), 70)
        for obstacle in obstacles:
            for ring in obstacle.rings + obstacle.holes:
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
                    for side in (1, -1):
                        offset = side * (obstacle.clearance + 1.0) / denominator
                        p = b[0] + (n1[0] + n2[0]) * offset, b[1] + (n1[1] + n2[1]) * offset
                        if h.in_bbox(p, bounds) and not self.is_blocked_point(p):
                            self.node(p)
        for t in terminals:
            owners = [o for o in obstacles if o.kind in BUILDINGS and o.contains_or_near(t.point)]
            radius = max((min(h.ring_distance(t.point, r) for r in o.rings) + o.clearance + 10
                          for o in owners), default=10.)
            ports = set()
            angles = [i * math.pi / 12 for i in range(24)]
            # A local facade normal remains available even when the district's
            # other buildings have an unrelated dominant orientation.
            for o in owners:
                for ring in o.rings:
                    if not h.point_in_ring(t.point, ring) and h.ring_distance(t.point, ring) > .02:
                        continue
                    edges = sorted(zip(ring, ring[1:]), key=lambda e: h.point_segment_distance(t.point, *e))[:8]
                    angles.extend(math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi / 2 for a, b in edges)
                    angles.extend(math.atan2(b[1] - a[1], b[0] - a[0]) - math.pi / 2 for a, b in edges)
            angles = sorted({round(a % (2 * math.pi), 3) for a in angles})
            for angle in angles:
                for reach in (radius, radius + 5):
                    p = t.point[0] + reach * math.cos(angle), t.point[1] + reach * math.sin(angle)
                    if self.is_blocked_point(p):
                        continue
                    try:
                        validate_path([t.point, p], h.select_diameter(t.flow_tph), obstacles, [t.point], True)
                    except ValueError:
                        continue
                    ports.add(self.node(p))
            self._terminal_ports[t.cell] = ports
        # Pairwise shared-junction hypotheses; their economic merit is decided
        # later by full flow/DN/reconstruction evaluation.
        for i, a in enumerate(terminals):
            for b in terminals[i + 1:]:
                if h.dist(a.point, b.point) < 220:
                    for t in (.25, .5, .75):
                        p = tuple(a.point[k] * (1 - t) + b.point[k] * t for k in (0, 1))
                        if not self.is_blocked_point(p):
                            self.node(p)
        regular = [(n, p) for n, p in self.extra_points.items() if n not in self.terminal_cells]
        pairs = set()
        positions = np.array([p for _, p in regular])
        for node, p in regular:
            delta = positions - p
            distances = np.hypot(delta[:, 0], delta[:, 1])
            sectors = ((np.arctan2(delta[:, 1], delta[:, 0]) + math.pi) * 8 / math.pi).astype(int) % 16
            attempted, accepted = [0] * 16, [0] * 16
            for index in np.argsort(distances, kind="stable"):
                other, q = regular[index]
                sector = sectors[index]
                if other == node or attempted[sector] >= 24 or accepted[sector] >= neighbors:
                    continue
                attempted[sector] += 1
                if self.line_clear(p, q):
                    pairs.add(h.edge_key(node, other))
                    accepted[sector] += 1
        for a, b in sorted(pairs):
            self.connect(a, b)
        for terminal, ports in self._terminal_ports.items():
            for port in ports:
                self.connect(terminal, port)

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

    def legal(self, a, b, diameter):
        key = h.edge_key(a, b), diameter
        if key not in self._legal:
            if a not in self.terminal_cells and b not in self.terminal_cells:
                # Graph edges already respect the sum-of-demands DN envelope.
                self._legal[key] = True
                return True
            try:
                validate_path([self.extra_points[a], self.extra_points[b]], diameter,
                              self.obstacles, self.terminal_points, True)
                self._legal[key] = True
            except ValueError:
                self._legal[key] = False
        return self._legal[key]

    def least_cost_path(self, sources, target, blocked, rub_per_m, bend_cost_rub=0., banned=frozenset()):
        if target is None or not sources:
            return []
        key = (tuple(sorted(sources.items())), target, frozenset(blocked), rub_per_m, bend_cost_rub, frozenset(banned))
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
            heapq.heappush(queue, (price + h.dist(self.extra_points[source], self.extra_points[target]) * rub_per_m,
                                   price, source, (-3, -3)))
        found = []
        while queue:
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
                if not self.legal(node, nxt, diameter):
                    continue
                weighted = self.crossing_weight(node, nxt, diameter, length)
                if not math.isfinite(weighted):
                    continue
                turn = prior is not None and h.polyline_bend_count([self.extra_points[prior], self.extra_points[node], self.extra_points[nxt]], 5.)
                value = price + weighted * rub_per_m + bend_cost_rub * bool(turn)
                next_state = nxt, node
                if value >= best.get(next_state, math.inf) - 1e-6:
                    continue
                best[next_state], previous[next_state] = value, (node, prior)
                estimate = h.dist(self.extra_points[nxt], self.extra_points[target]) * rub_per_m
                heapq.heappush(queue, (value + estimate, value, nxt, node))
        if len(self.route_cache) > 5000:
            self.route_cache.clear()
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
