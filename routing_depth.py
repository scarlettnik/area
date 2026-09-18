"""Appendix §§4–6: pipe envelopes, special passages and connected depth profiles.

Lengths and slopes use horizontal UTM chainage, as in the routing cost model.
The surface is flat. Depth is to the TOP of the pair, not its centreline.
No forbidden polygon is made traversable by changing depth.
"""
from dataclasses import dataclass, field
import heapq
import itertools
import math
from functools import cached_property


ENVELOPE = dict(zip(
    (50, 65, 80, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400),
    ((.400, .125), (.430, .140), (.470, .160), (.510, .180), (.600, .225),
     (.650, .250), (.880, .315), (1.050, .400), (1.150, .450), (1.370, .560),
     (1.670, .710), (1.850, .800), (2.050, .900), (2.250, 1.000),
     (2.450, 1.100), (2.650, 1.200), (3.100, 1.425), (3.450, 1.600))))
BUILDINGS = {"oks", "oks_existing", "oks_future"}
SPECIAL = {"gas_pipeline": 1.25, "power_cable": 1.15, "heat_network": 1.05,
           "road": 1.60, "tram_tracks": 1.75}
SLOPE = .10
NORMAL_DEPTH = 3.0
MIN_DEPTH = .7


def clearance(kind, diameter):
    """Required distance from the pipe axis to a forbidden polygon boundary."""
    margin = (5 if diameter < 500 else 7 if diameter <= 800 else 9) if kind in BUILDINGS else 1
    if kind == "railway":
        margin = 2  # Not specified by the appendix: conservative closed local policy.
    return margin + ENVELOPE[diameter][0] / 2


def depth_factor(depth):
    return 1 + .10 * max(0, depth - NORMAL_DEPTH)


@dataclass
class CrossingObject:
    id: str
    kind: str
    lines: list = field(default_factory=list)
    polygons: list = field(default_factory=list)  # polygon -> [shell, holes...]
    width: float = 0
    height: float = 0
    top: float = 0
    vertical_gap: float = 0
    horizontal_gap: float = 0

    @property
    def factor(self):
        return SPECIAL[self.kind]

    @cached_property
    def bounds(self):
        points = [p for line in self.lines for p in line]
        points += [p for poly in self.polygons for ring in poly for p in ring]
        if not points:
            raise ValueError(f"{self.id}: empty crossing geometry")
        return (min(p[0] for p in points), min(p[1] for p in points),
                max(p[0] for p in points), max(p[1] for p in points))

    def near(self, bounds, margin):
        x0, y0, x1, y1 = self.bounds
        return not (bounds[2] < x0 - margin or bounds[0] > x1 + margin
                    or bounds[3] < y0 - margin or bounds[1] > y1 + margin)


def read_crossing_objects(features, project):
    result = []
    for feature in features:
        props = feature.get("properties") or {}
        kind = props.get("object_type")
        if kind == "restriction":
            kind = props.get("restriction_type")
        if kind not in SPECIAL:
            continue
        geometry = feature.get("geometry") or {}
        coordinates = project(geometry)
        obj = CrossingObject(str(props["id"]), kind)
        if kind in {"road", "tram_tracks"}:
            if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
                raise ValueError(f"{obj.id}: road/tram passage requires polygon geometry")
            obj.polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
            obj.horizontal_gap = 1.5
        else:
            if geometry.get("type") not in {"LineString", "MultiLineString"}:
                raise ValueError(f"{obj.id}: utility requires linear geometry")
            obj.lines = [coordinates] if geometry["type"] == "LineString" else coordinates
            if kind == "gas_pipeline":
                obj.width, obj.height, obj.top, obj.vertical_gap, obj.horizontal_gap = .4, .4, 2.8, .2, 2
            elif kind == "power_cable":
                obj.width, obj.height, obj.top, obj.vertical_gap, obj.horizontal_gap = .2, .2, 2.7, .5, 2
            else:
                diameter = props.get("diameter")
                if diameter not in ENVELOPE:
                    raise ValueError(f"{obj.id}: missing/unsupported existing pipe diameter")
                obj.width, obj.height = ENVELOPE[diameter]
                obj.top, obj.vertical_gap, obj.horizontal_gap = 3, .5, 1
        result.append(obj)
    return result


def transformed_objects(objects, transform):
    from dataclasses import replace
    return [replace(o, lines=[[transform(p) for p in line] for line in o.lines],
                    polygons=[[[transform(p) for p in ring] for ring in poly] for poly in o.polygons])
            for o in objects]


def intersection_parameter(a, b, c, d):
    """Point intersection on ab; parallel/collinear pairs handled by clearance checks."""
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = d[0] - c[0], d[1] - c[1]
    denominator = vx * wy - vy * wx
    if abs(denominator) < 1e-10:
        return None
    dx, dy = c[0] - a[0], c[1] - a[1]
    t = (dx * wy - dy * wx) / denominator
    u = (dx * vy - dy * vx) / denominator
    return min(1., max(0., t)) if -1e-8 <= t <= 1 + 1e-8 and -1e-8 <= u <= 1 + 1e-8 else None


def chainages(points):
    values = [0.]
    for a, b in zip(points, points[1:]):
        values.append(values[-1] + math.dist(a, b))
    return values


def point_at(points, distances, value):
    for i, end in enumerate(distances[1:]):
        if value <= end + 1e-8:
            start = distances[i]
            t = max(0., min(1., (value - start) / (end - start)))
            return tuple(a + (b - a) * t for a, b in zip(points[i], points[i + 1]))
    return points[-1]


@dataclass
class Passage:
    segment: int
    start: float
    end: float
    obstacle: CrossingObject
    above: float | None
    below: float


@dataclass
class DepthPiece:
    segment: int
    points: list
    start_depth: float
    end_depth: float
    special_factor: float = 1.
    crossing_ids: tuple = ()

    @property
    def length(self):
        return sum(math.dist(a, b) for a, b in zip(self.points, self.points[1:]))

    @property
    def depth_coefficient(self):
        return (depth_factor(self.start_depth) + depth_factor(self.end_depth)) / 2


def segment_passages(index, segment, objects, tie_points):
    """Extract actual crossings, reject overlaps/parallel encroachment and shallow angles."""
    from heat_route_builder import point_in_ring, segments_distance, point_segment_distance
    points = segment.points
    stations = chainages(points)
    length = stations[-1]
    width, height = ENVELOPE[segment.diameter]
    passages = []
    bounds = (min(p[0] for p in points), min(p[1] for p in points),
              max(p[0] for p in points), max(p[1] for p in points))
    for obj in objects:
        if not obj.near(bounds, obj.horizontal_gap + (width + obj.width) / 2):
            continue
        windows = []
        ties = []
        if obj.lines:
            hits = set()
            for i, (a, b) in enumerate(zip(points, points[1:])):
                for line in obj.lines:
                    for c, d in zip(line, line[1:]):
                        t = intersection_parameter(a, b, c, d)
                        if t is not None:
                            hits.add(round(stations[i] + t * math.dist(a, b), 7))
            if obj.kind == "heat_network":
                # Projection/unprojection can leave an exported tie a fraction
                # of a millimetre from the existing line endpoint. Treat that
                # endpoint as the tie when both coordinates are within the
                # existing 2 cm tie tolerance, even without an exact crossing.
                for station, endpoint in ((0., points[0]), (length, points[-1])):
                    near_tie = any(math.dist(endpoint, tie) < .02 for tie in tie_points)
                    near_line = any(
                        point_segment_distance(endpoint, c, d) < .02
                        for line in obj.lines
                        for c, d in zip(line, line[1:])
                    )
                    if near_tie and near_line:
                        hits.add(round(station, 7))
            for position in sorted(hits):
                p = point_at(points, stations, position)
                is_tie = obj.kind == "heat_network" and any(math.dist(p, t) < .02 for t in tie_points)
                if is_tie:
                    ties.append(position)
                    continue
                if position < 2 - 1e-6 or position > length - 2 + 1e-6:
                    raise ValueError(f"{obj.id}: crossing plateau cannot fit before a route endpoint")
                windows.append((max(0., position - 2), min(length, position + 2)))
            required = obj.horizontal_gap + (width + obj.width) / 2
            # Split at crossing windows: nearby parallel sections outside a true
            # crossing/tie approach remain forbidden, even if deep enough.
            for i, (a, b) in enumerate(zip(points, points[1:])):
                midpoint = (stations[i] + stations[i + 1]) / 2
                for line in obj.lines:
                    for c, d in zip(line, line[1:]):
                        if segments_distance(a, b, c, d) >= required - 1e-7:
                            continue
                        va, vb = (b[0] - a[0], b[1] - a[1]), (d[0] - c[0], d[1] - c[1])
                        norm = math.hypot(*va) * math.hypot(*vb)
                        parallel = norm > 1e-10 and abs(va[0] * vb[0] + va[1] * vb[1]) / norm > .9999
                        approach = any(stations[i] - required <= t <= stations[i + 1] + required for t in ties)
                        crossing = any(stations[i] - required <= e and stations[i + 1] + required >= s for s, e in windows)
                        if parallel or not (approach or crossing):
                            raise ValueError(f"{obj.id}: insufficient horizontal utility clearance")
            above, below = obj.top - obj.vertical_gap - height, obj.top + obj.height + obj.vertical_gap
        else:
            cuts = {0., length}
            boundaries = [(a, b) for poly in obj.polygons for ring in poly for a, b in zip(ring, ring[1:])]
            for i, (a, b) in enumerate(zip(points, points[1:])):
                for c, d in boundaries:
                    t = intersection_parameter(a, b, c, d)
                    if t is not None:
                        cuts.add(stations[i] + t * math.dist(a, b))
            ordered = sorted(cuts)
            for start, end in zip(ordered, ordered[1:]):
                p = point_at(points, stations, (start + end) / 2)
                if any(point_in_ring(p, poly[0]) and not any(point_in_ring(p, hole) for hole in poly[1:])
                       for poly in obj.polygons):
                    if start < 3 - 1e-7 or end + 3 > length + 1e-7:
                        raise ValueError(f"{obj.id}: special passage requires 3 m outside each boundary")
                    axis = max(boundaries, key=lambda edge: math.dist(*edge))
                    dx, dy = axis[1][0] - axis[0][0], axis[1][1] - axis[0][1]
                    for i, (a, b) in enumerate(zip(points, points[1:])):
                        if stations[i] >= end or stations[i + 1] <= start:
                            continue
                        norm = math.dist(a, b) * math.hypot(dx, dy)
                        if norm and abs((b[0] - a[0]) * dx + (b[1] - a[1]) * dy) / norm > math.sqrt(.5) + 1e-8:
                            raise ValueError(f"{obj.id}: crossing angle below 45 degrees")
                    windows.append((start - 3, end + 3))
            # No polygon crossing is not a licence to route along its shoulder.
            if not windows:
                if any(segments_distance(a, b, c, d) < 1.5 + width / 2 - 1e-7
                       for a, b in zip(points, points[1:]) for c, d in boundaries):
                    raise ValueError(f"{obj.id}: insufficient road/tram horizontal clearance")
            above, below = None, 1.0 if obj.kind == "road" else 1.2
        for start, end in windows:
            passages.append(Passage(index, start, end, obj, above if above is not None and above >= MIN_DEPTH else None, below))
    return passages


def _propagate(adjacency, seeds, minimum=True):
    """Multi-source shortest paths propagate upper (or negated lower) bounds."""
    values = [math.inf] * len(adjacency)
    queue = []
    for node, value in seeds:
        v = value if minimum else -value
        if v < values[node]:
            values[node] = v
            heapq.heappush(queue, (v, node))
    while queue:
        value, node = heapq.heappop(queue)
        if value > values[node] + 1e-10:
            continue
        for neighbor, distance in adjacency[node]:
            candidate = value + SLOPE * distance
            if candidate < values[neighbor] - 1e-10:
                values[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return values if minimum else [-v for v in values]


def build_depth_profiles(segments, objects, root_cells, terminal_cells, tie_points, rates):
    """Solve a connected Lipschitz profile; junction depths are shared variables.

    All-above is tried first: when feasible its Kdepth=1 proves the lowest
    possible price for these fixed corridors. Up to eight binary crossings are
    otherwise enumerated; larger cases use bounded deterministic improvement.
    This is NOT a global optimum certificate for the horizontal routing.
    """
    events = [p for i, s in enumerate(segments) for p in segment_passages(i, s, objects, tie_points)]
    if not events:
        return [[DepthPiece(i, s.points, 3., 3.)] for i, s in enumerate(segments)], []
    nodes = {}
    adjacency = []
    fixed = set()
    paths = []
    edges = []
    plateau_nodes = [set() for _ in events]

    def node(key):
        if key not in nodes:
            nodes[key] = len(adjacency)
            adjacency.append([])
        return nodes[key]

    for i, segment in enumerate(segments):
        stations = chainages(segment.points)
        cuts = set(stations)
        for p in events:
            if p.segment == i:
                cuts.update((p.start, p.end))
        positions = sorted(cuts)
        sequence = []
        for j, position in enumerate(positions):
            cell = segment.start_cell if j == 0 else segment.end_cell if j == len(positions) - 1 else None
            v = node(("cell", cell) if cell is not None else ("cut", i, position))
            if cell in root_cells or cell in terminal_cells:
                fixed.add(v)
            sequence.append((v, position, point_at(segment.points, stations, position)))
        paths.append(sequence)
        for (a, start, p), (b, end, q) in zip(sequence, sequence[1:]):
            active = [k for k, passage in enumerate(events) if passage.segment == i
                      and passage.start <= (start + end) / 2 <= passage.end]
            for k in active:
                plateau_nodes[k].update((a, b))
            # Zero metric distance means equal depth across the entire plateau.
            metric_length = 0. if active else end - start
            adjacency[a].append((b, metric_length))
            adjacency[b].append((a, metric_length))
            edges.append((i, a, b, p, q, end - start, active))

    def solve(choices):
        lower = [(v, 3.) for v in fixed]
        upper = [(v, 3.) for v in fixed]
        for k, passage in enumerate(events):
            if choices[k] and passage.above is not None:
                upper.extend((v, passage.above) for v in plateau_nodes[k])
            else:
                lower.extend((v, max(MIN_DEPTH, passage.below)) for v in plateau_nodes[k])
        lo = _propagate(adjacency, lower, minimum=False)
        hi = _propagate(adjacency, upper)
        if any(max(MIN_DEPTH, l) > u + 1e-7 for l, u in zip(lo, hi)):
            return None
        pieces = [[] for _ in segments]
        cost = deviation = 0.
        for i, a, b, p, q, length, active in edges:
            if active:
                depth = max(MIN_DEPTH, lo[a], min(3., hi[a]))
                profile = [(0., depth), (length, depth)]
            else:
                functions = [(lo[a], -SLOPE), (lo[b] - SLOPE * length, SLOPE),
                             (hi[a], SLOPE), (hi[b] + SLOPE * length, -SLOPE),
                             (3., 0.), (MIN_DEPTH, 0.)]
                locations = {0., length}
                for (c, m), (d, n) in itertools.combinations(functions, 2):
                    if abs(m - n) > 1e-10 and math.isfinite(c) and math.isfinite(d):
                        x = (d - c) / (m - n)
                        if 1e-7 < x < length - 1e-7:
                            locations.add(x)
                profile = []
                for x in sorted(locations):
                    low = max(MIN_DEPTH, lo[a] - SLOPE * x, lo[b] - SLOPE * (length - x))
                    high = min(hi[a] + SLOPE * x, hi[b] + SLOPE * (length - x))
                    profile.append((x, max(low, min(3., high))))
            factor = max((events[k].obstacle.factor for k in active), default=1.)
            ids = tuple(sorted({events[k].obstacle.id for k in active}))
            # Overlapping special zones have no combination rule in the appendix.
            # Reject rather than quietly inventing a discounted max/product tariff.
            if len(ids) > 1:
                return None
            for (start, h1), (end, h2) in zip(profile, profile[1:]):
                if end - start < 1e-7:
                    continue
                interpolate = lambda s: tuple(x + (y - x) * s / length for x, y in zip(p, q))
                piece = DepthPiece(i, [interpolate(start), interpolate(end)], h1, h2, factor, ids)
                pieces[i].append(piece)
                cost += (end - start) * rates[segments[i].diameter] * piece.depth_coefficient * factor
                deviation += (end - start) * (abs(h1 - 3) + abs(h2 - 3)) / 2
        return cost, deviation, pieces

    above = tuple(p.above is not None for p in events)
    result = solve(above)
    if result is not None and all(piece.start_depth <= 3 + 1e-8 and piece.end_depth <= 3 + 1e-8
                                  for group in result[2] for piece in group):
        best = result
    else:
        binary = [i for i, p in enumerate(events) if p.above is not None]
        best = result
        if len(binary) <= 8:
            options = itertools.product((True, False), repeat=len(binary))
        else:
            options = [tuple(False for _ in binary), tuple(True for _ in binary)]
            options += [tuple(i != j for i in range(len(binary))) for j in range(len(binary))]
        for option in options:
            choices = [False] * len(events)
            for i, choice in zip(binary, option):
                choices[i] = choice
            trial = solve(choices)
            if trial is not None and (best is None or trial[:2] < best[:2]):
                best = trial
    if best is None:
        raise ValueError("No feasible depth profile: slope, plateau, endpoint depth or overlapping special zones")
    # Merge artificial graph cuts, but never merge a slope change or Kdepth=1 boundary.
    for i, group in enumerate(best[2]):
        merged = []
        for piece in group:
            if merged:
                old = merged[-1]
                slope1 = (old.end_depth - old.start_depth) / old.length
                slope2 = (piece.end_depth - piece.start_depth) / piece.length
                if (old.crossing_ids == piece.crossing_ids and abs(slope1 - slope2) < 1e-8
                        and abs(old.end_depth - piece.start_depth) < 1e-7
                        and not (min(old.start_depth, piece.end_depth) < 3 - 1e-8
                                 and max(old.start_depth, piece.end_depth) > 3 + 1e-8)):
                    old.points.extend(piece.points[1:])
                    old.end_depth = piece.end_depth
                    continue
            merged.append(piece)
        best[2][i] = merged
    return best[2], events
