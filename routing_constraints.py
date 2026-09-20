"""Hard horizontal rules from the corrected appendix and participant answers.

The own-OKS entry exception deserves special care.  The appendix requires the
last leg to cross a closest facade point and be straight to the connection
point, while explicitly exempting that leg from the own-building setback.  A
turn immediately outside the facade may therefore lie inside the nominal
setback.  The preceding leg is allowed to *leave* that setback from the turn,
but it may not run along it, re-enter it, or cross the building itself.

This interpretation is important for concave footprints in the corrected
sample: the closest facade can open into a narrow exterior notch whose width is
smaller than twice the nominal setback.  Treating the turn itself as if it had
to be at full setback incorrectly makes otherwise reachable consumers appear
unreachable.
"""
import math

from shapely import LineString, Point

from routing_depth import BUILDINGS, clearance


def nearest_exits(obstacle, endpoint):
    """Closest feet on the containing OKS *outer facade*.

    Interior rings describe courtyards/voids, not a facade through which an
    external heat-network route can reach the building.  Treating a hole as the
    target boundary can strand an otherwise reachable connection inside a
    closed courtyard (this occurs in the corrected sample).
    """
    from heat_route_builder import dist
    feet = []
    for ring in obstacle.rings:
        for a, b in zip(ring, ring[1:] + ring[:1]):
            length2 = dist(a, b) ** 2
            if length2 < 1e-16:
                continue
            t = max(0., min(1., sum((endpoint[k] - a[k]) * (b[k] - a[k]) for k in (0, 1)) / length2))
            foot = tuple(a[k] + t * (b[k] - a[k]) for k in (0, 1))
            feet.append((dist(endpoint, foot), foot))
    minimum = min((distance for distance, _ in feet), default=math.inf)
    return sorted({foot for distance, foot in feet if distance <= minimum + 1e-6})


def validate_turns(points, tolerance=1e-7):
    for a, b, c in zip(points, points[1:], points[2:]):
        u = b[0] - a[0], b[1] - a[1]
        v = c[0] - b[0], c[1] - b[1]
        norm = math.hypot(*u) * math.hypot(*v)
        if norm > 1e-12 and (u[0] * v[0] + u[1] * v[1]) / norm < -tolerance:
            raise ValueError("Route turn exceeds 90 degrees")


def _line_intervals(line, area, tolerance=1e-7):
    """Return merged chainage intervals where ``line`` lies inside ``area``."""
    hit = line.intersection(area)
    intervals = []
    for geom in getattr(hit, "geoms", [hit]):
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            values = [line.project(Point(geom.coords[0])), line.project(Point(geom.coords[-1]))]
            intervals.append((min(values), max(values)))
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                values = [line.project(Point(part.coords[0])), line.project(Point(part.coords[-1]))]
                intervals.append((min(values), max(values)))
        elif geom.geom_type == "Point":
            value = line.project(geom)
            intervals.append((value, value))
        elif geom.geom_type == "MultiPoint":
            intervals.extend((line.project(part), line.project(part)) for part in geom.geoms)
    intervals.sort()
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1] + tolerance:
            merged[-1] = merged[-1][0], max(merged[-1][1], hi)
        else:
            merged.append((lo, hi))
    return merged


def entry_departure_allowed(a, b, shape, required, tolerance=2e-5):
    """Allow one edge to depart from an own-building entry turn inside setback.

    ``a`` is the first point outside the facade and ``b`` is the next route
    point.  The edge may start inside ``shape.buffer(required)`` but must leave
    that envelope once, without crossing the building or later re-entering the
    envelope.  This is an endpoint relief only; it never licenses a parallel
    run in the setback.
    """
    line = LineString([a, b])
    if line.length <= 1e-9:
        return False
    # The departure leg is outside the building.  Boundary touches caused by
    # floating-point round trips are harmless, but positive transit is not.
    if line.intersection(shape).length > tolerance:
        return False
    envelope = shape.buffer(required)
    intervals = _line_intervals(line, envelope, tolerance)
    if not intervals:
        return True
    # Any setback overlap must be one prefix attached to the entry turn.
    if intervals[0][0] > tolerance or len(intervals) != 1:
        return False
    # The next graph/geometry vertex has to be outside the envelope so a second
    # turn cannot be hidden inside the own-building setback.
    return intervals[0][1] < line.length - tolerance or envelope.boundary.distance(Point(b)) <= tolerance


def nearest_entry_obstruction(terminal, owners, diameter):
    """Report only a true closest-facade dead end, not a narrow setback notch.

    A closest exit is locally usable as soon as there is positive exterior room
    for the first turn.  Full nominal setback at that turn is *not* required;
    the departure edge is validated separately by :func:`entry_departure_allowed`.
    """
    from heat_route_builder import dist
    witnesses = []
    for owner in owners:
        rays = []
        feet = nearest_exits(owner, terminal.point)
        for foot in feet:
            nearest = dist(terminal.point, foot)
            if nearest < 1e-6:
                break
            bounds = owner.geometry.bounds
            reach = 2 * math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]) + 20
            end = tuple(terminal.point[k] + (foot[k] - terminal.point[k]) * reach / nearest for k in (0, 1))
            ray = LineString([terminal.point, end])
            free = ray.difference(owner.geometry)
            spans = sorted((min(ray.project(Point(g.coords[0])), ray.project(Point(g.coords[-1]))),
                            max(ray.project(Point(g.coords[0])), ray.project(Point(g.coords[-1]))))
                           for g in getattr(free, "geoms", [free]) if g.geom_type == "LineString" and g.length > 1e-6)
            first = next((span for span in spans if abs(span[0] - nearest) <= .002), None)
            if first is None or first[1] - first[0] <= .05:
                rays.append({"nearest_boundary_distance_m": nearest,
                             "outside_interval_m": 0. if first is None else first[1] - first[0]})
        if rays and len(rays) == len(feet):
            witnesses.append({"building_id": owner.id, "all_nearest_rays_blocked": True, "rays": rays})
    return witnesses


def validate_path(points, diameter, obstacles, terminal_points=(), allow_entry=False):
    from heat_route_builder import dist, remove_collinear
    if len(points) < 2 or any(not math.isfinite(x) for p in points for x in p):
        raise ValueError("Empty/non-finite route")
    validate_turns(points)
    line = LineString(points)
    if not line.is_simple:
        raise ValueError("Route self-intersection")
    for source in obstacles:
        required = clearance(source.kind, diameter)
        shape = source.geometry
        if shape.distance(line) >= required - 1e-6:
            continue
        owners = [p for p in terminal_points if source.kind in BUILDINGS
                  and shape.covers(Point(p))
                  and min(dist(p, points[0]), dist(p, points[-1])) < .002]
        if not allow_entry or len(owners) != 1:
            raise ValueError(f"{source.id}: route crosses forbidden DN{diameter} envelope")
        endpoint = owners[0]
        lead = remove_collinear(points if dist(endpoint, points[0]) < .002 else points[::-1])
        entry = LineString(lead[:2])
        feet = nearest_exits(source, endpoint)
        if not any(entry.distance(Point(foot)) < .002 for foot in feet):
            raise ValueError(f"{source.id}: entry must use nearest building boundary")
        inside = entry.intersection(shape)
        nearest = min(dist(endpoint, foot) for foot in feet)
        # The final straight entry may cross the own polygon only from the
        # closest facade to the target.  Re-entry before the first turn remains
        # forbidden even though the setback itself is waived.
        if abs(inside.length - nearest) > .002:
            raise ValueError(f"{source.id}: building transit or re-entry")
        if len(lead) > 2:
            # The first turn may be inside the nominal own-building setback.
            # The immediately preceding leg must leave that setback once and
            # the rest of the route must obey the normal clearance again.
            if not entry_departure_allowed(lead[1], lead[2], shape, required):
                raise ValueError(f"{source.id}: invalid departure from building entry setback")
            if len(lead) > 3 and shape.distance(LineString(lead[2:])) < required - 1e-6:
                raise ValueError(f"{source.id}: building transit or re-entry")


def validate_segments(segments, obstacles, terminals, allow_entry):
    terminal_points = [t.point for t in terminals]
    incoming = {s.end_cell: s for s in segments}
    for segment in segments:
        validate_path(segment.points, segment.diameter, obstacles, terminal_points, allow_entry)
        parent = incoming.get(segment.start_cell)
        if parent:
            validate_turns([parent.points[-2], segment.points[0], segment.points[1]])
