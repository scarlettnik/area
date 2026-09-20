"""Hard horizontal rules from the corrected appendix and participant answers."""
import math

from shapely import LineString, Point

from routing_depth import BUILDINGS, clearance


def nearest_exits(obstacle, endpoint):
    """All equidistant closest boundary feet, including holes and tied facades."""
    from heat_route_builder import dist
    feet = []
    for ring in obstacle.rings + obstacle.holes:
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


def nearest_entry_obstruction(terminal, owners, diameter):
    """A geometric witness, independent of search: every closest ray re-enters
    the same building before any bend can clear its setback.

    Between two boundary hits distance to the polygon is at most half the
    interval length. This upper bound requires no sampling or optimizer result.
    """
    from heat_route_builder import dist
    witnesses = []
    for owner in owners:
        required = clearance(owner.kind, diameter)
        rays = []
        feet = nearest_exits(owner, terminal.point)
        for foot in feet:
            nearest = dist(terminal.point, foot)
            if nearest < 1e-6:
                break
            bounds = owner.geometry.bounds
            reach = 2 * math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]) + 4 * required
            end = tuple(terminal.point[k] + (foot[k] - terminal.point[k]) * reach / nearest for k in (0, 1))
            ray = LineString([terminal.point, end])
            free = ray.difference(owner.geometry)
            spans = sorted((min(ray.project(Point(g.coords[0])), ray.project(Point(g.coords[-1]))),
                            max(ray.project(Point(g.coords[0])), ray.project(Point(g.coords[-1]))))
                           for g in getattr(free, "geoms", [free]) if g.geom_type == "LineString" and g.length > 1e-6)
            if not spans:
                break
            start, finish = spans[0]
            if abs(start - nearest) > .002 or finish >= reach - .002 or (finish - start) / 2 >= required - .002:
                break
            rays.append({"nearest_boundary_distance_m": nearest, "outside_interval_m": finish - start,
                         "maximum_possible_clearance_m": (finish - start) / 2, "required_axis_clearance_m": required})
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
        if shape.distance(Point(lead[1])) < required - 1e-6:
            raise ValueError(f"{source.id}: bend inside building entry envelope")
        inside = entry.intersection(shape)
        nearest = min(dist(endpoint, foot) for foot in feet)
        if inside.length > nearest + .002:
            raise ValueError(f"{source.id}: building transit or re-entry")
        if len(lead) > 2 and shape.distance(LineString(lead[1:])) < required - 1e-6:
            raise ValueError(f"{source.id}: building transit or re-entry")


def validate_segments(segments, obstacles, terminals, allow_entry):
    terminal_points = [t.point for t in terminals]
    incoming = {s.end_cell: s for s in segments}
    for segment in segments:
        validate_path(segment.points, segment.diameter, obstacles, terminal_points, allow_entry)
        parent = incoming.get(segment.start_cell)
        if parent:
            validate_turns([parent.points[-2], segment.points[0], segment.points[1]])
