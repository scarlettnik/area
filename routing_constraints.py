"""Hard horizontal constraints, applied after routing and every geometry edit."""
import math

from routing_depth import BUILDINGS, clearance


def validate_path(points, diameter, obstacles, terminal_points=(), allow_entry=False):
    from heat_route_builder import (Obstacle, bbox, expand_bbox, dist, ring_distance,
                                    point_in_ring, remove_collinear)
    if len(points) < 2 or any(not math.isfinite(x) for p in points for x in p):
        raise ValueError("Empty/non-finite route")
    for source in obstacles:
        required = clearance(source.kind, diameter)
        obstacle = Obstacle(source.id, source.kind, source.rings, required,
                            expand_bbox(bbox([p for r in source.rings for p in r]), required), source.holes)
        if not any(obstacle.blocks_segment(a, b) for a, b in zip(points, points[1:])):
            continue
        owners = [p for p in terminal_points if source.kind in BUILDINGS
                  and (any(point_in_ring(p, r) for r in source.rings)
                       or min(ring_distance(p, r) for r in source.rings) < .02)
                  and not any(point_in_ring(p, r) for r in source.holes)
                  and min(dist(p, points[0]), dist(p, points[-1])) < .02]
        if not allow_entry or len(owners) != 1:
            raise ValueError(f"{source.id}: route crosses forbidden DN{diameter} envelope")
        endpoint = owners[0]
        lead = remove_collinear(points if dist(endpoint, points[0]) < .02 else points[::-1])
        # The FIRST bend must already be beyond the complete building setback.
        if len(lead) > 2 and obstacle.contains_or_near(lead[1]):
            raise ValueError(f"{source.id}: bend inside building entry envelope")
        if any(obstacle.blocks_segment(a, b) for a, b in zip(lead[1:], lead[2:])):
            raise ValueError(f"{source.id}: building transit or re-entry")
        allowed = min(ring_distance(endpoint, r) for r in source.rings) + required + 12.5
        # Exact boundary intersections detect even a thin second wall between
        # samples; the samples additionally check the circular setback buffer.
        from routing_depth import intersection_parameter
        a, b = lead[:2]
        length = dist(a, b)
        cuts = {0., 1.}
        for ring in obstacle.rings + obstacle.holes:
            for c, d in zip(ring, ring[1:] + ring[:1]):
                t = intersection_parameter(a, b, c, d)
                if t is not None:
                    cuts.add(t)
        ordered = sorted(cuts)
        left = False
        for lo, hi in zip(ordered, ordered[1:]):
            t = (lo + hi) / 2
            p = tuple(a[k] + (b[k] - a[k]) * t for k in (0, 1))
            inside = any(point_in_ring(p, r) for r in obstacle.rings) and not any(point_in_ring(p, r) for r in obstacle.holes)
            if inside and left:
                raise ValueError(f"{source.id}: entry crosses building twice")
            left |= not inside
        outside = False
        n = max(1, math.ceil(length / .2))
        for i in range(n + 1):
            p = tuple(a[k] + (b[k] - a[k]) * i / n for k in (0, 1))
            blocked = obstacle.contains_or_near(p)
            if blocked and (outside or dist(endpoint, p) > allowed):
                raise ValueError(f"{source.id}: long or repeated building entry")
            outside |= not blocked


def validate_segments(segments, obstacles, terminals, allow_entry):
    terminal_points = [t.point for t in terminals]
    for segment in segments:
        validate_path(segment.points, segment.diameter, obstacles, terminal_points, allow_entry)
