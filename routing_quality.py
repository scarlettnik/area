"""Geometry-quality diagnostics used after the official engineering evaluation.

These metrics never replace the official score.  They are diagnostics and safe
secondary criteria for equal-score / non-worsening geometry refinement only.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bend_angles(points: Sequence[Point]) -> list[float]:
    """Return interior direction-change angles in degrees (0=straight)."""
    result: list[float] = []
    for a, b, c in zip(points, points[1:], points[2:]):
        u = (b[0] - a[0], b[1] - a[1])
        v = (c[0] - b[0], c[1] - b[1])
        lu, lv = math.hypot(*u), math.hypot(*v)
        if lu <= 1e-9 or lv <= 1e-9:
            continue
        cosine = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (lu * lv)))
        result.append(math.degrees(math.acos(cosine)))
    return result


def polyline_quality(points: Sequence[Point], micro_leg_m: float = 5.0) -> dict:
    """Deterministic quality diagnostics for one polyline."""
    if len(points) < 2:
        return {"length_m": 0.0, "direct_m": 0.0, "detour_ratio": 1.0,
                "bend_count": 0, "right_angle_count": 0, "micro_bend_count": 0,
                "backtrack_count": 0}
    legs = [_dist(a, b) for a, b in zip(points, points[1:])]
    length = sum(legs)
    direct = _dist(points[0], points[-1])
    angles = bend_angles(points)
    micro = 0
    for i, angle in enumerate(angles):
        if angle >= 8 and (legs[i] <= micro_leg_m or legs[i + 1] <= micro_leg_m):
            micro += 1
    return {
        "length_m": length,
        "direct_m": direct,
        "detour_ratio": length / direct if direct > 1e-9 else 1.0,
        "bend_count": sum(a >= 8 for a in angles),
        "right_angle_count": sum(abs(a - 90.0) <= 2.0 for a in angles),
        "micro_bend_count": micro,
        "backtrack_count": sum(a > 90.0 + 1e-7 for a in angles),
    }


def feature_collection_quality(features: Iterable[dict]) -> dict:
    """Aggregate quality metrics from exported heat_network LineStrings."""
    totals = {"line_count": 0, "bend_count": 0, "right_angle_count": 0,
              "micro_bend_count": 0, "backtrack_count": 0, "max_detour_ratio": 1.0}
    for ft in features:
        props = ft.get("properties") or {}
        geom = ft.get("geometry") or {}
        if props.get("object_type") != "heat_network" or geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        # Export is WGS84; convert back to the mandated metric CRS before
        # measuring bends/ratios so longitude/latitude anisotropy cannot skew
        # the diagnostics.
        try:
            from heat_route_builder import lonlat_to_utm37
            metric = [lonlat_to_utm37(float(p[0]), float(p[1])) for p in coords]
        except ImportError:
            metric = [(float(p[0]), float(p[1])) for p in coords]
        q = polyline_quality(metric)
        totals["line_count"] += 1
        for key in ("bend_count", "right_angle_count", "micro_bend_count", "backtrack_count"):
            totals[key] += q[key]
        totals["max_detour_ratio"] = max(totals["max_detour_ratio"], q["detour_ratio"])
    return totals
