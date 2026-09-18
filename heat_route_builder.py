#!/usr/bin/env python3
"""
Obstacle-aware heat network routing prototype.

The script uses only the standard library. It projects WGS84 to UTM zone 37N,
builds a checked corridor graph, chooses tie-ins by construction cost, and
improves branch topology. Ranking minimizes the official weighted score; close
scores use route geometry as an internal presentation tie-break. The technical
appendix's weighted score is retained as a separate diagnostic field.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import cached_property, cmp_to_key
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from routing_depth import (BUILDINGS, SPECIAL, ENVELOPE, clearance, read_crossing_objects,
                           transformed_objects, build_depth_profiles)


Point = Tuple[float, float]
Cell = Tuple[int, int]
State = Tuple[Cell, int]


DIRECTIONS: List[Cell] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]

CARDINAL_DIRECTIONS: List[Cell] = [
    (0, -1),
    (-1, 0),
    (1, 0),
    (0, 1),
]


DIAMETERS = [
    # diameter, capacity_tph, max_len_m, new_rub_m, reconstruction_rub_m
    (50, 3.5, 181, 74023, 96180),
    (65, 8.3, 245, 78631, 109989),
    (80, 13.2, 327, 83530, 117582),
    (100, 22.3, 419, 89748, 133694),
    (125, 40.2, 554, 97275, 148030),
    (150, 65.1, 696, 105507, 152295),
    (200, 152.3, 1042, 120275, 181766),
    (250, 274.9, 1379, 135323, 202030),
    (300, 437.4, 1718, 150022, 228707),
    (400, 943.1, 2477, 190299, 271317),
    (500, 1663.4, 3245, 224137, 333884),
    (600, 2627.7, 4037, 264790, 372703),
    (700, 3735.1, 4775, 324298, 439571),
    (800, 5296.8, 5644, 325996, 489918),
    (900, 7165.0, 6518, 327693, 553607),
    (1000, 9391.8, 7419, 418777, 606679),
    (1200, 15012.8, 9288, 428074, 825692),
    (1400, 22501.9, 11276, 683417, 978584),
]


CAPACITY = {d: cap for d, cap, _ml, _n, _r in DIAMETERS}
MAX_LENGTH = {d: ml for d, _cap, ml, _n, _r in DIAMETERS}
NEW_COST = {d: n for d, _cap, _ml, n, _r in DIAMETERS}
RECON_COST = {d: r for d, _cap, _ml, _n, r in DIAMETERS}
TIE_IN_COST = 5_000_000
SCORE_TIE_EPSILON = 0.01


def chamber_cost(diameter: int) -> int:
    if diameter <= 200:
        return 3_000_000
    if diameter <= 500:
        return 5_000_000
    if diameter <= 1000:
        return 8_000_000
    return 12_000_000


def select_diameter(flow_tph: float, min_length: float = 0.0) -> int:
    if not math.isfinite(flow_tph) or flow_tph < 0 or not math.isfinite(min_length) or min_length < 0:
        raise ValueError("Flow and length must be finite and non-negative")
    for diameter, capacity, max_len, _new_cost, _recon_cost in DIAMETERS:
        if capacity + 1e-9 >= flow_tph and max_len + 1e-9 >= min_length:
            return diameter
    raise ValueError(f"No supported diameter for {flow_tph} t/h over {min_length} m")


def penalty_unconnected(flow_tph: float) -> float:
    return 100_000_000 + 500_000 * flow_tph


# WGS84 <-> UTM zone 37N. The technical appendix requires EPSG:32637.
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
UTM_K0 = 0.9996
UTM_FALSE_EASTING = 500000.0
UTM_ZONE_37_LON0 = math.radians(39.0)


def lonlat_to_utm37(lon: float, lat: float) -> Point:
    e2 = WGS84_F * (2.0 - WGS84_F)
    ep2 = e2 / (1.0 - e2)
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    tan_lat = math.tan(lat_r)
    n = WGS84_A / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = ep2 * cos_lat * cos_lat
    a = cos_lat * (lon_r - UTM_ZONE_37_LON0)
    m = WGS84_A * (
        (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2**3 / 256) * lat_r
        - (3 * e2 / 8 + 3 * e2 * e2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_r)
        + (15 * e2 * e2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_r)
        - (35 * e2**3 / 3072) * math.sin(6 * lat_r)
    )
    x = UTM_FALSE_EASTING + UTM_K0 * n * (
        a
        + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * a**5 / 120
    )
    y = UTM_K0 * (
        m
        + n
        * tan_lat
        * (
            a * a / 2
            + (5 - t + 9 * c + 4 * c * c) * a**4 / 24
            + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * a**6 / 720
        )
    )
    return x, y


def utm37_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    e2 = WGS84_F * (2.0 - WGS84_F)
    ep2 = e2 / (1.0 - e2)
    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))
    m = y / UTM_K0
    mu = m / (WGS84_A * (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2**3 / 256))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1 * e1 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )
    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)
    n1 = WGS84_A / math.sqrt(1.0 - e2 * sin_phi1 * sin_phi1)
    r1 = WGS84_A * (1.0 - e2) / (1.0 - e2 * sin_phi1 * sin_phi1) ** 1.5
    t1 = tan_phi1 * tan_phi1
    c1 = ep2 * cos_phi1 * cos_phi1
    d = (x - UTM_FALSE_EASTING) / (n1 * UTM_K0)
    lat = phi1 - (n1 * tan_phi1 / r1) * (
        d * d / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * ep2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * ep2 - 3 * c1 * c1)
        * d**6
        / 720
    )
    lon = UTM_ZONE_37_LON0 + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * ep2 + 24 * t1 * t1)
        * d**5
        / 120
    ) / cos_phi1
    return math.degrees(lon), math.degrees(lat)


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(points: Sequence[Point]) -> float:
    return sum(dist(a, b) for a, b in zip(points, points[1:]))


def turn_penalty(
    prev_dir: int,
    next_dir: int,
    penalty_for_90_deg: float,
    directions: Sequence[Cell],
) -> float:
    if prev_dir < 0 or prev_dir == next_dir or penalty_for_90_deg <= 0:
        return 0.0
    ax, ay = directions[prev_dir]
    bx, by = directions[next_dir]
    dot = ax * bx + ay * by
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)
    cos_angle = max(-1.0, min(1.0, dot / (la * lb)))
    angle = math.acos(cos_angle)
    if angle < math.radians(8.0):
        return 0.0
    return penalty_for_90_deg * angle / (math.pi / 2.0)


def polyline_bend_count(points: Sequence[Point], min_angle_deg: float = 15.0) -> int:
    if len(points) < 3:
        return 0
    count = 0
    for a, b, c in zip(points, points[1:], points[2:]):
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        l1 = math.hypot(*v1)
        l2 = math.hypot(*v2)
        if l1 <= 1e-9 or l2 <= 1e-9:
            continue
        cos_angle = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
        angle = math.degrees(math.acos(cos_angle))
        if angle >= min_angle_deg:
            count += 1
    return count


def polyline_micro_bend_count(points: Sequence[Point], max_adjacent_length: float) -> int:
    """Count bends next to short legs that make a route look stair-stepped."""
    if len(points) < 3:
        return 0
    count = 0
    for a, b, c in zip(points, points[1:], points[2:]):
        if (dist(a, b) <= max_adjacent_length or dist(b, c) <= max_adjacent_length) \
                and polyline_bend_count((a, b, c)):
            count += 1
    return count


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return dist(p, a)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def point_in_ring(p: Point, ring: Sequence[Point]) -> bool:
    x, y = p
    inside = False
    n = len(ring)
    if n < 3:
        return False
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        if (min(a[0], b[0]) - 1e-7 <= x <= max(a[0], b[0]) + 1e-7
                and min(a[1], b[1]) - 1e-7 <= y <= max(a[1], b[1]) + 1e-7
                and point_segment_distance(p, a, b) < 1e-7):
            return True
        xi, yi = a
        xj, yj = b
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi
        ):
            inside = not inside
    return inside


def segments_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    """Exact segment clearance, including intersections between raster nodes."""
    def cross(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    if (max(a[0], b[0]) >= min(c[0], d[0]) and max(c[0], d[0]) >= min(a[0], b[0])
            and max(a[1], b[1]) >= min(c[1], d[1]) and max(c[1], d[1]) >= min(a[1], b[1])
            and cross(a, b, c) * cross(a, b, d) <= 0
            and cross(c, d, a) * cross(c, d, b) <= 0):
        return 0.0
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def ring_distance(p: Point, ring: Sequence[Point]) -> float:
    if len(ring) < 2:
        return float("inf")
    return min(point_segment_distance(p, ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring)))


def bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def expand_bbox(bb: Tuple[float, float, float, float], amount: float) -> Tuple[float, float, float, float]:
    return bb[0] - amount, bb[1] - amount, bb[2] + amount, bb[3] + amount


def in_bbox(p: Point, bb: Tuple[float, float, float, float]) -> bool:
    return bb[0] <= p[0] <= bb[2] and bb[1] <= p[1] <= bb[3]


def iter_geojson_positions(geom: Dict[str, Any]) -> Iterable[Tuple[float, float]]:
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return
    if gtype == "Point":
        yield (coords[0], coords[1])
    elif gtype == "LineString":
        for c in coords:
            yield (c[0], c[1])
    elif gtype == "Polygon":
        for ring in coords:
            for c in ring:
                yield (c[0], c[1])
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for c in ring:
                    yield (c[0], c[1])


def project_geom_coords(geom: Dict[str, Any]) -> Any:
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Point":
        return lonlat_to_utm37(coords[0], coords[1])
    if gtype == "LineString":
        return [lonlat_to_utm37(c[0], c[1]) for c in coords]
    if gtype == "Polygon":
        return [[lonlat_to_utm37(c[0], c[1]) for c in ring] for ring in coords]
    if gtype == "MultiPolygon":
        return [
            [[lonlat_to_utm37(c[0], c[1]) for c in ring] for ring in poly]
            for poly in coords
        ]
    raise ValueError(f"Unsupported geometry type: {gtype}")


def unproject_point(p: Point) -> List[float]:
    lon, lat = utm37_to_lonlat(p[0], p[1])
    return [round(lon, 12), round(lat, 12)]


def unproject_linestring(points: Sequence[Point]) -> Dict[str, Any]:
    return {"type": "LineString", "coordinates": [unproject_point(p) for p in points]}


def unproject_point_geom(p: Point) -> Dict[str, Any]:
    return {"type": "Point", "coordinates": unproject_point(p)}


@dataclass
class Terminal:
    id: str
    point: Point
    flow_tph: float
    cell: Optional[Cell] = None


@dataclass
class TieCandidate:
    id: str
    point: Point
    existing_object_id: str
    existing_object_type: str
    existing_diameter: int
    is_existing_chamber: bool
    cell: Optional[Cell] = None


@dataclass
class Obstacle:
    id: str
    kind: str
    rings: List[List[Point]]
    clearance: float
    bbox: Tuple[float, float, float, float]
    holes: List[List[Point]] = field(default_factory=list)

    @cached_property
    def boundary_index(self) -> Dict[Cell, List[Tuple[Point, Point]]]:
        buckets: Dict[Cell, List[Tuple[Point, Point]]] = defaultdict(list)
        for ring in self.rings + self.holes:
            for a, b in zip(ring, ring[1:] + ring[:1]):
                for x in range(math.floor(min(a[0], b[0]) / 20), math.floor(max(a[0], b[0]) / 20) + 1):
                    for y in range(math.floor(min(a[1], b[1]) / 20), math.floor(max(a[1], b[1]) / 20) + 1):
                        buckets[x, y].append((a, b))
        return buckets

    def nearby_boundary(self, a: Point, b: Point) -> Iterable[Tuple[Point, Point]]:
        seen = set()
        bounds = expand_bbox(bbox([a, b]), self.clearance + 1e-8)
        for x in range(math.floor(bounds[0] / 20), math.floor(bounds[2] / 20) + 1):
            for y in range(math.floor(bounds[1] / 20), math.floor(bounds[3] / 20) + 1):
                for edge in self.boundary_index.get((x, y), []):
                    if edge not in seen:
                        seen.add(edge)
                        yield edge

    def contains_or_near(self, p: Point) -> bool:
        if not in_bbox(p, self.bbox):
            return False
        inside = any(point_in_ring(p, ring) for ring in self.rings)
        in_hole = any(point_in_ring(p, ring) for ring in self.holes)
        return (inside and not in_hole) or any(point_segment_distance(p, a, b) <= self.clearance
                                              for a, b in self.nearby_boundary(p, p))

    def blocks_segment(self, a: Point, b: Point) -> bool:
        if (max(a[0], b[0]) < self.bbox[0] or min(a[0], b[0]) > self.bbox[2]
                or max(a[1], b[1]) < self.bbox[1] or min(a[1], b[1]) > self.bbox[3]):
            return False
        return self.contains_or_near(a) or self.contains_or_near(b) or any(
            segments_distance(a, b, c, d) <= self.clearance + 1e-8
            for c, d in self.nearby_boundary(a, b)
        )


@dataclass
class CoordinateTransform:
    origin: Point
    angle_rad: float

    def forward(self, p: Point) -> Point:
        if abs(self.angle_rad) <= 1e-12:
            return p
        dx = p[0] - self.origin[0]
        dy = p[1] - self.origin[1]
        c = math.cos(self.angle_rad)
        s = math.sin(self.angle_rad)
        # Rotate by -angle so the dominant city axis becomes local X.
        return c * dx + s * dy, -s * dx + c * dy

    def inverse(self, p: Point) -> Point:
        if abs(self.angle_rad) <= 1e-12:
            return p
        c = math.cos(self.angle_rad)
        s = math.sin(self.angle_rad)
        return (
            self.origin[0] + c * p[0] - s * p[1],
            self.origin[1] + s * p[0] + c * p[1],
        )


IDENTITY_TRANSFORM = CoordinateTransform(origin=(0.0, 0.0), angle_rad=0.0)


class RoutingGrid:
    def __init__(
        self,
        obstacles: List[Obstacle],
        points: List[Point],
        step: float = 5.0,
        margin: float = 120.0,
        cardinal_only: bool = False,
        coord_transform: CoordinateTransform = IDENTITY_TRANSFORM,
        allow_building_leads: bool = False,
    ):
        if not math.isfinite(step) or step <= 0:
            raise ValueError("Grid step must be finite and positive")
        if not points:
            raise ValueError("Routing requires at least one coordinate")
        self.obstacles = obstacles
        self.step = step
        self.cardinal_only = cardinal_only
        self.directions = CARDINAL_DIRECTIONS if cardinal_only else DIRECTIONS
        self.coord_transform = coord_transform
        self.allow_building_leads = allow_building_leads
        self.crossing_objects = []
        self.connection_adjustments = []
        self.depth_enabled = True
        self._crossing_edge_cache = {}
        minx = min(p[0] for p in points) - margin
        miny = min(p[1] for p in points) - margin
        maxx = max(p[0] for p in points) + margin
        maxy = max(p[1] for p in points) + margin
        self.x0 = math.floor(minx / step) * step
        self.y0 = math.floor(miny / step) * step
        self.nx = int(math.ceil((maxx - self.x0) / step)) + 1
        self.ny = int(math.ceil((maxy - self.y0) / step)) + 1
        self.blocked = [[False] * self.nx for _ in range(self.ny)]
        self.extra_points: Dict[Cell, Point] = {}
        self.access_nodes: Set[Cell] = set()
        self.corridor_contacts: Dict[Point, Cell] = {}
        self.split_edges: Dict[Tuple[Cell, Cell], List[Cell]] = defaultdict(list)
        self.split_neighbors: Dict[Cell, List[Tuple[Cell, float, int]]] = defaultdict(list)
        self.extra_neighbors: Dict[Cell, List[Tuple[Cell, float, int]]] = defaultdict(list)
        self.edge_geometry: Dict[Tuple[Cell, Cell], List[Point]] = {}
        self.terminal_cells: Set[Cell] = set()
        self._neighbor_cache: Dict[Cell, List[Tuple[Cell, float, int]]] = {}
        self._bucket_size = max(40.0, step * 8)
        self._obstacle_buckets: Dict[Cell, List[Obstacle]] = defaultdict(list)
        for obs in obstacles:
            for key in self._buckets(obs.bbox):
                self._obstacle_buckets[key].append(obs)
        self._build_blocked()

    def _buckets(self, bounds: Tuple[float, float, float, float]) -> Iterable[Cell]:
        size = self._bucket_size
        for x in range(math.floor(bounds[0] / size), math.floor(bounds[2] / size) + 1):
            for y in range(math.floor(bounds[1] / size), math.floor(bounds[3] / size) + 1):
                yield x, y

    def point_to_cell_float(self, p: Point) -> Tuple[float, float]:
        return (p[0] - self.x0) / self.step, (p[1] - self.y0) / self.step

    def point_to_cell(self, p: Point) -> Cell:
        fx, fy = self.point_to_cell_float(p)
        return int(round(fx)), int(round(fy))

    def cell_to_point(self, c: Cell) -> Point:
        if c in self.extra_points:
            return self.extra_points[c]
        return self.x0 + c[0] * self.step, self.y0 + c[1] * self.step

    def to_utm(self, p: Point) -> Point:
        return self.coord_transform.inverse(p)

    def in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny

    def is_blocked_cell(self, c: Cell) -> bool:
        if c in self.extra_points:
            return False
        if not self.in_bounds(c):
            return True
        return self.blocked[c[1]][c[0]]

    def is_blocked_point(self, p: Point) -> bool:
        return any(obs.contains_or_near(p) for obs in self.obstacles)

    def _build_blocked(self) -> None:
        for obs in self.obstacles:
            ix0 = max(0, int(math.floor((obs.bbox[0] - self.x0) / self.step)) - 1)
            iy0 = max(0, int(math.floor((obs.bbox[1] - self.y0) / self.step)) - 1)
            ix1 = min(self.nx - 1, int(math.ceil((obs.bbox[2] - self.x0) / self.step)) + 1)
            iy1 = min(self.ny - 1, int(math.ceil((obs.bbox[3] - self.y0) / self.step)) + 1)
            for iy in range(iy0, iy1 + 1):
                y = self.y0 + iy * self.step
                row = self.blocked[iy]
                for ix in range(ix0, ix1 + 1):
                    if row[ix]:
                        continue
                    p = (self.x0 + ix * self.step, y)
                    if obs.contains_or_near(p):
                        row[ix] = True

    def nearest_free_cell(self, p: Point, max_radius: float = 80.0) -> Optional[Cell]:
        start = self.point_to_cell(p)
        max_r = int(math.ceil(max_radius / self.step))
        best: Optional[Tuple[float, Cell]] = None
        for r in range(max_r + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    cell = (start[0] + dx, start[1] + dy)
                    if self.in_bounds(cell) and not self.is_blocked_cell(cell):
                        d = dist(p, self.cell_to_point(cell))
                        if best is None or d < best[0]:
                            best = (d, cell)
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    cell = (start[0] + dx, start[1] + dy)
                    if self.in_bounds(cell) and not self.is_blocked_cell(cell):
                        d = dist(p, self.cell_to_point(cell))
                        if best is None or d < best[0]:
                            best = (d, cell)
            if best is not None:
                return best[1]
        return None

    def neighbors(self, cell: Cell) -> Iterable[Tuple[Cell, float]]:
        for nxt, length, _direction in self.neighbors_with_dirs(cell):
            yield nxt, length

    def neighbors_with_dirs(self, cell: Cell) -> Iterable[Tuple[Cell, float, int]]:
        if cell in self._neighbor_cache:
            return self._neighbor_cache[cell]
        result = list(self.extra_neighbors.get(cell, [])) + list(self.split_neighbors.get(cell, []))
        if cell in self.extra_points:
            return result
        x, y = cell
        for dir_idx, (dx, dy) in enumerate(self.directions):
            nxt = (x + dx, y + dy)
            split = self.split_edges.get(edge_key(cell, nxt))
            if split:
                contact = min(split, key=lambda c: dist(self.cell_to_point(cell), self.cell_to_point(c)))
                if self.line_clear(self.cell_to_point(cell), self.cell_to_point(contact)):
                    result.append((contact, dist(self.cell_to_point(cell), self.cell_to_point(contact)), dir_idx))
                continue
            if self.is_blocked_cell(nxt):
                continue
            if not self.line_clear(self.cell_to_point(cell), self.cell_to_point(nxt)):
                continue
            if dx and dy:
                if self.is_blocked_cell((x + dx, y)) or self.is_blocked_cell((x, y + dy)):
                    continue
                result.append((nxt, self.step * math.sqrt(2.0), dir_idx))
            else:
                result.append((nxt, self.step, dir_idx))
        self._neighbor_cache[cell] = result
        return result

    def edge_points(self, a: Cell, b: Cell) -> List[Point]:
        key = edge_key(a, b)
        points = self.edge_geometry.get(key)
        if points is None:
            return [self.cell_to_point(a), self.cell_to_point(b)]
        return points if a == key[0] else list(reversed(points))

    def add_access(self, point: Point, terminal: bool = False) -> Optional[Cell]:
        """Connect the actual node to several visible corridor cells, never teleport it.

        Input building connection points can lie inside their own footprint. Only
        their short service lead may leave that footprint; other obstacles remain
        closed. Multiple exits avoid snapping a consumer into a sealed courtyard.
        """
        owners = [o for o in self.obstacles if o.kind in {"oks", "oks_existing", "oks_future"}
                  and o.contains_or_near(point)] if terminal and self.allow_building_leads else []
        if self.is_blocked_point(point) and not owners:
            return None
        owner_ids = {id(o) for o in owners}
        exit_distance = max((min(ring_distance(point, ring) for ring in o.rings)
                             + o.clearance for o in owners), default=0.0)
        # A physical access envelope must not shrink with mesh refinement:
        # otherwise a finer grid can snap all exits into an enclosed courtyard.
        radius = exit_distance + max(12.5, 2.5 * self.step)
        start = self.point_to_cell(point)
        if dist(point, self.cell_to_point(start)) < 1e-8 and not self.is_blocked_cell(start):
            if terminal:
                self.terminal_cells.add(start)
            return start
        count = math.ceil(radius / self.step)
        exits: List[Tuple[float, Cell, List[Point]]] = []
        for dx in range(-count, count + 1):
            for dy in range(-count, count + 1):
                cell = start[0] + dx, start[1] + dy
                if self.is_blocked_cell(cell):
                    continue
                end = self.cell_to_point(cell)
                if dist(point, end) > radius:
                    continue
                paths = [[point, (point[0], end[1]), end], [point, (end[0], point[1]), end]]
                if owners or not self.cardinal_only:
                    paths = [[point, end]]
                for path in paths:
                    path = [p for i, p in enumerate(path) if i == 0 or dist(p, path[i - 1]) > 1e-8]
                    if len(path) == 1:
                        path.append(end)
                    if any(o.blocks_segment(a, b) for o in self.obstacles if id(o) not in owner_ids
                           for a, b in zip(path, path[1:])):
                        continue
                    # The lead must leave its own building once, with no re-entry.
                    valid = True
                    for owner in owners:
                        outside = False
                        for a, b in zip(path, path[1:]):
                            samples = max(1, math.ceil(dist(a, b) / 0.25))
                            for i in range(samples + 1):
                                p = (a[0] + (b[0] - a[0]) * i / samples,
                                     a[1] + (b[1] - a[1]) * i / samples)
                                blocked = owner.contains_or_near(p)
                                if outside and blocked:
                                    valid = False
                                    break
                                outside |= not blocked
                            if not valid:
                                break
                    if valid:
                        exit_cell, lead = self.first_corridor_contact(path, cell)
                        exits.append((path_length(lead), exit_cell, lead))
        if not exits:
            return None
        node = (-1, len(self.extra_points))
        self.extra_points[node] = point
        self.access_nodes.add(node)
        if terminal:
            self.terminal_cells.add(node)
        used_exits: Set[Cell] = set()
        for length, cell, path in sorted(exits):
            if cell in used_exits:
                continue
            used_exits.add(cell)
            key = edge_key(node, cell)
            self.edge_geometry[key] = path if key[0] == node else list(reversed(path))
            # Direction of the last leg on arrival at the regular grid.
            a, b = path[-2:]
            vector = b[0] - a[0], b[1] - a[1]
            direction = max(range(len(self.directions)), key=lambda i:
                            (vector[0] * self.directions[i][0] + vector[1] * self.directions[i][1])
                            / math.hypot(*self.directions[i]))
            reverse = self.directions.index(tuple(-v for v in self.directions[direction]))
            self.extra_neighbors[node].append((cell, length, direction))
            self.extra_neighbors[cell].append((node, length, reverse))
            self._neighbor_cache.pop(cell, None)
        return node

    def first_corridor_contact(self, path: List[Point], fallback: Cell) -> Tuple[Cell, List[Point]]:
        """A service lead ends at the first free grid node it touches.

        Hiding intermediate corridor nodes inside a long connector creates
        duplicate overlapping pipes and junctions that the flow graph cannot see.
        """
        for index, (a, b) in enumerate(zip(path, path[1:])):
            low = self.point_to_cell((min(a[0], b[0]), min(a[1], b[1])))
            high = self.point_to_cell((max(a[0], b[0]), max(a[1], b[1])))
            contacts = []
            if self.cardinal_only and (abs(a[0] - b[0]) < 1e-8 or abs(a[1] - b[1]) < 1e-8):
                if abs(a[0] - b[0]) < 1e-8:
                    for y in range(low[1] - 1, high[1] + 2):
                        p = a[0], self.y0 + y * self.step
                        if point_segment_distance(p, a, b) < 1e-8 and not self.is_blocked_point(p):
                            contacts.append((dist(a, p), p))
                elif abs(a[1] - b[1]) < 1e-8:
                    for x in range(low[0] - 1, high[0] + 2):
                        p = self.x0 + x * self.step, a[1]
                        if point_segment_distance(p, a, b) < 1e-8 and not self.is_blocked_point(p):
                            contacts.append((dist(a, p), p))
                if contacts:
                    _distance, point = min(contacts)
                    lead = path[:index + 1]
                    if dist(lead[-1], point) > 1e-8:
                        lead.append(point)
                    if len(lead) >= 2:
                        return self.insert_corridor_contact(point), lead
                continue
            for x in range(low[0] - 1, high[0] + 2):
                for y in range(low[1] - 1, high[1] + 2):
                    cell = x, y
                    point = self.cell_to_point(cell)
                    if not self.is_blocked_cell(cell) and point_segment_distance(point, a, b) < 1e-8:
                        contacts.append((dist(a, point), cell, point))
            if contacts:
                _distance, cell, point = min(contacts)
                lead = path[:index + 1]
                if dist(lead[-1], point) > 1e-8:
                    lead.append(point)
                if len(lead) >= 2:
                    return cell, lead
        return fallback, path

    def insert_corridor_contact(self, point: Point) -> Cell:
        """Split a corridor at the real service-junction coordinate, even off-grid."""
        nearest = self.point_to_cell(point)
        if dist(self.cell_to_point(nearest), point) < 1e-7:
            return nearest
        point = round(point[0], 8), round(point[1], 8)
        if point in self.corridor_contacts:
            return self.corridor_contacts[point]
        fx, fy = self.point_to_cell_float(point)
        if abs(fy - round(fy)) < 1e-6:
            a, b = (math.floor(fx), round(fy)), (math.floor(fx) + 1, round(fy))
        else:
            a, b = (round(fx), math.floor(fy)), (round(fx), math.floor(fy) + 1)
        node = (-2, len(self.corridor_contacts))
        self.corridor_contacts[point] = node
        self.extra_points[node] = point
        key = edge_key(a, b)
        self.split_edges[key].append(node)
        ordered = sorted(self.split_edges[key], key=lambda c: dist(self.cell_to_point(a), self.cell_to_point(c)))
        for cell in ordered:
            self.split_neighbors[cell] = []
        for left, right in zip([a] + ordered, ordered + [b]):
            p, q = self.cell_to_point(left), self.cell_to_point(right)
            if self.is_blocked_cell(left) or self.is_blocked_cell(right) or not self.line_clear(p, q):
                continue
            direction = max(range(len(self.directions)), key=lambda i:
                            (q[0] - p[0]) * self.directions[i][0] + (q[1] - p[1]) * self.directions[i][1])
            reverse = self.directions.index(tuple(-v for v in self.directions[direction]))
            if left in self.extra_points:
                self.split_neighbors[left].append((right, dist(p, q), direction))
            if right in self.extra_points:
                self.split_neighbors[right].append((left, dist(p, q), reverse))
        self._neighbor_cache.pop(a, None)
        self._neighbor_cache.pop(b, None)
        return node

    def dijkstra_from_sources(self, sources: Set[Cell], targets: Set[Cell]) -> Tuple[Dict[Cell, float], Dict[Cell, Cell]]:
        dist_map: Dict[Cell, float] = {}
        prev: Dict[Cell, Cell] = {}
        heap: List[Tuple[float, Cell]] = []
        for s in sources:
            if self.is_blocked_cell(s):
                continue
            dist_map[s] = 0.0
            heapq.heappush(heap, (0.0, s))
        found = 0
        remaining_targets = set(targets)
        while heap and found < len(targets):
            cur_dist, cur = heapq.heappop(heap)
            if cur_dist != dist_map.get(cur):
                continue
            if cur in remaining_targets:
                remaining_targets.remove(cur)
                found += 1
            for nxt, weight in self.neighbors(cur):
                nd = cur_dist + weight
                if nd < dist_map.get(nxt, float("inf")):
                    dist_map[nxt] = nd
                    prev[nxt] = cur
                    heapq.heappush(heap, (nd, nxt))
        return dist_map, prev

    def reconstruct_to_source(self, target: Cell, prev: Dict[Cell, Cell], sources: Set[Cell]) -> List[Cell]:
        if target in sources:
            return [target]
        path = [target]
        cur = target
        while cur not in sources:
            if cur not in prev:
                return []
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path

    def heading_dijkstra_from_sources(
        self,
        sources: Set[Cell],
        targets: Set[Cell],
        turn_penalty_m: float,
    ) -> Tuple[Dict[Cell, float], Dict[State, Optional[State]], Dict[Cell, State]]:
        dist_state: Dict[State, float] = {}
        prev_state: Dict[State, Optional[State]] = {}
        best_cell_dist: Dict[Cell, float] = {}
        best_cell_state: Dict[Cell, State] = {}
        heap: List[Tuple[float, int, State]] = []
        seq = 0
        for source in sources:
            if self.is_blocked_cell(source):
                continue
            state = (source, -1)
            dist_state[state] = 0.0
            prev_state[state] = None
            heapq.heappush(heap, (0.0, seq, state))
            seq += 1

        remaining_targets = set(targets)
        while heap and remaining_targets:
            cur_dist, _seq, state = heapq.heappop(heap)
            if cur_dist != dist_state.get(state):
                continue
            cell, prev_dir = state
            if cell not in best_cell_dist:
                best_cell_dist[cell] = cur_dist
                best_cell_state[cell] = state
                remaining_targets.discard(cell)
            for nxt, weight, next_dir in self.neighbors_with_dirs(cell):
                nd = cur_dist + weight + turn_penalty(prev_dir, next_dir, turn_penalty_m, self.directions)
                nxt_state = (nxt, next_dir)
                if nd < dist_state.get(nxt_state, float("inf")):
                    dist_state[nxt_state] = nd
                    prev_state[nxt_state] = state
                    heapq.heappush(heap, (nd, seq, nxt_state))
                    seq += 1
        return best_cell_dist, prev_state, best_cell_state

    def reconstruct_heading_to_source(
        self,
        target_state: State,
        prev_state: Dict[State, Optional[State]],
    ) -> List[Cell]:
        cells: List[Cell] = []
        cur: Optional[State] = target_state
        while cur is not None:
            cells.append(cur[0])
            cur = prev_state.get(cur)
        cells.reverse()
        compact: List[Cell] = []
        for cell in cells:
            if not compact or compact[-1] != cell:
                compact.append(cell)
        return compact

    def heading_astar_to_target(
        self,
        sources: Set[Cell],
        target: Cell,
        turn_penalty_m: float,
    ) -> Tuple[float, List[Cell]]:
        dist_state: Dict[State, float] = {}
        prev_state: Dict[State, Optional[State]] = {}
        heap: List[Tuple[float, int, State]] = []
        seq = 0
        target_point = self.cell_to_point(target)
        for source in sources:
            if self.is_blocked_cell(source):
                continue
            state = (source, -1)
            g = 0.0
            h = dist(self.cell_to_point(source), target_point)
            dist_state[state] = g
            prev_state[state] = None
            heapq.heappush(heap, (g + h, seq, state))
            seq += 1

        while heap:
            _f, _seq, state = heapq.heappop(heap)
            cur_cost = dist_state.get(state)
            if cur_cost is None:
                continue
            cell, prev_dir = state
            if cell == target:
                return cur_cost, self.reconstruct_heading_to_source(state, prev_state)
            for nxt, weight, next_dir in self.neighbors_with_dirs(cell):
                nd = cur_cost + weight + turn_penalty(prev_dir, next_dir, turn_penalty_m, self.directions)
                nxt_state = (nxt, next_dir)
                if nd < dist_state.get(nxt_state, float("inf")):
                    dist_state[nxt_state] = nd
                    prev_state[nxt_state] = state
                    h = dist(self.cell_to_point(nxt), target_point)
                    heapq.heappush(heap, (nd + h, seq, nxt_state))
                    seq += 1
        return float("inf"), []

    def line_clear(self, a: Point, b: Point, endpoint_relief: float = 0.0) -> bool:
        checked: Set[int] = set()
        for bucket in self._buckets(bbox([a, b])):
            for obs in self._obstacle_buckets.get(bucket, []):
                if id(obs) in checked:
                    continue
                checked.add(id(obs))
                if obs.blocks_segment(a, b):
                    return False
        return True

    def service_line_clear(self, a: Point, b: Point) -> bool:
        """A straight, short final lead may enter only the endpoint's own building."""
        if self.line_clear(a, b):
            return True
        if not self.allow_building_leads:
            return False
        endpoints = [self.cell_to_point(c) for c in self.terminal_cells]
        if any(dist(a, p) < 1e-8 for p in endpoints):
            endpoint, outside = a, b
        elif any(dist(b, p) < 1e-8 for p in endpoints):
            endpoint, outside = b, a
        else:
            return False
        owners = [o for o in self.obstacles if o.kind in BUILDINGS and o.contains_or_near(endpoint)]
        if not owners or self.is_blocked_point(outside):
            return False
        for obstacle in self.obstacles:
            if not obstacle.blocks_segment(a, b):
                continue
            if obstacle not in owners:
                return False
            max_length = min(ring_distance(endpoint, ring) for ring in obstacle.rings) + obstacle.clearance + max(12.5, 2.5 * self.step)
            if dist(a, b) > max_length:
                return False
            exited = False
            samples = max(1, math.ceil(dist(a, b) / .25))
            for i in range(samples + 1):
                point = tuple(endpoint[k] + (outside[k] - endpoint[k]) * i / samples for k in (0, 1))
                blocked = obstacle.contains_or_near(point)
                if exited and blocked:
                    return False
                exited |= not blocked
        return True

    def least_cost_path(
        self, sources: Dict[Cell, float], target: Cell, blocked: Set[Cell],
        rub_per_m: float, bend_cost_rub: float = 0.0,
    ) -> List[Cell]:
        """Multi-source heading A*: source prices include opening/upgrading a tree.

        Manhattan/Euclidean distance is an admissible lower bound; stale queue
        entries are discarded. Equal monetary paths prefer fewer turns.
        """
        target_point = self.cell_to_point(target)
        def heuristic(cell: Cell) -> float:
            p = self.cell_to_point(cell)
            dx, dy = abs(p[0] - target_point[0]), abs(p[1] - target_point[1])
            return (dx + dy if self.cardinal_only else math.hypot(dx, dy)) * rub_per_m

        diameter = min(NEW_COST, key=lambda d: abs(NEW_COST[d] - rub_per_m))
        best: Dict[State, Tuple[float, int]] = {}
        previous: Dict[State, Optional[State]] = {}
        heap = []
        for cell, price in sorted(sources.items()):
            state = cell, -1
            best[state] = (price, 0)
            previous[state] = None
            heapq.heappush(heap, (price + heuristic(cell), 0, price, state))
        while heap:
            _estimate, turns, price, state = heapq.heappop(heap)
            if best.get(state) != (price, turns):
                continue
            cell, incoming = state
            if cell == target:
                return self.reconstruct_heading_to_source(state, previous)
            for nxt, length, direction in self.neighbors_with_dirs(cell):
                if nxt in blocked or (nxt in self.access_nodes and nxt != target) or (nxt in self.terminal_cells and nxt != target):
                    continue
                first_direction = direction
                internal_turns = 0
                if edge_key(cell, nxt) in self.edge_geometry:
                    points = self.edge_points(cell, nxt)
                    internal_turns = polyline_bend_count(points)
                    headings = []
                    for a, b in ((points[0], points[1]), (points[-2], points[-1])):
                        vector = b[0] - a[0], b[1] - a[1]
                        headings.append(max(range(len(self.directions)), key=lambda i:
                            (vector[0] * self.directions[i][0] + vector[1] * self.directions[i][1])
                            / math.hypot(*self.directions[i])))
                    first_direction, direction = headings
                extra_turns = internal_turns + int(incoming >= 0 and incoming != first_direction)
                weighted_length = self.crossing_weight(cell, nxt, diameter, length)
                if not math.isfinite(weighted_length):
                    continue
                new_price = price + weighted_length * rub_per_m + extra_turns * bend_cost_rub
                new_turns = turns + extra_turns
                new_state = nxt, direction
                value = new_price, new_turns
                if value < best.get(new_state, (float("inf"), 0)):
                    best[new_state] = value
                    previous[new_state] = state
                    heapq.heappush(heap, (new_price + heuristic(nxt), new_turns, new_price, new_state))
        return []

    def crossing_weight(self, a_cell, b_cell, diameter, length):
        """Nonnegative special-passage lower bound; exact profiles price each trial.

        Unlike a cosmetic postprocessor, crossing tariffs participate in A*.
        Final connected depth feasibility remains the responsibility of pricing.
        """
        if not self.crossing_objects:
            return length
        from routing_depth import intersection_parameter
        key = edge_key(a_cell, b_cell), diameter
        if key in self._crossing_edge_cache:
            return self._crossing_edge_cache[key]
        points = self.edge_points(a_cell, b_cell)
        bounds = bbox(points)
        weighted = length
        for obj in self.crossing_objects:
            required = obj.horizontal_gap + (ENVELOPE[diameter][0] + obj.width) / 2
            if not obj.near(bounds, required):
                continue
            for a, b in zip(points, points[1:]):
                leg_length = dist(a, b)
                for line in obj.lines:
                    for c, d in zip(line, line[1:]):
                        if segments_distance(a, b, c, d) < required - 1e-7:
                            norm = leg_length * dist(c, d)
                            cosine = abs((b[0] - a[0]) * (d[0] - c[0]) + (b[1] - a[1]) * (d[1] - c[1])) / norm if norm else 0
                            if cosine > .9999:
                                self._crossing_edge_cache[key] = math.inf
                                return math.inf
                        t = intersection_parameter(a, b, c, d)
                        if t is not None and 1e-7 < t < 1 - 1e-7:
                            # A tie-in at a source costs no independent crossing.
                            weighted += 4 * (obj.factor - 1)
                for poly in obj.polygons:
                    cuts = {0., 1.}
                    for ring in poly:
                        for c, d in zip(ring, ring[1:]):
                            t = intersection_parameter(a, b, c, d)
                            if t is not None:
                                cuts.add(t)
                    ordered = sorted(cuts)
                    for lo, hi in zip(ordered, ordered[1:]):
                        t = (lo + hi) / 2
                        p = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
                        if point_in_ring(p, poly[0]) and not any(point_in_ring(p, hole) for hole in poly[1:]):
                            weighted += (hi - lo) * leg_length * (obj.factor - 1)
        self._crossing_edge_cache[key] = weighted
        return weighted


def remove_collinear(points: Sequence[Point]) -> List[Point]:
    out = []
    for p in points:
        if out and dist(out[-1], p) < 1e-8:
            continue
        while len(out) >= 2:
            a, b = out[-2:]
            cross = (b[0] - a[0]) * (p[1] - b[1]) - (b[1] - a[1]) * (p[0] - b[0])
            dot = (b[0] - a[0]) * (p[0] - b[0]) + (b[1] - a[1]) * (p[1] - b[1])
            if abs(cross) > 1e-6 * max(1., dist(a, p)) or dot < 0:
                break
            out.pop()
        out.append(p)
    return out


def smooth_polyline(points: List[Point], grid: RoutingGrid) -> List[Point]:
    if len(points) <= 2:
        return points
    if grid.cardinal_only:
        return remove_collinear(points)
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1:
            if grid.service_line_clear(points[i], points[j]):
                break
            j -= 1
        out.append(points[j])
        i = j
    return out


def read_input(path: str) -> Tuple[List[Terminal], List[TieCandidate], List[Obstacle], List[Point], Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    projected_features: List[Dict[str, Any]] = []
    all_points: List[Point] = []
    for ft in data.get("features", []):
        geom = ft.get("geometry") or {}
        proj = project_geom_coords(geom)
        projected_features.append({"feature": ft, "projected": proj})
        for lon, lat in iter_geojson_positions(geom):
            all_points.append(lonlat_to_utm37(lon, lat))

    terminals: List[Terminal] = []
    heat_networks: List[Dict[str, Any]] = []
    chambers: List[Dict[str, Any]] = []
    obstacles: List[Obstacle] = []

    future_flows = {str(item["feature"].get("properties", {}).get("id")):
                    item["feature"].get("properties", {}).get("flow_tph", 0.0)
                    for item in projected_features
                    if item["feature"].get("properties", {}).get("object_type") == "oks_future"}
    for item in projected_features:
        ft = item["feature"]
        props = ft.get("properties") or {}
        obj_type = props.get("object_type")
        if obj_type == "oks_connection_point":
            flow = float(props.get("flow_tph", future_flows.get(str(props.get("oks_id")), 0.0)))
            select_diameter(flow)  # Reject unsupported/invalid demand before routing.
            terminals.append(Terminal(str(props.get("id")), item["projected"], flow))
        elif obj_type == "heat_network":
            heat_networks.append(item)
        elif obj_type == "heat_chamber":
            chambers.append(item)
        elif obj_type in {"restriction", "oks_future", "oks_existing"}:
            kind = str(props.get("restriction_type", "restriction")) if obj_type == "restriction" else obj_type
            if kind in SPECIAL:
                continue  # Traversable only through a priced, validated special passage.
            # For this dataset, buildings are the critical no-go zones. The
            # appendix gives 5 m for DN < 500; water and railway are kept closed
            # with smaller buffers because they are not target corridors.
            clearance = 5.0 if kind in {"oks", "oks_existing", "oks_future"} else 1.0
            if kind == "railway":
                clearance = 2.0
            rings: List[List[Point]] = []
            holes: List[List[Point]] = []
            proj = item["projected"]
            gtype = ft.get("geometry", {}).get("type")
            if gtype == "Polygon":
                if proj:
                    rings.append(proj[0])
                    holes.extend(proj[1:])
            elif gtype == "MultiPolygon":
                for poly in proj:
                    if poly:
                        rings.append(poly[0])
                        holes.extend(poly[1:])
            if rings:
                flat = [p for ring in rings for p in ring]
                obstacles.append(
                    Obstacle(
                        id=str(props.get("id")),
                        kind=kind,
                        rings=rings,
                        clearance=clearance,
                        bbox=expand_bbox(bbox(flat), clearance),
                        holes=holes,
                    )
                )

    candidates: List[TieCandidate] = []
    chamber_points = [(str(x["feature"]["properties"].get("id")), x["projected"]) for x in chambers]
    network_lines: List[Tuple[str, int, List[Point]]] = []
    for hn in heat_networks:
        props = hn["feature"].get("properties") or {}
        network_lines.append((str(props.get("id")), int(props.get("diameter", 300)), hn["projected"]))

    def max_adjacent_diameter(p: Point) -> int:
        best = 0
        for _hid, diam, line in network_lines:
            for a, b in zip(line, line[1:]):
                if point_segment_distance(p, a, b) <= 8.0:
                    best = max(best, diam)
        return best or 500

    chamber_diameters = {str(x["feature"]["properties"].get("id")):
                         x["feature"]["properties"].get("diameter") for x in chambers}
    for chamber_id, p in chamber_points:
        candidates.append(
            TieCandidate(
                id=f"tie_chamber_{chamber_id}",
                point=p,
                existing_object_id=chamber_id,
                existing_object_type="heat_chamber",
                existing_diameter=int(chamber_diameters[chamber_id] or max_adjacent_diameter(p)),
                is_existing_chamber=True,
            )
        )

    for line_id, diameter, line in network_lines:
        sample_no = 0
        for a, b in zip(line, line[1:]):
            seg_len = dist(a, b)
            count = max(1, int(math.floor(seg_len / 25.0)))
            # Include projections of consumers: a 25 m sample alone can miss a
            # cheap, perpendicular service connection by half a sample interval.
            parameters = {i / count for i in range(count + 1)}
            if seg_len > 1e-8:
                for terminal in terminals:
                    parameters.add(max(0.0, min(1.0, ((terminal.point[0] - a[0]) * (b[0] - a[0])
                        + (terminal.point[1] - a[1]) * (b[1] - a[1])) / seg_len**2)))
            for t in sorted(parameters):
                p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                candidates.append(
                    TieCandidate(
                        id=f"tie_net_{line_id}_{sample_no}",
                        point=p,
                        existing_object_id=line_id,
                        existing_object_type="heat_network",
                        existing_diameter=diameter,
                        is_existing_chamber=False,
                    )
                )
                sample_no += 1

    # Deduplicate very close pipe samples.
    deduped: List[TieCandidate] = []
    seen: Set[Cell] = set()
    for cand in candidates:
        key = (round(cand.point[0] / 2.0), round(cand.point[1] / 2.0))
        if key in seen and not cand.is_existing_chamber:
            continue
        seen.add(key)
        deduped.append(cand)

    meta = {
        "input_feature_count": len(data.get("features", [])),
        "terminal_count": len(terminals),
        "candidate_count": len(deduped),
        "obstacle_count": len(obstacles),
        "has_existing_flow": any("flow_tph" in (x["feature"].get("properties") or {}) for x in heat_networks),
        "has_upstream": any("upstream_object_id" in (x["feature"].get("properties") or {}) for x in heat_networks),
        "terminal_building_access": "Forbidden; explicit endpoint normalization is required for blocked input points.",
    }
    from routing_existing import ExistingNetwork
    meta["existing_network"] = ExistingNetwork(data.get("features", []), project_geom_coords)
    meta["crossing_objects"] = read_crossing_objects(data.get("features", []), project_geom_coords)
    return terminals, deduped, obstacles, all_points, meta


def estimate_city_axis(obstacles: List[Obstacle]) -> float:
    # A circular mean across unrelated blocks can point along no real facade.
    # First locate the dominant 90-degree-equivalent orientation cluster, then
    # average only that cluster. Long facades carry more weight than details.
    observations: List[Tuple[float, float]] = []
    for obs in obstacles:
        if obs.kind not in {"oks", "oks_existing", "oks_future"}:
            continue
        for ring in obs.rings:
            for a, b in zip(ring, ring[1:] + ring[:1]):
                length = dist(a, b)
                if length < 8.0:
                    continue
                angle = (math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi / 4) % (math.pi / 2) - math.pi / 4
                observations.append((angle, length))
    if not observations:
        return 0.0
    def separation(a: float, b: float) -> float:
        return abs((a - b + math.pi / 4) % (math.pi / 2) - math.pi / 4)

    bins = [math.radians(i / 2) for i in range(-90, 90)]
    center = max(bins, key=lambda value: sum(length for angle, length in observations
                                            if separation(value, angle) <= math.radians(2)))
    cluster = [(angle, length) for angle, length in observations
               if separation(center, angle) <= math.radians(3)]
    sum_cos = sum(length * math.cos(4 * angle) for angle, length in cluster)
    sum_sin = sum(length * math.sin(4 * angle) for angle, length in cluster)
    angle = math.atan2(sum_sin, sum_cos) / 4.0
    while angle < -math.pi / 4:
        angle += math.pi / 2
    while angle >= math.pi / 4:
        angle -= math.pi / 2
    return angle


def transform_bbox(rings: List[List[Point]]) -> Tuple[float, float, float, float]:
    flat = [p for ring in rings for p in ring]
    return bbox(flat)


def apply_coordinate_transform(
    terminals: List[Terminal],
    candidates: List[TieCandidate],
    obstacles: List[Obstacle],
    all_points: List[Point],
    transform: CoordinateTransform,
) -> Tuple[List[Terminal], List[TieCandidate], List[Obstacle], List[Point]]:
    if abs(transform.angle_rad) <= 1e-12:
        return terminals, candidates, obstacles, all_points

    new_terminals = [
        Terminal(id=t.id, point=transform.forward(t.point), flow_tph=t.flow_tph, cell=t.cell)
        for t in terminals
    ]
    new_candidates = [
        TieCandidate(
            id=c.id,
            point=transform.forward(c.point),
            existing_object_id=c.existing_object_id,
            existing_object_type=c.existing_object_type,
            existing_diameter=c.existing_diameter,
            is_existing_chamber=c.is_existing_chamber,
            cell=c.cell,
        )
        for c in candidates
    ]
    new_obstacles: List[Obstacle] = []
    for obs in obstacles:
        rings = [[transform.forward(p) for p in ring] for ring in obs.rings]
        new_obstacles.append(
            Obstacle(
                id=obs.id,
                kind=obs.kind,
                rings=rings,
                clearance=obs.clearance,
                bbox=expand_bbox(transform_bbox(rings), obs.clearance),
                holes=[[transform.forward(p) for p in ring] for ring in obs.holes],
            )
        )
    return new_terminals, new_candidates, new_obstacles, [transform.forward(p) for p in all_points]


def prepare_grid(
    terminals: List[Terminal],
    candidates: List[TieCandidate],
    obstacles: List[Obstacle],
    all_points: List[Point],
    step: float,
    cardinal_only: bool = False,
    coord_transform: CoordinateTransform = IDENTITY_TRANSFORM,
    normalize_terminals: bool = False,
    allow_building_leads: bool = False,
) -> RoutingGrid:
    grid_points = all_points + [t.point for t in terminals] + [c.point for c in candidates]
    grid = RoutingGrid(
        obstacles=obstacles,
        points=grid_points,
        step=step,
        cardinal_only=cardinal_only,
        coord_transform=coord_transform,
        allow_building_leads=allow_building_leads,
    )
    for cand in candidates:
        cand.cell = grid.add_access(cand.point)
    if normalize_terminals:
        normalize_connection_points(terminals, candidates, grid)
    for terminal in terminals:
        terminal.cell = grid.add_access(terminal.point, terminal=True)
    return grid


def normalize_connection_points(terminals, candidates, grid):
    """Move blocked inputs to the nearest reachable outer facade clearance.

    This is an explicit input correction, never an invisible grid snap. Keep
    exact old/new coordinates and resolve reachability before selecting a face.
    """
    reachable = {c.cell for c in candidates if c.cell is not None}
    for terminal in terminals:
        if not grid.is_blocked_point(terminal.point):
            continue
        original = terminal.point
        owners = [o for o in grid.obstacles if o.kind in BUILDINGS and o.contains_or_near(original)]
        if not owners:
            continue  # Never move an input out of an unrelated forbidden area.
        options = []
        for owner in owners:
            for ring in owner.rings:
                for a, b in zip(ring, ring[1:] + ring[:1]):
                    length = dist(a, b)
                    if length < 1e-8:
                        continue
                    t = max(0., min(1., ((original[0] - a[0]) * (b[0] - a[0])
                                        + (original[1] - a[1]) * (b[1] - a[1])) / length**2))
                    foot = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
                    for sign in (-1, 1):
                        offset = owner.clearance + .05
                        p = (foot[0] - sign * (b[1] - a[1]) / length * offset,
                             foot[1] + sign * (b[0] - a[0]) / length * offset)
                        if not grid.is_blocked_point(p):
                            options.append((dist(original, p), p, owner.id))
        for displacement, point, owner_id in sorted(options):
            access = grid.add_access(point)
            if access is None:
                continue
            # add_access may split a corridor edge AFTER reachability was
            # calculated. A new contact is not an unreachable component.
            # A* proves a path to an existing root without flooding the map.
            path = grid.least_cost_path({c: 0. for c in reachable}, access, set(), 1.)
            if not path:
                continue
            reachable.update(path)
            terminal.point = point
            grid.connection_adjustments.append({
                "terminal_id": terminal.id, "building_id": owner_id,
                "original_coordinates": unproject_point(grid.to_utm(original)),
                "connection_coordinates": unproject_point(grid.to_utm(point)),
                "displacement_m": round(displacement, 3),
                "axis_clearance_m": next(o.clearance for o in owners if o.id == owner_id),
                "reason": "Input point within forbidden building/envelope; nearest reachable external facade",
            })
            break


def edge_key(a: Cell, b: Cell) -> Tuple[Cell, Cell]:
    return (a, b) if a <= b else (b, a)


@dataclass
class BuiltTree:
    roots: List[TieCandidate]
    edges: Set[Tuple[Cell, Cell]]
    connected_terminal_ids: Set[str]
    unconnected_terminal_ids: Set[str]
    segment_paths: Dict[Tuple[Cell, Cell], List[Point]] = field(default_factory=dict)
    node_points: Dict[Cell, Point] = field(default_factory=dict)


def build_forest(
    terminals: List[Terminal],
    roots: List[TieCandidate],
    grid: RoutingGrid,
    turn_penalty_m: float = 0.0,
    bend_cost_rub: float = 0.0,
    order: str = "cheapest",
    standalone_paths: Optional[Dict[str, List[Cell]]] = None,
) -> BuiltTree:
    """Cost-aware rooted forest with optional tie-ins and branch reattachment.

    The root list is a candidate pool, not a set of prepaid mandatory roots.
    Every accepted move is priced again using the complete exported network,
    including upstream diameter increases, chambers, bends and terminal leads.
    This is a deterministic discrete heuristic, not a global-optimality claim.
    """
    tree = BuiltTree([], set(), set(), {t.id for t in terminals})
    pending = sorted((t for t in terminals if t.cell is not None), key=lambda t: t.id)
    cache = standalone_paths if standalone_paths is not None else {}
    for terminal in pending:
        if terminal.id not in cache:
            diameter = select_diameter(terminal.flow_tph)
            sources = {r.cell: opening_cost(r, diameter) for r in roots if r.cell is not None}
            cache[terminal.id] = grid.least_cost_path(
                sources, terminal.cell, set(), NEW_COST[diameter],
                bend_cost_rub + turn_penalty_m * NEW_COST[diameter],
            )
    # If no candidate root can reach a terminal in the static free space,
    # adding a branch cannot make it reachable. Do not repeat that full search.
    pending = [t for t in pending if cache[t.id]]

    def priced(value: BuiltTree) -> Variant:
        return materialize_variant("trial", "cost search", value, terminals, grid, {}, bend_cost_rub)

    current = priced(tree)
    while pending:
        choices = []
        for terminal in pending:
            if order == "demand" and terminal != max(pending, key=lambda t: (t.flow_tph, t.id)):
                continue
            for candidate in connection_options(tree, terminal, roots, terminals, grid,
                                                cache[terminal.id], bend_cost_rub, turn_penalty_m):
                try:
                    variant = priced(candidate)
                except ValueError:
                    continue
                added_cost = variant.summary["calculated_cost"] - current.summary["calculated_cost"]
                # Remove the known disconnection penalty when comparing the
                # construction increments of consumers with different demand.
                increment = variant.score - current.score + .7 * penalty_unconnected(terminal.flow_tph) / 25_000_000
                choices.append((increment, variant.summary["route_bend_count"], terminal.id, variant))
        if not choices:
            if order == "demand":
                pending.remove(max(pending, key=lambda t: (t.flow_tph, t.id)))
                continue
            break
        _increment, _bends, tid, selected = min(choices, key=lambda v: v[:3])
        pending = [t for t in pending if t.id != tid]
        if variant_key(selected) < variant_key(current):
            tree, current = selected.tree, selected

    # Remove and reconnect a whole leaf branch, pruning abandoned tie-ins. A
    # move is accepted only when the full cost decreases (or equal cost has
    # fewer bends), so smoothing cannot silently make the solution dearer.
    for _pass in range(2):
        improved = False
        for terminal in sorted(terminals, key=lambda t: t.id):
            if terminal.id not in tree.connected_terminal_ids:
                continue
            reduced = detach_terminal(tree, terminal)
            for candidate in connection_options(reduced, terminal, roots, terminals, grid,
                                                cache.get(terminal.id, []), bend_cost_rub, turn_penalty_m):
                try:
                    variant = priced(candidate)
                except ValueError:
                    continue
                if variant_key(variant) < variant_key(current):
                    tree, current = candidate, variant
                    improved = True
        if not improved:
            break
    return improve_subtrees(tree, terminals, roots, grid, bend_cost_rub, turn_penalty_m)


def opening_cost(root: TieCandidate, diameter: int) -> float:
    if root.is_existing_chamber:
        return TIE_IN_COST + (chamber_cost(diameter) if diameter > root.existing_diameter else 0)
    return TIE_IN_COST + chamber_cost(max(diameter, root.existing_diameter))


def variant_key(variant: Variant) -> Tuple[float, int, float]:
    return (variant.score, variant.summary["route_bend_count"],
            variant.summary["calculated_cost"])


def variant_geometry_key(variant: Variant) -> Tuple[int, int, int, float, float, float]:
    summary = variant.summary
    return (
        summary.get("micro_bend_count", 0),
        summary.get("route_bend_count", 0),
        summary.get("route_right_angle_count", 0),
        summary.get("calculated_cost", float("inf")),
        summary.get("length", float("inf")),
        variant.score,
    )


def variant_is_better(candidate: Variant, incumbent: Variant) -> bool:
    """Prefer score, then geometry when scores are within the presentation epsilon."""
    tolerance = SCORE_TIE_EPSILON * max(abs(incumbent.score), 1e-9)
    if candidate.score < incumbent.score - tolerance:
        return True
    if candidate.score > incumbent.score + tolerance:
        return False
    return variant_geometry_key(candidate) < variant_geometry_key(incumbent)


def compare_variants(left: Variant, right: Variant) -> int:
    if variant_is_better(left, right):
        return -1
    if variant_is_better(right, left):
        return 1
    return 0


def connection_options(
    tree: BuiltTree, terminal: Terminal, candidates: List[TieCandidate], terminals: List[Terminal],
    grid: RoutingGrid, standalone: List[Cell], bend_cost_rub: float, turn_penalty_m: float,
    forbidden: Optional[Set[Cell]] = None,
) -> Iterable[BuiltTree]:
    parent, depth, flows, adjacency = orient_and_flow(tree, terminals)
    root_by_cell = {r.cell: r for r in tree.roots}
    candidate_by_cell = {r.cell: r for r in candidates if r.cell is not None}
    diameter = select_diameter(terminal.flow_tph)
    costs: Dict[Cell, float] = {}
    sources: Dict[Cell, float] = {}
    for cell in sorted(parent, key=lambda c: (depth[c], c)):
        par = parent[cell]
        if par is None:
            root = root_by_cell[cell]
            existing_flow = sum(flows.get(edge_key(cell, n), 0.0) for n in adjacency[cell])
            before = select_diameter(existing_flow)
            after = select_diameter(existing_flow + terminal.flow_tph)
            costs[cell] = max(0, opening_cost(root, after) - opening_cost(root, before))
        else:
            flow = flows[edge_key(cell, par)]
            length = path_length(grid.edge_points(cell, par))
            costs[cell] = costs[par] + length * (
                NEW_COST[select_diameter(flow + terminal.flow_tph)] - NEW_COST[select_diameter(flow)]
            )
        degree = len(adjacency[cell])
        if cell in grid.terminal_cells or degree >= (2 if cell in root_by_cell else 4):
            continue
        branch_cost = chamber_cost(diameter) if degree == 2 and cell not in root_by_cell else 0
        sources[cell] = costs[cell] + branch_cost

    paths = [standalone]
    if sources:
        paths.append(grid.least_cost_path(
            sources, terminal.cell, set(parent) | (forbidden or set()), NEW_COST[diameter],
            bend_cost_rub + turn_penalty_m * NEW_COST[diameter],
        ))
    for path in paths:
        if not path:
            continue
        intersections = [i for i, cell in enumerate(path) if cell in parent]
        new_roots = list(tree.roots)
        if intersections:
            path = path[intersections[-1]:]
            if path[0] not in sources:
                continue
        elif path[0] in candidate_by_cell:
            new_roots.append(candidate_by_cell[path[0]])
        else:
            continue
        if len(path) < 2:
            continue
        yield BuiltTree(new_roots, tree.edges | {edge_key(a, b) for a, b in zip(path, path[1:])},
                        tree.connected_terminal_ids | {terminal.id}, tree.unconnected_terminal_ids - {terminal.id})


def detach_terminal(tree: BuiltTree, terminal: Terminal) -> BuiltTree:
    adjacency: Dict[Cell, Set[Cell]] = defaultdict(set)
    for a, b in sorted(tree.edges):
        adjacency[a].add(b)
        adjacency[b].add(a)
    edges = set(tree.edges)
    cell = terminal.cell
    roots = {r.cell for r in tree.roots}
    while cell is not None and len(adjacency[cell]) == 1:
        nxt = next(iter(adjacency[cell]))
        edges.remove(edge_key(cell, nxt))
        adjacency[cell].remove(nxt)
        adjacency[nxt].remove(cell)
        if nxt in roots:
            break
        cell = nxt
    used_roots = [r for r in tree.roots if adjacency[r.cell]]
    return BuiltTree(used_roots, edges, tree.connected_terminal_ids - {terminal.id},
                     tree.unconnected_terminal_ids | {terminal.id})


def improve_subtrees(
    tree: BuiltTree, terminals: List[Terminal], candidates: List[TieCandidate], grid: RoutingGrid,
    bend_cost_rub: float, turn_penalty_m: float,
) -> BuiltTree:
    """Relocate complete downstream branches; leaf-only moves cannot shorten trunks."""
    current = materialize_variant("trial", "subtree search", tree, terminals, grid, {}, bend_cost_rub)
    for _pass in range(3):
        parent, depth, _flows, adjacency = orient_and_flow(tree, terminals)
        # Junctions are a small, topology-relevant neighborhood of the full grid.
        targets = sorted((c for c in parent if parent[c] is not None and len(adjacency[c]) >= 3),
                         key=lambda c: (depth[c], c), reverse=True)
        improved = False
        for target in targets:
            # Recompute after accepted moves: parents and downstream membership change.
            parent, _depth, _flows, adjacency = orient_and_flow(tree, terminals)
            if target not in parent or parent[target] is None:
                continue
            downstream: Set[Cell] = set()
            pending = [target]
            while pending:
                cell = pending.pop()
                downstream.add(cell)
                pending.extend(n for n in adjacency[cell] if parent.get(n) == cell)
            group = [t for t in terminals if t.cell in downstream and t.id in tree.connected_terminal_ids]
            if len(group) < 2:
                continue
            ids = {t.id for t in group}
            detached_edges = {e for e in tree.edges if e[0] in downstream and e[1] in downstream}
            kept_edges = tree.edges - detached_edges - {edge_key(target, parent[target])}
            # Prune the abandoned upstream chain to the next live junction/root.
            old = parent[target]
            degree = {c: len(ns) for c, ns in adjacency.items()}
            degree[old] -= 1
            roots_by_cell = {r.cell: r for r in tree.roots}
            while old not in roots_by_cell and degree[old] == 1 and parent[old] is not None:
                par = parent[old]
                kept_edges.remove(edge_key(old, par))
                degree[old] -= 1
                degree[par] -= 1
                old = par
            roots = [r for r in tree.roots if degree[r.cell] > 0]
            reduced = BuiltTree(roots, kept_edges, tree.connected_terminal_ids - ids,
                                tree.unconnected_terminal_ids | ids)
            demand = sum(t.flow_tph for t in group)
            synthetic = Terminal("__routing_subtree__", grid.cell_to_point(target), demand, target)
            diameter = select_diameter(demand)
            forbidden = downstream - {target}
            sources = {r.cell: opening_cost(r, diameter) for r in candidates
                       if r.cell is not None and r.cell not in downstream}
            path = grid.least_cost_path(sources, target, forbidden, NEW_COST[diameter],
                                       bend_cost_rub + turn_penalty_m * NEW_COST[diameter])
            for candidate in connection_options(reduced, synthetic, candidates, terminals, grid,
                                                path, bend_cost_rub, turn_penalty_m, forbidden):
                candidate.edges.update(detached_edges)
                candidate.connected_terminal_ids.discard(synthetic.id)
                candidate.connected_terminal_ids.update(ids)
                candidate.unconnected_terminal_ids.difference_update(ids)
                try:
                    variant = materialize_variant("trial", "subtree search", candidate,
                                                  terminals, grid, {}, bend_cost_rub)
                except ValueError:
                    continue
                if variant_key(variant) < variant_key(current):
                    tree, current = candidate, variant
                    improved = True
        if not improved:
            break
    return tree


@dataclass
class Segment:
    points: List[Point]
    start_cell: Cell
    end_cell: Cell
    flow_tph: float
    diameter: int
    length: float
    start_node_id: str
    end_node_id: str
    cost: float


@dataclass
class Variant:
    id: str
    label: str
    tree: BuiltTree
    features: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    score: float = float("inf")


def orient_and_flow(
    tree: BuiltTree,
    terminals: List[Terminal],
) -> Tuple[Dict[Cell, Optional[Cell]], Dict[Cell, int], Dict[Tuple[Cell, Cell], float], Dict[Cell, List[Cell]]]:
    adj: Dict[Cell, List[Cell]] = defaultdict(list)
    for a, b in sorted(tree.edges):
        adj[a].append(b)
        adj[b].append(a)
    if any(len(neighbors) > 4 for neighbors in adj.values()):
        raise ValueError("A heat chamber may have at most four incident pipes")
    if any(t.id in tree.connected_terminal_ids and len(adj[t.cell]) != 1 for t in terminals):
        raise ValueError("A consumer connection must terminate the route, never carry transit flow")
    root_cells = [r.cell for r in tree.roots if r.cell is not None]
    parent: Dict[Cell, Optional[Cell]] = {}
    depth: Dict[Cell, int] = {}
    q: deque[Cell] = deque()
    for root in root_cells:
        if root is None or root in parent:
            continue
        parent[root] = None
        depth[root] = 0
        q.append(root)
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt in parent:
                continue
            parent[nxt] = cur
            depth[nxt] = depth[cur] + 1
            q.append(nxt)

    if len(tree.edges) != len(parent) - len(set(root_cells)):
        raise ValueError("Routing topology must be an acyclic forest with exactly one tie-in per component")
    if any(t.id in tree.connected_terminal_ids and t.cell not in parent for t in terminals):
        raise ValueError("A connected consumer has no path to a tie-in")

    terminal_flow_by_cell: Dict[Cell, float] = defaultdict(float)
    for t in terminals:
        if t.id in tree.connected_terminal_ids and t.cell is not None:
            terminal_flow_by_cell[t.cell] += t.flow_tph

    subtotal: Dict[Cell, float] = defaultdict(float, terminal_flow_by_cell)
    edge_flow: Dict[Tuple[Cell, Cell], float] = {}
    for cell, _d in sorted(depth.items(), key=lambda item: item[1], reverse=True):
        par = parent.get(cell)
        if par is None:
            continue
        flow = subtotal[cell]
        edge_flow[edge_key(cell, par)] = flow
        subtotal[par] += flow
    return parent, depth, edge_flow, adj


def compress_segments(
    tree: BuiltTree,
    terminals: List[Terminal],
    grid: RoutingGrid,
) -> Tuple[List[Segment], Dict[Cell, str], Dict[Cell, float]]:
    parent, depth, edge_flow, adj = orient_and_flow(tree, terminals)
    root_ids = {r.cell: f"{tree.roots.index(r) + 1}" for r in tree.roots if r.cell is not None}
    terminal_by_cell: Dict[Cell, List[Terminal]] = defaultdict(list)
    for t in terminals:
        if t.cell is not None and t.id in tree.connected_terminal_ids:
            terminal_by_cell[t.cell].append(t)

    special: Set[Cell] = set()
    special.update(root_ids)
    special.update(terminal_by_cell)
    special.update({cell for cell, ns in adj.items() if len(ns) != 2})

    node_ids: Dict[Cell, str] = {}
    for r in tree.roots:
        if r.cell is not None:
            node_ids[r.cell] = f"tie_{r.id}"
    branch_no = 1
    for cell in sorted(special):
        if cell in node_ids:
            continue
        if cell in terminal_by_cell:
            terms = terminal_by_cell[cell]
            node_ids[cell] = terms[0].id
        elif len(adj[cell]) > 2:
            node_ids[cell] = f"new_chamber_{branch_no}"
            branch_no += 1
        else:
            node_ids[cell] = f"technical_node_{branch_no}"
            branch_no += 1

    visited_edges: Set[Tuple[Cell, Cell]] = set()
    segments: List[Segment] = []
    for start in sorted(special):
        for first in adj.get(start, []):
            ek = edge_key(start, first)
            if ek in visited_edges:
                continue
            chain = [start, first]
            visited_edges.add(ek)
            prev = start
            cur = first
            while cur not in special:
                next_cells = [n for n in adj[cur] if n != prev]
                if not next_cells:
                    break
                nxt = next_cells[0]
                visited_edges.add(edge_key(cur, nxt))
                chain.append(nxt)
                prev, cur = cur, nxt

            a_cell, b_cell = chain[0], chain[-1]
            if depth.get(a_cell, 10**9) > depth.get(b_cell, 10**9):
                chain = list(reversed(chain))
                a_cell, b_cell = chain[0], chain[-1]
            pts = [grid.cell_to_point(chain[0])]
            for a, b in zip(chain, chain[1:]):
                pts.extend(grid.edge_points(a, b)[1:])
            # Visibility shortcuts must be checked against the WHOLE network,
            # not independently here (they can cross another branch).
            pts = remove_collinear(pts)
            override = tree.segment_paths.get(edge_key(a_cell, b_cell))
            if override is not None:
                pts = override if a_cell <= b_cell else list(reversed(override))
            length = path_length(pts)
            flows = [edge_flow.get(edge_key(a, b), 0.0) for a, b in zip(chain, chain[1:])]
            flow = max(flows) if flows else 0.0
            diameter = select_diameter(flow)
            if length > MAX_LENGTH[diameter] + 1e-8:
                raise ValueError("Minimum flow diameter exceeds its length limit")
            cost = length * NEW_COST[diameter]
            segments.append(
                Segment(
                    points=pts,
                    start_cell=a_cell,
                    end_cell=b_cell,
                    flow_tph=flow,
                    diameter=diameter,
                    length=length,
                    start_node_id=node_ids[a_cell],
                    end_node_id=node_ids[b_cell],
                    cost=cost,
                )
            )

    # Chambers do not reset the maximum uninterrupted length at one diameter.
    # Process downstream chains and increase the diameter where necessary.
    incoming: Dict[Cell, Tuple[int, float]] = {}
    for seg in sorted(segments, key=lambda s: depth[s.start_cell]):
        prior_diameter, prior_length = incoming.get(seg.start_cell, (0, 0.0))
        continuous = seg.length + (prior_length if prior_diameter == seg.diameter else 0.0)
        if continuous > MAX_LENGTH[seg.diameter] + 1e-8:
            raise ValueError("Continuous minimum-diameter run exceeds its length limit")
        incoming[seg.end_cell] = seg.diameter, continuous
        seg.cost = seg.length * NEW_COST[seg.diameter]

    chamber_diameter_by_cell: Dict[Cell, float] = defaultdict(float)
    for seg in segments:
        chamber_diameter_by_cell[seg.start_cell] = max(chamber_diameter_by_cell[seg.start_cell], seg.diameter)
        chamber_diameter_by_cell[seg.end_cell] = max(chamber_diameter_by_cell[seg.end_cell], seg.diameter)
    return segments, node_ids, chamber_diameter_by_cell


def materialize_variant(
    variant_id: str,
    label: str,
    tree: BuiltTree,
    terminals: List[Terminal],
    grid: RoutingGrid,
    meta: Dict[str, Any],
    bend_cost_rub: float = 0.0,
) -> Variant:
    variant = Variant(id=variant_id, label=label, tree=tree)
    segments, node_ids, diameter_by_cell = compress_segments(tree, terminals, grid)
    for i, first in enumerate(segments):
        for second in segments[i + 1:]:
            if polylines_conflict(first.points, second.points):
                raise ValueError("New pipes intersect or overlap outside a shared endpoint")
    features: List[Dict[str, Any]] = []
    root_required_flow: Dict[str, float] = defaultdict(float)
    root_required_diameter: Dict[str, int] = {}
    construction_cost = 0.0
    new_length = 0.0
    route_bend_count = 0
    route_right_angle_count = 0
    micro_bend_count = 0
    terminal_cells = {t.cell for t in terminals}
    exported_node_ids = {cell: name if cell in terminal_cells else f"{variant_id}_{name}"
                         for cell, name in node_ids.items()}

    profiles, passages = build_depth_profiles(
        segments, grid.crossing_objects, {r.cell for r in tree.roots}, terminal_cells,
        [r.point for r in tree.roots], NEW_COST)
    from routing_constraints import validate_segments
    validate_segments(segments, grid.obstacles, terminals, grid.allow_building_leads)

    for i, seg in enumerate(segments, start=1):
        points = seg.points
        length = path_length(points)
        diameter = seg.diameter
        cost = sum(p.length * NEW_COST[diameter] * p.depth_coefficient * p.special_factor
                   for p in profiles[i - 1])
        bend_count = polyline_bend_count(points)
        route_bend_count += bend_count
        route_right_angle_count += polyline_bend_count(points, 80.) - polyline_bend_count(points, 100.)
        micro_bend_count += polyline_micro_bend_count(points, max(2.0 * grid.step, 8.0))
        construction_cost += cost
        new_length += length
        if seg.start_node_id.startswith("tie_"):
            root_required_flow[seg.start_node_id] += seg.flow_tph
            root_required_diameter[seg.start_node_id] = max(root_required_diameter.get(seg.start_node_id, 0), diameter)
        for part_no, part in enumerate(profiles[i - 1], start=1):
            last = part_no == len(profiles[i - 1])
            start_id = exported_node_ids[seg.start_cell] if part_no == 1 else f"{variant_id}_profile_{i}_{part_no - 1}"
            end_id = exported_node_ids[seg.end_cell] if last else f"{variant_id}_profile_{i}_{part_no}"
            if not last:
                features.append({"type": "Feature", "geometry": unproject_point_geom(grid.to_utm(part.points[-1])),
                                 "properties": {"id": end_id, "object_type": "technical_node",
                                                "variant_id": variant_id, "depth": round(part.end_depth, 6)}})
            features.append(
            {
                "type": "Feature",
                "geometry": unproject_linestring([grid.to_utm(p) for p in part.points]),
                "properties": {
                    "id": f"{variant_id}_new_{i}_{part_no}",
                    "object_type": "heat_network",
                    "variant_id": variant_id,
                    "start_node_id": start_id,
                    "end_node_id": end_id,
                    "flow_tph": round(seg.flow_tph, 3),
                    "diameter": diameter,
                    "length": round(part.length, 3),
                    "laying_method": "special" if part.crossing_ids else "base",
                    "depth_start": round(part.start_depth, 6),
                    "depth_end": round(part.end_depth, 6),
                    "depth_factor": round(part.depth_coefficient, 9),
                    "special_factor": part.special_factor,
                    "crossing_object_ids": list(part.crossing_ids),
                    "bend_count": polyline_bend_count(part.points),
                    "bend_penalty_cost": 0.0,
                    "cost": round(part.length * NEW_COST[diameter] * part.depth_coefficient * part.special_factor, 2),
                },
            }
        )

    existing = getattr(grid, "existing_network", None)
    injections = [(r.existing_object_id, grid.to_utm(r.point), root_required_flow.get(f"tie_{r.id}", 0.))
                  for r in tree.roots]
    reconstructed, reconstruction_cost, reconstruction_length, existing_required = (existing.evaluate(injections, variant_id)
        if existing is not None else ([], 0., 0., {}))
    features.extend(reconstructed)
    tie_cost = 0.0
    chamber_construction_cost = 0.0
    chamber_reconstruction_cost = 0.0
    for idx, root in enumerate(tree.roots, start=1):
        tie_id = f"tie_{root.id}"
        required_diameter = max(root_required_diameter.get(tie_id, 50),
                                select_diameter(root_required_flow.get(tie_id, 0.0)))
        tie_cost += TIE_IN_COST
        features.append(
            {
                "type": "Feature",
                "geometry": unproject_point_geom(grid.to_utm(root.point)),
                "properties": {
                    "id": f"{variant_id}_{tie_id}",
                    "object_type": "tie_in",
                    "variant_id": variant_id,
                    "existing_object_id": root.existing_object_id,
                    "existing_object_type": root.existing_object_type,
                    "existing_diameter": root.existing_diameter,
                    "required_diameter": required_diameter,
                    "cost": TIE_IN_COST,
                },
            }
        )
        if root.is_existing_chamber:
            chamber_required = existing.chamber_diameter(root.existing_object_id, required_diameter, existing_required) if existing else required_diameter
            if chamber_required > root.existing_diameter:
                required_diameter = chamber_required
                c_cost = chamber_cost(required_diameter)
                chamber_reconstruction_cost += c_cost
                features.append(
                    {
                        "type": "Feature",
                        "geometry": unproject_point_geom(grid.to_utm(root.point)),
                        "properties": {
                            "id": f"{variant_id}_chamber_reconstruction_{idx}",
                            "object_type": "heat_chamber_reconstruction",
                            "variant_id": variant_id,
                            "existing_object_id": root.existing_object_id,
                            "existing_diameter": root.existing_diameter,
                            "required_diameter": required_diameter,
                            "cost": c_cost,
                        },
                    }
                )
        else:
            c_cost = chamber_cost(max(required_diameter, root.existing_diameter))
            chamber_construction_cost += c_cost
            features.append(
                {
                    "type": "Feature",
                    "geometry": unproject_point_geom(grid.to_utm(root.point)),
                    "properties": {
                        "id": f"{variant_id}_tie_chamber_{idx}",
                        "object_type": "heat_chamber",
                        "variant_id": variant_id,
                        "diameter": max(required_diameter, root.existing_diameter),
                        "cost": c_cost,
                    },
                }
            )

    for cell, node_id in sorted(node_ids.items(), key=lambda item: item[1]):
        if cell not in terminal_cells and node_id.startswith("technical_node_"):
            features.append({"type": "Feature", "geometry": unproject_point_geom(grid.to_utm(tree.node_points.get(cell, grid.cell_to_point(cell)))),
                             "properties": {"id": exported_node_ids[cell], "object_type": "technical_node",
                                            "variant_id": variant_id}})
        if not node_id.startswith("new_chamber_"):
            continue
        diameter = int(diameter_by_cell.get(cell, 50))
        c_cost = chamber_cost(diameter)
        chamber_construction_cost += c_cost
        features.append(
                {
                    "type": "Feature",
                    "geometry": unproject_point_geom(grid.to_utm(tree.node_points.get(cell, grid.cell_to_point(cell)))),
                    "properties": {
                    "id": f"{variant_id}_{node_id}",
                    "object_type": "heat_chamber",
                    "variant_id": variant_id,
                    "diameter": diameter,
                    "cost": c_cost,
                },
            }
        )

    unconnected_penalty = 0.0
    terminal_by_id = {t.id: t for t in terminals}
    for tid in tree.unconnected_terminal_ids:
        unconnected_penalty += penalty_unconnected(terminal_by_id[tid].flow_tph)

    bend_penalty_cost = 0.0  # Search bias is never an official monetary expense.

    calculated_cost = (
        construction_cost
        + chamber_construction_cost
        + tie_cost
        + reconstruction_cost
        + chamber_reconstruction_cost
        + unconnected_penalty
        + bend_penalty_cost
    )
    total_length = new_length + reconstruction_length
    score = 0.7 * (calculated_cost / 25_000_000.0) + 0.3 * (total_length / 100.0)
    summary = {
        "id": f"{variant_id}_summary",
        "object_type": "variant_summary",
        "variant_id": variant_id,
        "rank": 0,
        "label": label,
        "construction_cost": round(construction_cost, 2),
        "chamber_construction_cost": round(chamber_construction_cost, 2),
        "tie_in_cost": round(tie_cost, 2),
        "reconstruction_cost": round(reconstruction_cost, 2),
        "chamber_reconstruction_cost": round(chamber_reconstruction_cost, 2),
        "bend_penalty_cost": round(bend_penalty_cost, 2),
        "unconnected_penalty": round(unconnected_penalty, 2),
        "calculated_cost": round(calculated_cost, 2),
        "new_network_length": round(new_length, 3),
        "reconstruction_length": round(reconstruction_length, 3),
        "length": round(total_length, 3),
        "score": round(score, 6),
        "route_bend_count": route_bend_count,
        "route_right_angle_count": route_right_angle_count,
        "micro_bend_count": micro_bend_count,
        "unconnected_oks_ids": sorted(tree.unconnected_terminal_ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)),
        "connected_oks_count": len(tree.connected_terminal_ids),
        "tie_in_count": len(tree.roots),
        "depth_routing": True,
        "special_passage_count": len(passages),
        "min_depth_m": round(min((min(p.start_depth, p.end_depth) for group in profiles for p in group), default=3.), 6),
        "max_depth_m": round(max((max(p.start_depth, p.end_depth) for group in profiles for p in group), default=3.), 6),
        "connection_adjustments": grid.connection_adjustments,
        "building_entry_allowed": grid.allow_building_leads,
        "optimization_objective": "official_score",
        "optimality": "best found by cost-aware growth and local branch reattachment; no global guarantee",
        "reconstruction_status": "complete" if existing and existing.complete else "unknown_missing_input",
        "score_status": "complete" if existing and existing.complete else "partial_excludes_unknown_reconstruction",
        "note": "" if existing and existing.complete else "Reconstruction cannot be determined: existing flows/upstream/chamber attributes are missing. Zero is a placeholder, not proof that reconstruction is unnecessary.",
    }
    features.append({"type": "Feature", "geometry": None, "properties": summary})
    variant.features = features
    variant.summary = summary
    variant.score = score
    return variant


def make_variants(
    terminals: List[Terminal],
    candidates: List[TieCandidate],
    grid: RoutingGrid,
    meta: Dict[str, Any],
    turn_penalty_m: float = 0.0,
    bend_cost_rub: float = 0.0,
    max_topologies: int = 0,
) -> List[Variant]:
    if not terminals:
        raise ValueError("Input has no consumer connection points")
    raw_variants: List[Tuple[str, str, BuiltTree]] = []
    standalone: Dict[str, List[Cell]] = {}
    for order in ("cheapest", "demand"):
        if max_topologies and len(raw_variants) >= max_topologies:
            break
        print(f"Building topology: {order}", file=sys.stderr, flush=True)
        tree = build_forest(terminals, candidates, grid, turn_penalty_m, bend_cost_rub, order, standalone)
        raw_variants.append((f"cost_{order}", f"cost-aware forest, {order} insertion with branch reattachment", tree))
    materialized: List[Variant] = []
    seen_signatures = set()
    for variant_id, label, tree in raw_variants:
        tree = polish_corridors(tree, terminals, grid, bend_cost_rub)
        if meta.get("refine_geometry", True):
            print(f"Refining geometry: {variant_id}", file=sys.stderr, flush=True)
            tree = refine_geometry(tree, terminals, grid, bend_cost_rub)
        signature = (tuple(sorted(r.id for r in tree.roots)), frozenset(tree.edges),
                     frozenset(tree.connected_terminal_ids))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        materialized.append(
            materialize_variant(
                variant_id,
                label,
                tree,
                terminals,
                grid,
                meta,
                bend_cost_rub=bend_cost_rub,
            )
        )
    materialized.sort(key=variant_key)
    for rank, variant in enumerate(materialized, start=1):
        variant.summary["rank"] = rank
        for ft in variant.features:
            props = ft.get("properties") or {}
            if props.get("object_type") == "variant_summary":
                props["rank"] = rank
    return materialized


def polylines_conflict(first: Sequence[Point], second: Sequence[Point], tolerance: float = 1e-7) -> bool:
    """Only a shared endpoint may touch; crossings and parallel overlaps need a node."""
    shared = [p for p in (first[0], first[-1])
              if any(dist(p, q) <= tolerance for q in (second[0], second[-1]))]
    for a, b in zip(first, first[1:]):
        for c, d in zip(second, second[1:]):
            if segments_distance(a, b, c, d) > tolerance:
                continue
            # At a shared node, collinear positive overlap is still a duplicate pipe.
            if point_segment_distance(c, a, b) <= tolerance and point_segment_distance(d, a, b) <= tolerance:
                if dist(c, d) > tolerance:
                    return True
            if point_segment_distance(a, c, d) <= tolerance and point_segment_distance(b, c, d) <= tolerance:
                if dist(a, b) > tolerance:
                    return True
            allowed = any((dist(p, a) <= tolerance or dist(p, b) <= tolerance)
                          and (dist(p, c) <= tolerance or dist(p, d) <= tolerance) for p in shared)
            if not allowed:
                return True
    return False


def tidy_orthogonal_path(points: List[Point], grid: RoutingGrid,
                         barriers: Sequence[Sequence[Point]]) -> List[Point]:
    """Remove stair steps with checked L-shaped shortcuts; never add pipe length."""
    current = smooth_polyline(points, grid)
    while True:
        old_length = path_length(current)
        old_key = polyline_bend_count(current), round(old_length, 7)
        best, best_key = current, old_key
        for i in range(len(current) - 2):
            if grid.is_blocked_point(current[i]):
                continue
            for j in range(i + 2, len(current)):
                if grid.is_blocked_point(current[j]):
                    continue
                a, b = current[i], current[j]
                for corner in ((a[0], b[1]), (b[0], a[1])):
                    shortcut = [a, corner, b]
                    shortcut = [p for k, p in enumerate(shortcut) if k == 0 or dist(p, shortcut[k - 1]) > 1e-8]
                    if any(not grid.line_clear(p, q) for p, q in zip(shortcut, shortcut[1:])):
                        continue
                    candidate = smooth_polyline(current[:i] + shortcut + current[j + 1:], grid)
                    length = path_length(candidate)
                    key = polyline_bend_count(candidate), round(length, 7)
                    if length > old_length + 1e-7 or key >= best_key:
                        continue
                    if any(polylines_conflict(candidate, other) for other in barriers):
                        continue
                    if any(segments_distance(candidate[u], candidate[u + 1], candidate[v], candidate[v + 1]) < 1e-7
                           for u in range(len(candidate) - 1) for v in range(u + 2, len(candidate) - 1)):
                        continue
                    best, best_key = candidate, key
        if best is current:
            return current
        current = best


def polish_corridors(tree: BuiltTree, terminals: List[Terminal], grid: RoutingGrid,
                     bend_cost_rub: float = 0.0) -> BuiltTree:
    if not grid.cardinal_only:
        return tree
    segments, _nodes, _diameters = compress_segments(tree, terminals, grid)
    result = BuiltTree(tree.roots, tree.edges, tree.connected_terminal_ids, tree.unconnected_terminal_ids,
                       dict(tree.segment_paths))
    before = materialize_variant("polish", "corridor refinement", result, terminals, grid, {}, bend_cost_rub)
    for index, segment in enumerate(segments):
        barriers = [s.points for i, s in enumerate(segments) if i != index]
        points = tidy_orthogonal_path(segment.points, grid, barriers)
        if points == segment.points:
            continue
        key = edge_key(segment.start_cell, segment.end_cell)
        previous = result.segment_paths.get(key)
        result.segment_paths[key] = points if segment.start_cell <= segment.end_cell else list(reversed(points))
        try:
            after = materialize_variant("polish", "corridor refinement", result, terminals, grid, {}, bend_cost_rub)
        except ValueError:
            after = None
        if after is not None and variant_key(after) <= variant_key(before):
            segment.points = points
            before = after
        elif previous is None:
            del result.segment_paths[key]
        else:
            result.segment_paths[key] = previous
    return result


def safe_shortcuts(points: List[Point], grid: RoutingGrid,
                   barriers: Sequence[Sequence[Point]]) -> Iterable[List[Point]]:
    """Straight shortcuts and trimmed corners, with fixed endpoints and no collisions."""
    def clear(path):
        return (not any(polylines_conflict(path, other) for other in barriers)
                and not any(segments_distance(a, b, c, d) < 1e-7
                            for i, (a, b) in enumerate(zip(path, path[1:]))
                            for c, d in zip(path[i + 2:], path[i + 3:])))

    candidates = []
    for i in range(len(points) - 2):
        for j in range(i + 2, len(points)):
            if grid.service_line_clear(points[i], points[j]):
                path = remove_collinear(points[:i + 1] + points[j:])
                if clear(path):
                    candidates.append(path)
    yield from sorted(candidates, key=lambda p: (path_length(p), polyline_bend_count(p)))


def refine_geometry(tree: BuiltTree, terminals: List[Terminal], grid: RoutingGrid,
                    bend_cost_rub: float = 0.) -> BuiltTree:
    """Coordinate descent on corridors and junctions; accept only full-price improvements."""
    result = BuiltTree(tree.roots, tree.edges, tree.connected_terminal_ids, tree.unconnected_terminal_ids,
                       dict(tree.segment_paths), dict(tree.node_points))
    before = materialize_variant("refine", "geometry refinement", result, terminals, grid, {}, bend_cost_rub)

    def attempt(paths, positions=None):
        nonlocal before
        old_paths, old_points = dict(result.segment_paths), dict(result.node_points)
        for segment, points in paths:
            key = edge_key(segment.start_cell, segment.end_cell)
            result.segment_paths[key] = points if key[0] == segment.start_cell else list(reversed(points))
        result.node_points.update(positions or {})
        try:
            after = materialize_variant("refine", "geometry refinement", result, terminals, grid, {}, bend_cost_rub)
        except ValueError:
            after = None
        if (after is not None and variant_key(after) < variant_key(before)
                and after.summary["calculated_cost"] <= before.summary["calculated_cost"]):
            before = after
            return True
        result.segment_paths, result.node_points = old_paths, old_points
        return False

    for _pass in range(5):
        improved = False
        segments, _, _ = compress_segments(result, terminals, grid)
        for index, segment in enumerate(segments):
            barriers = [s.points for i, s in enumerate(segments) if i != index]
            for _ in range(12):
                for points in safe_shortcuts(segment.points, grid, barriers):
                    if attempt([(segment, points)]):
                        segment.points = points
                        improved = True
                        break
                else:
                    break
        # Weighted geometric median reduces the total price of incident pipes.
        # Backtracking keeps chambers outside buildings and away from other pipes.
        incident = defaultdict(list)
        for segment in segments:
            incident[segment.start_cell].append((segment, True))
            incident[segment.end_cell].append((segment, False))
        fixed = {r.cell for r in tree.roots} | {t.cell for t in terminals}
        for cell, neighbors in sorted(incident.items()):
            if cell in fixed or len(neighbors) < 3:
                continue
            origin = result.node_points.get(cell, grid.cell_to_point(cell))
            anchors = [(s.points[1] if start else s.points[-2], NEW_COST[s.diameter])
                       for s, start in neighbors]
            target = origin
            for _ in range(60):
                weights = [(p, rate / max(.01, dist(target, p))) for p, rate in anchors]
                total = sum(w for _, w in weights)
                nxt = tuple(sum(p[k] * w for p, w in weights) / total for k in (0, 1))
                if dist(target, nxt) < .01:
                    break
                target = nxt
            for scale in (1., .5, .25, .125):
                point = tuple(origin[k] + scale * (target[k] - origin[k]) for k in (0, 1))
                if dist(point, origin) < .05 or any(dist(point, p) < .5 for p, _ in anchors):
                    continue
                if any(not grid.line_clear(point, p) for p, _ in anchors):
                    continue
                changed = [(s, [point] + s.points[1:] if start else s.points[:-1] + [point])
                           for s, start in neighbors]
                updated = {id(s): p for s, p in changed}
                paths = [updated.get(id(s), s.points) for s in segments]
                if any(polylines_conflict(a, b) for i, a in enumerate(paths) for b in paths[i + 1:]):
                    continue
                if attempt(changed, {cell: point}):
                    for s, p in changed:
                        s.points = p
                    improved = True
                    break
        if not improved:
            break
    return result


def write_geojson(path: str, features: List[Dict[str, Any]], include_crs: bool = True) -> None:
    collection = {
        "type": "FeatureCollection",
        "features": features,
    }
    if include_crs:
        collection["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False, indent=2)


def gis_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def gis_safe_feature(feature: Dict[str, Any]) -> Dict[str, Any]:
    props = feature.get("properties") or {}
    return {
        "type": "Feature",
        "geometry": feature.get("geometry"),
        "properties": {str(k): gis_safe_value(v) for k, v in props.items() if v is not None},
    }


def thin_feature(feature: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    props = feature.get("properties") or {}
    return {
        "type": "Feature",
        "geometry": feature.get("geometry"),
        "properties": {
            str(k): gis_safe_value(props[k])
            for k in keys
            if k in props and props[k] is not None
        },
    }


def write_gis_friendly_layers(output_dir: str, variants: List[Variant]) -> None:
    layer_dir = os.path.join(output_dir, "openable_layers")
    os.makedirs(layer_dir, exist_ok=True)

    best_features = [gis_safe_feature(ft) for ft in variants[0].features if ft.get("geometry") is not None]
    best_lines = [
        ft
        for ft in best_features
        if (ft.get("geometry") or {}).get("type") == "LineString"
    ]
    best_points = [
        ft
        for ft in best_features
        if (ft.get("geometry") or {}).get("type") == "Point"
    ]

    ranked_spatial = [
        gis_safe_feature(ft)
        for variant in variants
        for ft in variant.features
        if ft.get("geometry") is not None
    ]
    ranked_lines = [
        ft
        for ft in ranked_spatial
        if (ft.get("geometry") or {}).get("type") == "LineString"
    ]
    ranked_points = [
        ft
        for ft in ranked_spatial
        if (ft.get("geometry") or {}).get("type") == "Point"
    ]

    line_keys = [
        "id",
        "variant_id",
        "start_node_id",
        "end_node_id",
        "flow_tph",
        "diameter",
        "length",
        "bend_count",
        "bend_penalty_cost",
        "cost",
    ]
    point_keys = [
        "id",
        "object_type",
        "variant_id",
        "existing_object_id",
        "existing_object_type",
        "existing_diameter",
        "required_diameter",
        "diameter",
        "cost",
    ]

    strict_best_lines = [thin_feature(ft, line_keys) for ft in best_lines]
    strict_best_points = [thin_feature(ft, point_keys) for ft in best_points]
    strict_ranked_lines = [thin_feature(ft, line_keys) for ft in ranked_lines]
    strict_ranked_points = [thin_feature(ft, point_keys) for ft in ranked_points]
    minimal_line_keys = ["id", "variant_id", "diameter", "length"]
    minimal_point_keys = ["id", "object_type", "variant_id"]
    minimal_best_lines = [thin_feature(ft, minimal_line_keys) for ft in best_lines]
    minimal_best_points = [thin_feature(ft, minimal_point_keys) for ft in best_points]

    write_geojson(os.path.join(layer_dir, "best_lines.geojson"), strict_best_lines, include_crs=False)
    write_geojson(os.path.join(layer_dir, "best_points.geojson"), strict_best_points, include_crs=False)
    write_geojson(os.path.join(layer_dir, "best_lines_minimal.geojson"), minimal_best_lines, include_crs=False)
    write_geojson(os.path.join(layer_dir, "best_points_minimal.geojson"), minimal_best_points, include_crs=False)
    write_geojson(os.path.join(layer_dir, "best_spatial.geojson"), best_features, include_crs=False)
    write_geojson(os.path.join(layer_dir, "ranked_lines.geojson"), strict_ranked_lines, include_crs=False)
    write_geojson(os.path.join(layer_dir, "ranked_points.geojson"), strict_ranked_points, include_crs=False)
    write_geojson(os.path.join(layer_dir, "ranked_spatial.geojson"), ranked_spatial, include_crs=False)

    with open(os.path.join(layer_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Open these files if your GIS cannot load the competition-format GeoJSON.\n"
            "best_lines.geojson: only new heat network LineString features for the best variant.\n"
            "best_points.geojson: tie-in and heat chamber Point features for the best variant.\n"
            "best_lines_minimal.geojson: minimal line attributes for strict web map importers.\n"
            "best_points_minimal.geojson: minimal point attributes for strict web map importers.\n"
            "best_spatial.geojson: best variant without null geometry summary records.\n"
            "ranked_lines.geojson / ranked_points.geojson: top ranked variants split by geometry type.\n"
            "No CRS member, geometry=null features, null properties, or array/dict property values are included here.\n"
        )


def write_report(path: str, variants: List[Variant], meta: Dict[str, Any], output_dir: str) -> None:
    lines: List[str] = []
    lines.append("# Routing report")
    lines.append("")
    lines.append("Input profile:")
    lines.append(f"- features: {meta['input_feature_count']}")
    lines.append(f"- connection points: {meta['terminal_count']}")
    lines.append(f"- tie candidates: {meta['candidate_count']}")
    lines.append(f"- forbidden obstacles: {meta['obstacle_count']}")
    lines.append("- objective: minimum official weighted score among feasible candidates")
    lines.append(
        f"- geometry tie-break: fewer micro-bends, bends and right-angle turns within "
        f"{meta.get('score_tie_epsilon', SCORE_TIE_EPSILON) * 100:.1f}% score"
    )
    lines.append(f"- optional routing turn bias = {meta.get('turn_penalty_m', 0)} m")
    lines.append(f"- bend penalty in ranking = {meta.get('bend_cost_rub', 0)} rub per bend")
    if meta.get("orthogonal_corridors"):
        lines.append(
            f"- corridor mode: rotated orthogonal city-block grid, axis = {meta.get('city_axis_degrees')} degrees"
        )
    if not (meta.get("has_existing_flow") and meta.get("has_upstream")):
        lines.append("- reconstruction propagation: skipped, dataset has no heat_network flow_tph/upstream_object_id")
    lines.append("")
    lines.append("Ranked variants:")
    for v in variants:
        s = v.summary
        lines.append(
            f"{s['rank']}. {v.id}: S={s['score']}, cost={s['calculated_cost']:,} rub, "
            f"length={s['length']} m, connected={s['connected_oks_count']}/{meta['terminal_count']}, "
            f"bends={s.get('route_bend_count', 0)}, micro-bends={s.get('micro_bend_count', 0)}, "
            f"tie-ins={s['tie_in_count']}, "
            f"unconnected={s['unconnected_oks_ids']}"
        )
    if variants:
        best = variants[0]
        lines.append("")
        lines.append("Best result:")
        lines.append(
            f"- {best.id} ranked first because it has the lowest official score among the evaluated "
            f"topologies; geometry breaks ties within the configured score tolerance."
        )
        lines.append(f"- File: {os.path.join(output_dir, 'best_variant.geojson')}")
    lines.append("")
    lines.append("Routing model:")
    lines.append("- EPSG:4326 input is projected to EPSG:32637 by an internal UTM conversion.")
    lines.append("- OKS polygons are blocked with a 5 m clearance; water with 1 m; railway with 2 m.")
    lines.append("- Heading-aware A* compares the cost of a new tie-in with joining an existing branch.")
    lines.append("- Search prices include pipe diameter, upstream flow increases, tie-ins and chambers.")
    if meta.get("orthogonal_corridors"):
        lines.append("- Orthogonal corridor mode disables diagonal cuts and aligns routes to the dominant OKS block axis.")
    lines.append("- Leaf branches and complete downstream subtrees are reattached only when full network cost improves.")
    lines.append("- Every grid edge is checked against continuous obstacle geometry, not only occupied grid cells.")
    lines.append("- Actual connection coordinates are preserved; only short service leads may exit their own building.")
    lines.append("- Search bias and official score are separate; monetary bend surcharge is zero by default.")
    lines.append("- Near-score alternatives prefer fewer micro-bends, bends and right-angle turns.")
    lines.append("- Dominant facade orientation is estimated robustly; final L-shaped shortcuts remove stair steps without increasing cost or crossing other pipes.")
    lines.append("- Continuous same-diameter length is tracked across chambers; excessive demand is rejected.")
    lines.append("- Branching cells are emitted as new heat chambers; pipe tie-ins also get a new chamber.")
    lines.append("- Depth and special crossings are checked with shared-node profiles and utility clearances.")
    lines.append("- This is a discrete local optimization, not a proof of a global minimum. Grid resolution affects the result.")
    lines.append("- Building clearance is a fixed 5 m centerline buffer in this model; full diameter-dependent envelopes are not modeled.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build obstacle-aware heat network route variants.")
    parser.add_argument("--input", default="!!!_Датасет.geojson", help="Input GeoJSON path")
    parser.add_argument("--output-dir", default="routing_results", help="Output directory")
    parser.add_argument("--grid-step", type=float, default=5.0, help="Routing grid step in meters")
    parser.add_argument("--normalize-connections", action="store_true",
                        help="Explicitly move blocked OKS connection points outside buildings; export the corrections")
    parser.add_argument("--refine-geometry", action=argparse.BooleanOptionalAction, default=True,
                        help="Shorten free corridors and reposition junctions with complete cost/depth checks")
    parser.add_argument("--max-variants", type=int, default=3, help="How many top variants to export")
    parser.add_argument(
        "--max-topologies",
        type=int,
        default=0,
        help="Maximum topology candidates to build before ranking; 0 means full portfolio",
    )
    parser.add_argument(
        "--turn-penalty-m",
        type=float,
        default=0.0,
        help="Equivalent meters added for a 90-degree direction change during routing",
    )
    parser.add_argument(
        "--bend-cost-rub",
        type=float,
        default=0.0,
        help="Internal search bias per bend; never added to official cost or score",
    )
    parser.add_argument(
        "--orthogonal-corridors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route only along the dominant rotated city-block axes, like corridor/trunk utility plans",
    )
    args = parser.parse_args()
    if not math.isfinite(args.grid_step) or args.grid_step <= 0:
        parser.error("--grid-step must be finite and positive")
    if args.max_variants < 1 or args.max_topologies < 0:
        parser.error("--max-variants must be positive and --max-topologies non-negative")
    if any(not math.isfinite(v) or v < 0 for v in (args.turn_penalty_m, args.bend_cost_rub)):
        parser.error("Turn and bend penalties must be finite and non-negative")

    terminals, candidates, obstacles, all_points, meta = read_input(args.input)
    crossing_objects = meta.pop("crossing_objects")
    existing_network = meta.pop("existing_network")
    meta["existing_network_status"] = existing_network.status()
    envelope_diameter = select_diameter(sum(t.flow_tph for t in terminals))
    for obstacle in obstacles:
        obstacle.clearance = clearance(obstacle.kind, envelope_diameter)
        obstacle.bbox = expand_bbox(bbox([p for ring in obstacle.rings for p in ring]), obstacle.clearance)
    meta["search_envelope_diameter"] = envelope_diameter
    meta["special_objects_count"] = len(crossing_objects)
    coord_transform = IDENTITY_TRANSFORM
    city_axis_rad = 0.0
    if all_points:
        city_axis_rad = estimate_city_axis(obstacles)
        if all_points:
            origin = (
                sum(p[0] for p in all_points) / len(all_points),
                sum(p[1] for p in all_points) / len(all_points),
            )
        else:
            origin = (0.0, 0.0)
        coord_transform = CoordinateTransform(origin=origin, angle_rad=city_axis_rad)
        terminals, candidates, obstacles, all_points = apply_coordinate_transform(
            terminals,
            candidates,
            obstacles,
            all_points,
            coord_transform,
        )
    grid = prepare_grid(
        terminals,
        candidates,
        obstacles,
        all_points,
        args.grid_step,
        cardinal_only=args.orthogonal_corridors,
        coord_transform=coord_transform,
        normalize_terminals=args.normalize_connections,
        allow_building_leads=not args.normalize_connections,
    )
    grid.existing_network = existing_network
    grid.crossing_objects = transformed_objects(crossing_objects, coord_transform.forward)
    meta["connection_adjustments"] = grid.connection_adjustments
    meta["refine_geometry"] = args.refine_geometry
    meta["terminal_building_access"] = "Exact input coordinates; short terminal leads only" if grid.allow_building_leads else "Strict external connections"
    meta["grid_step"] = args.grid_step
    meta["grid_size"] = [grid.nx, grid.ny]
    meta["terminals_snapped"] = sum(1 for t in terminals if t.cell is not None)
    meta["candidates_snapped"] = sum(1 for c in candidates if c.cell is not None)
    meta["turn_penalty_m"] = args.turn_penalty_m
    meta["bend_cost_rub"] = args.bend_cost_rub
    meta["orthogonal_corridors"] = args.orthogonal_corridors
    meta["city_axis_degrees"] = round(math.degrees(city_axis_rad), 3)
    meta["score_tie_epsilon"] = SCORE_TIE_EPSILON

    variants = make_variants(
        terminals,
        candidates,
        grid,
        meta,
        turn_penalty_m=args.turn_penalty_m,
        bend_cost_rub=args.bend_cost_rub,
        max_topologies=args.max_topologies,
    )
    if not variants:
        raise RuntimeError("No route variants could be built")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "connection_adjustments.json"), "w", encoding="utf-8") as f:
        json.dump(grid.connection_adjustments, f, ensure_ascii=False, indent=2)
    if grid.connection_adjustments:
        with open(args.input, encoding="utf-8") as f:
            normalized = json.load(f)
        positions = {v["terminal_id"]: v["connection_coordinates"] for v in grid.connection_adjustments}
        for feature in normalized["features"]:
            if feature.get("properties", {}).get("object_type") == "oks_connection_point":
                replacement = positions.get(str(feature["properties"]["id"]))
                if replacement is not None:
                    feature["geometry"]["coordinates"] = replacement
        with open(os.path.join(args.output_dir, "normalized_input.geojson"), "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
    selected = variants[: args.max_variants]
    all_features: List[Dict[str, Any]] = []
    for variant in selected:
        all_features.extend(variant.features)
    write_geojson(os.path.join(args.output_dir, "variants_ranked.geojson"), all_features)
    write_geojson(os.path.join(args.output_dir, "best_variant.geojson"), selected[0].features)
    for variant in selected:
        write_geojson(os.path.join(args.output_dir, f"{variant.id}.geojson"), variant.features)
    write_gis_friendly_layers(args.output_dir, selected)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "variants": [v.summary for v in selected]}, f, ensure_ascii=False, indent=2)
    write_report(os.path.join(args.output_dir, "report.md"), selected, meta, args.output_dir)

    from routing_preview import write_preview
    write_preview(args.input, os.path.join(args.output_dir, "best_variant.geojson"),
                  os.path.join(args.output_dir, "preview.svg"))

    print(json.dumps({"output_dir": args.output_dir, "best": selected[0].summary, "variants": [v.summary for v in selected]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
