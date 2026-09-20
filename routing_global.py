"""Global rectilinear Steiner seeds, followed by the unchanged exact evaluator.

The construction graph is private. Only a selected forest enters the visibility
search. Terminal leads use the closest facade; source candidates honor chamber
reuse before metric closure. A proxy diameter guides corridors, never pricing.
"""

import copy
import heapq
import math
import time
from collections import defaultdict

import numpy as np
from shapely import distance, linestrings

import heat_route_builder as h
from routing_constraints import validate_path
from routing_depth import clearance

SUPER = (-9, -9)


def minimum_tree(edges, weights, required):
    """Kruskal followed by deletion of nonterminal leaves."""
    parent = {}

    def find(n):
        parent.setdefault(n, n)
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    adjacency = defaultdict(set)
    for a, b in sorted(edges, key=lambda e: (weights[e], e)):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            adjacency[a].add(b)
            adjacency[b].add(a)
    queue = [n for n in adjacency if len(adjacency[n]) == 1 and n not in required]
    while queue:
        n = queue.pop()
        if not adjacency[n]:
            continue
        other = adjacency[n].pop()
        adjacency[other].remove(n)
        if len(adjacency[other]) == 1 and other not in required:
            queue.append(other)
    return {h.edge_key(a, b) for a in adjacency for b in adjacency[a]}


def rectilinear_seed(
    ts, rs, original, dn=300, step=5.0, root_factor=1.0, deadline=math.inf
):
    """Return one global candidate; exact feasibility is the caller's decision."""
    if time.monotonic() >= deadline:
        return None
    if not ts or not rs:
        return h.BuiltTree([], set(), set(), {t.id for t in ts})
    g = copy.copy(original)
    g.extra_points = dict(original.extra_points)
    g._coordinate_nodes = dict(original._coordinate_nodes)
    g.graph = defaultdict(dict)
    g._crossing_edge_cache = {}
    positions = list(g.extra_points.values())
    # Keep memory bounded on large territories; this changes seed resolution,
    # never geometry admissibility or the visibility fallback.
    area = math.prod(
        max(p[k] for p in positions) - min(p[k] for p in positions) + 2 * step
        for k in (0, 1)
    )
    step = max(step, math.sqrt(area / 120_000))
    low = [math.floor(min(p[k] for p in positions) / step) for k in (0, 1)]
    hi = [math.ceil(max(p[k] for p in positions) / step) for k in (0, 1)]
    ns = {
        (i, j): g.node((i * step, j * step))
        for i in range(low[0], hi[0] + 1)
        for j in range(low[1], hi[1] + 1)
    }
    eds = [
        (ns[i, j], ns[x, y])
        for i, j in ns
        for x, y in ((i + 1, j), (i, j + 1))
        if (x, y) in ns
    ]
    reserved_entries = g.terminal_cells | set().union(*g._terminal_ports.values())
    ls = linestrings([[g.extra_points[a], g.extra_points[b]] for a, b in eds])
    rows, obs = g._spatial.query(ls)
    valid = np.ones(len(eds), dtype=bool)
    gap = distance(ls[rows], np.array(g._shapes, dtype=object)[obs])
    valid[
        rows[gap < np.array([clearance(o.kind, dn) + 0.05 for o in g.obstacles])[obs]]
    ] = False
    adj = defaultdict(dict)

    def add(a, b):
        if a != b:
            adj[a][b] = adj[b][a] = h.dist(g.extra_points[a], g.extra_points[b])
            g.connect(a, b)

    for obj in g.crossing_objects:
        if obj.kind == "heat_network":
            valid[
                distance(ls, obj.geometry)
                < obj.horizontal_gap + (h.ENVELOPE[dn][0] + obj.width) / 2 + 0.05
            ] = False
    for (a, b), ok in zip(eds, valid):
        if ok and a not in reserved_entries and b not in reserved_entries:
            add(a, b)

    # Root enters a split vertical grid edge horizontally, preserving 90-degree turns.
    roots = {}
    for r in sorted(rs, key=lambda r: (not r.is_existing_chamber, r.id)):
        if time.monotonic() >= deadline:
            return None
        if (
            r.cell is None
            or r.existing_degree >= 4
            or (not r.is_existing_chamber and not h.new_tie_allowed(r, [], set()))
        ):
            continue
        x, y = r.point
        i = math.floor(x / step)
        j = math.floor(y / step)
        options = []
        for gx in range(i - 4, i + 5):
            a, b = ns.get((gx, j)), ns.get((gx, j + 1))
            q = (gx * step, y)
            if b not in adj[a]:
                continue
            if not g.line_clear(r.point, q):
                continue
            qnode = g.node(q)
            if not math.isfinite(
                g.crossing_weight(r.cell, qnode, dn, h.dist(r.point, q))
            ):
                continue
            options.append((h.dist(r.point, q), a, b, q))
        if not options:
            continue
        _, a, b, q = min(options)
        n = g.node(q)
        if n not in (a, b):
            adj[a].pop(b)
            adj[b].pop(a)
            add(a, n)
            add(n, b)
        add(r.cell, n)
        roots[r.cell] = r
        adj[SUPER][r.cell] = adj[r.cell][SUPER] = (
            root_factor
            * h.opening_cost(r, dn)
            / (h.NEW_COST[dn] + h.SCORE_LENGTH_RUB_PER_M)
        )

    main = {SUPER}
    queue = [SUPER]
    for n in queue:
        for other in adj[n]:
            if other not in main:
                main.add(other)
                queue.append(other)
    # Attach inputs through one complete nearest-facade lead, with a legal turn at the grid.
    for t in ts:
        if time.monotonic() >= deadline:
            return None
        opts = []
        for port in sorted(g._terminal_ports[t.cell]):
            p = g.extra_points[port]
            ix, iy = round(p[0] / step), round(p[1] / step)
            for i in range(ix - 12, ix + 13):
                for j in range(iy - 12, iy + 13):
                    gate = ns.get((i, j))
                    if (
                        gate not in main
                        or gate in roots
                        or gate in g.terminal_cells
                        or gate in g._entry_port_owners
                    ):
                        continue
                    q = g.extra_points[gate]
                    pts = [t.point, p, q]
                    outward = [
                        n
                        for n in adj[gate]
                        if sum(
                            (q[k] - p[k]) * (g.extra_points[n][k] - q[k])
                            for k in (0, 1)
                        )
                        >= -1e-8
                    ]
                    if not outward:
                        continue
                    cost = h.path_length(pts)
                    opts.append((cost, port, gate, outward))
        for cost, port, gate, outward in sorted(opts):
            try:
                validate_path(
                    [t.point, g.extra_points[port], g.extra_points[gate]],
                    h.select_diameter(t.flow_tph),
                    g.obstacles,
                    [t.point],
                    True,
                )
            except ValueError:
                continue
            for n in list(adj[gate]):
                if n not in outward:
                    adj[n].pop(gate)
                    adj[gate].pop(n)
            add(t.cell, port)
            add(port, gate)
            break
        else:
            continue
    required = {SUPER} | {t.cell for t in ts if adj[t.cell]}
    paths = {}
    metric = {}
    for source in sorted(required):
        if time.monotonic() >= deadline:
            return None
        ds = {source: 0}
        prev = {}
        queue = [(0, source)]
        pending = required - {source}
        while queue and pending:
            if time.monotonic() >= deadline:
                return None
            value, n = heapq.heappop(queue)
            if value != ds[n]:
                continue
            if n in pending:
                pending.remove(n)
                key = h.edge_key(source, n)
                metric[key] = value
                path = [n]
                while path[-1] != source:
                    path.append(prev[path[-1]])
                paths[key] = path
            if n != source and n in required:
                continue
            for nxt, cost in adj[n].items():
                v = value + cost
                if v < ds.get(nxt, math.inf):
                    ds[nxt] = v
                    prev[nxt] = n
                    heapq.heappush(queue, (v, nxt))

    weights = {h.edge_key(a, b): v for a in adj for b, v in adj[a].items()}
    links = minimum_tree(metric, metric, required)
    expanded = {h.edge_key(a, b) for k in links for a, b in zip(paths[k], paths[k][1:])}
    edges = minimum_tree(expanded, weights, required)
    rr = [roots[b if a == SUPER else a] for a, b in edges if SUPER in (a, b)]
    edges = {e for e in edges if SUPER not in e}
    connected = {t.id for t in ts if any(t.cell in e for e in edges)}
    # Transfer only the selected geometry; dense construction edges stay private.
    mapping = {n: original.node(g.extra_points[n]) for edge in edges for n in edge}
    for a, b in edges:
        original.connect(mapping[a], mapping[b])
    return h.BuiltTree(
        rr,
        {h.edge_key(mapping[a], mapping[b]) for a, b in edges},
        connected,
        {t.id for t in ts} - connected,
    )
