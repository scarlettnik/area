"""Budgeted topology search. Proposals never substitute for exact evaluation."""
from collections import Counter
import math
import time

import heat_route_builder as h


def signature(tree):
    return (tuple(sorted(r.id for r in tree.roots)), frozenset(tree.edges), frozenset(tree.connected_terminal_ids),
            tuple(sorted((edge, tuple(points)) for edge, points in tree.segment_paths.items())),
            tuple(sorted(tree.node_points.items())))


def structure(tree, terminals):
    parent, depth, _, adjacency = h.orient_and_flow(tree, terminals)
    descendants = {n: set() for n in parent}
    for t in terminals:
        if t.id in tree.connected_terminal_ids:
            descendants[t.cell].add(t.id)
    for n in sorted(parent, key=lambda n: depth[n], reverse=True):
        if parent[n] is not None:
            descendants[parent[n]].update(descendants[n])
    groups = tuple(sorted((r.existing_object_id, tuple(sorted(descendants[r.cell]))) for r in tree.roots))
    clades = tuple(sorted(tuple(sorted(descendants[n])) for n in parent
                          if len(adjacency[n]) >= 3 and len(descendants[n]) > 1))
    return groups, clades


class Search:
    def __init__(self, terminals, candidates, graph, seconds=120., beam_width=5, routes=3, turn_m=8.,
                 seed_portfolio=True, initial_trees=()):
        self.terminals, self.candidates, self.graph = terminals, candidates, graph
        self.seconds, self.beam_width, self.routes, self.turn_m = seconds, beam_width, routes, turn_m
        self.deadline = math.inf
        self.evaluated = {}
        self.archive = {}
        self.rejections = Counter()
        self.standalone = {}
        self.evaluations = 0
        self.history = []
        self.best = None
        self.seed_portfolio = seed_portfolio
        self.seed_reports = []
        self.initial_trees = list(initial_trees)
        self.subtree_checked = set()

    def expired(self):
        return time.monotonic() >= self.deadline

    def evaluate(self, tree):
        key = signature(tree)
        if key in self.evaluated:
            return self.evaluated[key]
        self.evaluations += 1
        try:
            variant = h.materialize_variant("trial", "visibility topology search", tree, self.terminals, self.graph, {})
        except ValueError as error:
            self.rejections[str(error)] += 1
            self.evaluated[key] = None
            return None
        self.evaluated[key] = variant
        family = structure(tree, self.terminals)
        if family not in self.archive or h.search_key(variant) < h.search_key(self.archive[family]):
            self.archive[family] = variant
        if self.best is None or h.search_key(variant) < h.search_key(self.best):
            self.best = variant
            self.history.append({"seconds": round(time.monotonic() - self.started, 3),
                                 "score": variant.score, "connected": len(tree.connected_terminal_ids),
                                 "evaluations": self.evaluations})
        return variant

    def independent_paths(self, terminal):
        if terminal.id in self.standalone:
            return self.standalone[terminal.id]
        diameter = h.select_diameter(terminal.flow_tph)
        sources = {}
        for r in self.candidates:
            if r.cell is None or not h.new_tie_allowed(r, [], set()):
                continue
            price = h.opening_cost(r, diameter)
            sources[r.cell] = price
        paths = []
        roots = {r.cell: r for r in self.candidates if r.cell is not None}
        attempts = 0
        while sources and len(paths) < self.routes and attempts < 20:
            if self.expired():
                break
            attempts += 1
            route = self.graph.least_cost_path(sources, terminal.cell, set(), h.NEW_COST[diameter], self.turn_m * h.NEW_COST[diameter])
            if not route:
                break
            source = route[0]
            alternatives = [route]
            for path in alternatives:
                tree = h.BuiltTree([roots[source]], {h.edge_key(a,b) for a,b in zip(path,path[1:])},
                                   {terminal.id}, {t.id for t in self.terminals if t.id != terminal.id})
                if self.evaluate(tree) is not None:
                    paths.append(path)
                    break
            else:
                # A cheapest 2D path may fail its 3D profile. Keep the tie-in
                # and try a different corridor before rejecting that tie.
                for path in self.graph.alternatives({source: sources[source]}, terminal.cell, diameter, 3, self.turn_m)[1:]:
                    if self.expired():
                        break
                    tree = h.BuiltTree([roots[source]], {h.edge_key(a,b) for a,b in zip(path,path[1:])},
                                       {terminal.id}, {t.id for t in self.terminals if t.id != terminal.id})
                    if self.evaluate(tree) is not None:
                        paths.append(path)
                        break
            sources.pop(source, None)
        if self.expired():
            return paths[:self.routes]  # A truncated attempt must not poison the next phase.
        self.standalone[terminal.id] = paths[:self.routes]
        return self.standalone[terminal.id]

    def proposals(self, tree, terminal):
        yielded = set()
        feasible = False
        for candidate in self.direct_joins(tree, terminal):
            key = signature(candidate)
            if key not in yielded:
                yielded.add(key)
                yield candidate
                feasible |= self.evaluated.get(key) is not None
        paths = self.independent_paths(terminal) or [[]]
        self.graph.set_barriers(tree.edges)
        try:
            for standalone in paths:
                if self.expired():
                    return
                for candidate in h.connection_options(tree, terminal, self.candidates, self.terminals,
                                                      self.graph, standalone, 0., self.turn_m):
                    key = signature(candidate)
                    if key not in yielded:
                        yielded.add(key)
                        yield candidate
                        feasible |= self.evaluated.get(key) is not None
            # Reuse cached routes and trunk attachments first. A fresh search
            # around the entire forest is expensive and needed only when these
            # fail. ALNS removal still explores new roots and corridors.
            if not feasible and tree.edges and self.graph.graph[terminal.cell] and not self.expired():
                parent, _, _, _ = h.orient_and_flow(tree, self.terminals)
                diameter = h.select_diameter(terminal.flow_tph)
                sources = {r.cell: h.opening_cost(r, diameter) for r in self.candidates
                           if r.cell is not None and r.cell not in parent
                           and h.new_tie_allowed(r, tree.roots, tree.edges)}
                alternative = self.graph.least_cost_path(sources, terminal.cell, set(parent),
                    h.NEW_COST[diameter], self.turn_m * h.NEW_COST[diameter])
                if alternative and alternative not in paths:
                    for candidate in h.connection_options(tree, terminal, self.candidates, self.terminals,
                                                          self.graph, alternative, 0., self.turn_m):
                        if signature(candidate) not in yielded:
                            yield candidate
        finally:
            self.graph.set_barriers()

    def direct_joins(self, tree, terminal):
        """Add a service branch at a projection on a trunk, not just old vertices.

        The original pipe is split geometrically and topologically at the same
        point; the complete evaluator decides chamber cost and upstream sizing.
        """
        if not tree.edges or terminal.id in tree.connected_terminal_ids:
            return
        ports = self.graph._terminal_ports.get(terminal.cell, ())
        candidates = []
        for a, b in sorted(tree.edges):
            pa, pb = self.graph.cell_to_point(a), self.graph.cell_to_point(b)
            length2 = h.dist(pa, pb) ** 2
            if length2 < 1e-8:
                continue
            for port in sorted(ports):
                point = self.graph.cell_to_point(port)
                fraction = max(0., min(1., sum((point[k] - pa[k]) * (pb[k] - pa[k]) for k in (0, 1)) / length2))
                projection = tuple(pa[k] + fraction * (pb[k] - pa[k]) for k in (0, 1))
                candidates.append((h.dist(projection, point) + h.dist(point, terminal.point), a, b, port, projection))
        checked = set()
        for _, a, b, port, point in sorted(candidates)[:16]:
            if self.expired():
                return
            if not self.graph.line_clear(point, self.graph.cell_to_point(port)):
                continue
            join = self.graph.node(point)
            if (join, port) in checked or join in self.graph.terminal_cells:
                continue
            checked.add((join, port))
            edges = set(tree.edges)
            if join not in (a, b):
                edges.remove(h.edge_key(a, b))
                edges.update({h.edge_key(a, join), h.edge_key(join, b)})
                self.graph.connect(a, join); self.graph.connect(join, b)
            path = [join, port, terminal.cell] if join != port else [join, terminal.cell]
            for left, right in zip(path, path[1:]):
                edges.add(h.edge_key(left, right))
            yield h.BuiltTree(list(tree.roots), edges, tree.connected_terminal_ids | {terminal.id},
                              tree.unconnected_terminal_ids - {terminal.id})

    def select(self, variants, width):
        unique = {}
        for v in variants:
            if v is not None:
                key = signature(v.tree)
                if key not in unique or h.variant_key(v) < h.variant_key(unique[key]):
                    unique[key] = v
        ranked = sorted(unique.values(), key=h.search_key)
        selected, families = [], set()
        for v in ranked:
            family = structure(v.tree, self.terminals)
            if family not in families:
                selected.append(v)
                families.add(family)
                if len(selected) == width:
                    return selected
        selected.extend(v for v in ranked if v not in selected)
        return selected[:width]

    def run(self):
        self.started = time.monotonic()
        self.deadline = self.started + self.seconds
        self.graph.deadline = self.deadline
        empty = h.BuiltTree([], set(), set(), {t.id for t in self.terminals})
        baseline = self.evaluate(empty)
        for index, tree in enumerate(self.initial_trees):
            value = self.evaluate(tree)
            self.seed_reports.append({'name': f'imported-{index + 1}', 'valid_in_mode': value is not None,
                                      'score': value.score if value else None,
                                      'connected': len(tree.connected_terminal_ids) if value else 0})
        if self.initial_trees and not self.seed_portfolio and not self.best.tree.unconnected_terminal_ids:
            return self.finalists()
        nearest = lambda t: min((h.dist(t.point, r.point) for r in self.candidates if r.cell is not None), default=math.inf)
        orders = [sorted(self.terminals, key=lambda t: (-t.flow_tph, t.id)),
                  sorted(self.terminals, key=lambda t: (t.point[0], t.point[1], t.id))]
        if self.seed_portfolio:
            greedy_orders = [("demand", orders[0]),
                             ("near-first", sorted(self.terminals, key=lambda t: (nearest(t), t.id))),
                             ("hard-first", sorted(self.terminals, key=lambda t: (-nearest(t), t.id)))]
            for label, order in greedy_orders:
                state = baseline
                for terminal in order:
                    if self.expired():
                        break
                    values = [self.evaluate(p) for p in self.proposals(state.tree, terminal)]
                    values = [v for v in values if v is not None]
                    if values:
                        state = min(values, key=h.search_key)
                self.seed_reports.append({"name": label, "score": state.score,
                    "connected": len(state.tree.connected_terminal_ids)})
                print(f"Greedy {label}: {len(state.tree.connected_terminal_ids)}/{len(self.terminals)}, score={state.score:.6f}", flush=True)
                if state.tree.unconnected_terminal_ids and getattr(self.graph, 'search_clearance_diameter', 0):
                    # Reserved trunk width is a search preference. A narrow
                    # but legal service corridor must remain discoverable.
                    self.graph.search_clearance_diameter = 0
                    self.standalone.clear()
                if self.expired():
                    return self.finalists()
        for order in orders:
            beam = [baseline]
            for terminal in order:
                if self.expired():
                    break
                pool = list(beam)  # Incomplete states are exploration only; coverage wins selection.
                for state in beam:
                    for proposal in self.proposals(state.tree, terminal):
                        if self.expired():
                            break
                        value = self.evaluate(proposal)
                        if value is not None:
                            pool.append(value)
                beam = self.select(pool, self.beam_width)
            print(f"Seed completed: best score={self.best.score:.6f}, connected={len(self.best.tree.connected_terminal_ids)}, evaluations={self.evaluations}", flush=True)
            if self.expired():
                break
        # Relocate a consumer, split an uneconomic group, merge into a different
        # component, switch tie-ins, and replace its route through the same API.
        frontier = self.select(list(self.archive.values()), self.beam_width)
        for _ in range(8):
            before = self.best.score
            pool = list(frontier)
            for state in frontier:
                for terminal in self.terminals:
                    if self.expired():
                        return self.finalists()
                    reduced = h.detach_terminal(state.tree, terminal) if terminal.id in state.tree.connected_terminal_ids else state.tree
                    for proposal in self.proposals(reduced, terminal):
                        value = self.evaluate(proposal)
                        if value is not None:
                            pool.append(value)
            frontier = self.select(pool, self.beam_width)
            if self.best.score >= before - 1e-9:
                break
        return self.finalists()

    def finalists(self):
        return sorted(self.archive.values(), key=h.search_key)[:24]

    def exchange_subtrees(self, variant):
        """Move a whole downstream tree, retaining the best exact evaluation."""
        key = signature(variant.tree)
        if key in self.subtree_checked or variant.tree.segment_paths or variant.tree.node_points:
            return variant
        self.subtree_checked.add(key)
        deadline = self.graph.deadline
        self.graph.deadline = min(deadline, time.monotonic() + 20.)
        try:
            tree = h.improve_subtrees(variant.tree, self.terminals, self.candidates, self.graph, 0., self.turn_m)
            return self.evaluate(tree) or variant
        finally:
            self.graph.deadline = deadline
            self.graph.set_barriers()

    def report(self):
        return {"search_seconds": round(time.monotonic() - self.started, 3), "evaluations": self.evaluations,
                "valid_structural_families": len(self.archive), "route_searches": self.graph.searches,
                "route_cache_hits": self.graph.cache_hits, "anytime_history": self.history,
                "rejection_counts": dict(self.rejections.most_common(15)), "seeds": self.seed_reports,
                "standalone_paths": [{"input_id": t.input_id if t.input_id is not None else t.id,
                    "count": len(self.standalone.get(t.id, [])), "completed": t.id in self.standalone}
                    for t in self.terminals]}
