"""Budgeted topology search. Proposals never substitute for exact evaluation."""
from collections import Counter
import math
import time

import heat_route_builder as h


def signature(tree):
    return tuple(sorted(r.id for r in tree.roots)), frozenset(tree.edges), frozenset(tree.connected_terminal_ids)


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
    def __init__(self, terminals, candidates, graph, seconds=120., beam_width=5, routes=3, turn_m=8.):
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
        if family not in self.archive or h.variant_key(variant) < h.variant_key(self.archive[family]):
            self.archive[family] = variant
        if self.best is None or h.variant_key(variant) < h.variant_key(self.best):
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
        existing = getattr(self.graph, "existing_network", None)
        for r in self.candidates:
            if r.cell is None:
                continue
            price = h.opening_cost(r, diameter)
            if existing and existing.complete:
                _, cost, length, _ = existing.evaluate([(r.existing_object_id, self.graph.to_utm(r.point), terminal.flow_tph)])
                price += cost + length * (25_000_000 / .7) * .003
            sources[r.cell] = price
        paths = []
        for _ in range(self.routes):
            if self.expired():
                break
            route = self.graph.least_cost_path(sources, terminal.cell, set(), h.NEW_COST[diameter], self.turn_m * h.NEW_COST[diameter])
            if not route:
                break
            paths.append(route)
            # Alternate tie-ins are topologically meaningful alternatives.
            sources.pop(route[0], None)
        if paths and len(paths) < self.routes:
            alternatives = self.graph.alternatives({paths[0][0]: 0.}, terminal.cell, diameter, self.routes, self.turn_m)
            paths.extend(p for p in alternatives if p not in paths)
        self.standalone[terminal.id] = paths[:self.routes]
        return self.standalone[terminal.id]

    def proposals(self, tree, terminal):
        yielded = set()
        paths = self.independent_paths(terminal) or [[]]
        for standalone in paths:
            if self.expired():
                return
            for candidate in h.connection_options(tree, terminal, self.candidates, self.terminals,
                                                  self.graph, standalone, 0., self.turn_m):
                key = signature(candidate)
                if key not in yielded:
                    yielded.add(key)
                    yield candidate

    def select(self, variants, width):
        unique = {}
        for v in variants:
            if v is not None:
                key = signature(v.tree)
                if key not in unique or h.variant_key(v) < h.variant_key(unique[key]):
                    unique[key] = v
        ranked = sorted(unique.values(), key=h.variant_key)
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
        empty = h.BuiltTree([], set(), set(), {t.id for t in self.terminals})
        baseline = self.evaluate(empty)
        orders = [sorted(self.terminals, key=lambda t: (-t.flow_tph, t.id)),
                  sorted(self.terminals, key=lambda t: (t.point[0], t.point[1], t.id))]
        for order in orders:
            beam = [baseline]
            for terminal in order:
                if self.expired():
                    break
                pool = list(beam)  # Leaving an OKS disconnected is priced with the official penalty.
                for state in beam:
                    for proposal in self.proposals(state.tree, terminal):
                        if self.expired():
                            break
                        value = self.evaluate(proposal)
                        if value is not None:
                            pool.append(value)
                beam = self.select(pool, self.beam_width)
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
        return sorted(self.archive.values(), key=h.variant_key)[:12]

    def report(self):
        return {"search_seconds": round(time.monotonic() - self.started, 3), "evaluations": self.evaluations,
                "valid_structural_families": len(self.archive), "route_searches": self.graph.searches,
                "route_cache_hits": self.graph.cache_hits, "anytime_history": self.history,
                "rejection_counts": dict(self.rejections.most_common(15))}
