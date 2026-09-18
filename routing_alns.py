"""Adaptive destroy/repair over complete network evaluations, with a fixed RNG seed."""
import math
import random
import time

import heat_route_builder as h
from routing_search import Search


class AdaptiveSearch(Search):
    def __init__(self, *args, seed=17, iterations=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = random.Random(seed)
        self.seed, self.iterations = seed, iterations
        self.weights = {"expensive": 1., "nearby": 1., "component": 1., "reconstruction": 1.}
        self.operator_counts = {name: 0 for name in self.weights}
        self.improvements = 0
        self.completed_iterations = 0

    def destroy_ids(self, current, operator):
        connected = [t for t in self.terminals if t.id in current.tree.connected_terminal_ids]
        if not connected:
            return []
        parent, _, _, _ = h.orient_and_flow(current.tree, self.terminals)
        if operator in {"component", "reconstruction"}:
            roots = {}
            for t in connected:
                node = t.cell
                while parent[node] is not None:
                    node = parent[node]
                roots.setdefault(node, []).append(t)
            if operator == "reconstruction" and self.graph.existing_network.complete:
                impacts = []
                for root in current.tree.roots:
                    group = roots.get(root.cell, [])
                    _, cost, _, _ = self.graph.existing_network.evaluate(
                        [(root.existing_object_id, self.graph.to_utm(root.point), sum(t.flow_tph for t in group))])
                    impacts.append((cost, root.cell))
                selected = max(impacts)[1]
            else:
                selected = self.random.choice(sorted(roots))
            group = roots[selected]
            # Large components are rebuilt a local group at a time.
            pivot = self.random.choice(group)
            return sorted(group, key=lambda t: h.dist(t.point, pivot.point))[:4]
        if operator == "nearby":
            pivot = self.random.choice(connected)
            return sorted(connected, key=lambda t: h.dist(t.point, pivot.point))[:3]
        contributions = []
        for t in connected:
            if self.expired():
                break
            reduced = self.evaluate(h.detach_terminal(current.tree, t))
            if reduced is not None:
                marginal = current.score - reduced.score + .7 * h.penalty_unconnected(t.flow_tph) / 25_000_000
                contributions.append((marginal, t.id, t))
        return [item[2] for item in sorted(contributions, reverse=True)[:2]] or [self.random.choice(connected)]

    def run(self):
        budget = self.seconds
        self.seconds = max(1., budget * .45)
        super().run()
        self.seconds = budget
        self.deadline = self.started + budget
        current = self.best
        for iteration in range(self.iterations):
            if self.expired():
                break
            operator = self.random.choices(list(self.weights), weights=list(self.weights.values()))[0]
            self.operator_counts[operator] += 1
            removed = self.destroy_ids(current, operator)
            tree = current.tree
            for terminal in removed:
                tree = h.detach_terminal(tree, terminal)
            pending = list(removed) + [t for t in self.terminals if t.id not in current.tree.connected_terminal_ids]
            self.random.shuffle(pending)
            old_best = self.best.score
            while pending and not self.expired():
                # Regret insertion: prioritize an OKS whose second-best exact
                # insertion is much worse than its best feasible insertion.
                choices = []
                for terminal in pending:
                    values = [self.evaluate(proposal) for proposal in self.proposals(tree, terminal)]
                    values = sorted((v for v in values if v is not None), key=h.variant_key)
                    if values:
                        regret = values[1].score - values[0].score if len(values) > 1 else 100.
                        choices.append((regret, terminal.id, values[0]))
                    if self.expired():
                        break
                if not choices:
                    break
                _, tid, chosen = max(choices, key=lambda item: (item[0], item[1]))
                tree = chosen.tree
                pending = [t for t in pending if t.id != tid]
            candidate = self.evaluate(tree)
            reward = .2
            if candidate is not None:
                elapsed = (time.monotonic() - self.started) / budget
                temperature = max(.001, .03 * self.best.score * (1 - min(1., elapsed)))
                delta = candidate.score - current.score
                if delta < 0 or self.random.random() < math.exp(-max(0., delta) / temperature):
                    current = candidate
                    reward = 2. if delta < 0 else .5
            if self.best.score < old_best - 1e-9:
                reward = 8.
                self.improvements += 1
            self.weights[operator] = .8 * self.weights[operator] + .2 * reward
            self.completed_iterations += 1
            if iteration % 10 == 9:
                current = self.best
        return self.finalists()

    def report(self):
        return dict(super().report(), seed=self.seed, iterations=self.completed_iterations,
                    operator_weights=self.weights, operator_counts=self.operator_counts,
                    best_improvements=self.improvements)
