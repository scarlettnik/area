"""Adaptive destroy/repair over complete network evaluations, with a fixed RNG seed."""
import math
import random
import time

import heat_route_builder as h
from routing_search import Search


class AdaptiveSearch(Search):
    def __init__(self, *args, seed=17, iterations=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = random.Random(seed)
        self.seed, self.iterations = seed, iterations
        self.weights = {"expensive": 1., "nearby": 1., "component": 1., "retie": 1.,
                        "hotspot": 1., "blocked": 1., "random": 1.}
        self.operator_counts = {name: 0 for name in self.weights}
        self.improvements = 0
        self.completed_iterations = 0
        self.stagnation = 0
        self.repair_counts = {"regret": 0, "randomized": 0}

    def destroy_ids(self, current, operator):
        connected = [t for t in self.terminals if t.id in current.tree.connected_terminal_ids]
        if not connected:
            return []
        parent, _, _, _ = h.orient_and_flow(current.tree, self.terminals)
        if operator == "random":
            limit = max(2, math.ceil(len(connected) * (.6 if self.stagnation > 30 else .3)))
            return self.random.sample(connected, min(len(connected), self.random.randint(2, limit)))
        if operator == "blocked":
            missing = [t for t in self.terminals if t.id not in current.tree.connected_terminal_ids
                       and self.graph.graph[t.cell]]
            pivot = self.random.choice(missing or connected)
            return sorted(connected, key=lambda t: h.dist(t.point, pivot.point))[:4]
        if operator in {"component", "retie"}:
            roots = {}
            for t in connected:
                node = t.cell
                while parent[node] is not None:
                    node = parent[node]
                roots.setdefault(node, []).append(t)
            selected = self.random.choice(sorted(roots))
            group = roots[selected]
            if operator == "retie":
                # Remove an entire component.  Repair is then free to choose a
                # different existing-network tie-in and a different trunk family.
                return list(group)
            pivot = self.random.choice(group)
            count = len(group) if self.stagnation > 20 else min(4, len(group))
            return sorted(group, key=lambda t: h.dist(t.point, pivot.point))[:count]
        if operator == "hotspot":
            # Destroy the consumers sharing a busy branch point.  This is more
            # structural than removing geographically-near terminals and helps
            # ALNS escape expensive chamber/trunk local minima.
            _, depth, _, adjacency = h.orient_and_flow(current.tree, self.terminals)
            descendants = {node: [] for node in parent}
            terminal_by_cell = {t.cell: t for t in connected}
            for node in sorted(parent, key=lambda n: depth[n], reverse=True):
                if node in terminal_by_cell:
                    descendants[node].append(terminal_by_cell[node])
                if parent[node] is not None:
                    descendants[parent[node]].extend(descendants[node])
            hotspots = [(len(adjacency[node]), len(descendants[node]), depth[node], node)
                        for node in parent if len(adjacency[node]) >= 3 and len(descendants[node]) >= 2]
            if hotspots:
                _, _, _, node = max(hotspots)
                group = descendants[node]
                limit = min(len(group), 6 if self.stagnation > 20 else 4)
                return sorted(group, key=lambda t: (-t.flow_tph, t.id))[:limit]
            pivot = self.random.choice(connected)
            return sorted(connected, key=lambda t: h.dist(t.point, pivot.point))[:3]
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
        self.seconds = max(1., budget * .35)
        super().run()
        self.seconds = budget
        self.deadline = self.started + budget
        self.graph.deadline = self.deadline
        current = self.best
        if not self.expired():
            current = self.exchange_subtrees(current)
        for iteration in range(self.iterations):
            if self.expired():
                break
            operator = self.random.choices(list(self.weights), weights=list(self.weights.values()))[0]
            self.operator_counts[operator] += 1
            removed = self.destroy_ids(current, operator)
            tree = current.tree
            for terminal in removed:
                tree = h.detach_terminal(tree, terminal)
            pending = list(removed) + [t for t in self.terminals if t.id not in current.tree.connected_terminal_ids
                                       and self.graph.graph[t.cell]]
            self.random.shuffle(pending)
            old_best = h.search_key(self.best)
            repair = "randomized" if self.random.random() < .35 else "regret"
            self.repair_counts[repair] += 1
            while pending and not self.expired():
                # Regret insertion: prioritize an OKS whose second-best exact
                # insertion is much worse than its best feasible insertion.
                choices = []
                for terminal in pending:
                    values = [self.evaluate(proposal) for proposal in self.proposals(tree, terminal)]
                    values = sorted((v for v in values if v is not None), key=h.search_key)
                    if values:
                        regret = values[1].score - values[0].score if len(values) > 1 else 100.
                        chosen = values[0]
                        if repair == "randomized":
                            chosen = self.random.choices(values[:3], weights=[1., .35, .15][:len(values[:3])])[0]
                            regret *= self.random.uniform(.5, 1.5)
                        choices.append((regret, terminal.id, chosen))
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
                temperature = max(.001, .03 * max(self.best.score, 1e-6) * (1 - min(1., elapsed)))
                candidate_coverage = len(candidate.tree.connected_terminal_ids)
                current_coverage = len(current.tree.connected_terminal_ids)
                # Corrected LCT rules are lexicographic: maximize feasible
                # coverage first, then minimize official score.  Simulated
                # annealing may cross a score barrier, never a coverage barrier.
                if candidate_coverage >= current_coverage:
                    delta = candidate.score - current.score
                    better = h.search_key(candidate) < h.search_key(current)
                    same_coverage = candidate_coverage == current_coverage
                    anneal = same_coverage and self.random.random() < math.exp(-max(0., delta) / temperature)
                    if better or anneal:
                        current = candidate
                        reward = 2. if better else .5
            if h.search_key(self.best) < old_best:
                reward = 8.
                self.improvements += 1
                self.stagnation = 0
            else:
                self.stagnation += 1
            self.weights[operator] = .8 * self.weights[operator] + .2 * reward
            self.completed_iterations += 1
            if iteration % 10 == 9:
                current = self.random.choice(self.finalists()[:5]) if iteration % 30 == 29 else self.best
            if iteration % 25 == 24 and not self.expired():
                current = self.exchange_subtrees(current)
            if iteration % 25 == 24:
                print(f"ALNS {iteration + 1}: {len(self.best.tree.connected_terminal_ids)}/{len(self.terminals)}, score={self.best.score:.6f}", flush=True)
        return self.finalists()

    def report(self):
        return dict(super().report(), seed=self.seed, iterations=self.completed_iterations,
                    operator_weights=self.weights, operator_counts=self.operator_counts,
                    best_improvements=self.improvements, repair_counts=self.repair_counts)
