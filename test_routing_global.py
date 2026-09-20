"""Behavioral regressions for global seeds and directed junction attachment."""

import time
import unittest

import heat_route_builder as h
from routing_global import rectilinear_seed
from routing_visibility import VisibilityGraph
from test_heat_route_builder import root


class GlobalRoutingTests(unittest.TestCase):
    def test_existing_chamber_replaces_nearby_new_tie_and_shares_trunk(self):
        terminals = [h.Terminal("a", (80, 10), 5), h.Terminal("b", (80, -10), 5)]
        old = root("old", (0, 0))
        new = root("new", (5, 0), chamber=False)
        new.nearby_chambers = ((old.existing_object_id, old.existing_degree),)
        graph = VisibilityGraph(terminals, [old, new], [])
        before = sum(map(len, graph.graph.values()))
        tree = rectilinear_seed(terminals, [old, new], graph, dn=150)
        value = h.materialize_variant("global", "", tree, terminals, graph, {})
        self.assertEqual(value.summary["connected_oks_count"], 2)
        self.assertEqual(tree.roots, [old])
        self.assertLess(value.summary["new_network_length"], 150)
        # A dense temporary grid must not leak into the later visibility search.
        self.assertLess(sum(map(len, graph.graph.values())) - before, 100)

    def test_expired_seed_does_not_mutate_search_graph(self):
        terminal = h.Terminal("t", (20, 20), 1)
        source = root("r", (0, 0))
        graph = VisibilityGraph([terminal], [source], [])
        before = {n: dict(edges) for n, edges in graph.graph.items()}
        self.assertIsNone(
            rectilinear_seed([terminal], [source], graph, deadline=time.monotonic() - 1)
        )
        self.assertEqual(graph.graph, before)

    def test_attachment_preserves_both_endpoint_directions_and_cache(self):
        terminal = h.Terminal("t", (-10, 10), 1)
        source = root("r", (0, 0))
        graph = VisibilityGraph([terminal], [source], [])
        prior = graph.node((-10, 0))
        corner = graph.node((0, 10))
        successor = graph.node((-20, 10))
        graph.graph.clear()
        for a, b in (
            (source.cell, terminal.cell),
            (source.cell, corner),
            (corner, terminal.cell),
        ):
            graph.connect(a, b)
        plain = graph.least_cost_path(
            {source.cell: 0}, terminal.cell, set(), h.NEW_COST[50]
        )
        self.assertEqual(plain, [source.cell, terminal.cell])
        directed = graph.least_cost_path(
            {source.cell: 0},
            terminal.cell,
            set(),
            h.NEW_COST[50],
            source_priors={source.cell: prior},
            target_nexts=(successor,),
        )
        self.assertEqual(directed, [source.cell, corner, terminal.cell])
        self.assertEqual(
            plain,
            graph.least_cost_path(
                {source.cell: 0}, terminal.cell, set(), h.NEW_COST[50]
            ),
        )


if __name__ == "__main__":
    unittest.main()
