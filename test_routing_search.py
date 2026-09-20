import unittest

import heat_route_builder as h
from routing_search import Search, structure
from routing_visibility import VisibilityGraph
from routing_depth import clearance
from test_heat_route_builder import rectangle, root


class VisibilityTests(unittest.TestCase):
    def test_route_prices_include_the_official_length_term(self):
        terminal = h.Terminal('t', (100, 0), 1)
        roots = [root('far', (0, 0)), root('near', (60, 0))]
        graph = VisibilityGraph([terminal], roots, [])
        path = graph.least_cost_path({roots[0].cell: 0., roots[1].cell: 6_000_000.},
                                    terminal.cell, set(), h.NEW_COST[50])
        # The longer route is cheaper in rubles alone, but has a worse score.
        self.assertEqual(path[0], roots[1].cell)

    def test_network_barriers_change_routes_and_cache(self):
        terminals = [h.Terminal('a', (30, 0), 1)]
        candidates = [root('r', (0, 0))]
        graph = VisibilityGraph(terminals, candidates, [])
        sources = {candidates[0].cell: 0.}
        target = terminals[0].cell
        first = graph.least_cost_path(sources, target, set(), h.NEW_COST[50])
        barrier = h.edge_key(graph.node((15, -5)), graph.node((15, 5)))
        graph.set_barriers({barrier})
        self.assertTrue(any(graph.conflicts_with_network(a, b) for a, b in zip(first, first[1:])))
        second = graph.least_cost_path(sources, target, set(), h.NEW_COST[50])
        self.assertTrue(second)
        self.assertTrue(all(not graph.conflicts_with_network(a, b) for a, b in zip(second, second[1:])))
        graph.set_barriers()
        self.assertEqual(first, graph.least_cost_path(sources, target, set(), h.NEW_COST[50]))

    def test_straight_building_exit_and_exact_coordinates(self):
        building = rectangle(-10, -10, 10, 10, kind="oks")
        building.clearance = clearance("oks", 125)
        building.bbox = h.expand_bbox(h.bbox(building.rings[0]), building.clearance)
        terminal = h.Terminal("house", (.3, .7), 25)
        tie = root("tie", (60, 20))
        graph = VisibilityGraph([terminal], [tie], [building])
        paths = graph.alternatives({tie.cell: 0.}, terminal.cell, 125)
        self.assertTrue(paths)
        for path in paths:
            self.assertEqual(graph.cell_to_point(path[-1]), terminal.point)
            from routing_constraints import validate_path
            from shapely import Point
            self.assertFalse(building.geometry.contains(Point(graph.cell_to_point(path[-2]))))
            validate_path([graph.cell_to_point(n) for n in path], 125,
                          [building], [terminal.point], True)

    def test_beam_prunes_with_full_cost_and_retains_valid_best(self):
        terminals = [h.Terminal("a", (65, 15), 1), h.Terminal("b", (75, -10), 2)]
        candidates = [root("r1", (0, 0)), root("r2", (110, 0))]
        graph = VisibilityGraph(terminals, candidates, [])
        search = Search(terminals, candidates, graph, seconds=5, beam_width=2, routes=2)
        result = search.run()
        self.assertEqual(result[0].summary["connected_oks_count"], 2)
        self.assertLess(result[0].summary["calculated_cost"], 100_000_000)
        scores = [x["score"] for x in search.history]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len({structure(v.tree, terminals) for v in result}), len(result))

    def test_route_cache_key_includes_forbidden_edges(self):
        terminals = [h.Terminal("a", (30, 10), 1)]
        candidates = [root("r", (0, 0))]
        graph = VisibilityGraph(terminals, candidates, [])
        first = graph.least_cost_path({candidates[0].cell: 0}, terminals[0].cell, set(), h.NEW_COST[50])
        forbidden = {h.edge_key(*first[:2])}
        second = graph.least_cost_path({candidates[0].cell: 0}, terminals[0].cell, set(), h.NEW_COST[50], banned=forbidden)
        self.assertNotEqual(first, second)
        self.assertTrue(all(h.edge_key(a,b) not in forbidden for a,b in zip(second,second[1:])))


if __name__ == "__main__":
    unittest.main()
