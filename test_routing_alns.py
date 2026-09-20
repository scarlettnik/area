import unittest

import heat_route_builder as h
from routing_alns import AdaptiveSearch
from routing_visibility import VisibilityGraph
from test_heat_route_builder import root


class AdaptiveTests(unittest.TestCase):
    def test_destroy_repair_retains_best_and_produces_diverse_valid_networks(self):
        terminals = [h.Terminal("a", (65, 15), 1), h.Terminal("b", (75, -10), 2),
                     h.Terminal("c", (90, 30), 3)]
        roots = [root("r1", (0, 0)), root("r2", (110, 0))]
        graph = VisibilityGraph(terminals, roots, [])
        search = AdaptiveSearch(terminals, roots, graph, seconds=4, beam_width=2, routes=2, iterations=6)
        result = search.run()
        self.assertEqual(result[0].summary["connected_oks_count"], 3)
        self.assertGreater(search.completed_iterations, 0)
        self.assertEqual(sum(search.operator_counts.values()), search.completed_iterations)
        self.assertTrue(all(a["score"] >= b["score"] for a,b in zip(search.history,search.history[1:])))
        for variant in result:
            h.orient_and_flow(variant.tree, terminals)


if __name__ == "__main__":
    unittest.main()
