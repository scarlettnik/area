"""Behavioral regression checks; run with python -m unittest -v."""

import math
import unittest

import heat_route_builder as h


def rectangle(x0, y0, x1, y1, clearance=0.0, kind="water", name="wall"):
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return h.Obstacle(name, kind, [ring], clearance, h.expand_bbox(h.bbox(ring), clearance))


def root(name, point, chamber=True, diameter=300):
    return h.TieCandidate(name, point, name, "heat_chamber" if chamber else "heat_network",
                          diameter, chamber)


def setup(terminals, candidates, obstacles=(), step=5.0):
    points = [t.point for t in terminals] + [r.point for r in candidates]
    grid = h.RoutingGrid(list(obstacles), points, step=step, margin=25, cardinal_only=True)
    for t in terminals:
        t.cell = grid.add_access(t.point, terminal=True)
    for r in candidates:
        r.cell = grid.add_access(r.point)
    return grid


def route(terminals, candidates, grid):
    tree = h.build_forest(terminals, candidates, grid)
    return h.materialize_variant("test", "test", tree, terminals, grid, {})


class GeometryTests(unittest.TestCase):
    def test_service_lead_exposes_first_corridor_contact(self):
        grid = h.RoutingGrid([], [(0, 0), (30, 20)], margin=0, cardinal_only=True, step=5)
        endpoint = grid.add_access((2, 2), terminal=True)
        for cell, _length, _direction in grid.extra_neighbors[endpoint]:
            path = grid.edge_points(endpoint, cell)
            for a, b in zip(path, path[1:]):
                for x in range(grid.nx):
                    for y in range(grid.ny):
                        point = grid.cell_to_point((x, y))
                        if h.point_segment_distance(point, a, b) < 1e-8:
                            self.assertEqual((x, y), cell)

    def test_orthogonal_polishing_removes_stair_steps_without_extra_length(self):
        grid = h.RoutingGrid([], [(0, 0), (30, 20)], margin=0, cardinal_only=True)
        points = [(0, 0), (10, 0), (10, 10), (20, 10), (20, 20), (30, 20)]
        tidy = h.tidy_orthogonal_path(points, grid, [])
        self.assertEqual(tidy[0], points[0])
        self.assertEqual(tidy[-1], points[-1])
        self.assertEqual(h.polyline_bend_count(tidy), 1)
        self.assertLessEqual(h.path_length(tidy), h.path_length(points))

    def test_orthogonal_polishing_cannot_cut_through_buildings_or_other_pipes(self):
        wall = rectangle(15, -2, 18, 15)
        grid = h.RoutingGrid([wall], [(0, 0), (30, 20)], margin=5, cardinal_only=True)
        points = [(0, 0), (10, 0), (10, 20), (30, 20)]
        other = [(0, 10), (8, 10)]
        tidy = h.tidy_orthogonal_path(points, grid, [other])
        self.assertTrue(all(grid.line_clear(a, b) for a, b in zip(tidy, tidy[1:])))
        self.assertFalse(h.polylines_conflict(tidy, other))
        self.assertEqual(tidy, points)

    def test_pipe_contacts_require_shared_end_nodes(self):
        self.assertTrue(h.polylines_conflict([(0, 0), (20, 0)], [(10, -10), (10, 10)]))
        self.assertTrue(h.polylines_conflict([(0, 0), (20, 0)], [(0, 0), (10, 0)]))
        self.assertFalse(h.polylines_conflict([(0, 0), (20, 0)], [(20, 0), (20, 10)]))

    def test_city_axis_follows_dominant_facades_not_mean_of_unrelated_blocks(self):
        def rotated_building(angle, width, height, center):
            transform = h.CoordinateTransform(center, math.radians(angle))
            ring = [transform.inverse(p) for p in [(0, 0), (width, 0), (width, height), (0, height), (0, 0)]]
            return h.Obstacle(str(center), "oks", [ring], 0, h.bbox(ring))

        buildings = [rotated_building(-22, 120, 80, (i * 200, 0)) for i in range(8)]
        buildings += [rotated_building(0, 150, 100, (i * 200, 400)) for i in range(3)]
        self.assertAlmostEqual(math.degrees(h.estimate_city_axis(buildings)), -22, places=6)

    def test_city_axis_clusters_equivalent_angles_across_wrap_boundary(self):
        buildings = []
        for angle in (-44.5, 44.5):
            transform = h.CoordinateTransform((0, 0), math.radians(angle))
            ring = [transform.inverse(p) for p in [(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)]]
            buildings.append(h.Obstacle(str(angle), "oks", [ring], 0, h.bbox(ring)))
        self.assertAlmostEqual(abs(math.degrees(h.estimate_city_axis(buildings))), 45, places=6)

    def test_thin_obstacle_between_free_grid_nodes_cannot_be_crossed(self):
        wall = rectangle(4.9, -30, 5.1, 30)
        grid = h.RoutingGrid([wall], [(0, 0), (10, 0)], step=10, margin=0, cardinal_only=True)
        a, b = grid.point_to_cell((0, 0)), grid.point_to_cell((10, 0))
        self.assertFalse(grid.is_blocked_cell(a))
        self.assertFalse(grid.is_blocked_cell(b))
        self.assertFalse(grid.line_clear((0, 0), (10, 0)))
        self.assertNotIn(b, [n for n, _ in grid.neighbors(a)])

    def test_clearance_is_checked_along_entire_segment(self):
        grid = h.RoutingGrid([rectangle(4, 1, 6, 3, 1.1)], [(0, 0), (10, 0)], margin=0)
        self.assertFalse(grid.line_clear((0, 0), (10, 0)))
        self.assertTrue(grid.line_clear((0, -1), (10, -1)))

    def test_polygon_hole_remains_free(self):
        obstacle = rectangle(0, 0, 40, 40, clearance=1, kind="oks")
        obstacle.holes = [[(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)]]
        self.assertFalse(obstacle.contains_or_near((20, 20)))
        self.assertTrue(obstacle.contains_or_near((10.5, 20)))
        self.assertFalse(obstacle.blocks_segment((15, 20), (25, 20)))
        self.assertTrue(obstacle.blocks_segment((5, 20), (35, 20)))

    def test_blocked_tie_in_is_rejected_instead_of_teleported(self):
        obstacle = rectangle(-5, -5, 5, 5, kind="oks")
        grid = h.RoutingGrid([obstacle], [(0, 0), (20, 0)], margin=10)
        self.assertIsNone(grid.add_access((0, 0)))

    def test_actual_terminal_is_connected_by_short_own_building_lead(self):
        terminal = h.Terminal("building", (0, 0), 1)
        candidates = [root("root", (30, 0))]
        owner = rectangle(-3, -3, 3, 3, 1, "oks")
        grid = setup([terminal], candidates, [owner])
        result = route([terminal], candidates, grid)
        self.assertEqual(result.summary["connected_oks_count"], 1)
        segments, _, _ = h.compress_segments(result.tree, [terminal], grid)
        self.assertEqual(segments[-1].points[-1], terminal.point)
        self.assertAlmostEqual(result.summary["length"], 30, places=3)

    def test_terminal_lead_cannot_cross_another_obstacle(self):
        grid = h.RoutingGrid([rectangle(-3, -3, 3, 3, 1, "oks"),
                              rectangle(-20, -20, 20, 20, kind="water")],
                             [(0, 0), (30, 0)], margin=10, cardinal_only=True)
        self.assertIsNone(grid.add_access((0, 0), terminal=True))

    def test_mesh_refinement_keeps_exterior_exit_from_closed_courtyard(self):
        building = rectangle(-24, -24, 24, 24, clearance=5, kind="oks")
        building.holes = [[(-8, -8), (8, -8), (8, 8), (-8, 8), (-8, -8)]]
        for step in (5.0, 2.5):
            with self.subTest(step=step):
                terminals = [h.Terminal("t", (12, 0), 1)]
                candidates = [root("r", (50, 0))]
                grid = setup(terminals, candidates, [building], step=step)
                result = route(terminals, candidates, grid)
                self.assertEqual(result.summary["connected_oks_count"], 1)
                self.assertAlmostEqual(result.summary["length"], 38, places=3)


class CostRoutingTests(unittest.TestCase):
    def test_off_grid_service_branches_do_not_overlap_the_trunk(self):
        terminals = [h.Terminal("a", (12.3, 15), 1), h.Terminal("b", (2.2, 15), 1)]
        candidates = [root("r", (40, 0))]
        grid = setup(terminals, candidates)
        result = route(terminals, candidates, grid)
        segments, _, _ = h.compress_segments(result.tree, terminals, grid)
        for i, first in enumerate(segments):
            for second in segments[i + 1:]:
                self.assertFalse(h.polylines_conflict(first.points, second.points))
        self.assertEqual(result.summary["connected_oks_count"], 2)

    def test_single_consumer_has_analytically_minimum_rectilinear_price(self):
        terminals = [h.Terminal("t", (20, 20), 1)]
        candidates = [root("near", (0, 0)), root("far", (-100, 0))]
        result = route(terminals, candidates, setup(terminals, candidates))
        self.assertEqual([r.id for r in result.tree.roots], ["near"])
        self.assertAlmostEqual(result.summary["length"], 40, places=3)
        self.assertAlmostEqual(result.summary["calculated_cost"], 40 * h.NEW_COST[50] + h.TIE_IN_COST)

    def test_tie_in_choice_accounts_for_obstacle_detour(self):
        terminals = [h.Terminal("t", (15, 0), 1)]
        candidates = [root("near_but_blocked", (0, 0)), root("reachable", (50, 0))]
        grid = setup(terminals, candidates, [rectangle(4, -50, 6, 50)])
        result = route(terminals, candidates, grid)
        self.assertEqual([r.id for r in result.tree.roots], ["reachable"])
        self.assertAlmostEqual(result.summary["length"], 35, places=3)

    def test_two_nearby_tie_ins_beat_one_long_trunk(self):
        terminals = [h.Terminal("left", (0, 20), 1), h.Terminal("right", (200, 20), 1)]
        candidates = [root("a", (0, 0)), root("b", (200, 0))]
        result = route(terminals, candidates, setup(terminals, candidates))
        self.assertEqual(result.summary["tie_in_count"], 2)
        self.assertAlmostEqual(result.summary["length"], 40, places=3)

    def test_shared_trunk_sums_flow_and_is_cheaper_than_two_independent_routes(self):
        terminals = [h.Terminal("a", (90, -10), 1), h.Terminal("b", (90, 10), 1)]
        candidates = [root("r", (0, 0))]
        grid = setup(terminals, candidates)
        result = route(terminals, candidates, grid)
        self.assertEqual(result.summary["connected_oks_count"], 2)
        self.assertLess(result.summary["length"], 180)
        _parent, _depth, flows, adj = h.orient_and_flow(result.tree, terminals)
        self.assertIn(2.0, flows.values())
        self.assertLessEqual(max(map(len, adj.values())), 4)
        self.assertLess(result.summary["calculated_cost"], 200 * h.NEW_COST[50] + h.TIE_IN_COST)

    def test_chamber_price_includes_existing_pipe_diameter(self):
        terminals = [h.Terminal("t", (20, 0), 1)]
        candidates = [root("r", (0, 0), chamber=False, diameter=900)]
        result = route(terminals, candidates, setup(terminals, candidates))
        self.assertEqual(result.summary["chamber_construction_cost"], h.chamber_cost(900))

    def test_node_ids_and_geometry_match_exported_features(self):
        terminals = [h.Terminal("a", (90, -10), 1), h.Terminal("b", (90, 10), 1)]
        candidates = [root("r", (0, 0))]
        grid = setup(terminals, candidates)
        result = route(terminals, candidates, grid)
        nodes = {t.id: h.unproject_point(t.point) for t in terminals}
        nodes.update({f["properties"]["id"]: f["geometry"]["coordinates"] for f in result.features
                      if f["geometry"] and f["geometry"]["type"] == "Point"})
        for feature in result.features:
            if feature["properties"]["object_type"] != "heat_network":
                continue
            props, geometry = feature["properties"], feature["geometry"]["coordinates"]
            self.assertEqual(geometry[0], nodes[props["start_node_id"]])
            self.assertEqual(geometry[-1], nodes[props["end_node_id"]])

    def test_unreachable_consumer_is_reported(self):
        terminals = [h.Terminal("blocked", (0, 0), 1)]
        candidates = [root("r", (40, 0))]
        grid = setup(terminals, candidates, [rectangle(-20, -20, 20, 20)])
        result = route(terminals, candidates, grid)
        self.assertEqual(result.summary["unconnected_oks_ids"], ["blocked"])
        self.assertEqual(result.summary["tie_in_count"], 0)
        self.assertEqual(result.summary["length"], 0)

    def test_consumer_input_order_does_not_change_result(self):
        terminals = [h.Terminal("a", (90, -10), 1), h.Terminal("b", (90, 10), 1)]
        candidates = [root("r", (0, 0))]
        grid = setup(terminals, candidates)
        first = route(terminals, candidates, grid)
        second = route(list(reversed(terminals)), candidates, grid)
        self.assertEqual(first.tree.edges, second.tree.edges)
        self.assertEqual(first.summary, second.summary)


class HydraulicTests(unittest.TestCase):
    def test_invalid_or_unsupported_flow_is_not_silently_capped(self):
        for value in (-1, math.inf, math.nan, h.CAPACITY[1400] + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                h.select_diameter(value)

    def test_chamber_does_not_reset_same_diameter_length_limit(self):
        grid = h.RoutingGrid([], [(0, 0), (250, 0)], step=5, margin=0, cardinal_only=True)
        r = root("r", (0, 0))
        r.cell = (0, 0)
        terminals = [h.Terminal("a", (250, 0), 1, (50, 0)),
                     h.Terminal("b", (100, 5), 1, (20, 1))]
        edges = {h.edge_key((i, 0), (i + 1, 0)) for i in range(50)}
        edges.add(h.edge_key((20, 0), (20, 1)))
        tree = h.BuiltTree([r], edges, {"a", "b"}, set())
        segments, _, _ = h.compress_segments(tree, terminals, grid)
        downstream = next(s for s in segments if s.end_node_id == "a")
        self.assertGreater(downstream.diameter, 50)

    def test_cyclic_network_is_rejected(self):
        r = root("r", (0, 0))
        r.cell = (0, 0)
        edges = {h.edge_key((0, 0), (1, 0)), h.edge_key((1, 0), (1, 1)),
                 h.edge_key((1, 1), (0, 0))}
        with self.assertRaises(ValueError):
            h.orient_and_flow(h.BuiltTree([r], edges, set(), set()), [])


if __name__ == "__main__":
    unittest.main()
