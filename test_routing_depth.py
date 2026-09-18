"""Analytical depth/clearance regressions from the technical appendix."""
import math
import unittest

import heat_route_builder as h
import routing_depth as d


def segment(points, start="r", end="t", diameter=50):
    return h.Segment(points, start, end, 1, diameter, h.path_length(points), start, end, 0)


def utility(kind="gas_pipeline", x=20, name="utility"):
    props = {"id": name, "object_type": "restriction", "restriction_type": kind, "diameter": 300}
    if kind == "heat_network":
        props["object_type"] = kind
    return d.read_crossing_objects([{"properties": props, "geometry": {
        "type": "LineString", "coordinates": [(x, -50), (x, 50)]}}], lambda g: g["coordinates"])[0]


def profiles(segments, objects, roots={"r"}, terminals={"t"}, tie_points=[]):
    return d.build_depth_profiles(segments, objects, roots, terminals, tie_points, h.NEW_COST)


class DepthTests(unittest.TestCase):
    def assert_profile(self, pieces):
        for i, piece in enumerate(pieces):
            self.assertGreaterEqual(min(piece.start_depth, piece.end_depth), .7 - 1e-8)
            self.assertLessEqual(abs(piece.start_depth - piece.end_depth) / piece.length, .1 + 1e-8)
            if i:
                self.assertAlmostEqual(piece.start_depth, pieces[i - 1].end_depth)
                self.assertEqual(piece.points[0], pieces[i - 1].points[-1])

    def test_no_crossing_has_normal_depth_and_unchanged_cost(self):
        groups, events = profiles([segment([(0, 0), (40, 0)])], [])
        self.assertEqual(events, [])
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(groups[0][0].start_depth, 3)
        self.assertEqual(groups[0][0].depth_coefficient, 1)

    def test_gas_above_has_four_meter_plateau_and_two_ramps(self):
        groups, events = profiles([segment([(0, 0), (40, 0)])], [utility()])
        pieces = groups[0]
        self.assert_profile(pieces)
        special = [p for p in pieces if p.crossing_ids]
        self.assertEqual(len(events), 1)
        self.assertEqual(len(special), 1)
        self.assertAlmostEqual(special[0].length, 4)
        self.assertAlmostEqual(special[0].start_depth, 2.475)
        self.assertEqual(special[0].end_depth, special[0].start_depth)
        self.assertEqual(special[0].special_factor, 1.25)
        self.assertEqual(pieces[0].start_depth, 3)
        self.assertEqual(pieces[-1].end_depth, 3)
        cost = sum(p.length * p.depth_coefficient * p.special_factor * h.NEW_COST[50] for p in pieces)
        self.assertAlmostEqual(cost, 41 * h.NEW_COST[50])

    def test_below_selected_when_above_cannot_fit_slope(self):
        # Gas requires 5.25 m to rise, but only 4.5 m are available.
        # Going below requires 4 m, so the alternative must be considered.
        groups, _ = profiles([segment([(0, 0), (13, 0)])], [utility(x=6.5)])
        pieces = groups[0]
        self.assert_profile(pieces)
        special = next(p for p in pieces if p.crossing_ids)
        self.assertAlmostEqual(special.start_depth, 3.4)
        self.assertAlmostEqual(special.length, 4)
        self.assertGreater(special.depth_coefficient, 1)

    def test_infeasible_crossing_is_rejected_not_exported_with_steep_ramp(self):
        with self.assertRaisesRegex(ValueError, "No feasible depth profile"):
            profiles([segment([(0, 0), (10, 0)])], [utility(x=5)])

    def test_junction_depth_is_shared_and_ramp_propagates_upstream(self):
        segments = [segment([(0, 0), (20, 0)], end="j"),
                    segment([(20, 0), (40, 0)], start="j"),
                    segment([(20, 0), (20, 20)], start="j", end="t2")]
        groups, _ = profiles(segments, [utility(x=23)], terminals={"t", "t2"})
        junction = groups[0][-1].end_depth
        self.assertLess(junction, 3)
        self.assertAlmostEqual(junction, groups[1][0].start_depth)
        self.assertAlmostEqual(junction, groups[2][0].start_depth)
        for group in groups:
            self.assert_profile(group)

    def test_existing_heat_tie_is_not_an_independent_crossing(self):
        groups, events = profiles([segment([(0, 0), (40, 0)])],
                                 [utility("heat_network", x=0)], tie_points=[(0, 0)])
        self.assertEqual(events, [])
        self.assertEqual(groups[0][0].start_depth, 3)

    def test_existing_heat_tie_tolerates_submillimetre_export_drift(self):
        groups, events = profiles([segment([(0.0005, 0), (40, 0)])],
                                  [utility("heat_network", x=0)],
                                  tie_points=[(0.0005, 0)])
        self.assertEqual(events, [])
        self.assertEqual(groups[0][0].start_depth, 3)

    def test_parallel_pipe_requires_gap_between_outer_envelopes(self):
        with self.assertRaisesRegex(ValueError, "horizontal utility clearance"):
            profiles([segment([(19, -20), (19, 20)])], [utility(x=20)])

    def test_all_envelope_values_and_diameter_dependent_building_clearance(self):
        self.assertEqual(d.ENVELOPE[50], (.4, .125))
        self.assertEqual(d.ENVELOPE[1400], (3.450, 1.600))
        self.assertAlmostEqual(d.clearance("oks", 50), 5.2)
        self.assertAlmostEqual(d.clearance("oks", 500), 7.835)
        self.assertAlmostEqual(d.clearance("oks", 900), 10.225)

    def test_road_is_special_polygon_plus_three_meters_each_side(self):
        road = d.CrossingObject("road", "road", polygons=[[
            [(18, -50), (22, -50), (22, 50), (18, 50), (18, -50)]]])
        groups, _ = profiles([segment([(0, 0), (40, 0)])], [road])
        special = next(p for p in groups[0] if p.crossing_ids)
        self.assertAlmostEqual(special.length, 10)
        self.assertEqual(special.special_factor, 1.6)
        self.assertEqual(special.start_depth, 3)

    def test_road_shallow_crossing_angle_rejected(self):
        road = d.CrossingObject("road", "road", polygons=[[
            [(18, -50), (22, -50), (22, 50), (18, 50), (18, -50)]]])
        with self.assertRaisesRegex(ValueError, "angle below 45"):
            profiles([segment([(0, -45), (40, 45)])], [road])


if __name__ == "__main__":
    unittest.main()
