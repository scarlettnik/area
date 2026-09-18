import unittest

import heat_route_builder as h
from routing_constraints import validate_path
from test_heat_route_builder import rectangle


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.building = rectangle(-10, -10, 10, 10, kind="oks")

    def test_straight_entry_ends_at_original_point(self):
        validate_path([(40, 20), (25, 0), (0, 0)], 125, [self.building], [(0, 0)], True)

    def test_corner_inside_house_or_setback_is_rejected(self):
        for corner in ((0, 5), (0, 14)):
            with self.subTest(corner=corner), self.assertRaises(ValueError):
                validate_path([(30, 20), corner, (0, 0)], 125, [self.building], [(0, 0)], True)

    def test_parallel_transit_and_other_building_are_rejected(self):
        for points, endpoints in (([(-30, 0), (30, 0)], []),
                                  ([(40, 0), (0, 0)], [(40, 0)])):
            with self.assertRaises(ValueError):
                validate_path(points, 125, [self.building], endpoints, True)

    def test_actual_diameter_changes_clearance(self):
        validate_path([(-30, 16), (30, 16)], 125, [self.building])
        with self.assertRaises(ValueError):
            validate_path([(-30, 16), (30, 16)], 500, [self.building])


if __name__ == "__main__":
    unittest.main()
