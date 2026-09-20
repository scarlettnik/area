import unittest

import heat_route_builder as h
from routing_constraints import validate_path
from test_heat_route_builder import rectangle


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.building = rectangle(-10, -10, 10, 10, kind="oks")

    def test_straight_entry_ends_at_original_point(self):
        validate_path([(40, 20), (25, 0), (0, 0)], 125, [self.building], [(0, 0)], True)

    def test_corner_inside_house_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_path([(30, 20), (0, 5), (0, 0)], 125, [self.building], [(0, 0)], True)

    def test_first_entry_turn_can_depart_own_setback_once(self):
        validate_path([(30, 20), (0, 14), (0, 0)], 125, [self.building], [(0, 0)], True)
        with self.assertRaises(ValueError):
            validate_path([(30, 20), (3, 14), (0, 14), (0, 0)], 125,
                          [self.building], [(0, 0)], True)

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
