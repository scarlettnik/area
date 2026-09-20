import unittest

from routing_quality import polyline_quality


class GeometryQualityTests(unittest.TestCase):
    def test_straight_has_no_bends(self):
        q = polyline_quality([(0., 0.), (10., 0.), (20., 0.)])
        self.assertEqual(q['bend_count'], 0)
        self.assertEqual(q['micro_bend_count'], 0)
        self.assertEqual(q['backtrack_count'], 0)
        self.assertAlmostEqual(q['detour_ratio'], 1.)

    def test_micro_bend_detected(self):
        q = polyline_quality([(0., 0.), (2., 0.), (2., 10.)])
        self.assertEqual(q['bend_count'], 1)
        self.assertEqual(q['right_angle_count'], 1)
        self.assertEqual(q['micro_bend_count'], 1)

    def test_backtrack_detected(self):
        q = polyline_quality([(0., 0.), (10., 0.), (5., 5.)])
        self.assertEqual(q['backtrack_count'], 1)


if __name__ == '__main__':
    unittest.main()
