import unittest

import heat_route_builder as h
from routing_existing import ExistingNetwork


def feature(oid, kind, geometry, **props):
    return {"properties": dict(id=oid, object_type=kind, **props), "geometry": {"coordinates": geometry}}


def network():
    return [feature("s", "source", (0, 0)),
            feature("u", "heat_network", [(0, 0), (100, 0)], diameter=50, flow_tph=3, upstream_object_id="s"),
            feature("c", "heat_chamber", (100, 0), diameter=50, upstream_object_id="u"),
            feature("v", "heat_network", [(100, 0), (200, 0)], diameter=50, flow_tph=2, upstream_object_id="c")]


class ReconstructionTests(unittest.TestCase):
    def model(self, features=None):
        return ExistingNetwork(features or network(), lambda g: g["coordinates"])

    def test_multiple_ties_accumulate_only_upstream_parts(self):
        features, cost, length, required = self.model().evaluate([("v", (150, 0), 2), ("v", (180, 0), 5)])
        parts = [f["properties"] for f in features]
        self.assertEqual([(p["existing_object_id"], p["length"], p["added_flow_tph"], p["required_diameter"]) for p in parts],
                         [("u", 100., 7, 80), ("v", 50., 7, 80), ("v", 30., 5, 65)])
        self.assertEqual(length, 180)
        self.assertEqual(cost, 150 * h.RECON_COST[80] + 30 * h.RECON_COST[65])
        self.assertEqual(required, {"u": 80, "v": 80})

    def test_reversed_coordinates_and_chamber_injection(self):
        data = network()
        data[-1]["geometry"]["coordinates"].reverse()
        model = self.model(data)
        features, cost, length, required = model.evaluate([("c", (100, 0), 7)])
        self.assertEqual(length, 100)
        self.assertEqual(model.chamber_diameter("c", 50, required), 80)
        self.assertEqual(model.station("v", (175, 0)), 75)

    def test_missing_is_unknown_not_assumed_zero_flow(self):
        data = network()
        del data[1]["properties"]["flow_tph"]
        model = self.model(data)
        self.assertFalse(model.complete)
        self.assertEqual(model.status()["status"], "unknown_missing_input")
        self.assertIn("u.flow_tph", model.missing)

    def test_cycles_dangling_and_disconnected_upstream_fail(self):
        for upstream in ("v", "absent"):
            data = network()
            data[1]["properties"]["upstream_object_id"] = upstream
            with self.assertRaises(ValueError):
                self.model(data)
        data = network()
        data[0]["geometry"]["coordinates"] = (1000, 1000)
        with self.assertRaises(ValueError):
            self.model(data)


if __name__ == "__main__":
    unittest.main()
