"""End-to-end export audit, including deliberate corruptions of the result."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import heat_route_builder as h
from validate_routing import validate_result


def crossing_example(directory):
    def feature(name, kind, points, **props):
        geometry = h.unproject_point_geom(points) if kind == "oks_connection_point" else h.unproject_linestring(points)
        return {"type": "Feature", "geometry": geometry,
                "properties": {"id": name, "object_type": kind, **props}}
    x, y = 500_000., 6_170_000.
    source = {"type": "FeatureCollection", "features": [
        feature("existing", "heat_network", [(x, y - 50), (x, y + 50)], diameter=300),
        feature("gas", "restriction", [(x + 20, y - 50), (x + 20, y + 50)], restriction_type="gas_pipeline"),
        feature("t", "oks_connection_point", (x + 40, y), flow_tph=1.),
    ]}
    input_path = Path(directory) / "input.geojson"
    input_path.write_text(json.dumps(source))
    terminals, candidates, obstacles, points, meta = h.read_input(input_path)
    grid = h.prepare_grid(terminals, candidates, obstacles, points, 5, cardinal_only=True)
    grid.crossing_objects = meta.pop("crossing_objects")
    grid.depth_mode = True
    tree = h.build_forest(terminals, candidates, grid)
    result = h.materialize_variant("depth", "gas crossing", tree, terminals, grid, meta)
    result.summary["rank"] = 1
    result_path = Path(directory) / "result.geojson"
    h.write_geojson(str(result_path), result.features)
    return input_path, result_path, result


class ExportValidationTests(unittest.TestCase):
    def test_real_export_has_ramps_plateau_and_correct_price(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, result = crossing_example(directory)
            report = validate_result(source, output)
            self.assertTrue(report["valid"])
            self.assertEqual(report["connected"], 1)
            self.assertEqual(result.summary["special_passage_count"], 1)
            self.assertLess(result.summary["min_depth_m"], 3)
            self.assertEqual(result.summary["max_depth_m"], 3)
            special = [f["properties"] for f in result.features if f["properties"].get("laying_method") == "special"]
            self.assertEqual(len(special), 1)
            self.assertAlmostEqual(special[0]["length"], 4, places=3)
            self.assertEqual(special[0]["special_factor"], 1.25)

    def test_validator_rejects_tampered_profile_price_endpoint_and_turn_surcharge(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, result = crossing_example(directory)
            for corruption in ("slope", "cost", "endpoint", "surcharge"):
                with self.subTest(corruption=corruption):
                    features = copy.deepcopy(result.features)
                    pipe = next(f for f in features if f["properties"]["object_type"] == "heat_network")
                    summary = next(f["properties"] for f in features if f["properties"]["object_type"] == "variant_summary")
                    if corruption == "slope":
                        pipe["properties"]["depth_end"] = 20
                    elif corruption == "cost":
                        pipe["properties"]["cost"] += 100_000
                    elif corruption == "endpoint":
                        pipe["geometry"]["coordinates"][0][0] += .001
                    else:
                        summary["bend_penalty_cost"] = 100_000
                        summary["calculated_cost"] += 100_000
                    h.write_geojson(str(output), features)
                    with self.assertRaises(AssertionError):
                        validate_result(source, output)


if __name__ == "__main__":
    unittest.main()
