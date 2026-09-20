"""Regressions for rules changed by the corrected appendix, using synthetic geometry."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import heat_route_builder as h
import routing_depth as d
from routing_constraints import validate_path, nearest_entry_obstruction
from routing_search import Search
from routing_visibility import VisibilityGraph
from test_heat_route_builder import rectangle, root, setup, route
from test_validate_routing import crossing_example
from validate_routing import validate_result


class CorrectedEngineeringTests(unittest.TestCase):
    def test_new_tie_near_existing_chamber_becomes_available_when_it_fills(self):
        with tempfile.TemporaryDirectory() as directory:
            x, y = 500_000., 6_170_000.
            def feature(identifier, kind, geometry, **properties):
                return {'type': 'Feature', 'geometry': geometry,
                        'properties': dict(id=identifier, object_type=kind, **properties)}
            source, output = Path(directory) / 'in.geojson', Path(directory) / 'out.geojson'
            source.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
                feature('main', 'heat_network', h.unproject_linestring([(x-50, y), (x+50, y)]), diameter=300),
                feature('spur', 'heat_network', h.unproject_linestring([(x, y), (x, y+50)]), diameter=300),
                feature('old', 'heat_chamber', h.unproject_point_geom((x, y))),
                feature('a', 'oks_connection_point', h.unproject_point_geom((x-20, y-20)), flow_tph=1),
                feature('b', 'oks_connection_point', h.unproject_point_geom((x+5, y-20)), flow_tph=1),
            ]}))
            ts, rs, obs, _, meta = h.read_input(source)
            graph = VisibilityGraph(ts, rs, obs, crossing_objects=meta['crossing_objects'])
            old = next(r for r in rs if r.is_existing_chamber)
            new = min((r for r in rs if not r.is_existing_chamber), key=lambda r: h.dist(r.point, (x+5, y)))
            self.assertLess(h.dist(new.point, old.point), 10)
            incomplete = h.BuiltTree([new], {h.edge_key(new.cell, ts[1].cell)}, {ts[1].id}, {ts[0].id})
            with self.assertRaisesRegex(ValueError, 'Eligible existing chamber'):
                h.materialize_variant('v', '', incomplete, ts, graph, {})
            tree = h.BuiltTree([old, new], incomplete.edges | {h.edge_key(old.cell, ts[0].cell)}, {t.id for t in ts}, set())
            variant = h.materialize_variant('v', '', tree, ts, graph, {})
            variant.summary['rank'] = 1
            h.write_geojson(str(output), variant.features)
            self.assertEqual(validate_result(source, output)['connected'], 2)

    def test_large_total_demand_can_use_separate_roots(self):
        ts = [h.Terminal('a', (0, 20), 12_000), h.Terminal('b', (100, 20), 12_000)]
        rs = [root('ra', (0, 0)), root('rb', (100, 0))]
        graph = VisibilityGraph(ts, rs, [])
        search = Search(ts, rs, graph, seconds=2, routes=1, beam_width=1)
        best = search.run()[0]
        self.assertEqual(best.summary['connected_oks_count'], 2)

    def test_obstruction_witness_proves_narrow_reentry_gap(self):
        ring = [(0, 0), (30, 0), (30, 20), (20, 20), (20, 5), (16, 5), (16, 20), (0, 20), (0, 0)]
        obstacle = h.Obstacle('u', 'oks', [ring], 5.2, h.expand_bbox(h.bbox(ring), 5.2))
        terminal = h.Terminal('t', (14, 10), 1)
        proof = nearest_entry_obstruction(terminal, [obstacle], 50)
        self.assertTrue(proof[0]['all_nearest_rays_blocked'])
        self.assertAlmostEqual(proof[0]['rays'][0]['outside_interval_m'], 4)
        self.assertAlmostEqual(proof[0]['rays'][0]['maximum_possible_clearance_m'], 2)

    def test_numeric_and_text_ids_are_distinct_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            x, y = 500_000., 6_170_000.
            data = {'type': 'FeatureCollection', 'features': [
                {'type': 'Feature', 'geometry': h.unproject_linestring([(x, y - 50), (x, y + 50)]),
                 'properties': {'id': 'existing', 'object_type': 'heat_network', 'diameter': 300}},
                *[{'type': 'Feature', 'geometry': h.unproject_point_geom((x + 30, y + dy)),
                   'properties': {'id': tid, 'object_type': 'oks_connection_point', 'flow_tph': 1}}
                  for tid, dy in [(1, 10), ('1', -10)]] ]}
            source, output = Path(directory) / 'input.geojson', Path(directory) / 'out.geojson'
            source.write_text(json.dumps(data))
            ts, rs, obs, pts, meta = h.read_input(source)
            self.assertEqual(len({t.id for t in ts}), 2)
            graph = VisibilityGraph(ts, rs, obs, crossing_objects=meta['crossing_objects'])
            search = Search(ts, rs, graph, seconds=3, routes=1, beam_width=2)
            best = search.run()[0]; best.summary['rank'] = 1
            h.write_geojson(str(output), best.features)
            report = validate_result(source, output)
            self.assertEqual(report['connected'], 2)
            # Imported export keeps opaque identifiers and is re-evaluated in
            # the new run rather than trusting saved monetary totals.
            from routing_seed import load_seed_trees
            ts2, rs2, obs2, _, meta2 = h.read_input(source)
            graph2 = VisibilityGraph(ts2, rs2, obs2, crossing_objects=meta2['crossing_objects'])
            trees = load_seed_trees(source, output, ts2, rs2, graph2)
            replay = Search(ts2, rs2, graph2, seconds=.05, initial_trees=trees).run()[0]
            replay.summary['rank'] = 1
            h.write_geojson(str(output), replay.features)
            self.assertEqual(validate_result(source, output)['connected'], 2)
            self.assertLessEqual(replay.score, best.score + 1e-5)

    def test_new_chamber_includes_tie_cost(self):
        t = h.Terminal('t', (20, 0), 1)
        r = root('r', (0, 0), chamber=False, diameter=300)
        result = route([t], [r], setup([t], [r]))
        self.assertEqual(result.summary['existing_chamber_tie_in_cost'], 0)
        self.assertEqual(result.summary['construction_cost'], 20 * h.NEW_COST[50] + h.chamber_cost(300))
        self.assertEqual(result.summary['score_status'], 'complete')
        self.assertTrue(all(f['properties']['object_type'] in {'heat_network', 'heat_chamber', 'technical_node', 'variant_summary'} for f in result.features))
        nodes = {f['properties']['id'] for f in result.features if f['geometry'] and f['geometry']['type'] == 'Point'}
        pipe = next(f['properties'] for f in result.features if f['properties']['object_type'] == 'heat_network')
        self.assertIn(pipe['start_node_id'], nodes)

    def test_existing_chamber_charges_each_incident_new_pipe(self):
        ts = [h.Terminal('a', (10, 0), 1), h.Terminal('b', (-10, 0), 1)]
        r = root('r', (0, 0)); grid = setup(ts, [r])
        tree = h.BuiltTree([r], {h.edge_key(r.cell, t.cell) for t in ts}, {'a', 'b'}, set())
        result = h.materialize_variant('v', 'two ties', tree, ts, grid, {})
        self.assertEqual(result.summary['existing_chamber_tie_in_count'], 2)
        self.assertEqual(result.summary['existing_chamber_tie_in_cost'], 10_000_000)
        self.assertEqual(result.summary['chamber_construction_cost'], 0)

    def test_transform_preserves_existing_chamber_degree_and_id(self):
        r = root('r', (10, 0)); r.existing_degree = 3; r.input_id = 106
        _, roots, _, _ = h.apply_coordinate_transform([], [r], [], [], h.CoordinateTransform((0, 0), .4))
        self.assertEqual(roots[0].existing_degree, 3)
        self.assertEqual(roots[0].input_id, 106)

    def test_long_constant_flow_selects_minimum_length_diameter(self):
        t = h.Terminal('t', (300, 0), 1); r = root('r', (0, 0)); grid = setup([t], [r])
        tree = h.BuiltTree([r], {h.edge_key(r.cell, t.cell)}, {'t'}, set())
        segment = h.compress_segments(tree, [t], grid)[0][0]
        self.assertEqual(segment.diameter, 80)
        self.assertEqual(segment.length, 300)

    def test_parallel_branch_lengths_are_not_summed(self):
        r = root('r', (0, 0)); r.cell = (0, 0)
        ts = [h.Terminal('a', (150, 0), 1, (30, 0)), h.Terminal('b', (0, 150), 1, (0, 30))]
        grid = h.RoutingGrid([], [(0, 0), (150, 150)], 5, margin=0)
        tree = h.BuiltTree([r], {h.edge_key(r.cell, t.cell) for t in ts}, {'a', 'b'}, set())
        self.assertEqual([s.diameter for s in h.compress_segments(tree, ts, grid)[0]], [50, 50])

    def test_nearest_building_entry_and_maximum_turn(self):
        obstacle = rectangle(-10, -10, 10, 10, kind='oks')
        validate_path([(1, 0), (20, 0)], 50, [obstacle], [(1, 0)], True)
        with self.assertRaisesRegex(ValueError, 'nearest'):
            validate_path([(1, 0), (-20, 0)], 50, [obstacle], [(1, 0)], True)
        with self.assertRaisesRegex(ValueError, '90'):
            validate_path([(0, 0), (10, 0), (5, 5)], 50, [])

    def test_dn_aware_edge_retains_narrow_corridor(self):
        ts = [h.Terminal('t', (30, 0), 1)]; rs = [root('r', (0, 0))]
        o = rectangle(8, 6, 22, 15, kind='oks'); o.clearance = d.clearance('oks', 50)
        graph = VisibilityGraph(ts, rs, [o])
        a, b = graph.node((0, 0)), graph.node((30, 0)); graph.connect(a, b)
        self.assertTrue(graph.legal(a, b, 50))
        self.assertFalse(graph.legal(a, b, 500))

    def test_connect_even_when_penalty_would_be_cheaper(self):
        t = h.Terminal('t', (1500, 0), 1); r = root('r', (0, 0))
        graph = VisibilityGraph([t], [r], [])
        search = Search([t], [r], graph, seconds=2, beam_width=1, routes=1)
        result = search.run()[0]
        self.assertEqual(result.summary['connected_oks_count'], 1)
        self.assertGreater(result.summary['calculated_cost'], h.penalty_unconnected(1))

    def test_overlap_uses_max_and_splits_at_each_change(self):
        segment = h.Segment([(0, 0), (50, 0)], (0, 0), (1, 0), 1, 50, 50, 'a', 'b', 0)
        road = d.CrossingObject('road', 'road', polygons=[[[ (15, -20), (25, -20), (25, 20), (15, 20), (15, -20) ]]], horizontal_gap=1.5)
        gas = d.CrossingObject('gas', 'gas_pipeline', lines=[[(20, -20), (20, 20)]], width=.4, height=.4, top=2.8, vertical_gap=.2, horizontal_gap=2)
        profiles, _ = d.build_plan_profiles([segment], [road, gas], set(), set(), [], h.NEW_COST)
        pieces = profiles[0]
        overlap = [p for p in pieces if len(p.crossing_ids) == 2]
        self.assertEqual(len(overlap), 1)
        self.assertEqual(overlap[0].special_factor, 1.6)
        self.assertEqual([round(p.length, 6) for p in pieces], [12, 6, 4, 6, 22])
        self.assertTrue(all(p.start_depth is None and p.depth_coefficient == 1 for p in pieces))
        deep, _ = d.build_depth_profiles([segment], [road, gas], {(0, 0)}, {(1, 0)}, [], h.NEW_COST)
        self.assertTrue(any(len(p.crossing_ids) == 2 and p.special_factor == 1.6 for p in deep[0]))

    def test_linear_road_and_bend_in_special_passage(self):
        obj = d.CrossingObject('road', 'road', lines=[[(10, -30), (10, 30)]], horizontal_gap=1.5)
        segment = h.Segment([(0, 0), (20, 0)], (0, 0), (1, 0), 1, 50, 20, 'a', 'b', 0)
        passage = d.segment_passages(0, segment, [obj], [])[0]
        self.assertEqual((passage.start, passage.end), (7, 13))
        segment.points = [(0, 0), (11, 0), (20, 4)]
        with self.assertRaisesRegex(ValueError, 'straight'):
            d.segment_passages(0, segment, [obj], [])

    def test_crossing_does_not_permit_an_adjacent_parallel_run(self):
        gas = d.CrossingObject('gas', 'gas_pipeline', lines=[[(10, -30), (10, 30)]], horizontal_gap=2, width=.4)
        segment = h.Segment([(0, 0), (12.1, 0), (12.1, 20)], (0, 0), (1, 0), 1, 50, 32.1, 'a', 'b', 0)
        with self.assertRaisesRegex(ValueError, 'clearance'):
            d.segment_passages(0, segment, [gas], [])

    def test_validator_accepts_reversed_coordinate_order(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, result = crossing_example(directory)
            for f in result.features:
                p = f['properties']
                if p['object_type'] == 'heat_network':
                    f['geometry']['coordinates'].reverse()
                    p['start_node_id'], p['end_node_id'] = p['end_node_id'], p['start_node_id']
                    p['depth_start'], p['depth_end'] = p['depth_end'], p['depth_start']
            h.write_geojson(str(output), result.features)
            self.assertTrue(validate_result(source, output)['valid'])

    def test_numeric_unconnected_identifier_keeps_input_type(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _ = crossing_example(directory)
            data = json.loads(source.read_text()); data['features'][-1]['properties']['id'] = 16
            source.write_text(json.dumps(data))
            terminals, roots, obstacles, points, meta = h.read_input(source)
            graph = VisibilityGraph(terminals, roots, obstacles)
            tree = h.BuiltTree([], set(), set(), {t.id for t in terminals})
            result = h.materialize_variant('v', 'empty', tree, terminals, graph, {})
            self.assertEqual(result.summary['unconnected_oks_ids'], [16])
            self.assertIs(type(result.summary['unconnected_oks_ids'][0]), int)


if __name__ == '__main__':
    unittest.main()
