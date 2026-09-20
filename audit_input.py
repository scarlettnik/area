"""Audit corrected-model inputs and explain closest-facade entry feasibility."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from shapely import Point

import heat_route_builder as h
from routing_constraints import nearest_entry_obstruction
from routing_depth import BUILDINGS


def audit_input(path):
    terminals, _, obstacles, _, _ = h.read_input(path)
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    conflicts = []
    for terminal in terminals:
        owners = [o for o in obstacles if o.kind in BUILDINGS and o.geometry.covers(Point(terminal.point))]
        witnesses = nearest_entry_obstruction(terminal, owners, h.select_diameter(terminal.flow_tph))
        if witnesses:
            conflicts.append({'input_id': terminal.input_id, 'flow_tph': terminal.flow_tph,
                              'reason': 'no_exterior_turn_room_after_closest_outer_facade',
                              'witnesses': witnesses})
    penalty = sum(h.penalty_unconnected(c['flow_tph']) for c in conflicts)
    return {'input_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'object_counts': dict(Counter(f['properties']['object_type'] for f in data['features'])),
            'interpretation': ('The final straight entry crosses a globally closest point on the containing OKS '
                               'outer shell. Interior rings are courtyards/voids, not external facades. The first '
                               'exterior turn may lie inside the own-building nominal setback; its departure leg '
                               'must leave that setback once without building transit or setback re-entry.'),
            'geometrically_blocked_entries': conflicts,
            'connected_count_upper_bound_from_local_entry_only': len(terminals) - len(conflicts),
            'unavoidable_penalty_rub_from_local_entry_only': penalty,
            'score_lower_bound_from_local_entry_penalty_only': .7 * penalty / 25_000_000,
            'proof': ('A local entry is declared blocked only when every tied closest outer-facade ray has no '
                      'positive exterior interval in which the first turn can be placed. Full nominal setback at '
                      'that first turn is intentionally not required.'),
            'scope': ('This is only a local entry audit. It does not prove whole-network feasibility or global '
                      'optimality. On the corrected sample it reports no locally blocked connection points.')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='Датасет скорректированный.geojson')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = json.dumps(audit_input(args.input), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding='utf-8')
    print(report)
