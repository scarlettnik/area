"""Audit the input and explain provably blocked closest-facade entries."""
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
                              'reason': 'nearest_straight_entry_reenters_own_building', 'witnesses': witnesses})
    penalty = sum(h.penalty_unconnected(c['flow_tph']) for c in conflicts)
    return {'input_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'object_counts': dict(Counter(f['properties']['object_type'] for f in data['features'])),
            'interpretation': 'The final straight entry passes through a globally nearest point on the containing polygon boundary.',
            'geometrically_blocked_entries': conflicts,
            'connected_count_upper_bound': len(terminals) - len(conflicts),
            'unavoidable_penalty_rub_under_this_interpretation': penalty,
            'score_lower_bound_from_penalty_only': .7 * penalty / 25_000_000,
            'proof': 'Before re-entering the same polygon, distance to its boundary cannot exceed half the free interval. Every nearest ray has a bound below the mandatory pipe-axis clearance.',
            'scope': 'This proves only these local entry obstructions, not feasibility or optimality of the other connections.'}


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
