"""Standalone, dependency-free SVG preview of real GeoJSON routing geometry."""

import argparse
import html
import json
from pathlib import Path

from heat_route_builder import bbox, lonlat_to_utm37, point_in_ring, input_key


def write_preview(input_path, result_path, output_path):
    source = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    summary = next(f['properties'] for f in result['features'] if f['properties'].get('object_type') == 'variant_summary')
    missing = {input_key(value) for value in summary['unconnected_oks_ids']}
    used_nodes = {input_key(f['properties'][key]) for f in result['features']
                  if f['properties'].get('object_type') == 'heat_network' for key in ('start_node_id', 'end_node_id')}
    terminals = [f for f in source["features"] if f.get("properties", {}).get("object_type") == "oks_connection_point"]
    terminal_positions = [f["geometry"]["coordinates"][:2] for f in terminals if input_key(f['properties']['id']) not in missing]
    positions = [lonlat_to_utm37(*f["geometry"]["coordinates"][:2]) for f in terminals]
    for f in result["features"]:
        if f.get("geometry", {}) and f["geometry"]["type"] == "LineString":
            positions.extend(lonlat_to_utm37(*p[:2]) for p in f["geometry"]["coordinates"])
    x0, y0, x1, y1 = bbox(positions)
    x0, y0, x1, y1 = x0 - 70, y0 - 70, x1 + 70, y1 + 70
    width, height = 1200, 1040
    scale = min((width - 60) / (x1 - x0), (height - 100) / (y1 - y0))

    def xy(p):
        x, y = lonlat_to_utm37(*p[:2])
        return (30 + (x - x0) * scale, 75 + (y1 - y) * scale)

    def line(coords, color, thickness=2, fill="none"):
        points = " ".join(f"{x:.3f},{y:.3f}" for x, y in map(xy, coords))
        return f'<polyline points="{points}" fill="{fill}" stroke="{color}" stroke-width="{thickness}" stroke-linejoin="round"/>'

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#111619"/>',
           '<defs><clipPath id="map"><rect x="0" y="60" width="1200" height="980"/></clipPath></defs>',
           '<g clip-path="url(#map)">']
    for f in source["features"]:
        geometry, props = f.get("geometry"), f.get("properties", {})
        if not geometry:
            continue
        kind = geometry["type"]
        if kind in {"Polygon", "MultiPolygon"}:
            polygons = [geometry["coordinates"]] if kind == "Polygon" else geometry["coordinates"]
            for polygon in polygons:
                commands = []
                for ring in polygon:
                    commands.append("M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in map(xy, ring)) + " Z")
                connected_building = any(point_in_ring(p, polygon[0]) for p in terminal_positions)
                color = "#254453" if props.get("restriction_type") == "water" else ("#497367" if connected_building else "#353d44")
                stroke = "#a3e8ce" if connected_building else "#4b535b"
                svg.append(f'<path d="{" ".join(commands)}" fill="{color}" fill-rule="evenodd" stroke="{stroke}" stroke-width="1.3"/>')
        elif kind == "LineString" and props.get("object_type") == "heat_network":
            svg.append(line(geometry["coordinates"], "#c47a80", 2.5))
        elif kind == 'Point' and props.get('object_type') == 'heat_chamber' and input_key(props['id']) in used_nodes:
            x, y = xy(geometry['coordinates'])
            svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="#7bb7d6"/>')
    for f in result["features"]:
        geometry, props = f.get("geometry"), f.get("properties", {})
        if not geometry:
            continue
        if geometry["type"] == "LineString":
            svg.append(line(geometry["coordinates"], "#101417", 6))
            svg.append(line(geometry["coordinates"], "#f5ce78", 3))
        elif geometry["type"] == "Point":
            x, y = xy(geometry["coordinates"])
            color = "#7bb7d6" if 'existing_object_id' in props else "#f5ce78"
            svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="{color}" stroke="#111619" stroke-width="1.5"/>')
    for f in terminals:
        x, y = xy(f["geometry"]["coordinates"])
        label = html.escape(str(f["properties"]["id"]))
        color = '#ef8790' if input_key(f['properties']['id']) in missing else '#c1efdb'
        svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="{color}"/>')
        svg.append(f'<text x="{x+6:.3f}" y="{y-6:.3f}" fill="#e8f6ef" font-family="sans-serif" font-size="13">{label}</text>')
    svg.append('</g><rect width="1200" height="60" fill="#111619"/>')
    title = f'Подключено {summary["connected_oks_count"]}/{len(terminals)} · {summary["length"]:,.1f} м · строительство {summary["construction_cost"]/1e6:.2f} млн ₽ · S={summary["score"]:.3f}'
    svg.append(f'<text x="30" y="38" fill="#f5ce78" font-family="sans-serif" font-size="22">{html.escape(title)}</text>')
    svg.append('<rect x="22" y="988" width="1156" height="36" rx="5" fill="#111619" fill-opacity="0.92"/>')
    for x, color, label in ((36, "#f5ce78", "Новая сеть"), (205, "#c47a80", "Существующая сеть"), (450, "#7bb7d6", "Присоединения"), (660, "#c1efdb", "Подключённый ОКС"), (920, '#ef8790', 'Не подключён')):
        svg.append(f'<circle cx="{x}" cy="1006" r="4" fill="{color}"/><text x="{x+12}" y="1011" fill="#e0e5e4" font-family="sans-serif" font-size="14">{label}</text>')
    svg.append('</svg>')
    Path(output_path).write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Датасет скорректированный.geojson")
    parser.add_argument("--result", default="routing_results_corrected/2d/best_variant.geojson")
    parser.add_argument("--output", default="routing_results_corrected/2d/preview.svg")
    args = parser.parse_args()
    write_preview(args.input, args.result, args.output)
