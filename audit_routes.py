"""Explain geometric turns and measure endpoint errors in an exported result."""
import argparse
import json
from pathlib import Path

import heat_route_builder as h
from routing_constraints import validate_path
from routing_depth import segment_passages


def audit(input_path, result_path):
    terminals, _, obstacles, _, meta = h.read_input(input_path)
    features = json.loads(Path(result_path).read_text())["features"]
    lines = [(f["properties"], [h.lonlat_to_utm37(*p[:2]) for p in f["geometry"]["coordinates"]])
             for f in features if f["properties"]["object_type"] == "heat_network"]
    ties = [h.lonlat_to_utm37(*f["geometry"]["coordinates"]) for f in features if f["properties"]["object_type"] == "tie_in"]
    terminal_points = [t.point for t in terminals]
    violations, turns, errors = [], [], []
    for props, points in lines:
        try:
            validate_path(points, props["diameter"], obstacles, terminal_points, True)
        except ValueError as error:
            violations.append({"pipe": props["id"], "reason": str(error)})
        for terminal in terminals:
            for key, p in (("start_node_id", points[0]), ("end_node_id", points[-1])):
                if props[key] == terminal.id:
                    errors.append(h.dist(p, terminal.point))
        points = h.remove_collinear(points)
        for index in range(1, len(points) - 1):
            if not h.polyline_bend_count(points[index - 1:index + 2], 1.):
                continue
            replacement = points[:index] + points[index + 1:]
            reason = None
            try:
                validate_path(replacement, props["diameter"], obstacles, terminal_points, True)
            except ValueError as error:
                reason = str(error)
            if reason is None and any(h.polylines_conflict(replacement, p) for other, p in lines if other["id"] != props["id"]):
                reason = "shortcut intersects another pipe outside a shared endpoint"
            if reason is None:
                segment = h.Segment(replacement, props["start_node_id"], props["end_node_id"], props["flow_tph"],
                                    props["diameter"], h.path_length(replacement), props["start_node_id"], props["end_node_id"], 0.)
                try:
                    passages = segment_passages(0, segment, meta["crossing_objects"], ties)
                    if passages or props.get("laying_method") == "special" or props.get("depth_start", 3) != props.get("depth_end", 3):
                        reason = "shortcut changes special crossing/depth profile; requires network evaluation"
                except ValueError as error:
                    reason = str(error)
            turns.append({"pipe": props["id"], "coordinate": h.unproject_point(points[index]),
                          "reason": reason or "removable at unchanged depth and DN",
                          "removable": reason is None})
    return {"geometry_violations": violations, "endpoint_max_error_m": max(errors, default=0.),
            "endpoint_count": len(errors), "turns_over_one_degree": len(turns),
            "removable_bend_count": sum(t["removable"] for t in turns), "turns": turns}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="!!!_Датасет.geojson")
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.input, args.result)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k,v in result.items() if k != "turns"}, ensure_ascii=False, indent=2))
