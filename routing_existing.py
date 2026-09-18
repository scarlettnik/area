"""Appendix §7: cached upstream chains and exact partial-edge reconstruction.

Missing legacy attributes are reported, never replaced by guessed flows. Malformed
complete networks (cycles, dangling references, disconnected geometry) are errors.
"""
from collections import defaultdict
import math

from routing_depth import chainages, point_at


class ExistingNetwork:
    def __init__(self, features, project):
        self.records = {}
        self.missing = []
        self.chains = {}
        self.lines = {}
        self.stations = {}
        for feature in features:
            p = feature.get("properties", {})
            if p.get("object_type") in {"source", "heat_network", "heat_chamber"}:
                self.records[str(p["id"])] = dict(p, geometry=project(feature["geometry"]))
        for oid, p in self.records.items():
            required = {"heat_network": ("diameter", "flow_tph", "upstream_object_id"),
                        "heat_chamber": ("diameter", "upstream_object_id"), "source": ()}[p["object_type"]]
            self.missing.extend(f"{oid}.{key}" for key in required if p.get(key) is None)
        self.complete = bool(self.records) and not self.missing
        if not self.complete:
            return
        from heat_route_builder import CAPACITY, point_segment_distance, select_diameter
        for oid, record in self.records.items():
            path, visited, current = [], set(), oid
            while self.records.get(current, {}).get("object_type") != "source":
                if current in visited or current not in self.records:
                    raise ValueError(f"{oid}: cyclic or dangling existing upstream chain")
                visited.add(current)
                path.append(current)
                current = str(self.records[current]["upstream_object_id"])
            self.chains[oid] = tuple(path)
            if record["object_type"] == "source":
                continue
            if record["diameter"] not in CAPACITY:
                raise ValueError(f"{oid}: unsupported existing diameter")
            if record["object_type"] != "heat_network":
                continue
            select_diameter(float(record["flow_tph"]))
            line = record["geometry"]
            upstream = self.records[str(record["upstream_object_id"])]
            geometry = upstream["geometry"]
            if upstream["object_type"] == "heat_network":
                distance = lambda p: min(point_segment_distance(p, a, b) for a, b in zip(geometry, geometry[1:]))
            else:
                distance = lambda p: math.dist(p, geometry)
            if min(distance(line[0]), distance(line[-1])) > .1:
                raise ValueError(f"{oid}: upstream geometry does not meet a pipe endpoint")
            if distance(line[-1]) < distance(line[0]):
                line = line[::-1]
            self.lines[oid] = line  # chainage zero is the upstream end
            self.stations[oid] = chainages(line)

    def status(self):
        return {"status": "complete" if self.complete else "unknown_missing_input",
                "missing_fields": self.missing}

    def station(self, oid, point):
        from heat_route_builder import point_segment_distance
        line, distances = self.lines[oid], self.stations[oid]
        options = []
        for i, (a, b) in enumerate(zip(line, line[1:])):
            length = math.dist(a, b)
            if length < 1e-9:
                continue
            t = max(0., min(1., sum((point[k] - a[k]) * (b[k] - a[k]) for k in (0, 1)) / length**2))
            options.append((point_segment_distance(point, a, b), distances[i] + t * length))
        error, station = min(options)
        if error > .02:
            raise ValueError(f"{oid}: tie-in is not on the existing pipe")
        return station

    def evaluate(self, injections, variant_id="trial"):
        """injections: iterable of (existing object id, UTM point, added flow)."""
        if not self.complete:
            return [], 0., 0., {}
        from heat_route_builder import select_diameter, RECON_COST, unproject_linestring
        loads = defaultdict(list)
        for oid, point, flow in injections:
            oid = str(oid)
            if not math.isfinite(flow) or flow < 0:
                raise ValueError("Invalid added existing-network flow")
            if oid not in self.records:
                raise ValueError(f"Unknown tie-in object {oid}")
            record = self.records[oid]
            for part in self.chains[oid]:
                if part not in self.lines:
                    continue
                end = self.station(part, point) if part == oid and record["object_type"] == "heat_network" else self.stations[part][-1]
                if flow and end > 1e-8:
                    loads[part].append((end, flow))
        features, cost, length, required = [], 0., 0., {}
        for oid, values in sorted(loads.items()):
            record = self.records[oid]
            cuts = sorted({0., *(end for end, flow in values)})
            line, distances = self.lines[oid], self.stations[oid]
            for start, end in zip(cuts, cuts[1:]):
                added = sum(flow for stop, flow in values if stop >= end - 1e-8)
                total = float(record["flow_tph"]) + added
                diameter = select_diameter(total)
                required[oid] = max(required.get(oid, record["diameter"]), diameter)
                if diameter <= record["diameter"]:
                    continue
                points = [point_at(line, distances, start)]
                points.extend(p for p, s in zip(line, distances) if start + 1e-8 < s < end - 1e-8)
                points.append(point_at(line, distances, end))
                part_length = end - start
                part_cost = part_length * RECON_COST[diameter]
                cost += part_cost
                length += part_length
                features.append({"type": "Feature", "geometry": unproject_linestring(points), "properties": {
                    "id": f"{variant_id}_reconstruction_{len(features) + 1}", "variant_id": variant_id,
                    "object_type": "heat_network_reconstruction", "existing_object_id": oid,
                    "existing_flow_tph": record["flow_tph"], "added_flow_tph": added,
                    "calculated_flow_tph": total, "existing_diameter": record["diameter"],
                    "required_diameter": diameter, "length": round(part_length, 3), "cost": round(part_cost, 2)}})
        return features, cost, length, required

    def chamber_diameter(self, oid, new_diameter, required):
        if not self.complete:
            return new_diameter
        p = self.records[str(oid)]["geometry"]
        diameters = [new_diameter]
        from heat_route_builder import point_segment_distance
        for hid, line in self.lines.items():
            if min(point_segment_distance(p, a, b) for a, b in zip(line, line[1:])) < .1:
                diameters.append(required.get(hid, self.records[hid]["diameter"]))
        return max(diameters)
