#!/usr/bin/env python3
"""Run several deterministic ALNS seeds and keep the best independently valid result.

This is an orchestration layer, not a new scoring model.  Every child run uses
optimize_network.py, which round-trip-validates the delivered GeoJSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def read_best(summary_path: Path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    variants = data.get("variants") or []
    if not variants:
        raise RuntimeError(f"No variants in {summary_path}")
    v = min(variants, key=lambda x: (-(x.get("connected_oks_count") or 0), x.get("score", float("inf"))))
    return v, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Датасет скорректированный.geojson")
    parser.add_argument("--output-dir", default="routing_results_portfolio")
    parser.add_argument("--mode", choices=["2d", "depth"], default="2d")
    parser.add_argument("--seeds", default="17,23,41,73,101")
    parser.add_argument("--search-seconds", type=float, default=180.)
    parser.add_argument("--refine-seconds", type=float, default=60.)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--routes", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=2)
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        parser.error("At least one seed is required")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    best = None

    for seed in seeds:
        out = root / f"seed_{seed}"
        cmd = [sys.executable, "optimize_network.py", "--input", args.input,
               "--output-dir", str(out), "--mode", args.mode, "--solver", "alns",
               "--search-seconds", str(args.search_seconds), "--refine-seconds", str(args.refine_seconds),
               "--iterations", str(args.iterations), "--beam-width", str(args.beam_width),
               "--routes", str(args.routes), "--neighbors", str(args.neighbors), "--seed", str(seed)]
        print("RUN", seed, flush=True)
        proc = subprocess.run(cmd)
        record = {"seed": seed, "returncode": proc.returncode, "output_dir": str(out)}
        summary = out / args.mode / "summary.json"
        valid = out / args.mode / "validation.json"
        if proc.returncode == 0 and summary.exists() and valid.exists():
            v, _ = read_best(summary)
            record.update({"connected": v.get("connected_oks_count"), "score": v.get("score"),
                           "cost": v.get("calculated_cost"), "length": v.get("new_network_length")})
            key = (-(v.get("connected_oks_count") or 0), v.get("score", float("inf")))
            if best is None or key < best[0]:
                best = (key, seed, out, v)
        records.append(record)
        (root / "portfolio.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if best is None:
        raise SystemExit("No independently valid portfolio run completed")

    _, seed, out, variant = best
    final = root / "best"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(out / args.mode, final)
    result = {"selected_seed": seed, "selection": "max coverage, then minimum official score",
              "best": variant, "runs": records,
              "global_optimum_proven": False}
    (root / "portfolio.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"BEST seed={seed} connected={variant.get('connected_oks_count')} score={variant.get('score')}")
    print(f"Result: {final / 'best_variant.geojson'}")


if __name__ == "__main__":
    main()
