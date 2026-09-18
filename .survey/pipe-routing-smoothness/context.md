# Context

## Workflow Context

Urban utility routing is normally treated as a constrained least-cost alignment problem, not as a shortest Euclidean line. The input geometry defines no-go objects, tie-in candidates, and connection points; the route should satisfy obstacle clearances while remaining buildable and legible.

## Affected Users

Designers and reviewers need route alternatives that can be inspected visually and ranked numerically. Excessive bends make a route look artificial and can imply more fittings, chambers, labor, local resistance, and design review risk.

## Current Workarounds

The first prototype used grid routing plus line-of-sight simplification. This removes some intermediate vertices but cannot fully fix a bad corridor choice, because the path already paid only for distance during search.

## Adjacent Problems

Related work in pipe routing often includes constraints for minimum straight lengths, obstacle clearance, bend count, parallel/grouped routing, and preference for easy installation corridors.

## User Voices

The target visual style is closer to the reference screenshot: longer continuous trunks, short building tie-ins, fewer random kinks, and branch points that look intentional.

