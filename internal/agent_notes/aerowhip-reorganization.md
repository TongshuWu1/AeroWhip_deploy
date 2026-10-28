---
name: aerowhip-reorganization
description: State and rules of the AeroWhip_deploy reorganization toward an open-source RA-L release
metadata:
  node_type: memory
  type: project
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-26T20:16:12.124Z
---

AeroWhip_deploy (~/Documents/PHD/AeroWhip_deploy) is being reorganized on branch `reorganize` (started 2026-10-26) for an open-source release accompanying an RA-L submission. Everything (simulation, learning, evaluation, fitting, takes, ROS/Crazyswarm deployment) stays in the public repo.

Phases 0–4 done on branch `reorganize` as of 2026-10-26 (commit 9750cdb): code in `aerowhip/` package, entry points in `scripts/`, experiment files with run/environment/task/ppo sections, generic env + Task interface, pytest core suite (90 cases), public `docs/`, internal record in `internal/` (procedure in internal/PUBLICATION.md). Not yet merged into main; the companion Crazyswarm repo's docker/start.sh path edit is uncommitted there. Two branches: development keeps internal notes (internal/, AGENTS.md, CLAUDE.md, detailed record); `publication` branch = development minus those. Stop for user review after each phase.

**Why:** user wants organized, understandable, no-extra-stuff code for publication.
**How to apply:** every refactor must pass `scripts/equivalence_check.py` (bitwise; reference recorded at commit 375b44e, kept out of Git) and the test suite; never change physics/reward values silently. Recording redesign comes after phases 3–4: [[aerowhip-recording-redesign]].
