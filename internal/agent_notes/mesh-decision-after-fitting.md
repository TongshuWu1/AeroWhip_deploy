---
name: mesh-decision-after-fitting
description: "10 vs 20 link cable mesh is decided only after the simulator and model fitting are finished, by comparing a full fit at each mesh"
metadata:
  node_type: memory
  type: project
  originSessionId: ab84d596-c67b-413a-a8dc-e2a7a0e5d9ae
  modified: 2026-10-28T01:29:15.213Z
---

On 2026-10-27 the user decided to keep the 10-link cable for now and settle the mesh later. Once the physical simulator and all model fitting are finished, run a complete fit at 10 links and another at 20 links, compare them, and only then choose the mesh.

Evidence so far (2026-10-27, adopted parameters fitted at 10 links, all cf_2 cable flights, 1 s segments): marker error 27.88 mm at 10 links vs 27.73 mm at 20 links. Physics speed per core was 8.3× real time at 10 links vs 1.6× at 20 links. EI and Cb depend on the mesh, so a fair comparison needs a refit at 20 links. Training currently requires the 10-link mesh (check_model, env n == 11).

**Why:** the user wants the drone and cable models final before judging the mesh, so the comparison isn't distorted by drone-model error that is still changing.

**How to apply:** don't propose a 20-link refit or mesh changes until fitting is declared done. When it is, raise this comparison (a 20-link refit run through the Model Fit window, see [[run-long-jobs-in-the-ui]]) before the first real training run.
