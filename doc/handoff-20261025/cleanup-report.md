# Current-code cleanup and handoff — 2026-10-25

## Scope and reason

The user stopped training, requested a portable committed/pushed project, asked for a
clear current Stage 1 specification, and explicitly selected **“Keep only current
training code; old-model replay may break.”** This cleanup removes the retired
training implementations. It does not redesign the current experiment or promote
any saved policy.

The supported task is now [documented in one place](../stage1.md): four seconds,
world +X, ten links, 64 lanes, 200 collections, fresh PPO actor/critic, budget-based
exponential entropy. The stopped production run completed zero collections and
zero optimizer updates. Temporary software checks described below are separate
from that run and are not research training results.

## Exact implementation changes and why

| Area | Change | Why |
|---|---|---|
| Learning/task presets | Retained only the current `whip_stage1_ppo_continuous_directional_four_second_v6.json` learning/task pair. Removed older Stage 1 variants and Stage 2–4 presets. | Prevent accidental selection of superseded objectives. |
| Configuration | `training/configuration.py` now accepts only current Stage 1 PPO, its approved control/reward/entropy contract, and the four-second task. Current compute/budget settings remain explicit. | Unsupported historical modes must fail clearly instead of entering partially removed code. |
| Rewards | Removed excitation, shape-release/delivery, fold-formation/release, unified-fold, retained-delivery, movement, first-hit, flexible-delivery, recovery, strike and target-sampling modules, plus the isolated retired offline Stage 1 candidate scorer. | These objectives are no longer the selected research task. |
| Continuous task | Specialized `continuous_whip.py` and its environment to v6: directed +X, plain Q−C return, four-second horizon. Removed older radial, transformed-return and fixed-failure-total branches. | Preserve the selected arithmetic while eliminating selectable historical alternatives. |
| Geometry and records | Kept `FoldGeometry`, fold-region extraction/tracking, continuous support, shared observations, native stepping, replay storage and `Stage1Score`; removed the old one-attempt reward/event controller. | Current fold support depends on this geometry, but not on the retired event task. |
| Shared environment | Kept `tasks/whip/sac_environment.py` as the shared observation/native-buffer implementation and current factory; removed its old SAC/task/recovery behavior. Its historical filename remains. | Avoid an unrelated import-path migration while retaining code used by current PPO. |
| PPO networks | Moved the unchanged history encoder, actor and scalar-metric helper from `training/sac.py` into `training/networks.py`; PPO imports that module. Deleted SAC. | Current PPO should not depend on an obsolete learner module. |
| PPO | Removed frozen-recovery actors, retired stage interfaces/actor-transfer helpers and obsolete entropy schedules. Kept AR(1) conditional densities, current encoder, budget-based exponential entropy, KL safeguards, optimizer rollback and checkpoint clocks. | Keep the implemented learner needed by the current experiment. |
| Worker and artifacts | Removed SAC dispatch, curriculum/plateau/demonstration paths and stage/actor handoffs. Kept fresh initialization, exact resume, source hashes, RNG/optimizer clocks, all-lane evidence, deterministic evaluation and cooperative stop/save. | Make the supported execution path explicit without relaxing continuation checks. |
| Evaluation | Frozen evaluation now uses the same current native environment and PPO actor. Removed recovery protocols and old target samplers. The manual tester states that target coordinates are metadata, not a Stage 1 goal. | Evaluation must measure the actual current task and avoid suggesting target adaptation. |
| Cost reporting | Renamed the top-level `acceleration_cost` log field to `workspace_cost` in the current training/evaluation workers. The component and workspace ledgers already contained the correct cost. | The returned quantity is workspace cost; Stage 1 has no acceleration penalty. No numerical cost changed. |
| Training UI / CLI | Exposed only Stage 1 and fresh/exact-resume modes. Removed actor-transfer entry points, inactive target/matching/mechanics settings and legacy reward plots. Current plots show output credit, workspace/numerical-abort costs, fold support, full-horizon completion and physical/numerical validity. | Users should see controls and diagnostics that actually apply. |
| Historical models | Removed bundled `models/milestone-20261009` and its milestone document. | They are not selected current models; they remain recoverable from Git. |
| Documentation | Added `doc/stage1.md`; replaced superseded training history in `current_implementation.md` with the current contract and retained physical/fitting notes; corrected README commands and handoff instructions. | Separate current implementation from historical research discussion. |
| Tests | Removed tests dedicated to removed objectives/SAC/stages. Updated surviving fixtures to current v6 and retained directional counterexamples, accounting, physical/numerical checks, AR(1)/PPO and UI/history tests. | Tests should cover supported features; a deleted feature's tests are not silently presented as current coverage. |

The current task JSON still carries some compatibility/display metadata (`request`,
`strike_surface`, `matching`, `mechanics`). It does not enable target-hit reward,
legacy scoring or additional training stages. Shared recorded-motion measurement,
fitting, simulation, deployment and viewer utilities were retained because they
still serve the project. This was not an indiscriminate deletion of every unused
helper or every historical document.

The complete changed/deleted path inventory is in
[cleanup-source-inventory.json](cleanup-source-inventory.json). The full previous
source and historical models are preserved at
[commit 3c9a49e](https://github.com/TongshuWu1/AeroWhip_deploy/tree/3c9a49e).

## What did not change

The continuous reward is still `rate = 10 F H² A² U/L`, integrated at 100 Hz,
minus the fixed-horizon average squared workspace excess. Fold geometry,
thresholds/windows, output alignment, physical diagnostics, numerical-abort
remaining-time cost, four-second deadline, ten-link model, control rates and
production exploration schedule remain unchanged. There is no new success gate,
contact model, reward tuning, training launch or automatic stage promotion.

Git comparison against the preservation commit shows no changes in `simulation/`,
`native/`, `controllers/`, `fitting/`, the physical model JSON or scene files.
This is source-preservation evidence, not physical validation.

## Runs, recordings and recovery

Before pruning, unique experiment data were packaged into 18 independent gzip tar
archives. All 10,706 files were read back and SHA-256 checked locally. The release's
22 assets (18 archives plus four manifests/checksum files) were then checked against
GitHub's SHA-256 digests and byte sizes: **8,545,338,190 bytes**, all matching.

[Published data release](https://github.com/TongshuWu1/AeroWhip_deploy/releases/tag/handoff-2026-10-25)

Only then were **10,676 files / 15,028,642,560 bytes** removed from old local `runs/`
and `tmp/` directories. Each deleted file matched the archived per-file hash and
size; no changed file was deleted. Nineteen obsolete stop-request markers were
also removed. Earlier duplicate/cached-file cleanup removed 2,859,528,247 bytes;
old snapshot duplicates were checked against the existing October 16 release.

Retained locally:

- `runs/ppo_stage1/20261025-215750-176045` and its stopped initial checkpoint;
- `tmp/stage1-directed-four-second-v6-20261025` implementation evidence;
- source recordings, fitting inputs and project output data;
- the compressed backup and detailed deletion manifests outside the checkout at
  `/home/tony-5090/PycharmProjects/AeroWhip-backup-20261025`;
- downloaded third-party papers outside the public upload tree.

A normal clone contains current source and documentation, not the large run data.
Use the [handoff instructions](README.md) to restore data. Restore historical
models/code into a **separate checkout of `3c9a49e`** when needed; historical
checkpoint execution is deliberately not guaranteed in the pruned checkout.
Exact-resume source/binary guards remain in force. The current zero-update
production checkpoint does not warrant bypassing those guards: start fresh.

## Verification and limitations

The final verification summary and logs are retained beside this report. Checks
include the current CPU unit suite, native before/after comparison, short worker
integration, frozen deterministic replay, UI entry-point construction, and static
local-import scanning. See [cleanup-verification.json](cleanup-verification.json)
for exact outcomes and [cleanup-equivalence.json](cleanup-equivalence.json) for
all 28 compared arrays.

Before pruning, the historical suite ran 570 tests with 10 skips. The pruned suite
runs 229 tests with 2 skips; the reduction includes retirement of old-feature tests,
not an improvement in scientific validity. Twelve additional tests for the isolated retired offline candidate scorer were
removed with that unused module during the final reference scan. Original intermediate failure logs
remain outside the checkout. No scientific acceptance threshold or numerical
tolerance was relaxed to make this cleanup pass.

The two-lane native prefix comparison is only 0.4 seconds under prescribed actions.
The worker smoke check uses two CPU lanes and one collection, not the production
64-lane GPU budget. It verifies collection, reward accounting, optimizer/checkpoint
execution and replay, not learning quality. No full CUDA campaign, another-computer
installation, robust policy evaluation, physical flight or successful-whip claim
was tested. Training remains stopped after cleanup.
