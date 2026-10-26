> Historical pre-cleanup snapshot. For the supported implementation, see [current Stage 1](../stage1.md). Legacy-mode compatibility mentioned below was subsequently removed with approval.

# Four-second Stage1 restart — detailed implementation report

Date:2026-10-25. User explicitly requested4seconds,200collections,restart.

## Result

Fresh random actor AND critic;64parallel environments,200collections,12800episodes.
No checkpoint initialization/resume. Previous seven-second v5run stopped at
collection11/704saved episodes; unfinished collection12and all saved artifacts retained.
Same random seed20261028, same10-link simulator,50Hzcontrol/600Hzphysics and+Xoutput rule.
Entropy restarted as.001+.019exp(-8n/12800); near floor around collection150.

## Exact behavioral changes

- Episode horizon7→4seconds.200control decisions instead of350;400reward observations
  and401recorded states includingreset instead of700/701.
- Keep the same instantaneous +X fold-supported reward rate10*F*H²*A²*U/L.
  Integrate only over the new4second horizon. No new reward bonus or success threshold.
- Workspace cost denominator700→400. It remains10times the fixed-horizon mean
  squared excess, with unchanged radius.55m and height1.5–2.6m.
- Numerical abort cost becomes4−tau instead of7−tau. Tilt/cable-height breaches still
  do not end finite/admissible simulation; they remain physical-invalidity diagnostics.
- Normalize time and fold/output-duration critic features by4, and give the policy
  a401sample history/time reference400. This prevents a nominal4second run from
  retaining stale7second temporal features or history metadata.
- UI full-horizon labels no longer hardcode7seconds. Completion and physical validity
  are displayed separately, including complete-but-physically-invalid episodes.

For identical constant output RATE, four seconds accrues4/7of the seven-second Q;
for identical average squared boundary excess, C stays the same. Consequently this
is a shorter task with unchanged mean workspace penalty, not an invariant return-scale
transformation. No additional weight adjustment was made.

## File-by-file implementation and purpose

- `tasks/whip/continuous_whip.py`: new SHORT_MODE/SHORT_CONTRACT(v6,horizon4). The
  attempt accepts a restricted4or7second horizon;4requires directed plain-workspace
  mode. Deadline, workspace denominator, numerical remaining-time cost, features and
  recorded version use that horizon. Default7second legacy behavior retained.
- `tasks/whip/continuous_whip_environment.py`: recognize v6, pass its validated horizon
  to each lane, retain full-horizon physical diagnostic behavior, preserve actual mode
  in event records. Reward accounting and native control flow otherwise unchanged.
- `tasks/whip/horizontal_fold_environment.py`: allow only the registered v6layout
  (11nodes/401states/200decisions); all other supported fold modes still require
  11nodes/701states/350decisions. No physics/solver change.
- `tasks/whip/stage1_configuration.py`, `training/configuration.py`: register exact
  v6contract and horizon. Wrong episode duration, contract horizon or direction rejected.
- `tasks/whip/sac_environment.py`: register continuous dispatch/critic layout; select401
  encoder samples for v6only. Existing time_samples already follows physical horizon.
- `training/ppo.py`: explicit401history/time-reference400 interface for v6; reject
  a701history passed to v6. Existing7second interface remains supported. No PPOloss,
  optimizer or exploration formula change.
- `tasks/whip/learned_recovery.py`, `training/worker.py`: register existing scorer and
  continuous evidence/reporting paths for v6; no extra stage/recovery behavior added.
- `training/artifacts.py`: extend the directed fresh-only initialization and later-stage
  guard to v6. Exact-resume source/identity checks are unchanged.
- `training/ui/window.py`: register v6with+Xsetup and dynamic duration display; generic
  full-horizon labels and separate physical-invalid completion label.
- `training/ui/replay_window.py`: register continuous v6replay; remove hardcoded7s and
  distinguish completed-but-invalid motion from a genuinely aborted prefix.
- `training/ui/whip_diagnostics_data.py`, `training/ui/whip_diagnostics.py`: apply existing
  fixed+Xreach/velocity plots to v6 and remove a v5-only notice label. No fold changes.
- New task/preset: `configs/tasks/whip_stage1_continuous_directional_four_second_v6.json`
  and `configs/learning/whip_stage1_ppo_continuous_directional_four_second_v6.json`.
  Previous presets untouched.64lanes,200collections,seed20261028,fresh12800episode clock.
- `tests/test_continuous_four_second.py`: four tests cover exact full horizon/history,
  learner clock,physical violation continuation,400sample mean cost/integrated output,
  numerical abort arithmetic,wrong-config rejection and retained7sconfiguration.
- `doc/current_implementation.md`: current status/preset/contract and dated evidence updated.
  Dated local monitor script points only at this run and opens the visible diagnostics UI.

## Verification

57focused CPU/unit/UIchecks passed (`tests-final.log`), including new four-second
checks and previous direction/accounting/entropy/replay regressions.14UI/replaychecks
rerun and passed after the final label correction (`ui-final.log`). No tolerances or
scientific criteria weakened. Constructed adapter cases are not physical validation.
All44simulator/controller/native source hashes match the stopped v5run.

Production startup confirmed collection1with64active lanes. Round0checkpoint retained:
zero episodes/updates,empty optimizers,null resume/initializer,entropy budget12800/offset0,
401history/time reference400 and matching execution hashes. UI visible/exposed/active,
correct run,Whipping diagnostics tab. See launch.json and initial-checkpoint-check.json.
No extra GPUpreflight,consistent with the previous waiver. Full first-collection results
and speedup were not yet measured at this startup check. No learned-success claim.

## Limits

Four seconds may constrain preparation/repeated whipping; learning effectiveness is
unverified. Shorter simulation should reduce work but no wall-clock speedup is promised.
This change does not optimize the existing per-lane CPUfold tracking or host/device
synchronization. Full duration does not establish valid whipping. No contact model was
added; finite below-ground motion can continue and remains physically invalid.

Active run: `/home/tony-5090/PycharmProjects/AeroWhip_deploy/runs/ppo_stage1/20261025-215750-176045`. WorkerPID29757;UIPID29758.
