# Current implementation

Last reviewed: 2026-10-25.

Current status (2026-10-25): **Training stopped for other-computer handoff.**
Latest run20261025-215750-176045 stopped during its first collection:0completed
collections/episodes,0gradientupdates. Its checkpoint is an untrained initialization.
The current default is the approved four-second directed+X Stage1v6 preset:
64environments,200collections/12800episodes,10links,fresh actor/critic,entropy
.001+.019exp(−8n/12800). There is no automatic restart or Stage2 advancement.

[Other-computer instructions and archive restoration](handoff-20261025/README.md),
[tracked exact contract](handoff-20261025/stage1-contract.md),
[tracked implementation report](handoff-20261025/stage1-implementation-report.md).
Large historical runs/evidence are preserved in the handoff release rather than Git.
Old local-artifact links below require restoring the data release.

Cleanup removed verified old-backup duplicates and generated caches; retained all
unique experiment evidence in the archive. Corrected the stale default(v4→approvedv6)
and historical tests that accidentally depended on changing defaults. CPU suite:
570tests,10skipped; original failure log and corrected pass log retained. No reward,
physics,optimizer or scientific threshold changed during this cleanup. Source/hash
checks remain enforced, so another-machine historical exact resume is not promised;
start v6fresh because it has no completed learning.

Historical source v3run20261025-163957-732295 remains stopped; its old2.18second
replay is unchanged.

Stopped at22completed v3collections/2816episodes; collection23interrupted.
Latest/best checkpoints preserved; worker process and lock absent. The campaign
had47completed collections including25beforev3 at handoff. Historical v4continued from that clock; fresh v5does not.
Collections12–22had only1/1408full horizons and tiny raw whip output despite
ongoing PPO updates. This is failed behavior, not successful convergence.
See [stop evidence](../tmp/stage1-continuous-distance-20261025/user-stop-and-diagnosis.json)
and [failure-incentive analysis](../tmp/stage1-continuous-distance-20261025/failure-incentive-analysis.md).

**Current task:**
[Stage1 directed +X four-second v6](../configs/learning/whip_stage1_ppo_continuous_directional_four_second_v6.json).
Earlier seven-second descriptions below are historical; the v6 contract above defines the active horizon and normalization.
User explicitly requested removing physical-failure early termination. V4runs
finite/admissible trajectories7s despite tilt>60degrees or cable height<0.05m.
Numerical corruption still aborts. Q−C, boundary dimensions/weights, fold rules
and noise schedule are unchanged. Physical violations have no extra penalty;
low reward for every violation is NOT guaranteed. No ground-contact model added.
Completion, numerical validity and original physical-envelope validity are
reported separately; full7s with a violation is still physically invalid.
The original v4launch used128lanes and153remaining collections, with actor-only
transfer and fresh optimizers. It was followed by a64-lane full-state
continuation, now stopped and superseded by fresh directed v5. Critic inputs remain219. Stopped v3and earlier
v4artifacts remain unchanged. See the
[contract](../tmp/stage1-continuous-full-horizon-20261025/contract.md) and
[file-by-file report](../tmp/stage1-continuous-full-horizon-20261025/implementation-report.md).
63focused software tests passed; constructed adapter states verify termination
and reporting, not physical realism. All44physical-source hashes unchanged.

The following describes the **stopped historical v3**, not the new v4stopping rules.

V3 keeps the continuous horizontal fold-transport/straightening output Q unchanged.
Boundary cost C is10times the fixed7s mean of squared radial/vertical excess in
metres at the drone tracking point (radius0.55m, world height1.5–2.6m).
Return is directly Q−C, without saturation or nonlinear compression. Workspace
crossing and whip output do not terminate episodes; usable simulation runs7s.
Physical/numerical abort at time tau preserves Q_prefix−C_prefix and subtracts
1*(7−tau). There is no fixed failure total, elapsed-time cost, effort/path cost,
target reward or event-completion stop. Physical tilt/cable-height and numerical
guards remain unchanged. The remaining-time rate is provisional and cannot
make failure worse than every possible unbounded-cost valid trajectory.

The stopped v2 run20261025-161103-645210 contributed3collections/384episodes,
after v1 run20261025-153330-415723 contributed22collections/2816episodes.
Both remain preserved and unapproved. V2 used F(Q−C), with F(x)=5x/(5−x) for
negative x and identity otherwise, and fixed−10 failure totals. Those rules
are **historical v2 behavior, not active v3**. Raw cost around4015 was compressed
near−5; this motivated restoring direct distance costs.

V3 transfers the stopped final v2 actor (encoder/mean/learned scales) with fresh
critic/optimizers/RNG. This is not exact optimizer resume.175collections remain
at128lanes (22400new episodes), completing the original200collection campaign.
Entropy retains N=25600, offset3200:0.001+0.019*exp(-8*(3200+n)/25600).
No noise reset, physical/control change, automatic Stage2 or budget extension.
Saved accounting separates Q, C, failure cost, net return and validity.
58focused software tests passed; additional GPU preflight was waived, and these
checks do not establish learned whipping. Production launch evidence is retained
under tmp/stage1-continuous-distance-20261025.

The prior retained-progress Stage2 run
[runs/ppo_stage2/20261025-133812-161285](../runs/ppo_stage2/20261025-133812-161285)
remains stopped at53collections/6784episodes with all artifacts preserved.
Its reward and termination below are **historical, explicit-preset behavior**,
not the active new Stage1 objective. See its
[implementation report](../tmp/stage2-retained-progress-implementation-20261025/implementation-report.md)
and the [diagnostics UI report](../tmp/whipping-diagnostics-ui-20261025/implementation-report.md).

The new detector is [retained_fold_delivery.py](../tasks/whip/retained_fold_delivery.py),
registered as `retained_fold_delivery_v1`. It reuses the existing ten-link
material-coordinate FoldGeometry and output-frame calculations. Formation
requires a proximal coherent150degree fold for three100Hz observations;
propagation requires the same unique track to travel0.20L and reach at least0.65L;
opening requires the frozen downstream turn sum at most60degrees for three
observations. The output sample must satisfy the existing world and
attachment-relative radial-speed/cone rules. These are experimental engineering
criteria, not a validated universal definition of physical whipping.

The episode retains the maximum coherent sequential event progress `P` in[0,1].
Formation, propagation, opening and directed outreach occupy four equal bands.
Only increases pay `5*delta(P)`; losing/replacing a candidate never replenishes
the episode credit budget. Formation/opening guides include persistence;
propagation reaches full credit only when both existing propagation gates hold.
Opening uses a fixed150-to60degree reference rather than a running peak.
At one eligible sample of the same opening, delivery guidance is
`H^2 * exp(-0.5*(distance_to_rectangle/0.7m)^2) * A^2`, with A the smaller
positive +X cosine of world and attachment-relative tip velocity.

A separately verified useful hit totals15 (complete remaining progress to1,
then add10). A completed miss or safe7s timeout returns `5P-2`. A failure returns
`5P_valid_prefix-10`; failed endpoints cannot earn progress. No time, acceleration,
travel-distance or extra soft-workspace penalty is applied. Hence every valid
unsuccessful attempt outranks every failed attempt, but expected-return
optimization still need not maximize hit probability. High partial progress is
not a hit or proof of successful whipping.

The fixed world target is the +X plane centered[1.1,0,2]m, width1m and height0.6m.
A useful hit must be a confirmed forward rectangle crossing in the same opening,
with H>=0.75, the existing outward velocity conditions and world vx>=1m/s.
Position, velocity, reach and opening geometry are checked at the same linearly
interpolated100Hz crossing state. Touching/starting on the plane does not qualify.
Pending crossings require opening confirmation and no intervening failure.
Safety has priority; an unconfirmed crossing is never promoted to a hit.

The [new adapter](../tasks/whip/retained_fold_delivery_environment.py) reuses the
existing native substep guards and per-lane freezing, without changes to forces,
solver, resolution or controller. Ten links/eleven nodes,50Hz bounded world
acceleration ([8,8,4]m/s²),100Hz observations and seven-second maximum remain.
The drone tracking point must remain within radius0.55m of reset XY and world
height1.5–2.6m. Existing60degree tilt and0.05m cable-height guards remain. The cable
may extend outside the drone cylinder. The critic gets24 task/progress features;
the actor retains its701-observation history. Gamma remains1, with terminal
bootstrap stopped at actual task endings.

The [new Stage2 preset](../configs/learning/whip_stage2_ppo_retained_fold_delivery_v1.json)
selects a fresh random actor and critic, seed20261026,128 parallel lanes and
25,600episodes/200collections. This task alone permits fresh Stage2 initialization;
older Stage2 actor-transfer guards remain. No old failed actor is promoted.
Entropy remains `0.001+0.019*exp(-8*n/N)`, N=25,600 saved on creation, n=completed
training episodes; at collection150 it is0.001047096291. Exact resume preserves
N and n.100ms AR(1) exploration and existing PPO settings remain. No automatic
later stage, extension or hardware execution is included.

Collection evidence retains every attempted trajectory, validity and failure
reason, formation/propagation/opening, progress credit, terminal outcome credit
and actual same-opening hit. Representative motions include low/median/high
returns and available hit/miss/failure examples. The UI displays progress and
actual hit rates separately, and replay labels do not treat progress as success.
A visible monitor attaches to one worker, supports cooperative Stop/Save and
cannot launch a duplicate; closing that monitor leaves its worker running.

The prior Stage1 run
[runs/ppo_stage1/20261025-122524-250328](../runs/ppo_stage1/20261025-122524-250328)
remains stopped after81collections/10,368episodes; its unfinished collection was
discarded and its heartbeat deleted. Its reward was5H²−0.1T for completion,
−2.7for timeout and−10−0.1T for failure, with terminal-cancelled potential shaping.
All87 files matching its saved execution hashes were archived before this change
in [the retained source archive](../tmp/stage2-retained-progress-implementation-20261025/stopped-source-archive.json).
At70collections all8,960attempts were invalid with no folds; saved motions showed
lower-boundary failures near0.56s. The early-failure incentive is mathematically
present, but is not proven to be the sole cause of failed learning. Earlier
native/preflight checks were execution evidence, not task feasibility. The
historical strict2e-5 re-forward-density failure remains a failure.

The earlier [horizontal-fold contract v1](../tmp/stage1-horizontal-fold-contract-20261025/proposal.md)
is preserved as the audited proposal history, **not implemented or approved for training**.
It proposes a coherent150-degree proximal fold, unique outward tracking by0.20L
to at least0.65L, then fixed-downstream-support opening below60degrees; completed
return `5*(horizontal attachment-relative reach/L)^2 -0.1*T`, timeout `-2.7`,
and task/physical failure `-10-0.1*T`. Bounded potential-difference guidance
cancels at every terminal state. Hard drone containment in that historical proposal is now implemented only in
the approved v2 task described above, not retroactively in earlier experiments.
Coherence/angle/event thresholds are engineering hypotheses, not paper-derived
or validated fold definitions. Worked return and shaping-accounting arithmetic
were checked. The subsequent [contract/root-bend audit](../tmp/stage1-contract-root-audit-20261025/review.md)
applied an offline interpretation to six saved cases: Stage4 geometry completes
at2.59s while horizontal reach is still increasing, but a workspace breach at1.41s
already invalidates it as a task attempt; five controls do not complete.
The [velocity follow-up](../tmp/stage1-contract-root-audit-20261025/outward-velocity.json)
also shows that larger reach at2.62s is not better outward delivery: world tip
radial velocity is -0.663m/s while attachment-relative radial velocity remains
+0.165m/s. The user requested reach credit during outward tip motion. Checking
both world and attachment-relative horizontal radial velocity is proposed;
exact eligibility tolerances and event stopping rules remain unimplemented.
The stopping rule needs review; native execution, contained feasibility and
learnability remain untested. Raw saved hover is straight; moving recordings
contain root/first-interior bending consistent with the implemented point-clamp
response, whose hardware accuracy is not established here. This project's
physics, training and fold evaluation remain **10 links / 11 nodes**. The20-link
model belongs exclusively to the separate standalone simulation and is outside
this task's scope. Its measurements were unnecessarily included in the audit;
they are retained only as historical evidence, not a basis for project changes.
No preset, production reward, termination or worker changed.
The subsequent [four-recording physics-trace review](../tmp/bend-tracking-review-20261025/review.md)
recommends against the first-arrival-plus-0.20 s stopping rule: it can terminate
while the prescribed excitation is still active. This offline review does not
establish a replacement stopping rule or approve a new reward.

The subsequent [continuous bending-region comparison](../tmp/continuous-bend-review-20261025/review.md)
is implemented only as an offline diagnostic. It compares natural cubic centreline
curvature with a continuous field of measured turning angles and local region
centres/widths. The controlled travelling-bend example loses the node staircase,
but saved physical traces still have ambiguous associations and bandwidth
sensitivity. A constructed U-turn and S-bend have identical unsigned turning
profiles despite different end-tangent directions: **bending regions are not
fold detections**. No fold threshold, reward, event termination or training change
has been adopted. Nine new geometry checks and 22 existing measurement checks
passed; these are not physical-whip validation. Source recordings are preserved.

An [original-source recheck](../tmp/whip-mechanics-research-20261025/research-note.md#original-source-attribution-audit--2026-10-25)
distinguishes the human paper's observed marker-speed cascade from the older
planar travelling-loop model. Neither supplies our fold detector or stopping
rule. The ideal full loop can have parallel boundary tangents after a complete
tangent rotation, so the earlier U/S example must not be generalized into a
universal opposing-end-tangents gate. Full local tangent evolution is now an
[implemented offline diagnostic](../tmp/tangent-evolution-review-20261025/review.md),
not an adopted fold criterion. It retains actual segment tangents, unsigned
turning, separately labelled signed XZ/XY projections with singular/ambiguous
steps broken, and local summaries on the existing bending basins. Whole-cable
world and attachment-relative speeds are synchronized with geometry. Constructed
U/S/loop and rigid-motion controls expose limits of unsigned bending, endpoint
reversal and speed alone. Ten new numerical checks, nine existing region checks
and 22 measurement checks passed. Four original recordings were reanalysed with
their source/snapshot hashes checked; no physics or training was rerun. No
classification threshold, reward or termination was changed. The synchronized shape/bending/speed
animation format is retained as the user's preference; its historical red window
is not an accepted paper-derived event boundary.

The same diagnostics have also been [applied to the preserved old Stage4 final
replay](../tmp/stage4-tangent-review-20261025/review.md), including a full recovery
animation and a preparation-to-strike crop. Near 2.30–2.57 s the cable doubles
back, a bending region progresses outward and opens while tip speed rises.
This is a candidate unfolding example, not a validated fold classification:
association ambiguity and boundary effects remain. The historical replay's
workspace breaches remain reported. No fresh simulation or policy evaluation,
reward change, fold threshold or model promotion was performed.

The user subsequently selected Stage4's 2.20–2.58 s interval as a **positive
visual fold/unfolding reference**. The [retained reference and counterexamples](../tmp/fold-reference-set-20261025/review.md)
record that annotation alongside static U/S/arc shapes, a rigidly moved loop,
a constructed travelling packet and an unresolved saved stroke. This supports
the proposed formation → outward propagation → opening objective; it does not
adopt a detector, numerical threshold, release endpoint or reward. Existing
descriptors expose static/rigid counterexamples, while coherent 3D fold identity
and tracking through region changes remain unresolved. The source policy is not
promoted; workspace breaches and association flags remain retained.
It is a fold-motion reference, **not a reference for adequate horizontal
outreach**. At its historical interpolated strike, the tip is about0.532 m
forward and0.762 m below the attachment (0.953 m rest cable length), and about
0.801 m below the drone. Those kinematic values were recomputed from the saved
replay at the saved crossing time; they do not validate energy transfer or
constitute a new success threshold. The user selected horizontal outreach to
address this downward-oriented delivery without prescribing the preparation.

Current research status, reassessed2026-10-22: Stage1 is **not accepted as a
successful whipping skill**. The earlier proxy-based advancement conclusion has
been withdrawn following the user’s rejection and Stage1-specific audit. Original
numeric results/reports remain preserved. Stage2 also failed its delivery gates;
there is no accepted two-stage policy from those runs. See
[Stage1 reassessment](../tmp/stage1-reassessment-20261022/reassessment.json).

Current measurement work,2026-10-22: the user approved implementing separate
excitation, distal-response and delivery evaluation after the paper review.
The [whole-attempt evaluator](../evaluation/whip_attempt_review.py) and
[retained comparison](../tmp/whip-attempt-evaluation-20261022/review/index.html)
are implemented and checked offline. They use explicit common attempt bounds,
full traces and separately attributed release intervals; they introduce no
physical-whip classifier, reward, stage transition or accepted policy.
The earlier [paper comparison](../tmp/whip-paper-validation-20261022/index.html)
retains author-released human aggregate data and diagnostic examples. Sidearm
and overhead preparation are both eligible; target direction is assessed
separately. The retained side-swing case's original rejection is historical,
and no saved policy is automatically promoted.

Current curriculum direction,2026-10-24: the user approved simplifying the four
stages to excitation, fixed-direction broad delivery, varied target/full-velocity
commands with precision, then recovery. The [fixed-direction Stage2 proposal](../tmp/stage2-curriculum-redesign-20261024/fixed-direction-v4.md)
retains one +X rectangle throughout training, with no distance ladder, target
sampler or architecture change. The user approved Stage2 implementation and launch subject to preflight. The
separate flexible_delivery_v1 scorer, fixed preset, transfer guard and reporting
are implemented; native preflight is complete and supports the bounded learning
experiment. The original run [20261024-205138-342045](../runs/ppo_stage2/20261024-205138-342045/)
stopped cleanly at60collections/7,680episodes for the approved first-hit revision.
The new first_hit_flexible_delivery_v2 default is implemented; native preflight passed; the remaining60-collection actor-transfer run
[20261024-233827-625394](../runs/ppo_stage2/20261024-233827-625394/) is active after exact resume.
See the latest dated entry. The prior distance
ladder and multi-direction Stage2 redesign are superseded; full desired velocity
conditioning is deferred to Stage3. The user approved a
fresh Stage1 restart with distance-sensitive workspace cost and a nonlinear net
return. The default [v3 preset](../configs/learning/whip_stage1_ppo_flexible_excitation_rational_ar1_v3.json)
retains flexible_excitation_v1 Q and the original uncapped quadratic workspace
cost C. Valid return is F(Q-C); failure is -10+F(-C_accrued), with F(x)=x for
x>=0 and 5x/(5-x) otherwise. Causal rewards are differences of transformed
running net score, with terminal failure correction. Raw Q, raw C, raw net,
compression adjustment and actual return are stored separately. This is an
experimental objective, not physical energy or validated whipping success.
Sidearm motion remains eligible, with no fold/direction/target/path/effort cost.

The prior v1 unbounded-cost run stopped at39collections; its cost made some
valid flights rank below failures. The v2 capped-density run stopped at11 and
its exact continuation [20261022-195928-651339](../runs/ppo_stage1/20261022-195928-651339/)
completed120total collections. The subsequent climb audit found peak excitation
near1.73s followed by unrewarded ascent to14.6m: the density cap removed added
distance cost far outside the workspace. Neither run is promoted. All original
artifacts and presets remain preserved.

The v3 correction is implemented with100unique passing CPU checks. Independent
native accounting and optimizer integration are complete. Fresh run
[20261022-221619-402166](../runs/ppo_stage1/20261022-221619-402166/)
stopped at24collections/3,072episodes on2026-10-22. On2026-10-24 the user
requested restarting; exact continuation
[20261024-160428-146323](../runs/ppo_stage1/20261024-160428-146323/)
completed the remaining96collections on2026-10-24, reaching120total collections
and15,360episodes with unchanged settings and matching execution hashes. Frozen
final/best evaluation and motion review are complete:381/384independent
policy-noise trials were valid and positive-return, but only1/384valid trials
stayed entirely inside the soft workspace. Excitation is repeatable under this
proxy; containment, recovery and aimed delivery remain incomplete. No policy is
automatically promoted. See [completed review](../tmp/stage1-rational-review-20261024/review.md). Round0
actor/critic of the original fresh run matched seed5weights
bit-for-bit; both optimizers are empty, no initializer/resume is configured,
and entropy starts at0.02with zero episode offset and15360-episode budget.
Exactly one training worker was observed. See
[launch verification](../tmp/stage1-distance-sensitive-restart-20261022/verified-startup/launch-verification.json). The strict float32 reduction check remains
failed by1.91e-6 against1e-6, with exact stored vectors and float64 accounting
verified separately; see the dated preflight note below. User approval
covers fresh random actor AND critic,128lanes/120collections/15,360episodes;
physics, workspace dimensions, control, AR1 noise and exponential entropy stay
unchanged. No automatic Stage2 transition. See
[proposal and offline checks](../tmp/stage1-distance-sensitive-proposal-20261022/proposal-and-checks.json)
and [restart evidence](../tmp/stage1-distance-sensitive-restart-20261022/).
Negative compression ensures valid returns >-5 and failed returns <=-10 in real
arithmetic, but weakens differences between extreme excursions and changes
expected-return preferences across stochastic policies. It does not guarantee
containment, learning, or a physically useful whip. Historical failed precision
checks remain retained and are not relabeled by this restart.

The preceding unified-fold campaign is now historical. Original fresh run
[20261022-121145-434399](../runs/ppo_stage1/20261022-121145-434399/)
stopped after81collections/10368episodes for the approved exponential entropy
continuation. Continuation [20261022-131442-557003](../runs/ppo_stage1/20261022-131442-557003/)
completed the original120-collection/15360-episode budget. Its summary and both
runs are preserved. Completion does not establish whipping success or validate
the changed excitation reward. The new campaign starts from random actor and
critic, without inheriting those policies. Earlier [schedule evidence](../tmp/stage1-exponential-20261022/contract.json)
and [unified-fold launch evidence](../tmp/stage1-unified-20261022/contract.json)
remain retained.

The user subsequently requested improving the measure and completely restarting
Stage1. Fresh-run settings and prior execution sources are preserved in
[restart preparation](../tmp/stage1-restart-20261022/restart-preparation.json).
The user selected an outward-traveling fold that opens at the tip, as for a far
strike. The experimental fold-release prototype has not passed its native
coverage preflight:0 qualifying valid releases in256 new no-update trials.
The user previously approved a fold-formation warm-up followed by release learning within
120 total collections. The new warm-up and aligned release presets are now
implemented; native preflight passed and fresh formation run
[20261022-105933-962042](../runs/ppo_stage1/20261022-105933-962042/) stopped
at its scheduled30-collection review boundary. Frozen final and saved-best
evaluations and motion review are complete: both failed the approved transition
criteria. Release training has not started; no extension or promotion is approved
by this review. See [formation review](../tmp/stage1-fold-curriculum-20261022/formation-review.json).
See [approved curriculum contract](../tmp/stage1-fold-curriculum-20261022/contract.json).
Following the user's approval to restore soft workspace boundaries, a separate
[soft-workspace formation v2 preset](../configs/learning/whip_stage1_ppo_fold_formation_soft_workspace_ar1_v2.json)
is implemented and checked offline, but **not launched or validated by training**.
It replaces the path cost with the original outside-only quadratic workspace
formula, using a provisional coefficient0.01. Retained trajectory checks verify
arithmetic only; they do not validate this coefficient or predict new-policy behavior.
The old presets and failed reviews remain unchanged; this is selectable with
an explicit settings path; it is not the new unified restart default.
This cost change does not resolve the whip-definition problem. Details and
limitations are retained in the [soft-workspace contract](../tmp/stage1-soft-workspace-20261022/contract.json).
This supersedes the earlier proposed-only restart status; the failed sparse
release preflight remains retained in [restart readiness](../tmp/stage1-fold-restart-20261022/readiness.json).

The deterministic side swing remains a hash-pinned offline example, now eligible
under the user's revised motion-family scope. Its earlier negative goal label is
historical. The preserved provisional release/focusing diagnostic rejects that
recording under its own rules, supports events in related policy trials and has
no independently reviewed physical positive set. It is **not an accepted
whipping measure or training reward**; see the historical
[offline challenge report](../tmp/whip-side-swing-validation-20261022/report.json).

This record describes the current simulation, learning pipeline and retained
milestone models. The external development archive
`AeroWhip-development-archive-20261009` remains separate from this checkout.
On2026-10-10 the user stopped the restored-radius Stage2 campaign and explicitly
requested deleting all project `runs/` contents and restarting Stage1.
All2,169 old run files were deleted after the worker stopped; the previous
training, motion-search, cloning and reward-review artifacts are no longer
available in `runs/`. Existing `models/`, `data/` and configurations were preserved.
Historical experiment narratives that relied on those deleted artifacts have
been removed from this active record; their prior text is retained only in
`tmp/stage1-clean-restart-20261010/current_implementation.before.md`.

The random-policy restart stopped after44 collections/5,632 episodes. Its
demonstration-seeded replacement stopped after121 collections/15,488 episodes
when nominal reach and height declined despite improved containment. Both runs
and the reconstructed demonstration are retained. The user approved a new
Stage1 reward with linear height credit and separate outreach/height/joint
contributions, transferring the stopped actor into a fresh critic and optimizers
with the existing exploration schedule restarted. The new experiment uses
150 collections (19,200 episodes),128 lanes and radius0.55 m. The first launch
was paused for simulation work before any completed collection or PPO update.
On2026-10-11 the user requested relaunching training. The
[Stage1 run](../runs/ppo_stage1/20261011-052639-788494/) completed all150 collections
with the paused actor, fresh critic/Adam/clocks and the same approved settings.
GPU scheduling source changed during the pause, so that launch was actor-only
initialization under current execution hashes, not exact resume. Its final native
evaluation improved reach and drone-relative height, but was not contained and
missed the strict crossing angle. The user requested continuing. A one-off
handoff-script error was corrected in a separate
[Stage2/3 continuation driver](../tmp/stage23-continuation-150-20261011/campaign.py),
preserving the failed driver and completed Stage1 artifacts. The
[Stage2 run](../runs/ppo_stage2/20261011-063417-108510/) and
[Stage3 run](../runs/ppo_stage3/20261011-073423-438409/) also completed their150
collections. Stage3 finished at target level1 (0.6×0.4 m), with no native final
0.1×0.1-m hits, and failed the existing Stage4 handoff. Completed budgets do
not imply that the precision curriculum was learned;
their existing Gaussian-height reward definitions remain unchanged. Each later
stage takes the preceding final actor with fresh critic, optimizers and
stage-local exploration clock. Stage4 training is not included. Preparation,
launch and completion records are separated below; deleted-run outcomes are not
presented as retained scientific evidence.

On2026-10-13 the user approved restarting Stage2 with actual-crossing priority:
`0.2W + max_crossing(C + 2G + S) - workspace_cost`, defined below. This separate
150-collection/128-lane experiment transfers the completed Stage2 final actor
with fresh critic, optimizers and exploration clock. The
[preflight](../tmp/stage2-crossing-restart-20261013/preflight.json) reproduced the
source's nominal native motion exactly and verified the new reward. Training
launch/status is recorded in the dated note below. It does not automatically
launch Stage3; a revised Stage3 target ladder remains proposed.

On2026-10-15 the user explicitly skipped the proposed control-learning pilot
and requested training the whip directly from scratch. The first acceleration
[Stage1 run](../runs/ppo_stage1/20261015-111659-848428/) completed all80 collections
(10,240 episodes) from a new random actor and critic, with fresh optimizers and
clocks. It used the unchanged compact Stage1 reward,128 lanes,50-Hz acceleration
actions and its existing plateau rule. The policy deteriorated: every episode
in collections29–80 was invalid, and the final native replay ended on the cable
height envelope. The [retained-results review](../tmp/stage1-acceleration-review-20261015/analysis.json)
records the evidence and reward-accounting diagnosis; completion is not learning
success. No training worker remains active at review time.
No demonstration, saved actor, resume, pilot or automatic later-stage launch is
part of this run. Launch, initial verification and completed-run diagnosis are
recorded separately below.

## Pipeline and entry points

[run_simulation.py](../run_simulation.py) runs the native physical plant.
[run_custom_simulation.py](../run_custom_simulation.py) runs that plant and the
flight viewer with the streaming Crazyswarm ROS adapter in the existing Docker
container. The Crazyswarm checkout's `./docker/start.sh --custom-sim` invokes this
host entry point; `--real` and stock `--sim` retain their existing launch paths.
[run_custom_simulation_test.py](../run_custom_simulation_test.py) sends the separate
scripted PVA connection test through the upstream streaming serializer and saves
its ROS feedback in the existing Crazyswarm `logs/` mount. These simulation entry points do
not initialize a learned actor or run a real-flight server.
[run_custom_real.py](../run_custom_real.py), invoked by Crazyswarm's
`./docker/start.sh --custom-real`, launches the stock real Crazyswarm/OptiTrack
stack on domain42 with RViz disabled and opens the shared read-only GPU viewer.
`--attach` opens the viewer for an existing real stack; Ctrl+C also stops its
matching local ROS launch. Display poses come
from the actual OptiTrack capture, not the onboard estimate or a simulated plant.
Its native cable-only simulation is driven by measured attachment motion and
displayed separately as simulated; it is not a measured/corrected cable state.
[run_training.py](../run_training.py) opens Training; `--start` explicitly starts
a run. The default and `--stage 1` now select the approved distance-sensitive flexible-
excitation v3 Stage1 preset, with120collections and exponential entropy decay. The
historical formation-only preset still stops at30 when explicitly selected.
`--stage 2/3` select acceleration variants of the existing reward recipes;
`--stage 4` retains velocity recovery and cannot accept an acceleration actor.
The fold formation→release curriculum is described in the current status and
dated note below; later-stage CLI selection does not imply scientific approval.
`--settings`
selects an explicit settings file, and `--initialize-actor` performs actor-only
transfer. Exact resume is a separate, source-checked path. Its UI entry point
now uses the read-only [resume-history display](../training_history.py),
reusing the existing Training window and charts. Reward-component charts preserve
missing keys and explicit JSON null values as gaps before applying the selected
run's reward scale; recorded numeric zeros remain zeros. This applies to both
the general component chart and the crossing-reward chart.
[run_replay.py](../run_replay.py) renders saved arrays without inference/physics.
Its existing saved-outcome browser remains the default; `--flight-view` opens
the selected saved arrays in the same docked flight viewer as local simulation.
`--run`/`--file` and `--replay` select the existing run/file and NPZ basename.
The manual policy tester remains available from Training.
[Take Processing](../take_processing/run_take_processing.py) manages project-local
recordings and fitting. [validate_stages.py](../validate_stages.py) runs the small
self-contained current-pipeline regression suite; `--output` optionally writes
its result. It does not launch a training campaign.

[evaluation/motion_search.py](../evaluation/motion_search.py), launched with
`python -m evaluation.motion_search`, is a separate nominal Stage1 diagnostic.
It searches scripted velocity strokes through the existing plant and scorer;
it does not train an actor or change default stage presets.
[evaluation/motion_refinement.py](../evaluation/motion_refinement.py), launched
with `python -m evaluation.motion_refinement --output <new-run-folder>`, refines
those programs using separate geometry/speed diagnostics. Its `--source` selects
a source search. The prior search datasets, including its default source, were
deleted in the requested cleanup; a new search is needed before refinement.

[evaluation/reward_review.py](../evaluation/reward_review.py) is the separate,
offline reward-redesign evaluator approved on2026-10-21. It reads pinned saved
trajectories, measures linked mechanics/crossing events, and writes candidate
return intervals and traces. It does not run physics, infer policies, train PPO,
change the active task, or advance a curriculum. Its candidate and limitations
are specified in the offline review section below.

[simulation](../simulation) and [native](../native) own reusable physics.
[controllers](../controllers), [tasks/whip](../tasks/whip), [training](../training),
and [fitting](../fitting) own their respective behavior. Settings resolve from
the checkout through [project_paths.py](../project_paths.py), not the caller's
working directory. [configs/simulation.json](../configs/simulation.json) selects
physics, geometric controller, motors, and the unfitted aerodynamic schema.
[configs/scenes/default.toml](../configs/scenes/default.toml) sets the initial
state and viewer.
[deployment](../deployment) owns the external ROS command/feedback adapter and
connection test, separate from reusable physics, controllers and learning.

The sibling [AeroWhip_simulation](../../AeroWhip_simulation/) project now contains
a continuous native C++ drone/cable simulator. It originated from eight copied
native physics/runtime files. Its engine now contains standalone-only adaptive
substepping and experimental finite-thickness self-contact; its prescribed-boundary
entry points explicitly reject contact-enabled models. Material-frame, actuator,
and event-scheduler sources remain byte-identical to this checkout. See the
[contact implementation record](#2026-10-25-standalone-experimental-cable-self-contact).
[app/flight.cpp](../../AeroWhip_simulation/src/app/flight.cpp) connects the
[physical and motor defaults](../../AeroWhip_simulation/src/simulation/model.cpp),
the checked [geometric controller](../../AeroWhip_simulation/src/controller/geometric.cpp),
and the [trajectory source](../../AeroWhip_simulation/src/trajectories/trajectory.cpp). Its default cable mesh is 20 edges / 21 nodes: each of the ten
measured marker intervals is split in two, retaining all ten moving physical
markers, their masses, the 0.9525 m length, and the 0.017 kg total cable/marker
mass. `--links 10` selects the original 10-edge / 11-node baseline; `--links 20`
selects the new default in the viewer, headless, trace, raw, and benchmark modes.
This checkout's deployment/training configuration remains at 10 edges. The
material constants, projected Lie-BDF2 formulation, solver tolerances, and
controller/motor gains are unchanged. Adaptive retry and self-contact alter the
standalone folded-motion dynamics; mesh refinement also changes discrete dynamics
and computational cost.
It retains the 200 Hz controller, 30 ms command delay,
30 Hz PVA/body-rate references, and 100 Hz output clock. The viewer offers Hover,
Back-and-forth, Oscillation (X +/-0.5 m, 2 s period), and Figure eight
(X +/-0.5 m, Y +/-0.3 m, 6 s period). The two dynamic presets use a 4 s smooth
frequency startup. These are the default preset parameters; altitude and horizontal
center are editable, while yaw remains fixed at zero. Simulation state/history
is not reset at cycle boundaries. The viewer has an editable draft with X/Y
amplitudes, center X/Y, altitude, cycle time, and smooth startup. Blue geometry and
the moving blue target show the accepted trajectory; amber geometry shows an
unapplied draft. Selecting a preset or editing fields only changes the draft;
Apply & restart rebuilds the complete runtime at the draft's starting pose and
pauses at time zero. Run starts the accepted flight. Revert restores accepted
settings; Defaults previews the selected preset defaults. Reset restarts the
accepted flight while preserving an unapplied draft. Frame paths fits active and
draft geometry. Copy command places an equivalent `run.sh` command on the clipboard.
Displayed speed/acceleration are sampled command-peak estimates over startup and
one steady cycle, not guaranteed extrema or aircraft feasibility limits.
The root `./run.sh` builds the needed executable and forwards all CLI options;
`./run.sh --trajectory figure-eight` starts the viewer, while `--headless`,
`--trace`, `--benchmark`, and `--raw` select the Qt-free terminal executable.
Build messages go to stderr, preserving numeric/JSON stdout. The script resolves
its own directory, so it also runs from other working directories and paths
containing spaces. Its Makefile builds a native Qt6/OpenGL viewer by default;
`./build/simulation` opens the window, initially paused. `--headless` and `--trace`
run continuously without a GUI; `--duration SECONDS` bounds a terminal run.
`--trajectory hover|back-and-forth|oscillation|figure-eight` selects the reference.
Optional `--amplitude-x M`, `--amplitude-y M`, `--center-x M`, `--center-y M`,
`--altitude M`, `--period S`, and `--startup S` use the same typed settings as the
editor. They require an explicit preset and are unavailable in raw mode.
Amplitudes are half-spans for sine paths; Back-and-forth X amplitude specifies
full outbound travel from its start X. Center/altitude use world metres; timing
uses simulation seconds and is independent of playback speed. Mode-inapplicable
values, nonfinite numbers, negative amplitudes, and nonpositive moving-cycle or
dynamic-startup times are rejected. Preset-only CLI calls retain their previous
defaults. Custom settings also appear in benchmark JSON.
`--benchmark` retains Hover/Back-and-forth as its default 30 s baseline, with one
warm-up and three measured runs per trajectory. Select either dynamic preset
explicitly with `--benchmark --trajectory NAME` to use the same protocol.
`--raw --trace --links 10` retains the initial one-second direct-wrench extraction check.
`make headless` builds a separate `build/simulation-headless` without Qt or
graphics dependencies. Both paths are independent of Python, Torch, ROS, and an
IDE. Standalone self-contact uses a user-confirmed nominal 3.5 mm cable diameter,
with numerical barrier parameters kept separate from material parameters.
`--no-self-contact` explicitly disables contact while retaining adaptive retry;
it does not restore the pre-change fixed-substep failure behavior. The
[viewer](../../AeroWhip_simulation/src/app/viewer.cpp) uses the original assembled visual
geometry exported to [drone.glb](../../AeroWhip_simulation/assets/drone.glb), with a
display-only cable tube and markers at the native observation positions. See the
[extraction checks](../tmp/standalone-cpp-20261025/validation.json) and
[initial viewer checks](../tmp/standalone-viewer-20261025/viewer-validation.json).
Continuous flight, controller/trajectory parity, and timing results are retained
in [continuous-flight evidence](../tmp/continuous-flight-20261025/).
The standalone executable currently generates its own references and has no ROS,
CRTP, or radio adapter. The adjacent Crazyswarm `--custom-sim` launcher still starts
this checkout's `run_custom_simulation.py`, not the standalone C++ executable.
That existing adapter and the standalone simulator share their original native
formulation, but the new standalone contact/retry solver has not been ported back;
using the reorganized executable as a Crazyswarm backend requires an explicit
command/feedback/clock adapter. Its geometric controller does not establish
firmware controller or estimator equivalence. No hardware connection is added by
the folder reorganization.

## Filesystem and UI portability

Shared path validation resolves file-dialog selections and relative inputs inside
the checkout, rejecting outside/home/foreign-drive paths and resolved symlink or
junction escapes. Run-owned replay references must additionally stay inside their
owning folder. Shared artifact writers and settings/trajectory exports use these
checks; new model-source, actor-source, resume, and evaluation request references
are project-relative. Existing historical evidence is immutable. This is
application-level containment, not an OS sandbox or protection against hostile
concurrent filesystem replacement.

[model_catalog.py](../model_catalog.py) discovers manifest-packaged stages in the
policy-source/replay browsers. It verifies registered artifact bytes and maps the
original final-evaluation replay filename to `final-replay.npz` in memory only;
recorded metric checks remain unchanged. Default replay selects packaged Stage4.
The empty Take Replay window initializes its folder before opening the picker.
Invalid file selections are reported without outside writes. The low-level
Training default now matches the current Stage1 preset selected by the launcher.
The evaluation worker uses the same frozen actor factory as the policy tester,
including all current goal/phase columns; this does not change task scoring.

Training charts now follow `request.json` exact-resume links, assembling parent
and current-session records in memory. Each ancestor is cut off at the saved
`resumed` collection; before that event exists, the selected checkpoint supplies
the boundary. Later parent collections are excluded when resuming an older
checkpoint. Parent/child identities must agree (the existing identity excludes
only the episode-budget override), references remain project-contained, and
inherited copies of the same evaluation are deduplicated. Training returns,
reward components, losses, validity and exploration curves retain their original
collection numbers and selected-run reward scale. New stages and actor-only
transfers begin separate histories. Missing/incompatible ancestry produces a
visible warning while retaining available current-session records. JSONL logs,
checkpoints, replay files and accounting arrays are never rewritten by this
reader. Earlier evaluations retain their original run ownership for the latest
preview and selected/best/latest replay actions. An earlier evaluation is
identified as inherited; a new failed evaluation remains the latest result.
The display lives at the UI entry point outside pinned learner execution sources,
so no execution-hash comparison, trainer code, identity or scientific criterion
was weakened. Already-open windows keep their loaded code; during the previous
campaign an updated viewer was opened separately from its worker-owning window.
That campaign is now stopped. Later Training launches use the updated history
display.

## Physical model, units, and timing

World XYZ has Z up. Rotation maps body to world; angular velocity, torque, and
inertia use the body frame/about COM. Public drone position/velocity refer to
the tracked top-plate origin. Quaternions use XYZW. Cable nodes, including
attachment node 0, use world coordinates.

| Quantity | Default |
|---|---|
| Drone mass excluding cable | 0.145 kg |
| Cable and moving-marker mass | 0.017 kg |
| Cable length / mesh | 0.9525 m / 11 nodes, 10 edges |
| Tracking-to-COM / attachment offset | (0,0,-0.018) / (0,0,-0.051) m |
| Maximum physics step | 1/600 s |
| Observation / controller / policy rate | 100 / 200 / 50 Hz for acceleration; retained velocity presets 10 Hz |
| Command delay | 0.03 s |
| EI / Cb | 1.178939873e-8 N m² / 8.174015594e-5 N m² s |
| GJ / Ct | 1.178939873e-8 N m² / 8.174015594e-6 N m² s |

These are working model assumptions, not newly identified hardware parameters.
The ten measured marker intervals form the ten edges. The inextensible cable
is naturally straight and isotropic. A point clamp follows drone position and
body-down tangent; the first finite edge may bend. The free tip has no imposed
pose/end couple. Attachment force and direct material moment react on the drone
through one two-way coupled solve.

[native/material_frame.inc](../native/material_frame.inc) and the scalar/tensor
engine implement objective material bending/twist with Kelvin-Voigt damping.
Cb is internal material damping; external cable drag is a separate coefficient
in inverse seconds. Rayleigh dissipated power is twice its damping potential.
Material-angle history remains despite assumed negligible angular inertia.
This checkout's deployment/training solvers have no cable self-collision,
ground/wall contact, or measured wind. The separate standalone coupled simulator
now has experimental cable-only self-contact, described below.
Degenerate edges and antiparallel tangent singularities are rejected.
A pre-contact standalone 20-edge diagnostic with 0.5 m X-amplitude Oscillation,
1.5 s cycle and 4 s startup produces self-overlap from about 4.77 s, then fails
implicit endpoint convergence after the last accepted 20.85 s at a 1/600 s
maximum step. An in-memory 1/1200 s variant reaches 30 s but still self-overlaps;
no production default was changed by that diagnostic. The subsequently approved
standalone contact/retry change completes this 30 s case, but its folded-motion
prediction remains sensitive to timestep/barrier choices and physically unvalidated.
Earlier non-contact trajectory checks do not validate contact. See
[fold reproduction](../tmp/cable-fold-review-20261025/oscillation-1p5-reproduction.json)
and [step-size diagnostic](../tmp/cable-fold-review-20261025/timestep-1200-summary.json).

The solvers use float64 projected Lie-BDF2. Uniform steps use BDF2; startup and
excessive step growth restart at backward-Euler order. Endpoint geometry,
velocities, material rates, loads, and constraints iterate together. The cap
is 20 Picard slots, relative-change tolerance 2e-11, length tolerance 1e-10 m,
velocity-constraint tolerance 1e-9 m/s, and four position projections per slot.
Controller, command, and output event times remain exact. Solver convergence
does not establish physical accuracy.

The geometric controller retains its velocity-feedback path for `velocity_v1`.
The new `acceleration_direct_v1` path converts world XYZ acceleration requests
(excluding gravity) directly to nominal force with the existing static cable-weight
trim, then desired attitude, collective thrust and body torque. Initial yaw stays
fixed. This path has no position/velocity feedback or integrated references;
zero acceleration supplies weight trim, not braking or position hold. Both paths
use the existing attitude loop. There is no integral loop or exact dynamic
cable-force cancellation.
Motor inverse mixing is clipped per motor; first-order motor response supplies
the delivered wrench. Motor parameters are marked uncalibrated. Continuation
requires previous BDF2 state/step, motors, held wrench, command queue, and clocks.
Drone telemetry alone cannot exactly resume a simulated trajectory.

Captured CUDA tasks default to the GPU-controlled Picard loop on 64-bit Windows
and on64-bit Linux when Torch was built with CUDA12.8 or newer. Other platforms
and older Linux Torch CUDA builds default to host-controlled replay.
`AEROWHIP_CONDITIONAL_SOLVER=0` selects the host path; `1` selects the conditional
path and requires a CUDA driver API12.8+ and working NVRTC. An explicit task
constructor choice takes precedence over this environment variable. The
[conditional driver](../tasks/whip/conditional_driver.py) loads the Windows driver
or Linux `libcuda.so.1`, and locates NVRTC in Torch/NVIDIA package libraries or
the Linux system library search. Backend failures propagate rather than silently
changing the solver. The same compiled Picard slot, first-converged endpoint
mask,20-slot cap, and numerical acceptance gates remain in use. CPU/differentiable
fitting paths are separate.

The retained [Linux scheduler comparison](../tmp/simulation-performance-20261011/scheduler-parity.json)
checks50 synthetic100-Hz steps in both1- and128-lane batches. Host and conditional
paths match every physical, BDF2-history, motor and held-wrench field bitwise;
the one-lane scalar comparison's maximum absolute difference is3.17524e-14,
within the existing1e-9 numerical comparison tolerance. Both schedulers reject
collapsed first edges and retain the20-slot cap. This is numerical evidence,
not a model fit, policy result or real-flight result. A full current Stage1–4
Linux campaign and a fresh Windows/Linux execution comparison remain unverified.

The [local performance record](../tmp/simulation-performance-20261011/summary.json)
uses RTX4080, Torch2.11/CUDA13.0 and the existing11-node float64 plant. Across
three warm trials, the median of mean128-lane100-Hz step times is41.897 ms for
the host scheduler and37.056 ms for the conditional scheduler:11.55% less step
time, or13.06% more plant throughput. This measures the plant adapter without
policy inference, task scoring or PPO updates; it does not establish the same
gain for complete training. Native single-flight PVA steps average2.143 ms and
the prescribed cable adapter2.009 ms in separate mild synthetic motions; these
are not timing guarantees for whipping maneuvers or a causal real-flight observer.

The [viewer](../viewer/pyvista_viewer.py) renders through NVIDIA OpenGL. The prior
1440×900 off-screen display-motion baseline averaged2.159 ms per geometry update
and render with vertical-sync waiting disabled; default synchronization averaged
17.084 ms. Those timings precede the viewer rebuild and are not measurements of
the new complete UI.

### Local simulation execution and display

[viewer/sources.py](../viewer/sources.py) now gives the local native simulation
one background owner thread, independent of Qt render callbacks. Native stepping
releases the Python GIL, reacquiring it for the existing Python controller.
The worker polls the existing controller-input callback on its5-ms wake cycle
and calls the unchanged `Simulation.advance` with monotonic elapsed time.
The100-Hz output,200-Hz physical controller, maximum1/600-s physics substep,
timestamped reference events and configured transport delay remain unchanged.
Desktop scheduling is not a hardware real-time guarantee. Existing wall-time
debt/burst limits remain: overload slows simulated time without enlarging or
skipping physical steps. Input polling previously ran in the viewer timer.

A bounded64-command FIFO applies play/pause, single step, reset, source changes,
optional force diagnostics and pacing changes between output steps. User actions
wait for acknowledgement, with a5-s response bound; render/telemetry reads never
wait for integration. Pause and pacing changes discard paused wall time. Closing
the window stops and joins the worker. Failure pauses execution and publishes a
visible error with the last published physical state. Changing command sources
while running is rejected. The existing trajectory window uses the facade's
source/reset controls and the controller's pure `prepare_reference` for CSV
export; it cannot directly access native state through that facade.

The latest-frame mailbox owns immutable array bytes, keeping producer/consumer
state separate and bounding backlog to one display snapshot. Its source revision,
monotonic publication time, running/error state and simulation timeline are
distinct. Rendering may skip intermediate display snapshots; numerical states,
command queues and controller updates are not skipped. Recorded replay uses a
separate adapter over `StateReplay`, with no plant, actor inference or new scoring.

[viewer/display_state.py](../viewer/display_state.py) defines a read-only world
description and named-robot display frames. Positions use metres in the declared
world frame with Z up; XYZW quaternions rotate body to world. Velocity/acceleration
use world coordinates, angular velocity uses body coordinates. Each robot and
optional cable has its own timeline timestamp and simulated/measured/estimated
provenance. Missing telemetry remains absent rather than zero-filled. World/frame
names must match; source adapters are responsible for transforms. The local
adapter labels its robot `drone` and its cable simulated; recorded playback is
explicitly labeled recorded simulation. Synthetic measured/estimated display
fixtures demonstrate presentation only; the implemented OptiTrack live adapter
is described below and its synthetic checks do not establish real-flight evidence.
These frames contain display data, not BDF2 continuation or policy observations.
The thread-safe `DisplayFeed` accepts these immutable frames directly without
constructing a plant. `PyVistaViewer.show_source(feed)` uses the same window with
playback/trajectory actions absent. An external producer must supply frame
transforms, timestamps and connection errors; no middleware or estimator is
included in the feed. Optional force telemetry remains labeled model forces.

[viewer/robot_scene.py](../viewer/robot_scene.py) retains CAD, rotor, cable-tube,
marker and overlay actors per named robot, updating transforms/coordinates rather
than recreating meshes per frame. Robot addition/removal or cable-topology changes
rebuild only that robot's actors. Multiple robots and absent cables are supported.
CAD name plates now show display identity instead of a hardcoded drone number;
this does not identify or calibrate real hardware. Floor/grid contrast changed
only in the renderer; there is still no physical floor/contact model.

The [docked flight window](../viewer/simulation_window.py) has a central3D view,
Displays and Inspector docks, camera presets (orbit/top/front/side), scene/drone
framing, translation-only drone following, layer/path/force toggles and visual
force scaling. Playback controls retain0.25/0.5/1× pacing, step/reset and trajectory
editing; saved replay adds sample seeking. Force arrows start hidden. Qt checks
for new snapshots on a16-ms timer, limits source-driven rendering to60 Hz, and
refreshes telemetry at10 Hz; camera interactions can render separately. Running
snapshots older than0.25 s are labeled stale, paused snapshots are labeled paused,
and robot/cable timeline lags and unavailable fields remain explicit. Pose timeline
lags over0.25 s also mark a running source stale; older cables carry their own
stale label. A missing robot frame shows no data and removes old robot actors.
The corner orientation axes follow the camera and the world-axes display toggle.
This age threshold is a display indicator, not a scientific acceptance/safety
criterion.
Existing Training/replay outcome windows retain their scoring and artifact UI
while reusing the updated renderer. The streaming ROS source described below
shares this window, with local trajectory editing absent because ROS owns its
command stream. A real-flight cable observer, firmware-controller equivalence
and learned-policy deployment remain unimplemented or unverified.

### OptiTrack connection to real flight display

The user approved a separate real-flight viewer on2026-10-13 and explicitly
required real OptiTrack capture for the drone. The
[launcher](../run_custom_real.py), [host feed](../deployment/real_flight.py),
[ROS reader](../deployment/ros_real_bridge.py) and
[stock-stack supervisor](../deployment/real_stack.py) implement that scope.
The adjacent Crazyswarm `docker/start.sh --custom-real` launches the existing
`crazyflie launch.py` with `backend:=cflib`, `mocap:=True`, `teleop:=False` and
`rviz:=False` on domain42. Exact snapshots of both original YAML texts are passed
to temporary container files; hardware configuration is not rewritten. An
exclusive host launch lease and ROS graph check reject an already running real
stack; `--attach` starts only the display reader and can coexist with RViz.

All enabled Crazyflies with vendor or `librigidbodytracker` motion capture are
displayed by YAML name by default; repeated `--robot` restricts that list.
The YAML initial position is retained as configuration metadata, never shown
as a measured pose. The display begins empty. This adapter constructs a cable-only
simulation, with no drone dynamics, controller or policy. It uses the existing host `.venv` GPU window and container
ROS runtime, sending exact reader/supervisor source through Docker stdin.

The sole drone-pose source is the configured `/poses` `NamedPoseArray` publisher
`/motion_capture_tracking`. Read-only parameter queries verify its backend is
`optitrack` or `optitrack_closed_source` and its configured frame matches the
loaded YAML. The reader requires one discovered publisher endpoint belonging to
that node and rejects an ambiguous/replaced source until verification succeeds.
ROS Jazzy does not expose publisher GID in Python callback metadata; endpoint
checks use ROS graph discovery. No `/NAME/pose` estimator or drone TF
fallback supplies robot poses. The motion-capture node's TF path may replace
unavailable orientation with identity; `/poses` is read before that substitution
so invalid orientation is explicitly rejected here.

Positions are metres in the named capture/display frame; XYZW orientation
rotates the captured rigid-body frame into that frame. Pose age uses the capture
header timestamp on the real ROS clock (`use_sim_time=False`). The default
display frame is `topics.frame_id` in motion-capture YAML, currently `mocap`.
`--frame world` or another frame requires an existing TF transform at that
capture timestamp, including its rotation and translation. No identity
`mocap`-to-`world` assumption or alignment calibration is introduced. CAD uses
the existing shared model/visual attachment offset; this does not calibrate the
actual OptiTrack rigid-body pivot to the physical drone/cable attachment.

Duplicate-name frames and individual invalid/nonfinite poses or missing
orientations are rejected. Quaternion norm error above0.001 is rejected; accepted orientations
are normalized for the renderer. This is display validation, not a fitting or
scientific acceptance criterion. The reader preserves the last valid pose while
an independent60-Hz heartbeat advances the display timeline; tracking gaps become
stale under the existing0.25-s display-age threshold. Invalid source/transform
diagnostics are visible errors, and ROS clock regression clears poses and starts
a new timeline generation. Unknown capture names are ignored. Vendor tracking
is labeled measured; `librigidbodytracker` reconstruction is labeled estimated.
Measured-drone velocity, acceleration, body rates and forces remain absent.
The [live cable boundary driver](../simulation/live_cable.py) reuses the existing
native `Model.prescribed_advance` material-cable solver, preserving free-node
positions/velocities, material angles and BDF2 history separately for each robot.
The real drone is not integrated or overwritten in a coupled plant. Cable
provenance is **Simulated**; drone provenance retains the verified capture source.
The shared viewer already exposes both provenance labels and independent ages.

The relay batches every accepted selected-robot pose between its60-Hz display
heartbeats in `pose_samples`; latest poses still populate `robots` for display.
Accepted buffered samples are processed even if heartbeat source verification
changes after their capture callbacks. Source timestamps use integer ROS-header
nanoseconds relative to the relay epoch before conversion to seconds. These
remain capture-node receipt timestamps, not synchronized Motive exposure times.
Duplicate timestamps do not advance physics. A512-sample relay buffer bounds
memory; overflow is explicit and freezes the cable. The transport's owner thread
advances native physics independently of the Qt renderer; rendering cannot reset
or advance cable state. These connection traces are separate from the retained
source logger and do not constitute calibrated flight/fit evidence.

For the configured attachment offset, root position is `p + R*r_attachment` in
the selected Z-up display/capture frame. Arrival of the next pose permits causal
advancement over the completed interval: root endpoint velocity is a backward
secant, with cubic Hermite P/V/A retaining the previous endpoint velocity.
Attitude uses shortest SO(3) geodesic interpolation with piecewise derived body
rate. These are boundary-interpolant derivatives, not measured drone telemetry;
the method uses no future samples beyond the latest received endpoint and does
not extrapolate between arrivals. Existing native maximum substep and constraint
checks are unchanged. `--model` defaults to `configs/simulation.json`; model/source
hashes and initial-condition/timing assumptions are saved in launch metadata.
Scene attachment offset and marker-interval lengths must match the selected model.
An implicit endpoint-convergence, position-projection or velocity-constraint
rejection retries that completed interval from the same accepted state/BDF2
history. Each retry multiplies the baseline substep count by2,4, then8, so even
an interval shorter than the native maximum step is actually subdivided. Native
iteration limits and convergence/constraint tolerances remain unchanged; successful
output alone commits state. Invalid inputs, other failures and exhausted refinement
still freeze the cable. Launch metadata records `numerical_refinement_factors`
and the retry/subdivision rule. This is bounded numerical refinement of the
boundary driver, not a
parameter fit, smoothing filter or modification of the coupled/training integrator.

At the first accepted pose the model's hanging, stationary, untwisted cable is
an explicit prior; YAML positions do not supply real state. Frame/body-origin
calibration still requires physical verification. There is no ground contact,
so a ground-level attachment can yield cable geometry below the visual floor.
Gaps above0.25 s, timestamp regression, overflow or integration failure freeze
the last valid cable and show an error; viewer restart explicitly reinitializes
it. A source endpoint change or ROS clock generation change initializes a new
timeline. New flight-script identities clear trails without resetting cable
physics/history. No measured cable correction, force publication, artificial
propeller rotation or learned-policy deployment is introduced.

#### Experimental marker-cloud cable reconstruction

`--track-cable` adds a separate read-only observation path for one explicitly
selected robot, using [deployment/cable_tracking.py](../deployment/cable_tracking.py).
It does not feed the native cable integrator, controller, fitting or policy.
The selected model's cumulative `marker_interval_lengths_m` define material marker
arc distances; the user confirmed the real cable retains this spacing. Current
distances are0.063,0.150,0.250,0.350,0.450,0.550,0.650,0.7525,0.8525,0.9525 m
from the attachment. Root is the matching measured rigid-body pose plus the same
body attachment offset used by the display. Length overrides and heuristic
settings can be provided with `--cable-tracking-config`; the explicit example is
[configs/ros/cable_tracking.json](../configs/ros/cable_tracking.json). Actual settings
and source hashes are saved in launch metadata.

The optional relay subscribes to `/pointCloud` (`sensor_msgs/PointCloud2`), verifies
its sole publisher is the configured capture node and also checks the currently
verified pose endpoint. XYZ FLOAT32 decoding respects offsets, endian convention,
organized rows and padding. Current pinned `libmotioncapture` appends both NatNet
OtherMarkers and LabeledMarkers into this cloud, discarding marker IDs, asset and
visibility flags. This is therefore a **mixed candidate cloud**, not an
unlabeled-only or visibility-certified cable measurement. Point indices identify
only a row in one retained cloud; they are never persistent physical IDs. The
raw capture-frame XYZ array and separately transformed display-frame array are
recorded, with nonfinite coordinates represented by JSON null. Backend/frame
verification is the same as the pose path. Requested TF is evaluated at the
sample timestamp for both cloud and pose; there is no assumed frame alignment.

Clouds and selected poses pair by exact integer ROS header timestamp, which the
existing capture node sets once for each received frame. They are receipt stamps,
not Motive exposure timestamps. Either callback arrival order works. Pending
frames are bounded to32 and unmatched clouds expire after100 ms when subsequent
data arrives, retaining an explicit missing-pose record with no reconstruction.
Source/clock generation changes clear pairing and association history. Each paired
cloud is sent separately from the60-Hz display heartbeat. The relay caps frames
at2048 points, reports oversized/malformed clouds instead of truncating them, and
uses a1-MiB packet limit. `connection.jsonl` stores `capture_display` marker packets
before association, followed by `cable_reconstruction` results; all accepted cloud
callbacks are retained, while upstream/DDS drops are not detectable as camera
frame loss. This remains connection/reconstruction evidence, not lossless source
capture or a fitting dataset with verified identities.

Initialization requires a complete unambiguous approximately vertical hanging
cable, using world -Z and a3.5-cm spatial gate around the arc-distance prior.
It does not infer arbitrary coiled initial geometry. Tracking predicts each
material marker from its last accepted position and bounded constant velocity;
temporal gates widen with elapsed time. Velocity updates blend the new finite
difference with the previous estimate using `tau = position_noise_m /
maximum_speed_mps` (1 ms with current settings):
`v_new = (delta_position + tau*v_old)/(dt + tau)`, followed by the existing speed
cap. This damps spikes from bunched ROS receipt timestamps; it does not recover
camera exposure timing or constitute calibrated velocity estimation. An ordered beam search considers missing
nodes and one-to-one candidate assignments, with geometric upper bounds between
the attachment and markers and between all assigned marker pairs. Chord distance
may be shorter than arc spacing; no straight-segment equality is imposed. The
root-distance bound filters distant clutter, and1-mm spatial deduplication handles
duplicate list entries while preserving raw arrays. Rigid-body candidates are
not semantically excluded: their IDs/flags are unavailable upstream. Candidate-pair
distances are computed once per cloud over the retained candidate union and reused
across beam hypotheses; the matrix is not allocated over the full raw cloud.

Default beam width128, at most12 candidates per marker,15-mm chord tolerance,
12-mm noise scale,12-m/s motion bound, missing cost4 and ambiguity margin1 are
engineering settings, not scientifically calibrated probabilities or acceptance
thresholds. Results record whether search pruning occurred. Near-best retained
hypotheses that disagree on a marker leave that marker null; reported confidence
is a normalized score margin among retained hypotheses and cannot certify identity
outside the bounded search. Cloud capacity overflow produces no assignments;
more than12 candidates for one marker suppresses that marker's choices while
allowing other identities to remain visible. The existing root-distance bound
is applied before this limit so geometrically impossible points do not consume
the candidate budget. Partial occlusion preserves accepted
per-marker history. After0.25 s without an accepted observation, that marker
uses geometric recovery instead of stale constant-velocity extrapolation:
root/arc bounds and recent neighbours' reachable regions select candidates,
then the same one-to-one search and exact pairwise chord bounds test assignments.
Recovery candidates have equal geometric cost; disagreement among retained
near-best hypotheses still leaves the marker null. Accepted recovery resets its
velocity to zero. Results include one-based `reacquired_markers` and
`crowded_markers` lists for inspection. When all marker histories are older than
0.25 s, complete hanging reinitialization is required. These gates do not prove
material identity in clutter or after a fold. A flight-script trajectory epoch
does not reset associations. Arbitrary-shape initialization,
multiple interacting cables, model-based measurement correction and identity
ground truth are unimplemented.

[viewer/cable_tracking_scene.py](../viewer/cable_tracking_scene.py) renders separate
persistent gray candidate points and blue estimated marker identities/straight
links, alongside the existing green simulated cable. Links are only between
adjacent observed material markers (including the measured root); no gap filling
or smooth centerline inference is displayed. Labels remain Estimated because
identity is inferred and upstream visibility flags are unavailable. The inspector
shows count, age, status and minimum heuristic confidence. Separate visibility
controls select the estimate and raw candidates; overlays older than0.25 s are
hidden. Numerical integration, physical parameters and training stay unchanged.

#### Hand-test recording and offline playback

`run_custom_real.py --capture-only --robot NAME` launches only the existing
OptiTrack node plus the shared viewer and enables marker-cable tracking. It
does not start `crazyflie_server`, teleoperation or a radio connection. The
default domain is43; an explicit domain in0..232 is supported for isolated
capture. `--capture-only` and `--attach` are mutually exclusive. Existing nodes
and domain leases are checked before launching; no existing process is stopped.
Normal real-flight launch and attach still default to42. To attach to a default
capture-only stack, pass `--attach --track-cable --domain 43` explicitly.
[deployment/real_stack.py](../deployment/real_stack.py) builds capture parameters
from the same YAML bytes and stock Crazyswarm rigid-body selection rules, omitting
an empty `rigid_bodies` map because the direct ROS YAML parser rejects it.
It launches `ros2 run motion_capture_tracking motion_capture_tracking_node` with
that temporary configuration. Original YAML files and Motive settings are not
edited. The existing session/process-group ownership checks supervise this node;
viewer close, Ctrl+C and worker crash clean up an owned capture-only session.
The normal real-flight viewer's close/attach behavior remains as described below.

The shared window now offers **Record**, **Stop & save**, **Replay last**, and
**Open recording…**. Record controls are enabled for configured live real/capture
feeds; Open recording is also available in the other shared-viewer modes.
`--record` starts a clip at launch, including its initial waiting-for-capture
frame, and works with bounded headless operation. Each new clip gets a distinct
`output/custom_real/<session>/recordings/<UTC-time>-<id>/` directory. It contains
`recording.json` (schema `aerowhip_display_recording_v1`), `frames.jsonl`, and
`events.jsonl`. Scene/display configuration and the session's model, tracker,
frame, YAML/source identities and settings are copied into the manifest. This
is a display recording, not a native checkpoint or calibrated fitting dataset.

[viewer/recording.py](../viewer/recording.py) serializes the complete immutable
display contract: measured drone poses and absent telemetry, simulated cable
geometry, estimated cable marker identities, observed masks/confidence, finite
raw display points, original source timestamps, provenance, trajectory epoch and
error. Received pose/cloud packets, including raw capture-frame coordinates and
nonfinite/null markers, and association results are retained separately as events.
All accepted display publications are recorded at the relay heartbeat rate
(nominally60 Hz), independent of which frames Qt renders; every received packet
during the clip is an event. Recording does not reset live simulation/association
history and does not recompute prior motion. There is no claim of retaining
camera frames that were dropped before the relay. Starting a clip snapshots the
current display; physical initialization still uses the existing hanging prior.

The feed's owner lock makes Start/Stop and sample/frame writes atomic relative to
live processing. Writes use line-buffered append-only streams; Stop and orderly
window shutdown flush/fsync streams before atomically marking the manifest complete.
Write errors remain visible in recording status and prevent a complete mark. A
native crash can leave an incomplete recording whose complete lines are replayable
with a warning; only an interrupted final partial line is skipped. Malformed
interior lines and inconsistent completed frame counts are rejected. No recording
is overwritten or silently promoted to complete after interruption. Synchronous
disk writes can add capture latency; this is a desktop recorder, not a hard
real-time or lossless camera recorder.

[run_recording.py](../run_recording.py) opens a folder/manifest or a file picker
and uses `RecordingSource` with the same GPU viewer. It loads no model/native
plant, starts no ROS nodes, and reads saved results without rerunning physics or
association. Frame offsets are indexed once; only the requested frame is decoded
on seek. Play/pause, step, reset,0.25/0.5/1× pacing and a timeline use elapsed
host-monotonic recording time, while original ROS/pose/cable stamps and ages remain
unchanged inside displayed frames. This permits explicit replay of clock resets,
gaps and errors. Recorded errors/incomplete warnings remain visible but do not
disable playback. Seeking clears display trails; camera/trails are presentation,
not saved physical results. Offline replay requires neither Docker nor hardware.
`--inspect` validates/indexes frames and prints a summary without opening the GPU
window. Live and replayed blue estimated markers remain separate from green
simulated cable geometry; comparing them does not itself establish accuracy.

The existing [Docker receiver patch](../../Crazyflie/crazyswarm2_jazzy_whipping/docker/motion_capture_tracking-optitrack-interface.patch)
now also repairs NatNet unicast while preserving multicast. The server advertises
the mode at connection. Unicast retains the endpoint that sent `NAT_CONNECT`,
sends `NAT_KEEPALIVE` every1 s, and ignores interleaved frame packets while waiting
for server/model replies. Frame reception ignores control replies and retains
the latest complete frame. Command replies and missing frames time out after5 s
with an explicit error. Multicast still binds the advertised data port and joins
the advertised group on `interface_ip`. The running container was rebuilt and
future Docker builds apply the same patch to the pinned source. Motive mode changes
require ROS reconnect/restart; this work does not modify Motive settings.
The patched capture node also finishes normally when SIGINT invalidates its ROS
context during frame reception/publication or executor creation; exceptions during
active operation are preserved. This addresses an observed capture-only shutdown
abort and does not suppress source/integration failures during operation.

The adapter publishes no flight commands, invokes no command services and
changes no firmware parameters. Existing flight scripts still use the normal
Crazyswarm `./docker/shell.sh --real` domain42 shell. Closing the GPU window stops
only its reader; an owned real stack remains active until Ctrl+C in the launch
terminal, matching RViz lifecycle. The supervisor owns a separate process group;
cleanup verifies session identity, process environment and group ownership
before signalling it. With `--attach`, startup snapshots the matching local
`ros2 launch crazyflie launch.py backend:=cflib mocap:=True` process and its
children on the selected domain. Ctrl+C/SIGTERM also stops that captured stack.
[Stack lifecycle handling](../deployment/stack_lifecycle.py) checks PID start
time (`/proc/<pid>/stat` clock ticks), exact argv and ROS domain before each signal; it remembers child identities
if the launch parent exits first. SIGINT lets ROS launch shut down normally;
surviving captured descendants receive SIGTERM/SIGKILL only after timeouts.
An attached window close or bounded headless duration still leaves ROS running.
Absent matching local launches (including remote or manually assembled stacks)
are reported as viewer-only shutdown; ambiguous matches are refused. Docker and
independently started flight/logger processes are outside this captured tree.
The graphical real launcher now runs its viewer/reader in a separate worker
process with a host supervisor. Ctrl+C sets a shutdown event; a Qt timer closes
the window outside the Python signal handler. The shared window completes VTK
module initialization before importing PyVistaQt; this avoided the observed
intermittent native import/finalization crashes in the retained checks. Normal
Python finalization is retained. If a native worker nevertheless crashes before
a stop request, an owned launch keeps its supervisor terminal and ROS stack
running and prints attach/recovery instructions. Ctrl+C in that terminal then
performs session-checked cleanup. If the crash happens during requested shutdown,
the surviving supervisor performs that cleanup directly. The attached supervisor
also retains its startup snapshot for cleanup after a worker failure during a
stop request. No container-wide kill or automatic real-stack restart is performed.
Both custom modes isolate their Docker clients from terminal signals and defer
GUI shutdown through the shared Qt timer. Read-only feeds omit playback controls,
shortcuts and playback status hints in the shared window.

Each launch writes unique `output/custom_real/` connection evidence with session,
domain, robot selection, YAML/source hashes, frame and timestamped sampled
display packets. This is not a full-rate OptiTrack recording or fitting input.

Both ROS-connected viewers monitor endpoint identities on each selected robot's
`cmd_full_state` topic and `/all/cmd_full_state` at10 Hz. A newly discovered
publisher advances a per-robot display `trajectory_epoch`; disappearance alone
does not clear trails. The installed Crazyswarm Python client creates these
publishers when a script connects, including scripts using high-level commands.
Real capture packets carry the epochs to the host; the custom simulation uses a
separate display-session packet. On an epoch change the renderer clears the
drone trail, cable-tip trail and displayed reference, preserving pose, cable
geometry and provenance. **Clear trails** provides the same display-only cleanup
manually. No flight command, physical reset, clock change, plan cancellation or
recording deletion is caused by this display event. Discovery can lag a new
client by DDS discovery plus the polling interval; a long-running client reused
for another flight does not create a new endpoint and should use the button.
Publisher rediscovery may also clear display history; this is a display heuristic,
not a scientific flight-segmentation record. The prescribed-motion cable simulation
supplies explicitly simulated state; optional marker association supplies separate
estimated identities as described above, without correcting simulated state.
The retained [synthetic reconstruction checks](../tmp/custom-real-20261013/capture-checks.json)
cover direct capture, rotated/translated TF, multiple names, capture freezes,
invalid orientation, duplicate names and recovery with an unrelated estimator
still publishing. [GPU display checks](../tmp/custom-real-20261013/gui-checks.json)
and [capture](../tmp/custom-real-20261013/real-viewer-synthetic.png) show the shared
NVIDIA-rendered read-only window using synthetic poses.
[Source/frame rejection checks](../tmp/custom-real-20261013/source-frame-rejection-checks.json)
confirm that a non-OptiTrack source and an unavailable TF frame show diagnostics
without accepting a pose.
[Qt signal checks](../tmp/custom-real-20261013/gui-signal-checks.json) confirm that
Ctrl+C closes the actual GPU window/local event loop and joins its display reader.
[Mock lifecycle checks](../tmp/custom-real-20261013/stack-lifecycle-checks.json)
verify exact YAML/launch flags, keeping the stack after viewer close, Ctrl+C
cleanup and rejection of an unowned process. These checks used no real radio,
OptiTrack hardware or flight; actual capture alignment and hardware behavior
remain unverified. Physics, fitting, learned controllers and training are unchanged.

### Crazyswarm command connection to simulation flight

[deployment/custom_simulation.py](../deployment/custom_simulation.py) adapts the
existing native owner thread without modifying its plant. PVA remains the default;
an explicit ROS-only acceleration selector reuses the sampled nominal controller
with zero translation gains while armed.
The host `.venv` owns native physics and GPU visualization; the existing Docker
ROS Python runs [deployment/ros_bridge.py](../deployment/ros_bridge.py). The
launcher sends the exact checked adapter source to Docker Python stdin, sourcing
the installed ROS Jazzy/Crazyswarm setup. No image rebuild, host ROS installation,
container Torch installation or additional volume mount is required. The
simulation client patch is applied to the existing container's source symlink;
the adjacent Dockerfile includes it for future builds. This is a Linux/Docker
command adapter. Firmware high-level planning is reused; firmware motor-control
and estimator equivalence are not implemented.

`./docker/start.sh --custom-sim` requires the container already started with
`./docker/compose`. The default container is `crazyswarm2_jazzy_whipping_dev`;
`--container` can select another existing container. The launcher finds the
adjacent AeroWhip checkout, with `AEROWHIP_ROOT` as an explicit location override.
Alternatively, run `run_custom_simulation.py` directly in the host project/PyCharm.
The simulation starts playing after the ROS bridge connects. `--headless` omits
the viewer and `--duration` bounds headless wall time. Closing the viewer stops
the transport, ROS adapter and native owner. Ctrl+C/SIGTERM sets the same deferred
shutdown event in graphical and headless modes. The bridge is captured by its
unique session argument and ROS domain after connection; identity-checked
cleanup also handles a failed transport shutdown. Domain/port leases are released
without stopping Docker. The standalone `run_simulation.py` uses the same Qt
stop event and closes its controller receiver on Ctrl+C. The renderer remains on the host
GPU. Only one drone/cable plant is exposed. [Configuration selection](../deployment/configuration.py)
reads the hardware `crazyflies.yaml`: enabled Crazyflies, type/connection, URI and
reference-frame precedence follow the stock server. One enabled drone is inferred;
multiple enabled drones require an explicit `--robot` selection and only that
plant is exposed. Both PVA and direct acceleration require the `world` frame; other
frames fail explicitly. RViz itself consumes TF and a transient-local robot
description; initial states are established by the simulation server.

`--crazyflies-yaml` selects hardware metadata and the selected robot's exact
`initial_position` in world-frame metres. It is the default position for startup
and Reset; there is no implicit height offset or simulation YAML override. Reset
restores the configuration loaded at startup rather than rereading edited YAML.
`--simulation-yaml` is optional and overrides position only when explicitly
supplied. The retained [airborne example](../configs/ros/custom_simulation.yaml)
sets `cf_2` to `(0,0,1.1)` m and can reproduce the earlier airborne connection
setup by passing `--simulation-yaml configs/ros/custom_simulation.yaml` (relative
to the AeroWhip checkout). The hardware YAML remains `(0,0,0)` m. Original and
effective positions, paths/hashes of the files actually loaded and robot identity
remain in metadata. The effective position initializes the native plant and is
exposed through `robots.<name>.initial_position`. Orientation, velocities and
material-frame settings still come from the selected AeroWhip scene; explicitly
supplied cable geometry translates by the change in its drone origin.

Initialization does not infer a physical floor from the world origin or from
`viewer.ground_z_m`. The floor/grid are display geometry and no longer reject
otherwise valid native initial states. The unchanged native model is free-space:
omitted cable geometry hangs vertically, including below the displayed plane
when the configured drone position is near that plane. Native root/length,
velocity and material geometry checks remain active. A `z=0` initial position
does not implement resting contact, cable support, touchdown or unpowered flight.
General physical world/contact behavior remains unimplemented. Neither hardware
firmware parameters nor the YAML robot type override the explicit native model
or geometric controller; the adapter still owns one selected drone/cable plant.

ROS uses domain142 by default, separate from the existing real/stock simulation
domain42. The new launcher and test reject domain42. Both sides identify the
same robot, domain and fresh session over localhost TCP port47654 (configurable;
port0 chooses an unused port). This transport is distinct from the existing
manual-control TCP port47653. Physics never runs in a ROS or Qt callback.
An OS-released host domain lease prevents two custom launches from sharing the
global Crazyswarm service names, including launches using different TCP ports.
A ROS graph probe also rejects an existing `crazyflie_server` before advertising
the custom server. Neither check terminates another server. These checks do not
replace domain42 separation or form a distributed ROS locking protocol.
The listener enables `SO_REUSEADDR` for immediate restarts after TCP `TIME_WAIT`,
while preserving live-listener exclusivity. A busy port produces a short launcher
error; it never terminates the existing listener. If an older connection still
blocks binding and no simulation is running, `--port 0` selects a free port and
passes that actual port to the Docker adapter automatically.

The supported command is `crazyflie_interfaces/FullState` on
`/<robot>/cmd_full_state`. World positions are metres at the tracked top-plate
origin; velocity and acceleration are world-frame m/s and m/s², yaw is radians,
and angular-rate feedforward is the existing nominal desired-body XYZ rad/s
reference. Command orientation must be a normalized yaw-only XYZW quaternion.
The adapter accepts `world` and upstream `/world` headers, rejecting other
frames, nonfinite references and unsupported roll/pitch attitude commands.
Streaming clients must use `use_sim_time:=true`: `/clock` is the native plant
timeline. Headers must lie within `[latest_time-0.5, latest_time+0.02]` seconds,
with nondecreasing stamps and increasing transport sequence numbers. These are
transport validity limits, not fitted latency or flight acceptance thresholds.

Accepted commands enter a128-packet FIFO. At most one command becomes a native
request event at each next available100-Hz output boundary; bursts stay ordered
and queue overflow is explicitly rejected. Header time, published receipt time,
actual request boundary and applied-event time remain distinct in retained logs;
headers never backdate a causal live command. The existing native30-ms modeled
delay and200-Hz controller are preserved, so transport/receipt quantization adds
its measured latency to that modeled delay. Without a new command the plant
retains its last reference. Pause/reset/step and pacing controls remain available;
reset clears input/native history and changes the reset generation. Connection
loss or a2-s heartbeat stall pauses the host with a visible error and requires
restarting the custom simulation. This is desktop pacing, not a hard real-time
radio or firmware-watchdog implementation.

[Flight command arbitration](../deployment/flight_commands.py) supplies functional
arm, takeoff, land, absolute/relative go-to, polynomial upload/start, group masks,
stream release and emergency services under both `/<robot>/` and `/all/`. The ROS
node is named `crazyflie_server`, with the parameter services and robot metadata
needed by stock `CrazyflieServer()`/`Crazyswarm()` discovery. Takeoff/land/go-to
coefficients come from the pinned `cffirmware` high-level planner in Docker.
The host evaluates these XYZ/yaw polynomials at native100-Hz request boundaries,
including analytic P/V/A and nominal desired-body rate feedforward from firmware
flatness (`GRAV=9.81` m/s²). Uploaded trajectories support1..64 eight-coefficient
pieces, piece offsets without gaps, positive timescale, reverse playback and
relative XYZ shifts. This planner layer does not change native physics, the
200-Hz geometric controller,30-ms modeled delay or the motor model.

Streaming has priority until `notifySetpointsStop`; its validity interval begins
at the actual native notification boundary. Separate ROS service/topic clients
can reorder delivery: a high-level request arriving before notification remains
pending without actuation until notification releases it. Pre-notification
stream packets cannot cancel that release. The closing stream's cutoff includes
the20-ms permissible header-clock allowance at notification receipt. Packets
from it arriving after high-level activation are explicitly counted/logged as
discarded; newer streams can regain priority. API receipt/scheduling counts,
pending-release state, generated high-level samples and stream discard counts
remain distinct. Upstream services have empty responses; service failures appear
in `/aerowhip/simulation/status.command_error` and ROS logs. Writable parameter
emulation is limited to functional `commander.enHighLevel` and
`hlCommander.groupmask`, mirrored between robot and `all`; arbitrary firmware
gains/controllers are not emulated. Each ROS parameter transaction is fully
validated before sending one host batch; conflicting robot/`all` aliases are
rejected. The host validates again and commits the batch under one command lock
before acknowledging it, independently of the flight-command queue. Disarm
therefore cannot discard an accepted parameter update. Disabling high-level
control also cancels pending/active plans. A lost API acknowledgement or send
failure ends the bridge session rather than continuing with uncertain state;
reconnect requires restarting the session. These parameters are control metadata,
not changes to physical model parameters.

The separate boolean parameter `aerowhip.acceleration_only` selects the ROS-only
[controller wrapper](../deployment/acceleration_control.py), default false. It
must be set alone. Its owner-thread commit requires disarmed hold, an empty
command queue and matching native requested/applied references (mode transitions
must drain the modeled delay). In this mode an armed plant uses FullState world
acceleration and yaw, ignoring P/V placeholders; body-rate feedforward must be
zero and high-level trajectory requests are rejected. It uses the unchanged
nominal force equation `m*a - (m + sum(cable_node_mass))*gravity`, attitude loop,
motor dynamics, 200-Hz controller and 30-ms modeled command delay. No firmware
gain emulation or identification is implied. Disarmed hold uses PVA control.
Reset clears acceleration mode, and successful scripted completion explicitly
restores PVA. Status/evidence records `acceleration_only` and the selected
`controller_interface`; `reference_position_world_m` is unavailable in direct
mode because the transmitted P is a placeholder. Direct-mode checkpoint exports
use `aerowhip_ros_acceleration_runtime_v1` so generic PVA restore rejects them;
a dedicated continuation importer is not implemented.

The free-space plant holds its configured pose with control active and
API arming enabled. `arm(False)` gates new commands, cancels queued plans and
requests a controlled hold at the latest feedback pose while `/clock` continues.
It does not model motor power removal, falling or touchdown. Emergency immediately
suspends the native owner, clears incoming commands and latches an error; play
and step are blocked until explicit simulation reset/restart. Reset changes the
generation and restores the configured initial state. Other scripts issuing a
low landing height receive a free-space descent, not a ground/contact solution.

In the Crazyswarm checkout, plain `./docker/shell.sh` defaults to custom simulation
on domain142 with `AEROWHIP_SIMULATION=1`. The explicit alias
`./docker/shell.sh --custom-sim [domain]` allows a different simulation domain.
`./docker/shell.sh --real` and `--sim` explicitly select domain42 and
`AEROWHIP_SIMULATION=0` for real flight and stock simulation respectively.
This selection affects newly opened shells and their child scripts; it does not
retarget an existing terminal or change running ROS nodes. The simulation-only
[Crazyflie client patch](../../Crazyflie/crazyswarm2_jazzy_whipping/docker/crazyflie-py-custom-simulation.patch)
enables `use_sim_time`, verifies the custom server/session metadata and exposes
simulation status and configured return height. The existing parameter name
`aerowhip.airborne_landing_height_m` is retained for client compatibility; its
value is the effective initial world Z, including zero or negative coordinates.
Ordinary `TimeHelper` scripts
then follow `/clock`; the real/stock shell does not enable this behavior. The
[CSV repeat player](../../Crazyflie/crazyswarm2_jazzy_whipping/scripts/full_state_pva_csv_repeat.py)
additionally processes ROS clock/status callbacks during simulated playback and
resets each new hover's rate deadline in both real and simulation modes, avoiding
catch-up bursts after the CSV's separate scheduler has advanced time.
Simulation playback checks session/generation, command rejection and landing
reference completion. Its landing height explicitly returns to the configured
initial world Z; this is printed before arming and still represents a free-space
reference rather than touchdown. Emergency waits use wall
time because the plant clock stops. The real-flight wall-time playback scheduler,
landing-height default and hot-loop behavior are preserved.

Feedback uses the same native timestamps: `/<robot>/pose` is `PoseStamped`,
`/<robot>/simulation_state` is `FullState` with world linear velocity/net
acceleration and body angular velocity, `/tf` is world-to-robot, and
`/<robot>/odom` expresses its twist in the robot child/body frame (linear velocity
is `Rᵀ v_world`). `/<robot>/simulation_cable` is a `PoseArray` of simulated node
positions, with identity placeholder orientations; it does not publish measured
cable geometry or material frames. `/aerowhip/simulation/status` carries session,
generation, robot/frame, clock periods, error and command counts, plus API arm/mode
and pending-release state. Drone and cable are labeled simulated in the viewer.
`/<robot>/robot_description` is transient-local URDF, and `/<robot>/status` exposes
simulation API supervisor flags. Battery/radio fields in this required fixed
message remain zero placeholders, not simulated measurements. Hover, legacy
velocity and position streaming, arbitrary firmware parameter changes and
multiple simultaneous plants are not implemented by this adapter.

The separate [PVA test](../deployment/ros_test.py) uses the upstream
`Crazyflie.cmdFullState` serializer through a streaming-only client, without
initializing high-level services or a joystick. It waits for matching simulation
status, clock and feedback, then sends at nominal30 Hz:1-s hold, a6-s `sin⁴`
outward/return excursion of `(0.20,0.10,0.05)` m, and2-s hold. P/V/A are analytic;
yaw and rate feedforward are zero. These are connection-test settings, not a whip
definition/reward, calibrated flight maneuver or policy evaluation. The script
records actual stamps/missed slots and aborts on clock/session reset, stalled
feedback, command rejection or incomplete acceptance/scheduling.

Each host launch creates a new evidence directory under `output/custom_simulation`
(or project-contained `--output`), preserving existing files. `metadata.json`
retains scene/model/session/adapter identities, `connection.jsonl` records
receipts, request boundaries and latest output snapshots, `ros-bridge.log` retains
adapter diagnostics, and `final-checkpoint.npz` stores exact native BDF2/motor/
queue continuation at shutdown with the model in metadata. Output snapshots and
their last-step diagnostics may coalesce when a wall-pacing burst advances more
than one output; command receipt/request records remain separate. They are
connection/numerical evidence, not recordings of real flight or fitted models.
The native checkpoint is not an implemented ROS session/input-FIFO resume path.

## Policy observations and PPO

All current stages append the static world-axis input
`goal = (target_world - drone_initial_world) / cable_length`.
The origin is the episode's initial tracking point. It does not rotate with yaw
or become current drone-to-target error. The target plane is vertical: its normal
is the horizontal requested direction and its vertical axis is world+Z.
Independent tilted target planes are not implemented.

Actor history contains scaled tracked displacement, body-to-world rotation,
linear/angular velocity, attachment-relative cable position/velocity, previous
normalized action, and elapsed time. Physical frames record motor thrust and
held wrench, but these columns are excluded from the deployable actor encoder.
Cable observations are available in simulation. Stage1-3 use 99 frame columns;
Stage4 adds recovery phase/time for 101 and preserves 451-sample rolling whip
history. The goal has a separate projection. The Gaussian actor uses three tanh-bounded outputs. In retained `velocity_v1`
recipes these scale to world XYZ velocity, limited to (2,2,0.8) m/s. The default
`acceleration_direct_v1` Stage1 preset scales them to world XYZ acceleration,
limited to (8,8,4) m/s² at50 Hz. Acceleration means the nominal translational
request excluding gravity, not measured acceleration or IMU specific force.
Neither interface enforces workspace containment. The same99-column Stage1–3
layout is retained; the previous-action and privileged delay-queue columns take
the selected interface's units and scales. Acceleration checkpoints carry the
new run schema `ppo_stage1_acceleration_direct_v1` and explicit actor control
metadata; cross-interface actor transfer/resume is rejected. Existing artifacts
retain velocity semantics. Stage4's velocity residual and scripted return, and
velocity demonstration cloning, are explicitly unavailable for acceleration.

The separate value network has causal history plus privileged numerical, motor,
queue, and integration state; it is unnecessary for actor inference. Networks
and optimizers use float32; physics and analytic KL reduction use float64.
PPO uses five current-cohort epochs, minibatches 256, learning rate 0.0003,
clipped ratio 0.2, gamma 1, GAE lambda 0.95, value-loss weight 0.5, gradient norm
cap 0.5, and Gaussian mean-KL limit 0.02. Reward scale 100 is applied once before
GAE. Complete episodes are collected with frozen actor/value weights and their
data are discarded after optimization.

Actor/value Adam optimizers are independent. Up to eight post-step actor trials
restore identical pre-step weights/moments before trying smaller learning rates.
Accepted steps must be finite and satisfy the sampled-cohort KL gate. Rejection
stops actor steps while value fitting completes its passes. This empirical
mean-KL safeguard is not a per-state trust-region guarantee. Entropy counts
completed training episodes, including invalid lanes; evaluation does not advance
it. Actor transfer starts with fresh critic, Adam, RNG, and local clocks;
`transfer_entropy_clock` selects inherited exploration progress or a stage-local
restart. Exact resume
retains them and rejects changed execution sources/settings except its explicit
supported budget override.

## Shared policy interface (proposed)

The shared live policy runner and real-flight integration remain proposed.
On2026-10-14 the user approved direct acceleration outputs. That controller/action
profile is now implemented for PPO training and deterministic evaluation, with
separate Stage1–3 acceleration presets, starting with
[Stage1](../configs/learning/whip_stage1_ppo_acceleration_v1.json).
The earlier integrated-PVA proposal is superseded; no reference generator or new
reference-error observations were added. The ROS bridge defaults to PVA and now
has an explicit direct-acceleration mode for the standalone circle test; it has
not been connected to this policy. Existing velocity presets/checkpoints
retain their original behavior and units. No new reward/curriculum, fitted model,
onboard configuration or flight result is established by this change.

### Confirmed interface gap and recommended command path

[Training](../tasks/whip/sac_environment.py) dispatches explicitly by action interface.
Retained velocity recipes scale three normalized actions into world XYZ velocity
and call [the velocity controller](../controllers/full_state_controller/tensor_controller.py).
Its translational acceleration is `Kv * (v_requested - v_tracking)`. The new direct
acceleration path uses the scaled action itself in this force conversion, with
no translation feedback. Both hold initial yaw. The
[interactive native controller](../controllers/full_state_controller/geometric_controller.py)
instead uses `a_ref + Kp * (p_ref - p_tracking) + Kv * (v_ref - v_tracking)`.
The [ROS bridge](../deployment/ros_bridge.py) defaults to this PVA interface;
`aerowhip.acceleration_only=true` explicitly selects the nominal direct path.
Integrating policy velocity into a position reference therefore changes the
closed-loop system; setting position to the latest measured pose also fails to
remove position feedback after transport/controller delay.

The [installed-source audit](../tmp/policy-contract-20261011/source-and-codec-audit.json)
records Crazyswarm commit `fcf51e27fe675bbdb564a40f4ca1ee4336c062ea`, firmware-build
commit `16c5630707e92830dcd16455a57b5560e5838735`, cflib 0.1.28 and inspected file
hashes. The hardware YAML requests `stabilizer.controller=2` (Mellinger), while
the flashed firmware and active onboard parameters remain unverified. The
inspected firmware FullState decoder selects absolute XYZ control. Its Mellinger
translation branch uses P/V/A plus position integrals in that mode; the
non-absolute branch does not implement the training velocity-only servo.
Crazyswarm's real servers expose `cmd_velocity_world`, but the installed Python
client method is commented out and that topic is unsupported by our custom
bridge. A velocity subscription alone would not close the controller gap.

The user approved `acceleration_direct_v1` on2026-10-14: three bounded
world-acceleration requests feed force construction and the existing attitude
controller directly. This supersedes the unimplemented `acceleration_ref_pva_v1`
generator proposal. The direct path computes
`F_des = m_drone * a_requested - (m_drone+m_cable) * g_world`.
It does not add a COM/tracking-offset acceleration correction or cancel dynamic
cable loads. Requested and achieved acceleration are therefore distinct. This is
a nominal simulation controller, not a claim of Crazyflie Mellinger equivalence.

The separate nominal PVA controller computes
`F_des = m_drone * (a_ref + Kp*e_p + Kv*e_v) - (m_drone+m_cable)*g_world`,
then desired attitude, collective thrust and body torque. Static cable-weight
trim is not dynamic cable-load cancellation; motor response and cable loads
mean `a_actual` need not equal `a_ref`. The inspected firmware instead uses its
configured mass multiplying acceleration and gravity, plus force-domain P/V/I
terms. Its gains and mass are not interchangeable with nominal simulation
parameters. Acceleration feedforward is supported by both paths, but this does
not establish controller or force equivalence.

The deeper [Mellinger source/configuration review](../tmp/policy-acceleration-20261011/mellinger-review.json)
retains the force, attitude and command-output source excerpts and YAML hash.
For `FullState` absolute-XYZ mode the firmware computes
`F_des = m_fw*(a_ref-g_world) + Kp_force*e_p + Kd_force*e_v + Ki_force*I_p`,
where the integral is accumulated and clamped. It aligns desired body +Z with
`F_des`, constructs the other axes from desired yaw, and projects desired force
onto the actual body +Z for collective thrust. It uses the commanded quaternion
for yaw in this path; force determines desired roll/pitch. The subsequent
attitude/rate loop outputs legacy commands with thrust scaled by `massThrust`;
these are not the nominal simulator's SI body-torque commands.

The adjacent Crazyswarm `config/crazyflies.yaml` requests `ctrlMel.mass=0.144 kg`,
`massThrust=17700`, `kp_xy=0.4`, `kd_xy=0.3`, `kp_z=2.5`, `kd_z=0.4`,
`ki_z=1.5` and `i_range_z=15`. These are requested controller parameters, not
measured physical mass or verified active onboard values. Unlisted gains may
retain firmware defaults or persistent settings. The inspected source selects
`ATTITUDE_RATE=500 Hz` for Mellinger, whereas the nominal simulation uses 200 Hz
and has no integral loop. Matching command units alone does not close these
controller, timing, load-trim or actuator differences; the audit did not change
firmware/controller defaults.

The implemented training path is **direct acceleration-to-force control**, with
no position/velocity reference feedback. Learned-policy deployment remains
unimplemented; the standalone stock-Mellinger interface experiment below uses a
host PID instead of a learned policy. In
the inspected source, setting all XY/Z
translational P/D/I gains to zero would algebraically retain the acceleration
and gravity force terms in the same `FullState` mode. These gains are exposed as
persistent `ctrlMel` parameters, so new firmware is not necessarily required.
The user authorized a standalone takeoff/circle/landing script on2026-10-15;
its distinct controller configuration and restoration are described below.
This does not establish matching simulation/training or authorize policy flights. Simply filling the
`acc` field of `FullState` does not disable existing feedback, and gain changes
cannot be inferred from a packet. Other routes include legacy attitude/thrust
commands, which are not an SI force-vector interface and are unsupported by
this simulation bridge. No onboard gains or firmware were altered. The user
selected direct
acceleration for training; implementing its real-flight command path still
requires a separate controller/transport change and validation.

#### Standalone cf_2 acceleration-interface circle experiment

The user-requested
[`scripts/acceleration_circle_cf2.py`](../../Crazyflie/crazyswarm2_jazzy_whipping/scripts/acceleration_circle_cf2.py)
is a simulation/real-flight script in the adjacent Crazyswarm checkout.
`AEROWHIP_SIMULATION=1` (the default Docker client shell) selects simulation;
`--simulation` or `--real` explicitly overrides that selection. Normal invocation
attempts flight in the selected mode; `--check` performs read-only ROS preflight and `--dry-run` samples
the reference without ROS. It selects only `cf_2` and uses the shell's ROS domain.
Real mode requires one real Crazyswarm server, one OptiTrack `/poses` publisher, fresh
`cf_2` status, and stock Mellinger (`stabilizer.controller=2`) in the server mirror.
It starts from a stationary, level, disarmed pose assumed to be on the floor.

Default reference: rise1.3 m relative to that tracking-origin height over6 s,
hover3 s, execute one0.2-m-radius horizontal circle over16 s, hover2 s, descend
in6 s to1.5 cm below the starting height, then allow up to3 s for settling.
The circle center is0.2 m along world +X from the initial pose, so the circle
extends0.4 m in +X; a quintic angle schedule starts/stops at rest. Initial yaw
is held. Landing acceptance requires proximity to the starting position and low
measured speed for0.5 s; this is a pose-based cutoff heuristic, not a contact sensor.

The host runs `a = a_ref + Kp*(p_ref-p) + Kd*(v_ref-v) + Ki*integral` targeting50 Hz
using OptiTrack positions and a causal filtered velocity in real mode, or the
native tracking-origin position/velocity in simulation mode. Gains are4 s^-2,
3 s^-1 and0.6 s^-3 on each axis, with integral contribution limited to1.5 m/s^2;
integration freezes on acceleration saturation. Horizontal acceleration magnitude
is limited to2 m/s^2 and vertical acceleration to[-3,3] m/s^2. These are uncalibrated
experiment settings, not fitted dynamics or validated performance limits.

In real mode, before arming, the script saves the server-mirrored gains, zeros all XY/Z
translation P/D/I gains and both translation integral ranges, and leaves mass,
thrust calibration and attitude gains unchanged. Every motion phase uses
`/cf_2/cmd_full_state` with only acceleration changing, initial yaw, fixed initial
position placeholders, zero velocity and zero body rates. Acceleration is world
m/s^2 excluding gravity; onboard Mellinger adds gravity and constructs attitude
and thrust. There are no high-level takeoff/goTo/land requests. This is a host
trajectory controller testing an acceleration interface, not an acceleration-only
learned policy or an open-loop trajectory guarantee.

Simulation mode verifies the custom server identity, session, reset generation,
world frame, simulation clock and feedback before sending commands. It starts
from idle PVA hold and waits up to 10 wall seconds for the same stationary/level
limits; this does not establish that the cable has settled. It disarms, drains
the modeled delay, selects `aerowhip.acceleration_only=true` and arms. Motion uses
simulation time targeting 50 Hz, with no catch-up bursts; actual request timing
is retained in evidence. ROS service/clock-progress waits use bounded wall time.
The final stream must be consumed before disarm so topic/service reordering
cannot deliver a late packet after disarm. Successful completion restores PVA,
allowing an ordinary waypoint script in the same session. Pause, stale feedback,
clock stalls, reset/session changes or flight-bound failures stop the simulation;
Reset/Play or a restart is then required. This is a free-space return to the
starting height with the unchanged settling limits, not physical touchdown.
Residual cable motion can fail the settling check even when the drone appeared
stationary at entry. Nominal cable-weight trim and the SI wrench/motor model
remain distinct from stock Mellinger firmware; this test does not validate real
flight or deploy a learned policy.

In real mode, fresh onboard status confirms arm/disarm. Pose loss over0.15 s, status loss over
2.5 s, command-loop delays over0.12 s, graph changes, or configured motion bounds
cause a selected-robot emergency-stop/disarm request, not a recovery landing.
Ctrl+C/SIGTERM does the same. Changed parameters are restored after confirmed
disarm, or after a setup failure before any arm attempt. If disarm is unconfirmed,
restoration is withheld to avoid re-enabling feedback with placeholder setpoints.
SIGKILL, power loss and radio loss cannot guarantee cleanup. Per-run original
parameters, requested/measured motion, sent acceleration and cleanup outcome are
saved under the Crazyswarm `logs/acceleration-circle-cf2-*` folder.

The installed cflib ROS server acknowledges parameter writes when queued and
serves cached parameter reads. The script verifies these server mirrors and waits
before arming; this does **not** independently verify firmware application of the
zero gains or restoration. The actual flashed firmware, gain application, mass/
thrust calibration, contact behavior and flight remain unverified. No changes
were made to the coworker's force controller, radio server, YAML defaults,
training rewards or physical model.

Current `velocity_v1` checkpoints must not be relabeled as acceleration policies.
The implemented profile changes action units and controller semantics, requiring
compatible new training. Cross-interface actor initialization and exact resume
are rejected even though both networks have three outputs. Same-interface
transfers retain the existing source/compatibility checks. The selected bounds
are nominal experiment settings, not identified hardware limits; firmware
equivalence remains unverified.

### State and observation contract

The proposed source-neutral state envelope identifies robot, world frame,
tracked body origin, body-axis calibration, model/topology, session/reset
generation, sample sequence, source timestamp, receipt timestamp and aligned
control time. Fields carry validity and provenance independently; a measured
pose does not make a derived velocity or simulated cable measured.
The real viewer's prescribed-motion cable simulation is now implemented, but
policy-state/history assembly, measurement-corrected cable observation and
real-flight policy execution described here remain proposed and unvalidated.

| State field | Meaning and units | Simulation flight | Proposed real-flight source |
|---|---|---|---|
| Tracking position `p` | World XYZ, m, calibrated tracked top-plate origin | Native simulated state | Calibrated motion capture or an explicitly selected estimator |
| Orientation `q`, rotation `R` | Unit XYZW quaternion; body-to-world rotation | Native simulated state | Calibrated rigid-body orientation or onboard estimate |
| Tracking velocity `v` | Velocity of that same tracked origin, world m/s | Native simulated state | Timestamped estimator or causal derivative, labeled estimated |
| Angular velocity `omega` | Physical body XYZ, rad/s | Native simulated state | Calibrated gyro/estimator, with explicit unit/frame conversion |
| Cable `q_i`, `v_i` | Node/marker position and velocity, world m and m/s; ordered topology | Native simulated cable | Causal cable observer, labeled estimated |
| Net acceleration | World `dv/dt`, m/s²; optional diagnostic | Native endpoint diagnostic | Explicitly estimated; not raw IMU specific force |
| Issued reference and action | Requested acceleration, fixed yaw, normalized actor request and timing | Known command history | Proposed known command history, not claimed onboard applied state |

The topology and attachment transform come from the selected model/calibration;
the current 11-node cable is one supported model, not a global interface constant.
For offset `r_attachment`, attachment kinematics are
`p_attachment = p + R*r_attachment` and
`v_attachment = v + R*(omega cross r_attachment)`. COM, tracking origin and
motion-capture rigid-body origin must not be interchanged implicitly. Cable
observer dynamics are driven by measured/estimated attachment motion while
preserving cable history; repeatedly resetting the cable or overwriting a
two-way coupled simulated drone with measurements is not this observer design.

The actor adapter must reproduce its registered column order and scales:
tracked displacement `(p-p_episode_start)/L`, row-major `R`, tracking velocity
`v/(4 m/s)`, body rate `omega/(4 rad/s)`, attachment-relative cable position
`(q_i-q_0)/L` and velocity `(v_i-v_0)/(4 m/s)`, previous normalized actor action,
and checkpoint-defined normalized episode time. In retained velocity training the previous
action means normalized velocity; in direct-acceleration training it means
normalized acceleration.
Its static goal remains
`(target_world-p_episode_start)/L` in world axes. This is not a yaw-rotated goal
or current drone-to-target error. Motor/wrench, BDF2, delay-queue and material
state remain privileged/model diagnostics; they are not required actor inputs.
Absent required cable/velocity/history data cannot be replaced with zero fields.

Existing Stage1-3 frames are 99 columns and Stage4 frames 101 for the current
mesh, with motor/wrench columns excluded by the registered actor selection.
Direct acceleration keeps the existing99-column Stage1–3 actor layout, with no
integrated reference or extra reference-error fields. Its action units, limits
and policy rate are explicit in the checkpoint interface. Stage4's velocity
recovery architecture has not been adapted. Topology, selected columns, scales,
task clock, goal/phase fields, device and actor batch shape remain registered.

[History encoding](../training/sac.py) is causal and shape-dependent. Preserve
the checkpoint's sample spacing, history length, prefix mask and episode index;
future-frame masking does not authorize filling missing past samples. Stage4's
451-sample rolling history and phase logic remain checkpoint/task-specific. A
simulation-feedback source must deliver coherent 100-Hz samples before this
encoder is used; the current latest-frame mailbox can coalesce output steps and
is not sufficient evidence of an uninterrupted actor history.

### Implemented training action and timing contract

The default direct-acceleration preset uses50-Hz actions and100-Hz observations.
A finite three-vector `u` in `[-1,1]` gives
`a_requested = (8,8,4) * u` in world axes, in m/s² excluding gravity. These initial
bounds equal the retained velocity controller's zero-speed full-command requests:
`(4,4,5) s^-1 * (2,2,0.8) m/s`. This does not make the two controllers equivalent.
The downward bound must keep desired vertical force positive, preventing a
zero-force/heading singularity anywhere in the action box. No jerk, speed or
position clamp was added. Zero action does not brake an already moving drone.

Fixed initial yaw accompanies each acceleration command. Commands are held for
20 ms; the existing30-ms delay applies once on the100-Hz observation grid. The
attitude controller runs at200 Hz with zero desired body-rate feedforward.
Per-motor allocation/clipping and first-order response remain unchanged. Both
native evaluation and CUDA collection call the same selected tensor controller.
No translation-reference integrator, position feedback or velocity feedback is
present in the direct path. These clocks and limits are nominal simulation
settings, not measured firmware timing or identified hardware capabilities.

The default Stage1 preset copies the previous compact v5 task, reward, budget
and learner settings; only action interface, limits and policy rate change.
No training was started by this interface implementation. Stage2/3 launch presets
likewise copy their previous cone25 v4 tasks/rewards, changing only the control
interface, limits and rate; they require compatible acceleration actors. Stage4
and velocity demonstrations are blocked for acceleration
rather than reinterpreted. This change does not approve a new curriculum.

Saved acceleration replays identify action interface, command units, limits and
policy rate. `command` and `applied_command` contain requested/delayed acceleration
plus yaw; motor thrust and requested wrench retain their existing units. Physical
acceleration remains a separately derived diagnostic. Actor observations keep
100-Hz causal history; the action/critic delay queue is normalized by acceleration
limits. Checkpoint schemas and actor manifests prevent three velocity outputs from
being silently relabeled as acceleration.

Live policy ownership, inference deadlines, ROS/radio timing and a direct-force
firmware command path remain unimplemented. The bridge's default PVA behavior
is unchanged; its explicit acceleration selector applies only to custom simulation.
Sending acceleration in real `FullState` does not disable onboard P/V/I feedback. Transport and firmware equivalence require separate verification.

The audit's captured cflib FullState packet confirms signed-int16 millimetre,
mm/s, mm/s² and milliradian/s fields, plus compressed quaternion. The rate
packer and firmware decoder use rad/s despite the installed cflib docstring
listing degrees/s; no degrees conversion belongs in this FullState path.
Nonfinite or unrepresentable encoded fields must fail rather than wrap or be
silently clamped. The current ROS/native bridge does not emulate radio
quantization or quaternion compression. A future transport profile must retain
backend/codec identity and compare commands after the actual encode/decode;
nominal unquantized PVA and effective firmware reference are distinct evidence.
The no-radio probe checked scalar packing/range rejection, not quaternion
decoding, onboard timing or controller behavior.

### Lifecycle and staged validation

YAML startup/Reset still initializes the selected simulation robot exactly.
Policy activation is a separate explicit episode boundary after checking the
checkpoint's required initial domain, cable state and history availability.
An arbitrary YAML pose, including `z=0`, does not make a fixed-start actor
compatible. No flight start, takeoff, floor contact or trained-start tolerance
is inferred from this contract. Task/reward/phase logic stays outside physics
and retains the existing research definitions.

Session/reset changes must clear actor history and command state together. Invalid
frames, missing required data, stale input beyond configured limits, inference
overruns and transport rejection must be recorded and stop policy ownership;
continuation uses an explicitly chosen mode-specific supervisor action. The
simulation's pause/hold/emergency behavior is not a physical landing or real
emergency-stop model. Freshness/timeout parameters require explicit selection;
none is presented here as a calibrated flight limit.

Remaining live integration should proceed through separate evidence gates:
common encoder reconstruction; direct-acceleration controller/codec checks; one
compatible policy through the ROS simulation compared with direct execution at
recorded command boundaries; new-data drone/controller/actuator identification
and held-out flight comparison; then a causal cable observer compared against
simulated truth and newly recorded cable measurements. Live policy input must
use the observer's estimate in observer evaluation. Numerical equality,
transport success, model fitting, policy performance and real-flight results
remain separate. No scientific acceptance threshold or model promotion is
established by this design document.

## Four stages and scientific criteria

**Requested restoration,2026-10-22 — learning sequence selected; experimental
Stage1 excitation implemented, later-stage adaptation still proposed.** The user requested excitation
or "energy collection" first, delivery/strike through a broad plane second,
and a target hit third. The fourth stage retains the earlier recovery purpose.

| Stage | Intended learning responsibility | Evidence to retain separately |
|---|---|---|
| 1 — Excitation/preparation | Learn to excite the cable and develop useful cable motion; sidearm and overhand preparation are both eligible. No precision-hit requirement. | Shape evolution, nodal speed timing, distal response and raw amplitudes; an excitation guide is not automatically a physical-whip certificate. |
| 2 — Broad delivery | Turn the learned excitation into a release that actually crosses the broad target rectangle in the intended direction. | Actual interpolated forward tip crossings and their association with the same release; proximity alone does not establish a strike. |
| 3 — Precision | Improve where that strike lands, with progressively narrower target geometry. | Actual hit/error measurements at each target level, preserving validity and release quality. |
| 4 — Recovery | Return the drone and settle the cable after the strike while preserving the learned strike. | Post-strike drone return and cable-settling behavior. |

"Energy collection" describes the user's intended preparation behavior; neither
the historical mechanics proxy, unified-fold score nor new excitation guide measures
physical stored energy or energy transfer. Maximizing an unqualified total-energy
quantity could reward whole-system motion or height without useful distal output;
this is a design risk, not a newly observed training result. The recommended
Stage1 design therefore retains cable-response evidence as well as excitation.
The implemented Stage1 reward and fresh initialization are specified below.
Later-stage reward expressions and transition thresholds remain to be settled;
no automatic promotion rules are implied by this table. Earlier human
aggregate data and counterexample diagnostics remain measurements, not a fitted
classifier or grounds for declaring old policies successful.

A structural restoration should preserve the current 50 Hz acceleration interface,
unchanged cable physics, sidearm allowance and approved soft workspace handling.
A literal old-preset rollback would instead restore 10 Hz velocity commands,
historical path discounts, scales and exploration settings; that rollback has
not been performed. Retained Stage4 recovery currently supports velocity only,
and configuration validation rejects acceleration Stage4. It needs an explicit
adaptation before a complete acceleration-based four-stage campaign can be
launched. Earlier 120-collection-per-stage planning is not replaced by the old
150/80-collection budgets merely by choosing this structure. The user separately
authorized the fresh120-collection Stage1 excitation run described below.

### Fresh flexible-excitation Stage1,2026-10-22

The user authorized starting Stage1 after the paper-informed, direction-inclusive
measurement work. [ExcitationScorer](../tasks/whip/excitation.py) uses the existing
attachment-anchored weighted least-squares decomposition: subtract attachment
translation and the fitted whole-cable rigid rotation from nodal velocities.
Its geometric rest-length weights are not physical mass/energy weights.
At each100Hz post-initial sample, define

`q[k] = min(||v_tip_world[k]||, ||v_tip_shape[k]||)/(1 m/s)`,
`q[0]=0` (unpaid baseline), and `b[k]=max(q[0:k+1])`.

Current v3 valid return is `F(b[T]−C_workspace)`. A failed episode returns
`−10+F(−C_workspace_accrued)`, where `F(x)=x` for `x>=0` and `5x/(5-x)`
otherwise. Terminal accounting claws back all earlier guide credit. Differences
of `F(b[k]−C_accrued[k])` are aggregated into50Hz policy transitions, with
reward scale1 and discount1. The two speed norms always use the same sample.
There is no fold, peak-order, propagation, gain or target-direction gate, and no
new smoothing window. Sidearm and overhead deformation can both score. Pure
translation or rigid rotation cannot intentionally earn nonrigid credit, and a
stationary world tip receives zero credit. Numerical roundoff is not thresholded.
A positive guide, including tiny roundoff, is not a physical-whip success flag.
Mostly rigid preparation can receive little credit until deformation develops;
localized flapping can score. Independent full-motion review is therefore needed.

The current approved outside-only soft cost is uncapped:
`C_workspace = (1/450) * sum_k>=1 10*(radial_excess_m^2 + vertical_excess_m^2)`.
It uses `squared_excess_v1` with weight0.1and distance scale0.1m. Increasing
boundary excess increases raw cost at every finite distance; there is no
per-sample cap. Negative **net returns**, rather than individual charges, are
compressed by F. Valid returns exceed-5; failures lie in(-15,-10] in real
arithmetic. Extreme bad outcomes have weaker return sensitivity; this is not a
learning guarantee. Historical v1 had untransformed Q-C and v2 capped each
sample's density at5; both remain selectable historical presets. All use
horizontal radius0.55m and world height1.5–2.6m. Failure uses only the finite
observed prefix but keeps the full450-observation denominator. All movement
inside is free of workspace cost. Full drone path is recorded diagnostically;
path, acceleration and time costs are zero. Physics, cable model,60-degree tilt
and0.05m cable-height failure envelope are unchanged.

The preset uses seed5, random actor AND critic,128 lanes and120collections
(15360episodes),50Hz world acceleration limits(8,8,4)m/s²,4s active plus0.5s
settling, and100ms AR(1) action correlation. The approved exponential entropy
coefficient is `0.001+0.019*exp(-completed_episodes/2560)`; its initial value is0.02
and decay scale20collections, with no hold. No prior actor/resume is allowed for
this fresh launch. The preflight actor is never a production initializer.

All-lane guide, world/nonrigid speeds, path, workspace cost, validity and net
return are retained separately. Saved stochastic motions include median/weak
valid responses and a failed prefix when available, as well as maxima; target
crossings remain diagnostics. The displayed guide is not labeled whipping
success. No automatic Stage2 handoff or numerical acceptance threshold has been
established. This experiment must be judged under its own reward and retained
whole-attempt measurements, not treated as a continuation of older reward runs.

Previous run status,2026-10-22: the user requested addressing Stage1 and restarting.
The worker stopped cooperatively at39collections/4,992episodes; original final
and best checkpoint hashes and stopped summary are retained in
[stop evidence](../tmp/stage1-excitation-restart-20261022/stopped-run.json).
In collections1–3, valid mean returns were approximately−109/−117/−135versus
failed mean returns−32/−26/−36. Later32–39each had0/128valid episodes. Saved
finite failed trajectories ended at the cable-height limit. Terminal accounting
matched the configured objective; the evidence supports an objective-ranking
problem, not a demonstrated PPO or numerical-simulation defect.

Historical v2: the [then-approved correction](../tmp/stage1-excitation-restart-20261022/approved-contract.json)
used existing bounded-density cost machinery with weight5and distance scale
sqrt(0.5)m: `C=(1/450)*sum min(10*(radial_excess_m^2+vertical_excess_m^2),5)`.
It preserves the current near-boundary quadratic, zero cost inside, full-horizon
denominator, excitation, physical model, control and failure clawback. Since
q>=0and0<=C<=5, every valid episode has return>=−5and every failed episode has
return<=−10. Saturation beyond combined excess0.7071m can underprice extreme
excursions; containment and motion review remain necessary. This approved
research cost change is implemented as a separate preset without changing the
scorer or physics. Learning remains unvalidated. The fresh120-collection restart uses seed5,
random actor/critic, fresh optimizers and zero entropy offset with the same
exponential schedule. It does not resume the failed actor.

The [counterfactual arithmetic audit](../tmp/stage1-excitation-restart-20261022/counterfactual-cost-audit.json)
reconstructed old costs on73curated retained trajectories,34valid, to maximum
absolute error2.84e−14. The independent NumPy bounded cost matched existing
Torch bounded-density arithmetic to8.88e−16. These are saved-trajectory
accounting checks, not independent policy trials or a physics rerun. A narrow
configuration guard now admits exactly this bounded profile for flexible
excitation and preserves the older approved profiles. Seven new CPU accounting
checks and retained regression coverage pass:92total. One historical discovery
test initially failed because it used the mutable launcher default as its old
compact-acceleration baseline. It now explicitly selects that preserved baseline;
assertions and tolerances are unchanged, and the initial failure log is retained.
Native preflight is complete:32initial trials,17valid, unchanged initial actions
and physical histories bit-for-bit versus the matched original preflight, plus
independent projection/cost/return reconstruction. PPO performed117actor and120
critic updates with final conditional KL0.0198186245<0.02; the subsequent native
deterministic replay was valid. The preflight actor is never an initializer.
The strict float32 sum check remains FAILED at1.90735e−6against unchanged1e−6;
exact stored-vector casting passed and independent float64 telescoping error was
1.78e−15. This distinguishes rounding diagnostics from accounting correctness
without changing tolerance. See [prelaunch review](../tmp/stage1-excitation-restart-20261022/prelaunch-review.json).
The fresh production restart [20261022-160251-474373](../runs/ppo_stage1/20261022-160251-474373/)
was launched, stopped at11collections, and its exact continuation completed120
total collections before the climb audit. Round0 was captured and independently reconstructed from
random seed5; actor/critic bit identity, empty optimizers, null initializer/resume,
128lanes/15,360episodes, approved bounded workspace profile and initial entropy
clock0/0.02were verified. Only one training worker was running. This establishes
launch identity, not subsequent learning or successful whipping.

### Proposed broad-delivery Stage2 after flexible excitation,2026-10-22

The user endorsed the preceding Stage2 direction in principle and requested a
careful curriculum and flexible training budget. The [original proposal](../tmp/stage2-flexible-delivery-design-20261022/proposal.json)
is preserved; the [expanded curriculum v2](../tmp/stage2-flexible-delivery-design-20261022/curriculum-v2.json)
records the current **design only**, with new progression/budget/evaluation
details proposed. Stage2 has not been implemented or launched. That proposal
predates the approved v3 transformed-return Stage1 restart; its workspace-cost and return-transform inheritance
must be explicitly reconciled before Stage2 implementation. The Stage1 restart
approval does not launch Stage2. This supersedes no archived task.
The existing `directed_shape_delivery_v1` is unsuitable unchanged because it uses
withdrawn release windows and path cost, while current Stage1 records are speed
improvements rather than release intervals.

Proposed fixed rectangle: world center(1.1,0,2)m, normal+X, width1m, height0.6m.
At one observed or interpolated state let
`q=min(max(v_world_tip·normal,0),||v_shape_tip||)/(1m/s)` and `phi=q/(1+q)`.
Only actual world velocity is directed; the fitted residual's direction is not
a physical eligibility gate. Approach guidance is
`A=phi*exp(-0.5*(distance_to_finite_rectangle/0.70m)^2)<1`.
A confirmed forward rectangle passage has candidate utility `B=5+phi(q_cross)`.
The raw objective is `max(0,max A,max B)`, with initial sample unpaid. Valid
return subtracts the unchanged soft workspace cost; failure returns−10 minus
accrued workspace cost, even after an earlier hit. Maximum increments prevent
adding multiple hits, dwelling time or an unrelated global excitation peak.
The bounded quality premium limits reward preference for rare very fast hits;
it is not lexicographic hit-rate optimization. Bonus5, width0.70m, speed scale1m/s
are engineering choices, not physical definitions. The expanded curriculum
and its acceptance/selection rules are proposed for review before implementation.

Crossing evidence must confirm opposite-side arrival, not merely touch the
plane. For negative-to-positive intervals, interpolate the entire cable state
and recompute the anchored residual there. For an exact-zero plateau between
negative and positive states, use its **last zero sample before positive
confirmation**, including rectangle membership and velocity at that departure.
This explicitly differs from the existing offline first-zero arrival convention
and prevents an inside arrival/outside departure from being called a rectangle
passage. Positive interpolated world-normal velocity is required. No prior
negative sample, backward passage, touch-and-retreat or unconfirmed terminal
contact earns a hit bonus. Award only when confirmation is observed; retain the
full episode and all failures. This is sampled geometric evidence, not proof of
physical impact, force-mediated causality or an event detector.

Proposed curriculum: retain the same center, +X normal, target distance, reward,
workspace and control throughout. Default to the intended1.0m-wide by0.6m-high
rectangle. An optional2.0m-wide by1.2m-high finite rectangle is a single bridge,
not an infinite/closer plane or a separate reward. A128-trial frozen Stage1
baseline scores both rectangles on the same trajectories; bridge entry is proposed
only with at least122valid episodes, at most25valid final-rectangle hits, and at
least64valid larger-rectangle hits, plus motion evidence of an extent problem.
Stage1 must already have passed its independent excitation and validity review.
A completed budget or guide-positive return does not qualify for transfer.

The optional bridge is reviewed after20and40collections. Advancement requires
two independent128-trial checks,20collections apart, each with at least122valid
and103valid bridge hits, plus valid deterministic delivery and full motion review.
The baseline can count as the first check only if it meets all those conditions.
If unmet at40, diagnose and record a new decision rather than force a switch.
Always report intended-target hits too. Once on the intended1.0x0.6m rectangle,
train at least80collections there and at least120Stage2 collections overall.
Consolidation keeps this geometry and reward unchanged; center accuracy and
further target narrowing belong to Stage3. Current actor observations do not
include target dimensions, so this proposal uses a shared cohort geometry rather
than unobserved per-lane random sizes. Internal target changes occur only between
complete on-policy batches, retaining actor/critic/optimizer/RNG/entropy state.

Initial planned budget is120collections x128lanes =15,360new episodes, including
any bridge. At120and subsequent20-collection boundaries, review whether further
learning is useful. Each extension adds2,560episodes. Compare independent
actual-hit counts or actual forward-plane miss distances over recent checks,
net return, validity, workspace excursions and motion. Reward growth alone is
insufficient. One explicitly recorded20-collection confirmation block is allowed
for an inconclusive plateau; do not renew it indefinitely without new behavioral
evidence. Persistent failure/exploitation triggers diagnosis.240collections
(30,720episodes) is a mandatory research-review boundary, not a promise of
convergence. No silent reward, physics, safety or success-threshold revisions.
All these budget/bridge choices are proposed engineering rules.

Transfer uses the reviewed final Stage1 actor including learned scale, fresh
critic/optimizers/RNG and an explicit inherited entropy clock. The exponential
curve remains `beta(n)=0.001+0.019*exp(-n/2560)` on total inherited training
episodes. After120Stage1 collections, Stage2 starts at0.0010471. Neither internal
curriculum changes nor budget extensions reset exploration. Actual learned action
scale/spread and100ms AR1 correlation are reported separately from this coefficient.
Current transfer guards block this path; implementation needs a narrow reviewed
registration. Physics/control,4.5s horizon, outside-only workspace penalty and
zero additional path/acceleration/time costs remain unchanged.

Development checks use fresh128-trial seeds every20collections and deterministic
replay at serialized collection boundaries. Training RNG must be restored after
evaluation. Two final-geometry checks20collections apart should each reach
122valid and116valid confirmed broad hits out of128, alongside motion review,
before final assessment. Track the best final-geometry checkpoint separately;
a bridge reward maximum is not a comparable final-target checkpoint.

To avoid selecting a winner on its own acceptance test, evaluate frozen final
and saved-best final-geometry candidates on separate selection seeds
2026102501/2502/2503,128trials each. Retain both reports, select a primary using
motion acceptability, valid actual-hit count, validity and workspace cost, then
freeze that choice before using final holdout seeds2026102321/2322/2323.
The primary's each-seed final gate is at least122valid and116valid confirmed
broad hits out of all128; post-hit failures remain failures. Deterministic valid
actual delivery and full-motion review are also required. Failed holdout results
are not development data for this experiment; a revised experiment needs a new
recorded plan and untouched holdout seeds. These thresholds are observed sample
fractions, not confidence-bound guarantees of population reliability. Report
binomial uncertainty and limit conclusions to fixed-scene action-noise reliability.

Crossing-local speeds, residual magnitude, strict3m/s/30degree diagnostics,
containment and failures remain separate. Full motion review includes typical,
weakest-quality, high-return and high-cost attempts, misses/failures and post-hit
motion. Slow or nearly rigid passages earn the geometric bonus and cannot alone
establish flexible striking. No current policy has passed this proposed review;
there is no automatic Stage3 or hardware progression.

This performance-dependent difficulty principle is consistent with
[Florensa et al.2018](https://proceedings.mlr.press/v80/florensa18a.html), but the
paper does not establish these rectangles, reward coefficients, gates or budget
choices, and its GoalGAN algorithm is not being introduced.

A [read-only Stage1 snapshot through collection16](../tmp/stage2-flexible-delivery-design-20261022/stage1-readonly-health-through16.json)
found0/128valid full episodes in collections9–15and1/128in16. Lower workspace
cost and shorter failed trajectories accompanied higher return; retained failed
examples reached the cable-height limit. Not all failed lane traces are stored,
so this does not classify every failure. This is a current early-training health
warning, not a permanent-failure conclusion or evidence from obsolete policies.
Stage1 has not passed the handoff review. A [later snapshot through34](../tmp/stage2-flexible-delivery-design-20261022/stage1-readonly-health-through34.json)
found collections30–34each had0/128valid full episodes. This persistent training
health issue must recover or be diagnosed before transfer; it is not a final
frozen-policy result. All82production execution hashes still matched the active
run configuration. No worker/settings were changed.

The completed linear outreach/height campaign used these retained experimental
settings (the new crossing-priority Stage2 restart is described separately below):

| Stage | Learning preset | Budget and initialization |
|---|---|---|
| 1 | [outreach/height radius055 v12](../configs/learning/whip_stage1_ppo_outreach_height_radius055_150_v12.json) | 150 new collections; paused collection0 actor (identical to original collection121 actor), fresh critic/Adam/clocks, seed0 |
| 2 | [delivery radius055 v9](../configs/learning/whip_stage2_ppo_outreach_speed_radius055_150_v9.json) | 150 collections; final new Stage1 actor, fresh critic, seed1 |
| 3 | [precision radius055 v8](../configs/learning/whip_stage3_ppo_outreach_speed_radius055_150_v8.json) | 150 collections; final new Stage2 actor, fresh critic, seed1 |
| 4 | [recovery radius055 v6](../configs/learning/whip_stage4_ppo_outreach_speed_radius055_80_v6.json) | Prepared80-collection recipe; not launched by this restart |

Stages1–3 each completed19,200 new episodes at128 lanes in the retained campaign;
Stage3 did not reach its final target level. Their task presets are
[Stage1 v8](../configs/tasks/whip_stage1_outreach_height_radius055_v8.json),
[Stage2 v4](../configs/tasks/whip_stage2_outreach_speed_radius055_v4.json), and
[Stage3 v4](../configs/tasks/whip_stage3_outreach_speed_radius055_v4.json).
[Stage4 v4](../configs/tasks/whip_stage4_outreach_speed_radius055_v4.json) retains
learned recovery. Stage1 v8 was the preceding approved experiment; other stage recipes
retain their existing rewards. These do not replace launcher defaults. Bare `run_training.py`/`--stage 1`
now selects the experimental flexible-excitation Stage1 preset;
`--stage 2/3` select acceleration variants with their previous cone25 v4 tasks
and budgets. Stage4 remains velocity-only. The scratch acceleration Stage1 run
completed its budget but failed to learn; acceleration Stages2–4 have not run.
The separately approved2026-10-21 discovery preset has now started its first
training run; select it explicitly with `--settings`, as described below.
Older compact, level-outreach,40/80/160-collection and0.40 m presets remain
readable configuration variants, but their deleted runs are not active models.

The current nominal start is world(0,0,2) m and target center(1.1,0,2) m.
Stages1–3 execute the full4.5 s horizon. The requested horizontal direction is
world+X. The virtual target rectangle has no collision response.
Automatic evaluation uses one native physical lane, preserving CUDA actor batch
shape64: after the first collection, every512 training episodes, and at the end.
These evaluations are one nominal case, not independent robustness trials.

Mechanics proxy M combines bend transport/coherence, non-rigid release and
bounded forward motion over0.2/0.4/0.6/0.8 s windows,0.05 s stride and0.15 s tip
lag. U measures recovered chord/arc slack; D is squared positive forward
alignment with a plateau within25 degrees. Raw W=M*U*D remains a diagnostic.
These are approved task proxies, not an established physical whip classifier.
For rest cable length L=0.9525 m, the retained Gaussian-height event credit B
(earlier Stage1 speed presets and the unchanged Stage2–4 presets) is

`B(e) = [MU/(MU+0.05)] * D * r * H * [v_positive/(v_positive+3 m/s)]^2 * exp(-0.15*l/L)`.

Here `r=clip(dot(tip_world-attachment_world,requested_direction)/L,0,1)`,
`H=exp[-0.5*((tip_z-drone_z)/(0.35*L))^2]`, and `v_positive` is positive forward
tip velocity in m/s. The height width is0.333375 m; drone_z refers to the tracked
top plate at the same release, not the cable attachment or initial height.
The path l is actual tracked-drone travel in meters from episode start through
that observed release, calculated from100 Hz tracked-drone positions. All terms
share one complete observed event; future crossings or recovery states do not
rewrite earlier credit. Reach/height/speed alone cannot earn credit with MU=0.
The speed scale and height preference are research weights, not success thresholds.
`outreach_credit-preparation_cost` reconstructs B; raw `mechanics_credit` is W.

The completed linear outreach/height Stage1 experiment
(`directed_outreach_height_whip_v1`) instead selects
`max_e R(e)`, where

`K(e) = [MU/(MU+0.05)] * D * [v_positive/(v_positive+3 m/s)]^2 * exp(-0.15*l/L)`

`h(e) = clip(1 - abs(tip_z-drone_z)/L, 0, 1)`

`R(e) = K(e) * [0.5*r(e) + 0.5*h(e) + 2*r(e)*h(e)]`.

This linear height preference uses tracked-drone Z at the same observed release;
it is not a new physical success tolerance. The existing mechanics/unfolding,
alignment, forward-speed and actual-travel factors remain unchanged. All three
contributions share one complete event; separate maxima cannot be combined.
The new explicit weight fields are `outreach_reach_weight=0.5`,
`outreach_height_weight=0.5` and `outreach_joint_weight=2`. No Gaussian height
width is accepted in this mode. Diagnostic `level_height_factor` now means h,
and `level_outreach_quality` means r*h for this mode only. The three diagnostic
`outreach_*_credit` components are measured before path reduction and sum to
`outreach_credit`; subtract `preparation_cost` once to reconstruct R. They are
shown in Training alongside their total. Raw W remains a mechanics diagnostic.
Zero mechanics/unfolding or zero positive forward speed still earns zero credit.
The older Stage1 (`directed_outreach_speed_whip_v1`) selects `max_e B(e)`.
The retained Stage2 (`directed_outreach_speed_delivery_v1`) selects `max_e B(e)*(1+P(e))`.
P is the Gaussian of Euclidean distance from that same observed tip release to
the current target rectangle, with width0.35 m. Stage2's target is fixed1.0×0.6 m.
`approach_reward=B*P`, including the same preparation discount.
Stage3 (`directed_outreach_speed_precision_v1`) selects
`max_e B(e)*(1+P(e)+C(e))`, with
`C=exp[-||tip(e)-target_center||^2/(2*sigma^2)]` at the same world-coordinate
release. `center_reward=B*C`. Stage3 narrows its rectangle
1.0×0.6 →0.6×0.4 →0.3×0.2 →0.1×0.1 m with sigma0.25/0.15/0.075/0.05 m.
Current-target stochastic scores and final-target native scores are distinct.
`crossing_center_quality` is separately measured on the forward-plane interval;
it is zero without a crossing in that interval. Full-episode actual crossings
are measured independently of event selection.

The approved Stage2 restart uses
[crossing-priority learning v10](../configs/learning/whip_stage2_ppo_crossing_priority_radius055_150_v10.json)
and [task v5](../configs/tasks/whip_stage2_crossing_priority_radius055_v5.json).
Mode `crossing_priority_delivery_v1` defines positive credit

`0.2W + max_actual_forward_crossing [C + 2G + S]`,

where W is the approved linear Stage1 best-event R above divided by3, and
`C=1/[1+(lateral_error/0.25 m)^2+(vertical_error/0.15 m)^2]`. Errors are relative
to the target center at the interpolated actual crossing. G is the existing
geometric-hit indicator; S is its existing strict-qualified counterpart
(forward speed≥3 m/s, entry angle≤30 degrees). All three crossing contributions
come from one crossing; separate component maxima cannot be combined. No
crossing earns zero crossing credit. W and the crossing record are deliberately
independent and can occur at different times; crossing reward does not require
an eligible mechanics window and is not evidence of physical whip behavior.
The radius0.55 m, full4.5-s horizon, fixed1.0×0.6-m target centered at
(1.1,0,2) m, invalid-episode replacement and outside penalty are retained.
Before workspace costs, positive credit is bounded by4.2; a geometric miss
is bounded by1.2, a geometric hit earns at least2 and a strict hit at least3.
Workspace penalties can reverse net-return ordering.

The normalized whip support and the crossing each maintain a causal record.
Crossing credit becomes available at its observed interval endpoint, never
before that observation. PPO differences telescope to the same episode return;
a later invalid episode cancels positive credit with the existing failure rule.
`whip_support_reward`, `crossing_center_reward`, `crossing_geometric_bonus` and
`crossing_strict_bonus` reconstruct the positive reward. `crossing_reward` is
the total crossing credit; `crossing_reward_*` timing/errors describe its
selected crossing. Existing outreach/path components and event indices still
describe the unscaled linear Stage1 record. `target_proximity` means C at the
reward-selected crossing in this mode; whole-episode crossing diagnostics are
retained separately. Training plots show the actual four contributions plus
workspace cost. The new source is the completed Stage2
[latest actor](../runs/ppo_stage2/20261011-063417-108510/latest.pt), not a random
policy or an exact learner resume. Budget is150 new collections/19,200 episodes
at128 lanes; critic, Adam and stage-local exploration restart. Saved-array
review follows completion before any Stage3 decision. Existing presets/runs
and the Stage3/4 criteria remain preserved; no new Stage3 ladder is implemented.

For the older event-only modes, causal credit is the best complete event
available so far; repeated events do not accumulate credit. The new Stage1 event bound is3; its older Gaussian
variant has bound1. The unchanged Stage2/3 event bounds are2/3. Raw returns
across these different reward definitions are not directly comparable.
A separate outside-only squared-excess penalty uses weight0.1 and distance
scale0.1 m. The tracked drone's cylinder has radius0.55 m about initial XY and
world-Z bounds1.5–2.6 m. The cable is not constrained to the cylinder. This
boundary is soft: no command projection/filter or excursion termination is
introduced. Inside travel affects the existing event path factor, not an added
workspace charge. Acceleration cost remains zero.
Tilt above60 degrees, cable below0.05 m, nonfinite state or numerical failure
invalidates an episode; invalid event credit is replaced by failure return−1
while accrued workspace cost remains charged.

Geometric forward-plane crossings inside the current rectangle and strict
crossing diagnostics are separate. Strict crossing requires valid execution,
forward speed≥3 m/s and entry angle≤30 degrees. Final Stage3/4 geometry is
0.1×0.1 m, inclusive±0.05 m on both target-plane axes. The25-degree reward
plateau does not redefine qualification. Stage3 advancement requires all lanes
valid, at least80% geometric hits for three consecutive collections and an
updated-policy nominal geometric hit. Workspace is measured but does not gate
this curriculum. No multi-target, random-start or robustness learner is active.

Stage4 freezes the entire preceding Stage3 whip and trains only the internal PD
residual recovery head/logstd. It switches at the first causally observed
geometric hit or4.5 s miss timeout and then recovers toward episode-start drone
position for4 s. Recovery weight0.5 uses existing return/cable-settling terms;
a miss earns no recovery credit. Internal position gain is1.5 inverse seconds
and velocity damping0.4. The final1 s hold diagnostics require drone error≤0.1 m,
speed≤0.15 m/s, attachment-relative cable RMS speed≤0.2 m/s and horizontal RMS
extent≤0.1 m. Its positive bound is3.5. The existing campaign handoff requires
Stage3's final target level plus a valid, contained final native strict hit.
Failure is reported without bypassing a gate or relabeling a baseline as trained.
This restart evaluates that handoff after Stage3 without training Stage4.

Stage1's fresh entropy schedule is0.02→0.001; Stage2/3 use0.005→0.001. Hold
fraction0.1 and anneal-end fraction0.8 apply to each new19,200-episode budget.
Plateau detection keeps its thresholds and warns; it does not stop the approved
full budget. These are fresh runs, not exact resumes. Actor-only Stage2
initialization permits Stage1 or Stage2 sources; Stage3 permits Stage2 or
precision Stage3 sources. Role, identity, finite-weight, model/scene/control and
physical-source compatibility checks remain intact. Exact resume may change
only the total episode budget and retains the saved learner/exploration state.

[motion_search.py](../evaluation/motion_search.py),
[motion_refinement.py](../evaluation/motion_refinement.py), and
[demonstration_search.py](../evaluation/demonstration_search.py) remain available
for bounded simulated-motion diagnostics; [demonstration.py](../training/demonstration.py)
can clone simulated causal state/action sequences into an actor. These tools do
not alter plant physics or automatically install a policy. Their old datasets
and original searched clone were deleted with `runs/`. A replay of that clone's
closed-loop motion survived separately in
[the retained baseline](../tmp/stages1to4-80-20261010/stage1-seed-interface-baseline.npz).
The approved [reconstruction and cloning script](../tmp/stage1-replay-demo-150-20261010/build_seed.py)
recovered45 requested velocity actions at10 Hz from recorded samples1+10*k,
normalized by world-XYZ limits(2,2,0.8) m/s. Reintegration through the current
native plant reproduced tracked-drone position/velocity/rotation and cable
position/velocity exactly, remained valid for4.5 s, and was contained in the
current0.55 m workspace. The [new teacher dataset](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/demonstration.npz)
contains causal observation frames, action targets and encoder/interface
metadata; it represents the earlier closed-loop clone, not the original search
winner or a handheld recording. Its [reconstruction record](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/assessment.json)
retains replay, source-provenance and deleted-original-checkpoint hashes.

The fresh seed0 actor was fitted with the existing supervised bounded-command
MSE procedure for1,200 Adam steps at0.0003 learning rate, freezing scale-head
weights and selecting the lowest teacher-state error. There is no imitation
reward bonus. The [new seed checkpoint](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/cloning/actor.pt)
has zero PPO counters and fresh critic/optimizers/clocks. Its
[native closed-loop replay](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/cloning/cloned-actor.npz)
is separately evaluated from the supervised fit. The
[cloning assessment](../runs/motion_search/20261010-replay-derived-demo-radius055-v1/cloning/assessment.json)
reports teacher-state MSE9.36546e-5, nominal raw return0.0151354, selected-event
reach0.639620 m, forward tip speed5.01312 m/s and tip-to-drone height error
-0.737543 m, with full-horizon validity and no workspace breach. This is a useful
initial stroke, not attainment of near-level outreach or target/strict crossing
success. One nominal evaluation does not establish robustness or flight validity.
Handheld retargeting remains a proposal described under recordings below.

## Approved Stage1 discovery preset — implemented, training started

On2026-10-21 the user explicitly approved a separate directed-whip discovery
preset, initially without launching training; the user subsequently requested
starting Stage1. Select
[whip_stage1_ppo_acceleration_discovery_v1.json](../configs/learning/whip_stage1_ppo_acceleration_discovery_v1.json),
which uses [whip_stage1_discovery_bounded_v1.json](../configs/tasks/whip_stage1_discovery_bounded_v1.json).
The default and `--stage 1` still select the previous compact acceleration recipe.
From the repository root, this opens the selected setup without starting PPO:

```bash
.venv/bin/python run_training.py --settings configs/learning/whip_stage1_ppo_acceleration_discovery_v1.json
```

The new `directed_discovery_bounded_v1` mode in
[Stage1Scorer](../tasks/whip/stage1.py) uses `R = 0.5*max_e W(e) - 0.25*C` on
completed valid episodes and `R = -1` on failure. W is the inherited M*U*D proxy
described above, with the same windows, tip lag and25-degree alignment plateau.
It remains an experimental guide, not a validated genuine-whip classifier.
There is no reach multiplier, preparation-path discount, target accuracy/hit
bonus, acceleration penalty or time penalty in this preset. The fixed broad
target, geometric crossings and existing strict speed/angle hits are diagnostic.

The new `bounded_density_v1` [workspace penalty](../tasks/whip/workspace.py) is
`C = sum_t min(1, (radial_excess_t/0.1m)^2 + (vertical_excess_t/0.1m)^2)/450`,
using observed right endpoints on the100-Hz mesh, excluding the initial sample.
The denominator stays450 for failed prefixes; unobserved suffixes contribute
zero. C lies in[0,1]. Thus completed returns lie in[-0.25,0.5], strictly above
failure−1, and quiet contained hover earns0. The cylinder remains radius0.55 m
around initial XY, with world-Z1.5–2.6 m. Excursions incur soft cost; they do not
terminate execution. Existing physical failure conditions remain unchanged.
Capping cost reduces the distinction between large excursions; uncapped radial/
vertical excess, outside duration and containment remain recorded for review.

[Environment accounting](../tasks/whip/sac_environment.py) distributes increases
in the best causal weighted proxy minus accrued workspace charges. Terminal
correction on failure cancels all earlier credit and cost so the total is−1.
The pre-cast sum is checked at1e-12; float32 stored transition sums can differ
from the scalar return by rounding (about1.2e-7 in the retained probes).
`whip_proxy` stores unweighted W and `mechanics_credit` stores0.5W, both zero
for a valid no-event episode. `workspace_cost` is the effective charged cost
(zero on failure); `workspace_accrued_cost` and workspace metrics preserve
the accumulated capped cost even on failure. Historical modes still charge
their original unbounded workspace cost on failure.

Control stays world XYZ acceleration at50 Hz, limits(8,8,4) m/s²,100-Hz history,
200-Hz controller and30-ms delay. The full4.5 s contains225 policy decisions;
the final0.5 s is still policy-controlled, not a separate settling controller.
Physics, scene, safety limits and all PPO settings are copied from the previous
acceleration preset:128 lanes,10,240-episode budget, reward scale100, gamma1,
GAE lambda0.95, existing exploration and plateau-stop settings. Correct episode
ordering does not prove effective temporal credit assignment or learnability.
Implementation verification did not run an optimizer. The subsequently authorized
fresh seed0 training run is
[20261021-214537-277266](../runs/ppo_stage1/20261021-214537-277266/configuration.json),
started2026-10-21 at21:45 America/New_York through the visible Training window.
Its [request](../runs/ppo_stage1/20261021-214537-277266/request.json) selects no
resume or actor initialization. The [event log](../runs/ppo_stage1/20261021-214537-277266/events.jsonl)
confirms the CUDA physics backend initialized with128 environments. The budget
is10,240 episodes, subject to the unchanged plateau-stop settings. Completion,
learned whipping and later-stage readiness are not established by startup.
The first128-episode collection completed with67 valid and61 failed episodes;
all failures returned exactly−1 and completed returns stayed within the approved
bounds. Optimization was in progress; the initial round0 `latest.pt` loaded with
finite actor weights. [Startup verification](../tmp/stage1-discovery-20261021/training-start-verification.json)
records this initial check; it is not evidence of sustained improvement.

The run subsequently stopped cleanly after collection9 /1,152 episodes at
21:56 America/New_York; the event log records `termination_reason=user_stop`,
not a plateau or software error. Its round9 `latest.pt` is saved. The
[collection9 review](../tmp/stage1-discovery-20261021/collection9-review.json)
records126/128 physically valid episodes,3/128 contained,17 diagnostic geometric
hits and mean raw return−0.1040. Mean W among completed episodes is0.0552;
maximum W is0.2118. The latest scheduled native evaluation is still collection8:
W0.00153, no hit, return−0.11885, and maximum horizontal displacement4.669 m
from initial XY (4.119 m beyond the0.55-m radius). Collection9 has no scheduled
post-update evaluation. This is progress in survival/exploratory returns, not
validated directed-whip performance. No resume or reward change was made.

The subsequent [exploration audit](../tmp/exploration-review-20261021/summary.json)
inspected all nine collections and ran384 separate diagnostic episodes with no
optimizer updates. The initial zero-mean Gaussian heads imply requested-command
RMS approximately[4.30,4.30,2.15] m/s², with fresh noise each20-ms decision.
The entropy coefficient remains0.02 through collection8 and is0.01966 at9;
this coefficient is not the learned action standard deviation. Intermediate
all-lane action distributions were not retained; selected saved motions are
not an unbiased sample. See [history](../tmp/exploration-review-20261021/history.json).

The frozen round9 actor was evaluated at pre-tanh standard-deviation multipliers
0,0.25,0.5,1, with64 paired random sequences per stochastic setting across two
batches, using the original128-lane CUDA physics batch and nominal initial state.
Mean returns were−0.08749,−0.06928,−0.08065,−0.12349; completed-episode mean W
was0.00203,0.03634,0.04877,0.05905. Physically valid counts were64,64,63,61;
no episode remained inside the workspace. The zero-noise lanes repeat one
deterministic case, not64 independent robustness trials. Reduced noise improved
sampled return while reducing W; removing noise did not recover a contained whip.
These execution probes do not establish which exploration schedule learns best.
The [probe report](../tmp/exploration-review-20261021/report.json) verifies unchanged
actor weights, optimizer-update count, source checkpoint hash and execution hashes.

In the actual collection9, completed episodes averaged+0.02762 in weighted W
and−0.11744 in workspace charge. This supports investigating competing objectives,
but scalar magnitudes do not establish policy-gradient dominance or causality.
Workspace bounds already apply a soft charge, without wall forces or boundary
termination; acceleration-cost weight is already zero. A curriculum delaying
workspace/effort penalties was discussed, not implemented or experimentally
validated. The primary [Dream to Fly paper](https://rpg.ifi.uzh.ch/docs/ICRA26_Romero.pdf),
sections III-C and IV-A, starts its body-rate coefficient at zero and increases
it after total reward exceeds50; this is a different cost and task, and its
threshold is not an approved whip curriculum criterion.

There is no automatic target curriculum in this Stage1. Motion review is required
before designing Stage2. [Actor-source validation](../training/artifacts.py)
rejects a discovery checkpoint as a source for a later stage; a reviewed handoff
is not implemented. Same-Stage1 transfer and exact-resume identity rules remain.
No numerical proxy threshold, success classifier or new advancement gate is
established by this change. The offline full-task candidate below remains a
separate proposal; its crossing quality and success bonus are absent here.

Verification: [10 focused tests](../tmp/stage1-discovery-20261021/discovery-tests.log)
and [94 existing regression tests](../tmp/stage1-discovery-20261021/regression-tests.log)
passed, covering reward/prefix accounting, target independence, CUDA/CPU scoring,
configuration isolation, UI, serialization and transfer guards. The existing
checks cover scoring, acceleration control, pipeline, handoff and offline review.
[Native checks](../tmp/stage1-discovery-20261021/native-checks.json) compare three
identical command sequences under the old/new presets: hover returns0, constant
descent fails at0.78 s and saved final commands fail at2.01 s, both now returning−1.
Drone/cable state and motor-thrust arrays are exactly equal between each pair.
These are numerical regressions, not new policy-learning results.
[Saved-trace checks](../tmp/stage1-discovery-20261021/replay-checks.json) rescore198
hash-verified retained trajectories:176 completed returns range from−0.08014 to
0.23795;22 failures are assigned−1. Weighted W exactly matches half the prior
offline W, and maximum workspace-cost difference is1.4e-17. These traces include
duplicates and different historical policies; this is an accounting comparison,
not a fair comparison of trained policies. Full-suite and real-flight validation
were not performed.

The subsequent2026-10-21 [proxy audit](../tmp/whip-proxy-review-20261021/report.json)
and [figure](../tmp/whip-proxy-review-20261021/proxy-review.png) record a specific
limitation of the inherited unfolding term: `U=(rho_release-rho_min)/(1-rho_min)`,
clipped to[0,1], where rho is chord length divided by actual arc length. Large U
does not imply large absolute unfolding. Saved collection4 lane113 has W0.1515,
U0.9525 and a chord/arc-ratio increase of only0.00896. Its selected tip is only
0.0581 m forward of the attachment. This is a small geometric straightening
with measurable relative motion, not validated evidence of a genuine whip.

Prescribed, constant-edge-length traveling-bend controls confirm this sensitivity.
At0.1-rad bend amplitude, recovered chord/arc ratio is0.000532, U0.845 and W0.0524.
Adding uniform world+X translation at2 m/s leaves relative cable motion unchanged
but raises W to0.0915, through world-frame forward speed/alignment. This compares
the proxy alone: workspace cost can penalize that translation in the full return.
The synthetic controls are kinematically consistent, not dynamically validated.
Rigid translation alone scores0; standing-bend controls score below5e-8, and W
increases monotonically with prescribed traveling-bend amplitude. The inherited
cube root softens penalties from small multiplicative mechanics factors. The
proxy does not directly measure distal speed amplification or energy transfer.
No proxy threshold is promoted to a physical whip definition by these checks.

The pinned first four collections improve completion67/128→127/128 while mean
W among completed episodes changes0.0389→0.0426 and mean completed return
changes−0.1363→−0.1424. None of their episodes remained contained. Aggregate
return improvement primarily reflects fewer failures; these four collections
do not demonstrate learned whipping. The audit changed no reward, checkpoint,
controller or running process. Research changes to the proxy remain unapproved.

A second user-requested [recheck](../tmp/whip-proxy-recheck-20261021/report.json)
qualifies that interpretation: large U for small motion does not by itself
invalidate the complete W or its use as Stage1 partial credit. Eight fixed native
controls were run with the current plant, scene, safety rules and acceleration
interface, without optimizer updates or edits to the active training process.
Commands were hover or one/two finite cosine acceleration cycles, starting at
0.3 s, with predefined amplitudes1/2/4/8 m/s² and frequencies0.5/1/2 Hz; exact
combinations and input hashes are retained in the report. All eight completed.
Contained1-Hz controls at amplitudes1 and4 m/s² earn W0.0274/0.0884 and full
returns0.0137/0.0442. Hover earns0. Larger excursions remain charged separately.

Three retained milestone strokes, verified to share the model and initial scene,
rescore to W0.602/0.632/0.614 and current full returns0.291/0.297/0.256. Their
selected chord/arc-ratio recoveries are0.648/0.829/0.868 and tip-relative speeds
7.72/8.72/8.82 m/s. The early acceleration example has W0.152, recovery0.00896,
relative speed1.14 m/s and negative full return−0.0723. The
[comparison figure](../tmp/whip-proxy-recheck-20261021/full-proxy-comparison.png)
shows these fixed-state measurements. This supports basic motion/reward ordering,
not a comparison of policy learning under different action interfaces. Retained
milestone actors were trained with related mechanics rewards, so they are not
independent ground-truth validation of the proxy or proof that acceleration PPO
can discover these motions.

Uniform translation changes world-frame tip speed toward a fixed target; this
is an intentional factor in W, not necessarily a mechanics bug. Its intrinsic
bend/nonrigid measures reject rigid translation alone. Stage1 explicitly allows
partial progress, so the earlier statement that the proxy was too permissive
was not established by the first sensitivity audit. Remaining questions concern
calibration against independently reviewed motion and learnability. A new
absolute-fold or tip-amplification gate is a research decision, not an automatic
fix. Related [human-inspired whip research](https://pubmed.ncbi.nlm.nih.gov/39791118/)
optimizes tip-target distance and examines preparatory motion; it does not
validate this implementation's W formula. No new threshold, reward or curriculum
rule has been installed by either review.

## Reward redesign restart — proposed, not implemented

On2026-10-21 the user requested a complete reconsideration of the reward system
after the discovery/exploration reviews. The following is a design proposal,
not a replacement configuration or an approved physical definition. The existing
discovery preset and historical evaluator below remain executable as documented;
neither is treated as the selected final design. No new learning run is authorized
by this design record.

**Objective and separation.** Optimize a genuine cable whipping event that delivers
an accurate tip strike, then improve its economy. Distinguish (1) motion evidence,
(2) target delivery at the same event and (3) resource/workspace cost. A rigid
translation or pendulum-like sweep can reach the target but must not be reported
as verified whipping. Conversely, a strong whip missing the target is useful
discovery progress, not full-task success. A whip earlier in the episode cannot
qualify an unrelated later crossing. The target remains a virtual plane; this
simulation does not establish contact impulse or impact energy.

**Measurements before scalarization.** The offline replacement should expose
distal deformation propagation, tip speed relative to the attachment, speed
growth through the release, and the residual after fitting whole-cable rigid
translation/rotation. Measure chord/arc change in absolute units of cable-length
fraction, alongside the current recovered-slack fraction. A large fraction of
tiny initial slack is not a large unfolding motion. None of these individual
measurements is a validated whip classifier, an energy-transfer measurement or
a reason to require one specific preparation trajectory. Compare their utility
against reviewed motions before choosing normalization scales or combinations.
Retain world-frame tip velocity and target errors separately because those
measure delivery even when relative cable dynamics are unchanged.

**Candidate reward architecture.** Use one best observed attempt per episode,
with bounded partial credit for developing a directed cable motion and additional
credit for its linked target delivery. The delivery term should measure actual
forward crossing, center error, forward speed and entry direction at the same
event. Target precision should increase only after broad delivery is repeatable.
Motion-only credit must remain below verified full-task success. Avoid paying
repeatedly for an unchanged bend, stationary proximity, repeated oscillations
or repeated crossings. Partial credit should remain available for meaningful
near misses; a sparse conjunction of all final criteria should not be the sole
training signal. Exact formulas, weights, event support durations and success
thresholds remain research decisions pending offline comparison and approval.

Deliver causal increments in the chosen bounded best-attempt score, with explicit
failure/terminal accounting, rather than adding an arbitrary positive reward at
every timestep. Verify that transition sums reconstruct the intended return and
that no earlier transition consumes future motion information. Such redistribution
does not by itself prove effective credit assignment with PPO/GAE. Preserve the
current full-horizon physical-failure rule in the first prototype: a later failure
does not become a successful episode because an earlier strike was good.

**Curriculum proposal.** Discovery initially uses zero workspace and effort
weights while retaining the existing action and physical-failure limits. Delivery
then rewards accurate strikes linked to the discovered motion. Only after reviewed,
repeatable task performance should successive fixed-weight stages introduce
workspace and effort costs. Workspace usage and effort are separate quantities;
zero acceleration effort cost is already implemented today. The first offline
effort candidate is mean squared normalized commanded acceleration, explicitly a
command-magnitude proxy, not battery energy. Body-rate and command-change metrics
are diagnostics until a distinct need for them is established. Do not add all
penalties simultaneously. Raw returns under changing weights are incomparable;
retain a fixed evaluation task and separate success, accuracy, motion and cost
metrics. Advancement thresholds and ramp increments are not yet selected.

**Review before training.** Compare candidates on hash-pinned saved trajectories
and existing native controls: hover, tiny vibration, rigid translation/rotation,
pendulum-like sweeps, developed candidate whips, misses, linked strikes, unrelated
wave-plus-hit events, large excursions and post-strike failures. Obtain event-level
motion labels without displaying reward rankings; use a held-out subset when
calibrating scales. Historical policies trained with related proxies are comparison
cases, not independent ground truth. Report ambiguous cases instead of assigning
invented labels. Inspect frame/translation sensitivity, temporal resolution,
finite behavior, boundedness, reward reconstruction and event linkage. Synthetic
controls test scoring logic, not physical feasibility. Only after this review
should a separately selectable training preset and matched-budget multi-seed
learning comparison be considered. Keep exploration fixed in the initial reward
comparison to avoid confounding the two changes.

The next implementation was narrowed and approved on2026-10-22 to an offline
measurement evaluator with replay annotations, described below. It keeps the
measurements separate and does not produce candidate reward rankings, select a
training reward or automate curriculum advancement. The earlier design review
inspected scorer/environment code, retained audits and primary literature; it
did not run new simulation or training.

### Whipping measurement and cable-model literature review — 2026-10-22

The user narrowed the next step to measuring whipping, reviewing whip/casting/
mathematical literature and understanding the supplied Shen, Franchi and Gabellieri
paper, *Aerial Robots Carrying Flexible Cables: Dynamic Shape Optimal Control via
Spectral Method Model*, IEEE TRO41 (2025),3162–3182,
[DOI](https://doi.org/10.1109/TRO.2025.3562459). Its21-page supplied PDF was read
locally; equations and figures on PDF pages3,4,6,16 were also rendered and visually
checked. This narrows implementation priority; it does not approve new reward
weights, success thresholds or automatic stage transitions.

**What the literature establishes.**

- [Goriely and McMillen,2002](https://doi.org/10.1103/PhysRevLett.88.244301) and
  [McMillen and Goriely,2003](https://doi.org/10.1016/S0167-2789(03)00221-5)
  study travelling whip waves and acceleration, including tapering and boundary
  conditions, using elastic-rod mechanics. Their cracking problem concerns
  supersonic motion of tapered whips; that speed requirement does not define our
  untapered cable's target-striking task. Retrieved primary abstracts/excerpts
  support these statements; complete PDFs from the author's host were unavailable.
- [Preston, *The motion of whips and chains*](https://arxiv.org/html/1105.1944)
  studies an ideal inextensible string with a fixed end and no gravity. Its
  normalized equations are `r_tt=(T r_s)_s`, `|r_s|=1`, and
  `T_ss-|r_ss|² T=-|r_st|²`. Tension is a constraint response, not an elastic
  stretch law. The continuum/discrete convergence result under its assumptions
  does not validate our finite mesh, moving attachment, damping or integrator.
- [Nah et al.,2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10405071/) use a
  discretized50-DOF whip and a low-dimensional minimum-jerk arm action, optimized
  for closest approach to the target. [Edraki et al.,2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11715529/)
  compare striking-only and preparation-plus-strike trajectories with a25-DOF
  model, also optimizing target proximity. These objectives establish delivery
  performance in their setups, not an independent physical-whip classifier.
- [Tabata et al., *Casting manipulation of unknown string by robot arm*](https://arxiv.org/html/2401.17082v1)
  use a planar mass-spring-damper model with axial and hinge effects, drag and
  a grasp boundary, iteratively identifying parameters from observed casts.
  Their task criterion is tip entry into a target region. This reviewed version
  is distinct from the2024 IEEE Access extension to nonuniform strings; only
  the latter's abstract was accessible during this review.
- [Chi et al.,IRP,RSS2022](https://arxiv.org/html/2203.00663v1) optimize changes
  to a three-parameter rope-swing primitive using observed tip trajectories.
  Performance is minimum tip-to-target distance, not a wave-propagation label.
  Its iterative whole-attempt action search differs from our50-Hz action noise.
- [Bergou et al.,2008](https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf) provide
  the discrete-elastic-rod framework with centerline bending, material frames,
  twist and inextensibility treatment. It is conceptually closer to our model
  than a perfectly flexible extensible string, without certifying our particular
  implementation or parameters.

**Supplied spectral paper.** Material coordinate `s∈[0,L]` labels the unstretched
centerline `r(s,t)`. With constant mass per reference length `μ=ρc A`, their
equations(2–5,12) can be written

`ε=|r_s|-1`, `n=EA ε r_s/|r_s|`,

`μ r_tt = ∂s n - ρa cd |r_t| r_t - μ g e_z`.

The constitutive model is perfectly flexible: no bending/twisting moment, no
internal material damping term in this PDE, axial linear elasticity and
quadratic aerodynamic drag. The head is attached at the quadrotor CoM in the
model; the free tip satisfies zero strain and hence zero force. Real experiments
approximate the off-CoM attachment's acceleration by the CoM acceleration,
assuming attitude effects are small. Rotor downwash is neglected. The printed
axial law is not tension-clamped: if `|r_s|<1`, its scalar force is negative.
This is an algebraic observation about the published formula, not evidence that
compression occurred in their experiments. Slack/compression behavior requires
checking before using this law for large folding motions.

They first build an FDM/RK4 reference simulator, then obtain POD spatial modes
from simulated excitation trajectories and project the PDE onto those modes:
`r(s,t)≈r(0,t)+Σ a_i(t) φ_i(s)`, with vector coefficients `a_i∈R³`.
Three modes give nine shape coordinates, with velocities and attachment state
bringing the outer control model to24 states. This is a projected nonlinear
physics model with data-derived spatial bases, not just an SVD reconstruction
or a neural dynamics predictor. The simulation example uses100 intervals,
0.5-ms FDM steps and50-Hz NMPC with a0.4-s horizon; the physical experiment
uses a different100-Hz control setup. Identification uses50 intervals and
0.05-ms steps. These are distinct configurations, not one universal timestep.

The reported99.998% for three modes is a squared-singular-value share of the
sampled position representation (Eq.37); it is not proof of mechanical-energy,
curvature or peak-speed accuracy. The fixed-head release identification example
reports mean marker-position discrepancies within8 mm for the first3.5 s, with
ROM discrepancies later reaching roughly6 cm. This is not a demonstrated
high-speed whip validation or a held-out validation of our cable. A sharp,
travelling localized deformation may need modes absent from the training
snapshots. Model-order selection for whipping should evaluate event timing,
curvature transport and tip velocity as well as position reconstruction.

Our current model instead has11 nodes/10 inextensible edges, bending/twist
elasticity, objective Kelvin-Voigt material damping, linear external drag,
explicit moving-marker masses, and an offset point clamp transmitting forces
and moments to the drone. The paper is useful for reduced-order prediction
and shape-coordinate ideas; replacing our physical plant is not justified by
this comparison. Neither model is automatically more accurate for our hardware.

**Proposed measurement contract, before reward design.** One provisional event
family for review is a significant deformation propagating toward the free end
with a temporally associated concentration and increase of distal motion. This
is an operational hypothesis for our intended motion, not a universal necessary
condition for whipping or a literature-standard binary classifier. The deeper
review below qualifies both the direction and timing requirements. Keep these
measurements separate, retaining their times and units:

1. Map unsigned centerline curvature against material position and time. For
   measured edge tangents `t_j` and lengths `l_j`, use the finite-resolution
   estimator `κ_j=atan2(|t_(j-1)×t_j|,t_(j-1)·t_j)/((l_(j-1)+l_j)/2)`.
   Track coherent ridges across successive material positions; centroid motion
   alone can result from one bend disappearing and another appearing. Report
   propagation distance, duration, amplitude and fitting uncertainty before
   selecting gates. Global rigid motion cannot create this curvature transport.
2. Retain world tip speed and `|v_tip-v_attachment|`, including pre-event and
   release values. Relative speed removes bulk translation but not rigid rotation.
   Supplement it with the residual after a weighted best-fit rigid velocity
   field over the event history; do not require large residual motion at every
   release sample, since local terminal whirling can accompany snapping.
   A large tip/attachment speed ratio alone is insufficient and becomes
   unstable near zero attachment speed; any gain statistic needs a common event
   window and its raw numerator/denominator.
3. Keep absolute chord/arc recovery as a descriptive measurement. Do not require
   one prescribed fold, counter-movement or minimum-jerk trajectory as the
   definition. Target crossing geometry, direction and speed stay separate and
   must later link to the same motion event.
4. Treat kinetic-energy distribution as supporting information, not proof of
   transfer. Quantifying power flow also requires section force/moment and
   boundary work, gravity and dissipation accounting. Current saved replays have
   kinematics/material angles but no full section-force history; do not label a
   velocity ratio or kinematic residual as energy-transfer efficiency.

The [kinematic evidence](../tmp/whip-measurement-research-20261022/kinematic-evidence.json)
and [figure](../tmp/whip-measurement-research-20261022/kinematic-evidence.png) were
computed from three existing saved motions, with source SHA256 hashes. The
small1-m/s² sinusoidal control has peak estimated interior curvature0.6785 m⁻¹
and peak relative-tip speed0.5271 m/s; the frozen collection9 zero-noise trace has
0.4698 m⁻¹ and0.4141 m/s; the historical Stage1 reference has11.2456 m⁻¹ and
7.8483 m/s. The latter shows a visibly travelling curvature concentration. These
are selected descriptive cases, not a matched policy comparison or independent
whip labels. Peaks are not necessarily simultaneous. The estimator excludes
clamp-boundary curvature and is not the simulator's constitutive curvature.

The diagnostic checked array shapes, finite states, positive lengths and increasing
times; measured edge-length discrepancies against the common configured mesh
were below5e-16 m in these traces. The plot was visually inspected. No new
simulation, optimization, learned classifier, threshold or training reward was
introduced. Still required before using these measurements for rewards: rigid-
motion/tiny-noise controls, reviewed and held-out physical motions, spatial/time
resolution sensitivity, and event detector validation. With only nine interior
curvature samples, narrow bends can be underresolved; interpolating the same
markers adds no independent physical information. Existing stages remain unchanged.

**Deeper mechanics and counterexample review — 2026-10-22.** This is a refinement
of the research proposal, not approval or implementation of a detector/reward.

- [Dehadrai and Hanna](https://arxiv.org/html/2206.07864v2), sections III–VI,
  study a falling inextensible string. Its fast distal region shrinks, followed
  by terminal whirling; the curvature maximum can diminish before peak snapping.
  This rules out treating simultaneous curvature/speed maxima as a general
  requirement. Their particular scaling laws are not validated for our damped,
  bending-resistant, marker-loaded cable.
- [Guo, Zheng and Palffy-Muhoray](https://arxiv.org/html/2207.09901v1) analyze
  uniform chains and demonstrate snapping during free fall. World direction and
  active boundary input at the instant of release are not universal identifiers
  of whipping. Wrong-direction delivery can fail our task while remaining a
  physically whip-like event. Backward/downward preparation is not automatically
  a failed attempt.
- [Gatti-Bono and Perkins, 2003](https://irp.cdn-website.com/c49efa06/files/uploaded/CAROLINE_GATTI_PAPER_3.pdf)
  model a fly-casting loop, including moving-line, loop and terminal-fly kinetic
  energy and aerodynamic losses. A propagating loop can accelerate or decelerate
  the fly depending on the parameters. Their simplified model diverges from the
  fuller numerical solution near turnover. Casting and whipping share mechanisms;
  a travelling loop alone neither establishes amplification nor separates the
  task names mathematically.
- [Krotov et al., 2022](https://doi.org/10.1098/rsos.220581) observe a common
  proximal-to-distal peak-speed sequence in human whipping, without establishing
  measured energy flux or a clear relation between sequence differences and
  targeting performance. Their speed analysis uses the marker 6 cm from the tip
  because the endpoint derivative was noisy. Marker identity and uncertainty
  therefore matter. Also, [Nah et al., 2023](https://doi.org/10.1016/j.isci.2023.107395)
  optimize closest target distance over 25 whip nodes, not strictly the tip;
  their success criterion cannot validate our tip-only strike requirement.

There are three distinct directions: motion along material cable coordinates,
world motion of a deformation feature, and the tip's target-relative approach.
For a feature at material location `s_b(t)`, the chain rule gives
`d r(s_b(t),t)/dt = r_t + r_s ds_b/dt`. Thus a downward-moving feature can
still progress attachment-to-tip. Keep both signed projections
`v_tip · d` and `(v_tip-v_attachment) · d` for the requested direction `d`.
For example, attachment speed `3d` and tip speed `2d` give world-forward
delivery but attachment-relative motion `-d`. This illustrates why forward world
speed alone does not establish forward internal amplification; it does not
approve an instantaneous relative-speed gate.

Energy concentration is also insufficient by itself. As an analytic control,
a straight uniform cable rotating rigidly about a fixed end has `v(s)=omega*s`.
Its last 20% contains `1-0.8^3=48.8%` of kinetic energy despite unchanged shape.
This is an ideal kinematic example, not a simulation or a feasible flight claim.
Inspect changes in the spatial speed distribution together with deformation
history. A shrinking moving region can amplify speed even without increased
total energy, but gravity, boundary work and dissipation must be accounted for
before attributing any observed speed change to energy transfer.

The existing [scorer](../tasks/whip/stage1.py) uses a whole-window advection fit
and curvature-centroid displacement, without checking a tracked ridge's distal
arrival; it pairs these with a subsequent speed increase. This leaves possible
ambiguities from fading/growing bends, reflections and unrelated motion within
the lag window. These are implementation-supported vulnerabilities, not proof
that the trained policy exploited them. The scorer also includes world-forward
speed in its mechanical term: this is task-sensitive credit, not a frame-invariant
measure of internal dynamics. Its rigid fit uses material-length weights, not
the physical node masses, and remains a kinematic residual.

Any later energy diagnostic must use the configured distributed cable mass plus
the explicit marker masses. The current configuration includes a roughly 0.960-g
terminal marker; the native mesh adds its mass to the endpoint's lumped bare-cable
mass. An ideal massless free-end tension condition must not be imposed directly
upstream of that inertial marker. Mesh refinement must preserve the marker masses
and locations. The nine interior curvature samples and 100-Hz saved kinematics
also require resolution checks before certifying narrow or brief release events.

The next proposed validation remains event-based: review shape progression,
distal speed/localization history and linked delivery separately, with an
explicit unresolved category. Challenge the measurements using tiny waves,
rigid swings, changing bend amplitudes, reflections, wrong-direction releases
and unrelated wave-plus-hit sequences. A clean travelling-loop event can be
recognized as one candidate family; absence of that signature does not establish
absence of every physical whipping mechanism. No thresholds, new event windows,
reward weights or curriculum transitions were selected. This deeper review read
papers, implementation and retained diagnostics; no new simulation or training
was run.

## Offline whipping-event measurements — implemented 2026-10-22

**Current approved evaluation interface:**
[evaluate_attempt](../evaluation/whip_attempt_review.py) returns three separate
measurement groups, `excitation`, `distal_response` and `delivery`, plus full
saved-record traces. It requires explicit inclusive attempt sample bounds and
selection provenance. All nodal peak timing and curvature tracks use that same
whole attempt; shape evolution is not truncated at the display peak. This is an
offline review interface, not a causal event detector or enabled training reward.
The original candidate-basin evaluator below remains preserved with its original
semantics; this addition does not change its saved reports or training callers.

Excitation retains attachment speed, sampled path/displacement and shape ranges.
Path is a descriptive measurement, not a penalty. Distal response retains world,
attachment-relative and attachment-anchored nonrigid speed, raw nodal maxima,
ties/boundary/constancy flags, curvature tracks and same-sample tip vectors.
No instantaneous tip/root ratio, mandatory fold, amplitude normalization or
peak-order pass rule is introduced. A display sample is the earliest maximum
relative-tip speed in the optional release interval, otherwise in the attempt;
it is not a detected physical release or a success score. The rigid-fit residual
uses geometric weights and is not physical energy or causal energy transfer.

Delivery retains every saved-record plane passage, with signed velocity,
crossing coordinates and distance outside the rectangle at that passage.
Attempt containment and optional release containment are separate. Arrival and
opposite-side confirmation must both fall within the attributed interval; touches
and unresolved terminal contacts are excluded. `geometric_hit_ids` means forward
rectangle passages inside an explicitly supplied release interval only; it never
establishes a physical-whip link or task success. Without release bounds these
IDs are empty and linkage remains unresolved, even if the full attempt crosses.
The caller-supplied trial outcome (`valid`, `failed`, `unknown`) is preserved;
finite arrays do not imply valid flight. Invalid/padded samples are rejected
rather than silently dropped. No success-rate aggregation is implemented.

The [review and full data](../tmp/whip-attempt-evaluation-20261022/review/index.html)
cover eight preserved recordings plus two disjoint temporal views of the
early-wave/later-rigid-crossing fixture. Existing model geometry and the frozen
horizontal broad rectangle are used for diagnosis only. Human mean profiles
remain reference material rather than individual positive trials. Checked15 new
analytic tests and53 retained measurement/decomposition/timing/event-core tests;
all passed. Full source records, provenance and failed-outcome tests are retained.
No simulation, policy inference, training or physical classifier validation ran.

The approved first implementation measures deformation progression, distal speed
development and directed delivery separately within candidate intervals. It is
an offline review tool, not an automatic physical-whip classifier or a replacement
training reward. Every generated candidate retains `review_status: unreviewed`
and `association: temporal_support_only`. The subsequently approved event-output
core below adds explicit release measurements and separately attributed review
annotations. The earlier research proposals remain proposals where they go
beyond these measurements.

The separate [speed-peak timing helper](../evaluation/whip_peak_timing.py), added
2026-10-22, describes every node's raw sampled speed maxima inside one explicit
shared interval. It retains ties, boundary maxima, actual timestamps and speeds,
and flags numerically constant profiles rather than assigning spurious peak
order to rigid-motion roundoff. Its64-float64-epsilon scale is a numerical
diagnostic, not a physical motion threshold. World, attachment-relative and
attachment-anchored shape speed remain separate. The helper is not registered
in training and does not change existing event selection or release rules.

[Kinematic extraction](../evaluation/whip_measurements.py) reads saved positions,
velocities and actual strictly increasing timestamps. Node zero is the cable
attachment, not the drone center of mass. Material coordinates use cumulative
rest lengths from each case's frozen model, including mesh subdivisions. Interior
curvature is turning angle divided by the mean of its two measured edge lengths,
in `1/m`; total turning is in radians. Endpoint curvature is not estimated. The
retained eleven-node examples therefore have nine interior curvature samples.

Consecutive local curvature maxima are associated only when the match is unique
within one interior sample. Splits, merges, plateaus, gaps and boundary peaks are
flagged rather than bridged into a claimed physical wave. Backward tracks remain
visible. The floating-point floor is numerical, not a minimum physical bend or
whipping threshold. There is no smoothing or amplitude normalization that turns
a tiny wave into a strong one.

World velocity, attachment-relative velocity and signed components along the
task direction are retained separately. A per-frame best-fit rigid translation
and rotation produces a nonrigid velocity residual. Its quadrature weights are
material lengths, not physical masses. The spatial centroid and spread of
relative-speed squared use the same length convention and are undefined at rest.
These are kinematic localization diagnostics, not energy or energy-flux estimates.

[Event review](../evaluation/whip_event_review.py) retains every sampled local
relative-tip speed maximum, bounded by its adjacent troughs; explicit sample
intervals can instead be supplied in the manifest. A flat trough ends the earlier
interval at its first sample and starts the later interval at its last sample,
so a quiet gap cannot connect an earlier wave to a later crossing. The largest
sampled relative-tip speed chooses the default display only. No minimum speed,
duration, reward ranking or automatic positive/negative label is applied.
Automatic intervals use future samples and are not a causal training signal.

**Event-output core — implemented2026-10-22.** Following approval of the
mechanism-evidence / relative-tip-output / linked-delivery design,
[event review](../evaluation/whip_event_review.py) accepts an optional
`release_interval` on each explicit event. It contains inclusive `start_index`
and `end_index`, an earlier `baseline_index`, and `provenance` with nonempty
`reviewer`, `method` and `notes` strings. The required ordering is
`event_start <= baseline < release_start <= release_end <= event_end`.
The interval must be supplied independently of the output calculation; recording
provenance cannot verify that the supplier actually selected it independently.

Within that interval, `release_measurement.q_relative_tip_speed_mps` is
`max_t |v_tip(t)-v_attachment(t)|`, in m/s, using the physical cable attachment
at node0. Its sample index/time, world and relative velocity vectors, and signed
target-direction components are retained. Exact maximum ties choose the earliest
sample. The baseline is the explicitly selected raw prerelease sample, and
`signed_gain_mps` is Q minus that baseline without clipping. This is sampled
event output, not impact energy, energy-transfer efficiency or an automatic
whip label. Automatic candidates without explicit release intervals have no Q.
The original candidate start/peak/end, default display selection, and
start-to-peak deformation summaries remain unchanged and explicitly identified;
they must not be mistaken for the new release interval or Q timestamp.

An optional `mechanism_review` contains `status` (`supported`, `unsupported` or
`unresolved`) and the same provenance fields. Omission produces unresolved
status and null provenance. These are attributed supplied judgements, not
classifier outputs or independently validated physical truth. The candidate's
legacy `review_status` stays `unreviewed`, independently of this annotation.

Delivery retains geometrically confirmed `release_crossing_ids` only when both
arrival and opposite-side confirmation lie inside the explicit release interval.
An optional `delivery_review` names unique `linked_crossing_ids` with provenance;
it requires a supported manual mechanism review and release-contained crossings.
Its `linked_quality` retains raw crossing position, direction, speed and errors.
Neither temporal containment nor a supplied link establishes causation, correct
direction, minimum strike speed or task success. Unsupported/unresolved mechanism
reviews cannot supply reviewed delivery links. Existing v1 reports and viewer
loading remain compatible; new fields are additive. CSV/Markdown expose Q,
baseline, gain and review status; plots mark the release bounds, Q and baseline.

[Core verification](../tmp/whip-event-core-20261022/review/report.json) reprocessed
ten retained/synthetic cases and four predeclared synthetic release intervals;
all ten viewer loads and input hashes were checked. No positive physical-whip
label was assigned. Rigid controls retain unsupported attributed construction
reviews, and the earlier wave cannot borrow the later rigid crossing. The
annotated separation plot was visually inspected. Fourteen
[core tests](../tests/test_whip_event_core.py) and29 existing measurement/replay
tests passed, including boost invariance, release-local maxima, signed negative
gain, annotation validation, crossing confirmation and legacy compatibility.
These are software/kinematic checks, not training or physical validation.

**Stage1 shape-release integration history (2026-10-22); current path-cost revision below.** The user approved the attachment-anchored
shape-release utility and100-ms half-std AR(1) exploration after reviewing the
attribution and coverage checks below. A separate experimental preset is now
implemented: [whip_stage1_ppo_shape_release_ar1_v1.json](../configs/learning/whip_stage1_ppo_shape_release_ar1_v1.json).
The original offline candidate and its failed independent-sweep check are retained
as historical evidence. The revised output removes that identified rigid-speed
contamination; it is not a validated physical-whip classifier. Integration and
native optimizer checks, followed by the separately authorized fresh run, are
recorded below. The existing default `--stage 1` still selects the older compact
preset; select the new settings file explicitly. No physics or controller changes.

**Original provisional automatic Stage1 contract — retained offline diagnostic;
original utility rejected by a failed counterexample.**
The user requested exact rules before integration, then authorized the proposed
exploration comparison. [stage1_candidate.py](../evaluation/stage1_candidate.py)
implements the following detector, event utility and offline100-Hz reward
increments for that diagnostic. Its relative-tip-speed utility was not integrated. The approved shape-output
revision and conditional policy are implemented separately below.
The following numbers are provisional engineering choices, not calibrated whip
thresholds or conclusions from the nine early collections:

- Process existing100-Hz observations with prefix-only, unique curvature-peak
  associations, at most one interior-node step per frame. No gap bridging;
  split/merge/plateau ambiguity ends the segment. A boundary peak alone is allowed.
- Start an outward segment at material coordinate `s/L <= 0.55`, with local
  turning angle `kappa*measured_dual_length >= 0.10 rad` throughout. Stationary
  and outward steps are allowed. A backward step, lost/ambiguous association,
  subthreshold angle or duration above0.60 s ends the segment; the current sample
  may begin another segment if it meets the start conditions.
- Arm one event when that segment first reaches `s/L >= 0.75`, visits at least
  three distinct interior nodes, and has historical nonrigid evidence: the
  maximum whole-cable length-weighted RMS rigid-fit velocity residual during
  the segment is at least0.10 m/s. This is a kinematic diagnostic, not energy.
  It need not remain high at release. Each segment arms at most once.
- Freeze release bounds `[arming_time, arming_time+0.20 s]`; a valid finite
  horizon clips the upper bound and records the truncation. Baseline is raw
  attachment-relative tip speed at the segment's first sample. Finalize only
  when the release interval closes. Q is the largest saved relative-tip speed
  inside it, earliest exact tie; require `Q-baseline >= 0.25 m/s` for eligibility.
  Later track loss cannot retroactively alter the frozen evidence prefix.
- At the same Q sample define `v_forward = v_tip_world dot desired_direction`.
  Directed event utility is `S = min(Q, max(0, v_forward))/(1 m/s)` for an eligible
  event, otherwise0. Q remains the separately reported event-output measurement;
  S explicitly prefers both internal relative output and forward world delivery.
  It is not a universal whip-strength or impact-energy formula. There is no
  additional angle gate, hit reward or target-error reward in this Stage1.
- Valid episode return is the maximum finalized event S, with0 for no eligible
  event. At50 Hz pay only increases in that finalized best value. Failed episodes
  replace their entire return with−10, using terminal clawback. This penalty is
  an explicit utility choice equivalent to ten speed-derived reward units,
  not a physical constant or a guarantee of safe risk-taking. Workspace, effort
  and time costs are zero; current action and physical-failure limits remain.
- Fresh seed0 PPO,128 environments,10,240 episodes (80 collections),4.5-s full
  episodes,50-Hz acceleration actions with existing(8,8,4) m/s² limits. Proposed
  reward multiplier1, gamma1, lambda0.95 and otherwise existing PPO settings.
  Disable old plateau stopping; scheduled evaluation stays every512 episodes
  and at first/final collection. No automatic advancement or checkpoint transfer.
- Record provisional eligibility and each rejection reason separately from
  manual mechanism review; no automatic `supported` physical-whip annotation.
  Before a budgeted run, check rigid motion, tiny/inward waves, changing separated
  bends, unrelated wave-plus-hit, and a traveling bend superposed on an independent
  rigid sweep. Also check retained native motions without treating them as
  independent ground-truth labels, then a small no-update exploration probe.
  Failure of a required counterexample check or unusably sparse feedback must be
  reported for review, not fixed by silently changing thresholds or adding reward.

This detector restricts the first experiment to one resolvable outward-release
family. Fast or overlapping waves may remain unresolved at this mesh, and
kinematic association cannot establish force-mediated causation. The fixed
contract failed the traveling-bend-plus-independent-sweep test described below.
No thresholds or acceptance criteria were weakened to bypass it.

**2026-10-22 candidate check and exploration comparison.** The reproducible
[candidate checker](../tmp/stage1-exploration-20261022/check_candidate.py) pins
[results and constructions](../tmp/stage1-exploration-20261022/candidate-checks.json).
Rigid rotation/translation, tiny outward/inward waves, separated stationary bends
and an early wave followed by an unrelated later crossing all earn0. A resolved
outward bend arms an event; independently superposing a common angular
acceleration preserves its turning-angle history but earns4.135850. The separate
rigid sweep earns0. These constant-edge-length constructions use analytic
velocities; no force-mediated dynamics are simulated. This demonstrates an
attribution failure of the kinematic rule, not a physically realized policy
exploit or a real-flight result. Required research acceptance is **failed**.
Four retained native/historical controls were also rescored with pinned sources;
they have no independent physical-whip labels and do not validate classification.

The [no-update probe](../tmp/stage1-exploration-20261022/probe.py) compares the
exact fresh zero-head actor distribution against half its pre-tanh standard
deviation and two stationary AR(1) Gaussian processes. The AR(1) time constants
are0.10 and0.20 s, with `rho=exp(-0.02/tau)`; the first noise sample is standard
normal and subsequent innovations scale by `sqrt(1-rho^2)`. These are explicit
engineering probes, not literature-calibrated choices. Each condition has32
lanes per seed, paired innovations across conditions, and seeds2026102200–02:
96 episodes per condition,384 total. All use the same nominal initial state,
50-Hz acceleration limits(8,8,4) m/s²,4.5-s horizon and unchanged native physical
failure rules. The old discovery preset supplies the execution model and scene;
its reward is not used. The candidate is evaluated separately on finite saved
prefixes, with unfinished release windows finalized only at valid horizons.
The AR(1) samplers are not installed PPO policies; no optimizer, entropy schedule,
checkpoint initialization, training run or policy promotion occurs. Changing
training exploration would require compatible behavior/update probability and
KL accounting. This probe assesses initial coverage, not comparative learning.

All384 episodes completed collection on the host RTX4080 using the existing
`cuda_conditional_v1` native backend. [The report](../tmp/stage1-exploration-20261022/probe-report.json)
retains per-seed/per-lane results, hashes, rejection reasons and event records;
the three compressed batch files retain full numerical histories, planned
actions, executed masks and paired innovations. Selected paired-lane0 replays
are diagnostic examples, not representative success claims.

| Initial action process | Full-horizon valid /96 | Positive candidate and valid /96 | Mean peak relative-tip speed (m/s) | Mean maximum drone displacement (m) |
|---|---:|---:|---:|---:|
| Independent, original std | 49 | 9 | 2.010 | 2.963 |
| Independent, half latent std | 61 | 0 | 1.309 | 2.085 |
| AR(1),0.10-s correlation time | 34 | 27 | 5.123 | 6.858 |
| AR(1),0.20-s correlation time | 40 | 29 | 5.824 | 10.196 |

The half-std condition armed no events in any of its96 episodes. The original,
0.10-s and0.20-s conditions armed events in23,89 and90 episodes respectively.
Eligibility or positive score is **not** a genuine-whip success label, given the
failed attribution test. Validity means reaching4.5 s without the existing
physical/numerical failure conditions; it does not imply workspace containment.
Motion summaries use executed prefixes, so failed episodes have shorter exposure.
Scheduled budgets and initial states are equal, and planned as well as executed
action statistics are retained to expose that censoring. Half latent standard
deviation produces approximately2.65/2.65/1.33 m/s² executed command RMS because
of tanh; it does not halve post-tanh acceleration. The full-amplitude conditions
all remain near4.3/4.3/2.16 m/s², yet correlated commands produce much more motion.
At0.5 s, when all lanes are still observed, mean pairwise drone-position distance
is0.137/0.089/0.368/0.446 m in the same condition order. This documents early
trajectory dispersion, not useful strategy diversity or learned robustness.

These observations support evaluating temporal persistence separately from
amplitude, but select no training winner or revised exploration setting. A
candidate requiring further review is shorter/milder correlated excitation;
the immediate prerequisite for training remains resolving the reward attribution
failure without silently changing the intended task. [Plots](../tmp/stage1-exploration-20261022/exploration-comparison.png)
and the [counterexample figure](../tmp/stage1-exploration-20261022/candidate-counterexample.png)
were visually inspected. [Saved-array checks](../tmp/stage1-exploration-20261022/saved-array-verification.json)
verified all three batch hashes, executed episode accounting and unchanged
execution sources. No native rerun was performed for these verification checks.

The [verification log](../tmp/stage1-exploration-20261022/tests.log) records55
passing selected tests:12 new detector/accounting checks and43 prior event-core,
measurement and replay checks. They cover causal frozen evidence, release timing,
signed gain, direction at the Q sample, repeated-event accounting, terminal
clawback and basic negative kinematic controls. Passing software checks does not
override the failed scientific acceptance gate. No full test suite or training
experiment is claimed.

**2026-10-22 attachment-anchored output revision — diagnostic results and
subsequently approved training contract.** After the user requested
resolving the attribution gap and testing milder exploration,
[whip_output_decomposition.py](../evaluation/whip_output_decomposition.py) added
an offline geometric decomposition. For each observation define
`r_i = p_i-p_attachment`, `v_rel_i = v_i-v_attachment`, and rest-length
trapezoidal weights `w_i` normalized to sum1. Fit one whole-cable angular
velocity `omega* = argmin_omega sum_i w_i ||v_rel_i - omega cross r_i||²`.
The residual `u_i = v_rel_i - omega* cross r_i` is reported as shape velocity,
alongside the original relative velocity, fitted rigid velocity, angular
velocity and fit rank. The anchor translation is fixed to its measured velocity;
this differs from the earlier freely translated whole-body rigid fit used by
the mechanism gate. Both measurements remain available and that gate is unchanged.
Weights are geometric lengths, not masses. A straight chain may have an
unobservable axial angular velocity; fitted nodal velocities remain unique.

Under a superposed rigid motion `p'=R(t)p+b(t)`, attachment-relative velocities
transform as `v_rel'=R v_rel + Omega cross (R r)`. The fitted angular velocity
absorbs the added `Omega`, giving `u'=R u`; residual speed is unchanged. This is
a geometric projection property, not proof of energy transfer or a unique
physical allocation of nodal speed. Residual speed at one node can exceed its
original relative speed because fitted and residual vectors can oppose each
other. Neither should be interpreted as nonnegative additive speed/energy shares.

The separate [analysis script](../tmp/stage1-attribution-20261022/analyze.py)
evaluated this scalar change offline before approval, without altering the
original candidate module: retain all event-tracking and release-window
rules; set `Q_shape=max_release ||u_tip||`, with earliest exact ties, and require
`Q_shape-||u_tip(segment_start)|| >= 0.25 m/s`. At the Q_shape observation score
`min(max(0,u_tip dot target_direction), max(0,v_tip_world dot target_direction))/(1 m/s)`.
Otherwise score0. The direction and actual-world-motion requirements prevent a
positive shape residual from earning task credit when the world tip moves
backward. Episode max, failure−10, frozen finalization and zero effort/workspace/
time costs remain as proposed previously. Reusing0.25 m/s on this different
quantity is an explicit provisional research choice, not a calibrated threshold.
The user subsequently approved this output definition and both direction checks
for the separate experimental PPO preset below.

[Retained results](../tmp/stage1-attribution-20261022/attribution-review.json)
show the original counterexample changes from4.135850 to0: shape-speed gain is
0.141283 m/s, below the unchanged numerical0.25 floor. The same bend alone and
with the sweep have identical residual-speed curves. Four additional prescribed
rigid rotation/translation stress cases retain the same0.579295 m/s peak tip
shape speed while ordinary relative speed rises as high as29.718 m/s; maximum
vector-covariance error is8.46e−15 m/s. Thus the rejection is supported by the
invariance property as well as this fixture, not simply a changed numerical
threshold. This does not prove arbitrary bend-plus-sweep constructions should
all score0: genuine shape motion can remain and a rigid rotation can change
its world direction. No broader rejection criterion was silently declared passed.
The v1 scientific gate remains a retained failure; the revised test establishes
removal of added rigid-speed contamination, not a universal whip classifier.

A construction rotating only the last two links retains peak residual tip speed
0.800208 m/s versus original0.8 m/s, so this whole-cable subtraction does not
automatically erase localized terminal whirling. This matters because
[Dehadrai and Hanna](https://arxiv.org/html/2206.07864v2), sections III–VI, describe
a shrinking moving distal region and terminal whirling during snapping. It does
not establish general sensitivity for our11-node damped cable. Retained hover
and weak sinusoid remain0; the strong sinusoid changes from3.077882 to1.682394,
and the historical replay from2.099429 to1.337500. Those recordings are unlabelled
diagnostics, not independent true-positive validation.

The [follow-up native probe](../tmp/stage1-attribution-20261022/probe.py) executed
384 additional no-update episodes,96 per condition, with the same three paired
noise seeds,4.5-s horizon,128 lanes,50-Hz feedback, nominal scene and physical
limits as the previous comparison. It tested the original independent baseline,
50-ms AR(1) with half latent std,100-ms AR(1) with half latent std, and50-ms
AR(1) with three-quarter latent std. No new physics, reward, entropy schedule or
optimizer was used. New and prior traces were rescored offline with the proposed
utility. The repeated96 baseline trajectories match the prior saved positions,
velocities, sampled actions and failures exactly and are **not** additional
independent experimental evidence.

| Initial action process | Full-horizon valid /96 | Positive revised score and valid /96 | Mean maximum drone displacement (m) |
|---|---:|---:|---:|
| Independent, original std | 49 | 3 | 2.963 |
| AR(1),50 ms, half latent std | 42 | 9 | 3.733 |
| AR(1),100 ms, half latent std | 50 | 20 | 5.294 |
| AR(1),50 ms, three-quarter latent std | 33 | 15 | 4.305 |

The100-ms half-std condition is a candidate for a controlled training experiment,
not a demonstrated learning winner: positive-and-valid counts are6/8/6 across
the three seeds, versus1/1/1 for the repeated independent baseline, but its
excursions are larger. Equal initial exploration coverage does not guarantee
equal learning quality. Its approved initial marginal pre-tanh std is
`exp(-0.3)/2 = 0.370409`, with `rho=exp(-0.02/0.1)=0.818731`. These parameters
are provisional engineering choices, now selected only in the new experimental preset.

Training integration must account for the correlated distribution explicitly.
For a standardized AR(1) residual, the conditional raw-action mean is
`mu_t + rho*sigma_t*(raw_previous-mu_previous)/sigma_previous` and conditional
std is `sigma_t*sqrt(1-rho²)` after the first step; the first step uses the
stationary marginal `N(mu_0,sigma_0²)`. Previous raw actions are fixed logged
history, while current-policy means/stds at both adjacent observations must be
recomputed during PPO updates. Old independent-Gaussian log probabilities,
entropy and KL calculations cannot simply be retained. The implementation below now uses these conditional probabilities for sampling,
PPO likelihood ratios, entropy and analytic KL. Deterministic evaluation still
uses the marginal mean policy from episode start.

[Verification](../tmp/stage1-attribution-20261022/tests.log):62 selected tests
passed, including seven new projection/covariance/rank/local-terminal-motion
checks and the prior55 tests. [Saved-array checks](../tmp/stage1-attribution-20261022/saved-array-verification.json)
verified all three new batch hashes, unchanged execution sources and exact
repeated-baseline equality. [Subtraction](../tmp/stage1-attribution-20261022/rigid-motion-subtraction.png)
and [exploration](../tmp/stage1-attribution-20261022/milder-exploration.png) plots
were visually inspected. Section tension/power histories are not recorded in
these runs: the native constraint solves compute reactions internally, whereas
current public force telemetry reports the cable-on-drone reaction. A physically
causal energy-transfer claim would require additional validated force/power
accounting; this projection does not supply it. No training or real flight was run.

**2026-10-22 approved Stage1 integration.**
[shape_release.py](../tasks/whip/shape_release.py) uses the original prefix-only
peak detector for mechanism evidence and frozen release bounds, discards its
old relative-speed utility, and applies the approved anchored shape-speed gain
and same-sample directional minimum. Mechanism evidence still uses the freely
translated/rotated fit in `extract_measurements`; output uses the distinct
attachment-anchored fit in `decompose`. Both use rest-length quadrature, measured
world positions/velocities and node0 as the attachment. Curvature uses measured
dual lengths. Versioned thresholds are in the original detector and the new task
module; legacy mechanics-window/alignment fields retained for configuration
compatibility are not inputs to this new utility.

The CPU float64 reference scorer runs once per completed cohort and returns
running100-Hz best finalized credit. Only finite executed prefixes enter the
fits. Explicit physical-failure flags prevent an all-finite final-step failure
from being mistaken for a valid horizon and clipping an unfinished release.
The existing environment samples that causal potential at50 Hz, pays record
increments, and reconciles the terminal transition to exactly−10 on failure.
Episode returns are unbounded above by an artificial point scale, use a1 m/s
reference unit, and have multiplier1. Workspace, effort and time costs are zero;
physical/numerical failure rules and(8,8,4) m/s² acceleration limits are unchanged.
Target crossings remain diagnostics. No automatic supported-whip labels or
later-stage handoffs are enabled.

[PPO](../training/ppo.py) implements `ar1_v1` with tau0.10 s at50 Hz and initial
marginal logstd `-0.3-log(2)`. The worker explicitly passes the previous logged raw
action and observation index, clearing both on every environment reset; the
learner has no hidden exploration state. Update minibatches recompute the current
policy's marginal mean/std at BOTH current and previous observation histories,
with gradients through both. Logged behavior means/stds are conditional. The
first action uses marginal variance; later conditional variance is multiplied
by `1-rho²`. Squashed conditional log probabilities, conditional entropy and
conditional old-to-new Gaussian KL all refer to this same distribution. The old
single-prefix KL input cache is bypassed for this mode. Checkpoint interfaces
include noise version, time constant, rho and likelihood version, so independent
and correlated policies cannot silently transfer/resume. Checkpoints remain at
whole-collection boundaries. Deterministic evaluation has zero latent noise
from reset and uses the marginal mean, without stochastic-state leakage.

The original zero-path-cost task/preset fixed4.5-s full episodes, fresh seed0,128 lanes,10,240 episodes
(80 collections), gamma1, lambda.95, reward multiplier1 and existing PPO update
settings. Entropy weight decreases .02→.001 on the completed-episode clock,
holding through10% and reaching its floor at80% of budget; it now weights
conditional entropy. Old plateau stopping is absent. Fixed-state evaluation is
scheduled after the first collection, every512 episodes, and the final collection.
No training initialization from the preflight or older policies is used.

Each stochastic collection retains all-lane returns, reward sums, event/output
components and workspace diagnostics, selected replay motions, and a separate
`collection-NNNNN-events.json` with frozen evidence, finalized/truncated flags,
shape gain, same-sample directions and rejection reasons. `event_indices` mean
segment start, arming, selected shape-speed peak; finalization is recorded
separately in each event. These are kinematic eligibility records, not human
physical-mechanism annotations. Run execution hashes now also include the three
shared evaluation helper modules actually used by this scorer. The Training UI
labels this preset and plots selected shape output/gain without a bounded score
claim. The reusable simulator and original presets are unchanged.

Verification is retained in [stage1-integration-20261022](../tmp/stage1-integration-20261022/).
The [CPU log](../tmp/stage1-integration-20261022/tests-cpu.log) records87 tests
(84 passed,3 CUDA skips); the [host log](../tmp/stage1-integration-20261022/tests-host.log)
records50 focused tests, all passed including CUDA optimizer/checkpoint checks.
These overlapping suites are not137 independent tests. A float64 multivariate
Gaussian reference checks the joint action likelihood and gradients through
both adjacent marginals. Tests cover reset, incompatible checkpoints, terminal
clawback, unfinished releases on failure, output-peak selection, target geometry
independence, and retained event evidence. An initial reference-test failure was
caused by constructing its covariance powers in float32; constructing the
reference in float64 fixed it without relaxing the1e−12 tolerance.
[Saved-data reconstruction](../tmp/stage1-integration-20261022/retained-reconstruction.json)
reproduced all384 follow-up probe scores exactly and the nine retained kinematic
controls, including zero for the independent-sweep counterexample. This is
reconstruction of existing evidence, not new physical validation.

The [native preflight](../tmp/stage1-integration-20261022/preflight-report.json)
used32 environments and separate seed2026102203, the full approved4.5-s task,
unchanged PPO settings and `cuda_conditional_v1` on the RTX4080. Eighteen episodes
were full-horizon valid; seven of those earned positive shape-release utility.
Conditional behavior log probabilities reconstructed with maximum error0;
reward sums and−10 failure replacement passed. All120 actor and120 value updates
completed; final conditional cohort KL0.005300004 was below0.02. Checkpoint reload
and the native deterministic evaluation completed (valid, return0). Collection/
backend preparation took67.79 s, scoring0.80 s, optimizer plus reconstruction/
saving6.62 s, total76.85 s. Retained arrays/events and source hashes pin these
checks. The preflight policy is not used to initialize training.

Fresh experimental run [20261022-015843-738817](../runs/ppo_stage1/20261022-015843-738817/)
was launched through the visible Training UI with the approved preset, seed0,
128 lanes and10,240-episode budget. [Startup verification](../tmp/stage1-integration-20261022/launch-verification.json)
confirmed no resume/actor initialization, zero initial mean head, initial logstd
−0.99314719 in float32, and exact execution-source hash agreement.
[First-collection verification](../tmp/stage1-integration-20261022/first-collection-verification.json)
confirmed128 episodes (61 valid,20 positive-and-valid;67 failures all returned−10),
460 actor and460 value updates, final conditional cohort KL0.006101537, and a
finite round1 checkpoint. Float32 saved reward sums matched the saved returns
exactly; recomputing sums in float64 differs by at most4.77e−7 from the rounded
float32 return. The first native deterministic evaluation was valid with return0.
The worker continued into collection2. These are first-collection integration
checks, not a learning-quality conclusion. Software checks and this preflight do not establish learned whipping,
causal energy transport, robustness or real-flight performance.

In the original offline event-review reports, tracks are clipped to each
candidate's start-to-peak interval. Reports include
raw outward/inward travel, material endpoints, amplitude and ambiguity flags;
arrival at the last interior sample is descriptive, not a required whip condition.
Speed changes, direction and localization are reported alongside the tracks.
Target geometry comes from that case's frozen task and scene. Plane crossings use
linear interpolation and require observed opposite sides, with both the arrival
and its opposite-side confirmation inside the candidate interval. A touch,
dwell returning to the same side, or recording ending on the plane is not a
confirmed crossing. Continuous piecewise-linear closest approach is also retained.
The finite rectangle test is geometric and adds no contact or impact model.

Install the plotting extra with `pip install -e ".[review]"`, then run:

```bash
.venv/bin/python -m evaluation.whip_event_review \
  --manifest /absolute/path/inputs.json --output /absolute/path/new-review
.venv/bin/python run_replay.py \
  --measurement-report /absolute/path/new-review/report.json --case 0 --event 0
```

The input schema is `offline_whip_event_inputs_v1`. Each case supplies a unique
`name`, replay `path` and `sha256`, a `source_snapshot` with its own `path` and
`sha256`, and `evidence_kind` (`saved_simulation` or
`synthetic_kinematic_control`). Optional `events` contain `start_index`,
`peak_index` and `end_index`; otherwise automatic intervals are used. Paths are
absolute or relative to the manifest. Inputs must contain valid saved replay
arrays and at least two time samples. Each frozen snapshot needs a sibling
`scene.toml` matching its `scene_text` for viewer loading. Different frozen models
are allowed and recorded; these cases do not form a controlled policy comparison.
Inputs are hash checked, and the output directory must be new.

Outputs are `report.json`, `events.csv`, `review.md`, per-case raw `trace-*.npz`
and synchronized `case-*.png` plots. The report records input, manifest and
measurement-source hashes plus Python/NumPy versions. Curvature and speed maps,
world/relative/residual speeds and signed forward components share a time axis;
the displayed candidate and peak are marked. Color scales are per case, so use
their units and raw values for amplitude comparisons.

The [existing replay loader](../viewer/replay_data.py) verifies the source pins
and exposes optional measurement annotations to the
[existing replay window](../training/ui/replay_window.py). Case and event indices
are zero based. Omitting `--event` uses the display default; cases without a
candidate still open as full replays. An amber path, timeline interval and peak,
an unreviewed measurement card, and candidate jump controls accompany the original
recorded task outcome. A detached diagnostic is not assigned a fabricated training
outcome. `--audit` and `--snapshot <image.png>` work with measurement reports.
Playback does not step physics or run a policy.

The local example generator
`python -m evaluation.whip_measurement_examples --output <new-input-folder>`
collects retained hover/sinusoidal controls, a historical Stage1 replay and the
separate round9 zero-noise diagnostic, then constructs six analytic kinematic
controls. It requires those local retained sources. Controls include rigid
rotation/translation, tiny forward/backward waves, separated fading/growing bends
and an earlier wave with a later unrelated crossing. They exercise logic and do
not assert physically feasible flights or ground-truth whip labels.

Retained evidence for this implementation:

- [Eleven-case input manifest](../tmp/whip-event-review-20261022/inputs-v2/manifest.json).
- [Measurement review and synchronized plots](../tmp/whip-event-review-20261022/final-review/review.md),
  [machine-readable report](../tmp/whip-event-review-20261022/final-review/report.json)
  and [all candidate rows](../tmp/whip-event-review-20261022/final-review/events.csv).
- [Source-pin and eleven-case viewer-load verification](../tmp/whip-event-review-20261022/final-review/verification.json),
  [75-test log](../tmp/whip-event-review-20261022/verification-tests.log), and
  [rendered historical replay](../tmp/whip-event-review-20261022/final-review/viewer-historical.png).

The selected unittest suite passed 75 tests covering extraction, rigid-motion
controls, tracking ambiguity, event separation, plane touches/confirmation timing,
hashes, overwrite protection, viewer integration and existing relevant replay/
history/reward-review behavior. All eleven final cases loaded in the viewer and
their source pins matched; the historical plot and actual 3D replay rendering
were visually inspected. No new physics rollout, policy inference, training,
reward weight or curriculum change was part of this implementation. Independent
event labels, physical-whip classification, mesh/time convergence and real-flight
validation remain unverified. Overlapping/fading waves can still confuse local
peak tracking; temporal association alone does not prove mechanical causation.

## Offline reward redesign review — implemented diagnostics, earlier full-task candidate

On2026-10-21 the user selected genuine whipping followed by an accurate tip
strike as the objective and approved an offline evaluator before changing PPO.
The [evaluator](../evaluation/reward_review.py) is deliberately independent of
the training reward and the existing Stage1–4 transitions. Subsequent approval
implemented only the separate discovery preset above. Full-task success and
later-stage advancement rules remain proposals requiring review.

Run `.venv/bin/python -m evaluation.reward_review --manifest <inputs.json>
--output <new-folder>` from this checkout. The manifest schema is
`offline_whip_review_inputs_v1`; `reference_snapshot` and every case's replay and
`source_snapshot` specify a path and SHA-256. Relative paths resolve against the
manifest. Cases carry a unique name and distinguish `saved_simulation` from
`synthetic_kinematic_control`. The evaluator checks the common model and scene,
initial tracking position, proper body-to-world rotations and observed100-Hz
numerical prefix. Incomplete safe prefixes have no episode return; a retained
numerical-failure explanation can identify a failed prefix. Output folders must
be new. Inputs and old run rewards are never overwritten.

The offline candidate is `R=-1` on failure and `R=5*S+Q-0.25*C` on full-horizon
completion. `C` is the right-endpoint sum of
`min(1,(radial_excess/0.1m)^2+(vertical_excess/0.1m)^2)*0.01/T`, including the
executed prefix only and using fixed full horizon T. Thus C is in[0,1]. It uses
the unchanged0.55-m cylinder and1.5–2.6-m world-Z limits, and does not constrain
the cable to the cylinder. Full-episode containment is additionally required
for S. Existing tilt60 degrees and minimum cable height0.05 m remain failure
criteria. Failure after a strike still yields−1. This is an explicit candidate
preference, not a physical constraint solver or a guarantee of safe learning.

`W=M*U*D` reuses the existing directed-whip mechanics proxy without the old
preparation-path discount. Candidate window search covers the full4.5-s horizon,
retaining the inherited window durations/stride and0.15-s release lag. At each
actual forward crossing, the evaluator restricts the existing scorer to that
crossing and its preceding windows; a different earlier maximum cannot qualify
it. Crossing data become available at the observed interval endpoint. Positions
and velocities use the existing interpolated whole-cable crossing state, not a
new physics substep. Synthetic controls do not establish physical feasibility.

The first explicit offline partial-credit candidate is
`Q=max(0.5*best_sampled_W, max_crossing 0.5*W_crossing*(1+A_crossing))`, where
`A=exp(-0.5*(lateral_error^2+vertical_error^2)/sigma^2)` and
`sigma=max(0.05m,max(target_width,target_height)/4)`. All joint terms refer to the
same crossing; Q is in[0,1]. This partial-credit choice and scales are for offline
comparison, not validated training settings or a physical whip definition.
Continuous guidance before a crossing, per-step reward delivery, and GAE remain
unimplemented by this evaluator and require a separate decision/verification.

S requires completed, contained execution plus an existing strict crossing and
a reviewed whip label on that particular crossing. Optional case `whip_labels`
carry `interval_index`, boolean `is_whip`, `reviewer`, and `reason`; the replay
hash binds labels to the reviewed input. Without a label, an otherwise eligible
crossing leaves S unresolved and produces `[Q-0.25*C,5+Q-0.25*C]`. These are
classification bounds, not statistical confidence intervals. Positive W alone
never establishes genuine whipping. Threshold sweeps0.01/0.05/0.1/0.2/0.3 are
explicitly proxy sensitivity diagnostics, not accepted classifiers.

The evaluator reports the same candidate at each existing target size
1.0×0.6 →0.6×0.4 →0.3×0.2 →0.1×0.1 m. It reuses the strict crossing's inclusive
geometry, forward speed≥3 m/s and entry angle≤30 degrees. These tables do not
advance training. Current Stage3 still advances on80% geometric hits for three
collections, all lanes valid, and a valid updated-policy nominal geometric hit;
containment is diagnostic at that gate. Redesigning that gate to require linked
whipping, containment and reliable held-out performance remains proposed.
The acceleration campaign has only attempted Stage1; its Stage2/3 presets have
not established progression and its Stage4 recovery remains incompatible.

The retained [input manifest](../tmp/reward-review-20261021/inputs.json),
[verified report](../tmp/reward-review-20261021/verified-results/report.json),
[summary](../tmp/reward-review-20261021/summary.json) and
[figure](../tmp/reward-review-20261021/reward-review.png) document2026-10-21
offline scoring of198 saved simulation trajectories and4 synthetic controls.
The saved set consists of195 archived nominal evaluations and3 previously run
numerical probes, including a duplicate final-command reproduction; these are
not198 independent trials. The release data were restored to isolated
`tmp/training-audit-20261021/snapshot/`, not the active `runs/` tree.

Of those198 trajectories,176 completed,22 failed, and54 completed while fully
contained. Strict crossings at the four target sizes occurred in83/8/0/0
trajectories; requiring full completion and containment reduces this to6/0/0/0.
These counts do not include a validated genuine-whip classification. No retained
trajectory here provides a positive final-precision example. Hover has candidate
return0 and both retained descent-failure probes−1. An artificial early wave
followed by an unrelated rigid crossing retains a positive earlier W but zero
W at the crossing. Its unknown human classification stays explicitly unresolved.
These checks support the evaluator's arithmetic/event separation; they do not
prove learnability, classify real whip motion, or validate real flight.

## Retained models and evidence

[The milestone manifest](../models/milestone-20261009/manifest.json) records
one coherent completed chain: Stage1 `20261006-195224-532710` (80/10,240),
Stage2 `20261006-214007-568609` (80/10,240), Stage3 `20261007-001712-887951`
(120/15,360), and Stage4 `20261007-034135-720530` (60/7,680), with counts given
as collections/episodes. Each later initial actor is byte-identical to the
preceding final actor. Stage1 continued from a 36-collection run; its earlier
initial RNG byte-comparison gap remains unresolved.

The checkpoints preserve original 96-column Stage1-3 and 98-column Stage4
interfaces without the later relative goal. Current CPU loading, exact actor-state
restoration, finite synthetic inference, and physical-model/scene compatibility
were checked in milestone selection. Task/training execution sources have
changed, and unused study modules were archived; the final compatibility record
lists the earlier milestone differences, not approval of later source changes.
The subsequent containment/UI edits changed physically guarded Python files
(`simulation/model_configuration.py` and full-state controller UI/launcher paths),
without changing equations, parameters, rewards, thresholds, or timesteps.
The existing physical-source and binary-hash guards therefore reject fresh plant
reevaluation/actor transfer from the packaged historical checkpoints. Policy
Tester still loads them for inspection but disables incompatible fresh tests.
Saved replay and CPU synthetic inference work. Rebuilding native binaries on
Linux requires separate compatibility validation; these guards were not loosened.
Exact historical resume/reproduction is not claimed.
No checkpoint is relabeled as trained with current observations or the new
Stage1 compact reward. These retained models predate that reward change, and no
new trained model has been promoted. Evidence
contains saved settings/model/task/scene, summaries, initialization links,
historical audits, and final nominal evaluation/replay for each stage.

Saved final nominal results: Stage1 valid directed mechanics with unfolding 0.9911
and no geometric/strict hit; Stage2 valid geometric/strict broad hit; Stage3
reached the final target level and hit with 9.24 mm center error, 7.50 m/s forward
speed, and 21.22-degree angle. Stage4 retained the frozen whip hit and passed saved
recovery/cable diagnostics: final drone error 35.9 mm, speed 0.0158 m/s, held cable
relative RMS speed 0.0255 m/s, and horizontal RMS extent 6.25 mm. All four stages
breached the soft workspace; Stage3/4 maximum radial/vertical excess was 0.0883/
0.2085 m. These are saved single-case results, not fresh milestone training,
convergence, containment, varied-target robustness, or flight evidence.

Streaming Crazyflie command/feedback integration with the native simulation is
implemented on Linux as described above. Real-flight integration, cable-state
estimation from drone telemetry, firmware/controller equivalence, hardware
parameter identification, arbitrary starts, moving targets, and real-flight
safety remain unimplemented or unverified. No model default or physical fit was
promoted during cleanup or this connection work.

## Recordings and fitting

The adjacent
[motor/PVA logger](../../Crazyflie/crazyswarm2_jazzy_whipping/scripts/experiment_logger_optitrack_pva_motor_v3.py)
now subscribes to raw `motion_capture_tracking_interfaces/NamedPoseArray` on
`/poses` (`--poses-topic` overrides it), selecting the exact `--drone` name.
The existing script filename is retained, but new recordings use schema
`optitrack_poses_pva_motor_v4`: `pose_stamp_ns`, `pose_time_s` and `pose_dt_s`
replace the old `tf_*` columns. Metadata records source topic, pose name, parent
frame and source-clock limitations. No coordinate conversion is applied.
`--parent-frame` can require a particular frame; otherwise the first matching
parent is locked. Former `--tf-topic`/`--mocap-frame` options fail with migration
instructions because TF visualization may replace missing orientation with an
identity quaternion. The logger retains invalid raw quaternion samples with
`pose_valid=0` and invalidates finite-difference velocity across them and tracking
gaps; a legitimate measured identity quaternion remains valid. Missing named
poses do not produce invented samples. Command and motor sidecars retain their
existing fields, clocks and validity rules. Existing recordings are untouched;
this change does not migrate old schemas or establish clock synchronization.

The adjacent `force_controller.py`, `force_controller_cal.py` and
`force_controller_v1.py` mark their saved gain state dirty before the first
parameter write, so partial write failures enter the existing restoration path.
The v1 exception handler also attempts emergency even if restoration fails.
This is best-effort cleanup: asynchronous parameter writes and physical gain
restoration still require hardware verification. No force definition, gain
default, calibration formula or learned-policy interface changed.

Original source recordings remain external and untouched. For active use, copy
the collection inside `data/recordings/` with its original `data/flight_batches`
and linked rehearsal layout. Take Processing selects only project-contained
collections; it no longer infers an external root from historical catalog paths.
The prior personal catalog was removed from tracked files during the earlier
containment cleanup. Its former project-run backup is no longer available.
New catalogs and local preferences are ignored by Git. A clone
has an empty recording folder and needs user-supplied data for fitting.
Fitting checks catalog/alignment/source
hashes and complete 100 Hz windows. Missing observations are masked, not invented.
Causal history initializes velocities; twist and initial motors remain assumptions
where telemetry is absent. Command-driven and measured-boundary replay answer
different model-validation questions.

Handheld OptiTrack demonstrations are a proposed next source of motion evidence;
no such recording is currently available in the project. The recommended pilot
uses the same marked cable with a rigid tracked handle and a mounting boundary
matching the drone clamp. Capture handle position/orientation and all ten cable
markers, retain original TAK/CSV files at native capture rate, and record the
attachment offset and mounting direction in the handle's local frame. A brief
stationary lead-in/out and several complete preparation/drive/pullback attempts
would support tracking checks and motion comparison. Use meters, world
coordinates, quaternion rotations, explicit world-Z-up convention and CSV
headers; export physical marker observations rather than substituting solved
constraint positions for occluded cable markers. These controls are described
in [OptiTrack's CSV export documentation](https://docs.optitrack.com/motive/data-export/data-export-csv).

This is a capture/retargeting proposal, not an implemented imitation pipeline.
[load_take](../take_processing/take_data.py) currently expects a drone identity
and ten named cable markers; existing fitting preparation additionally requires
verified flight-command clock alignment and contiguous native 100 Hz data, with
no resampling. A hand take should not be relabeled as drone flight or supplied
invented command logs. It provides observed motion, not drone action labels.
The proposed sequence is measured-boundary cable replay, then a search for
feasible bounded drone commands and timing, then imitation from those simulated
state/action trajectories. The existing prescribed-cable engine can represent
attachment P/V/A and mounting rotation/angular rate, but hand-specific import,
fixture calibration, derived-data resampling if needed, retargeting objectives
and imitation remain unimplemented. Wrist motion and drone actuation may differ;
physical parameters, rewards and curriculum are not changed automatically to
match a demonstration.

[fitting/run_refit.py](../fitting/run_refit.py) requires an explicit base model.
It fits cable EI/Cb/external drag with measured attachment motion, then vehicle
controller response scales with cable/motors frozen in the coupled plant. Default
selection is four training and thirteen historically exposed comparison takes.
`--data`, `--split`, and `--prepare-only` select/record inputs. Candidates and
comparisons are saved under `runs/refit` without automatic default installation.

[fitting/run_residual_fit.py](../fitting/run_residual_fit.py) separately fits
structured nonnegative aerodynamic coefficients: cable first, then drone with
cable weights frozen. It requires a calibrated physical base and gradient gates.
The active residual remains unfitted/disabled. Low fitting loss does not establish
identifiability or reliable whole-maneuver cable prediction.

## Build, validation, and dated change

[build_engine.py](../build_engine.py) invokes the C++17 extensions in
[setup.py](../setup.py); installed Torch enables the tensor build. Rebuild for
each OS/Python ABI. There is no Python solver fallback. Windows requires MSVC/
SDK; non-Windows builds use the supplied strict floating-point compiler flags.
Existing Linux native/tensor binaries and local `.venv` remain unchanged.
The current suite checks configuration, rewards/causality, numerical behavior,
actor/control contracts, checkpoint restoration, recovery and UI provenance.
It does not establish learned-policy or real-flight success. Previous validation
logs under `tmp/` remain local historical checks; this cleanup does not imply
a full suite, model fit, simulation or training experiment was rerun.

2026-10-10 approved clean Stage1 restart: the previous restored-radius Stage2
run was safely stopped after five complete collections/640 episodes; Stage3
was not launched. The old campaign terminated on `user_stop`. The user explicitly
approved deleting all old project runs,150 Stage1 collections and a new random
policy. The [cleanup manifest](../tmp/stage1-clean-restart-20261010/cleanup-manifest.json)
records2,169 deleted files/1,095,648,064 bytes, including old PPO runs, motion
searches, cloned demonstration and reward-review evidence. The old Training
windows were closed after their workers stopped. `runs/` was verified empty
before the new experiment. All108 checked `models/`, `data/` and existing
configuration files remained unchanged. The maintained record was shortened to
replace deleted-run references and describe the active recipe; a prior text
snapshot is retained in the cleanup folder. Original external recordings were
not part of the deletion. The restart changes only the Stage1 budget from its
prepared80-collection counterpart to150, and uses fresh actor/value initialization;
rewards, physics, workspace, scientific criteria and source code are unchanged.
No old run outcome is promoted or presented as retained flight evidence.
Preparation, launch and new experiment outcomes are recorded below.

2026-10-10 clean-restart preparation:
the [preflight](../tmp/stage1-clean-restart-20261010/preflight.json) loaded all
three150-collection/128-lane/0.55 m configurations and exercised the actual
Stage1 create-run path with `resume=null` and `initialize_actor=null`. No initial
actor payload is present. CUDA is available; no physical rollout was performed
by this check. The Stage1 settings differ from the prepared80-collection recipe
only in total episode budget. All108 pre-cleanup protected files stayed exact;
the new campaign pins109 input files and71 execution hashes.
The one-off [campaign driver](../tmp/stage1-clean-restart-20261010/campaign.py)
starts fresh Stage1 in the Training UI and retains an accounting/native review
after each completed stage. It transfers each final actor into the next stage
with a fresh critic/Adam/exploration clock, records native transfer checks and
stops the sequence if the user stops a run. After Stage3 it records the existing
Stage4 handoff assessment without starting Stage4 training. The retained-array
[reviewer](../tmp/stage1-clean-restart-20261010/review.py) distinguishes fresh
Stage1 initialization from later actor transfers and preserves new logs and
source/ownership references. This is preparation, not a completed policy result.


2026-10-10 Clean Stage1–3 150-collection restart: Stage1 launched in [20261010-215414-837444](../runs/ppo_stage1/20261010-215414-837444); its subsequent status and review are recorded below.


2026-10-10 fresh Stage1 launch verification:
the [live initial checkpoint](../tmp/stage1-clean-restart-20261010/initial-live-check.json)
confirmed counters0, no actor payload/provenance, empty actor/value Adam states,
zero exploration offsets, seed0 and19,200-episode entropy budget. The active
[Stage1 run](../runs/ppo_stage1/20261010-215414-837444/) completed its first new
collections and one nominal native evaluation. Its
[interim saved-array review](../runs/ppo_stage1/20261010-215414-837444/interim-review.json)
checked256 stochastic episodes for return/workspace/transition accounting and
valid native event scoring/crossing diagnostics. Maximum float32 return
reconstruction error was0 and transition-sum error4.77e-7. This is a launch
check, not evidence of learned success or a completed150-collection result.
The [cleanup/ownership preservation check](../tmp/stage1-clean-restart-20261010/launch-checks.json)
verified all2,169 old run files absent, exactly one new Stage1 run, an active
campaign/UI/worker and all109 pinned input files/71 execution hashes unchanged.
This describes the initial random-policy launch, now stopped. The subsequent
approved demonstration-seeded replacement is recorded below.

2026-10-10 approved replay-demonstration restart preparation:
the [stopped random run](../runs/ppo_stage1/20261010-215414-837444/summary.json)
ended at44 complete collections/5,632 episodes, with its logs and checkpoints
preserved; its driver terminated without launching Stage2. Collection40's
native evaluation was valid with no numerical failure, selected reach0.042924 m
and forward tip speed0.052959 m/s; its base event credit was8.38948e-10, before
workspace cost. The saved evaluations suggest an exploration/optimization
problem; they do not prove an eventual random-policy run could never learn.
The [approved request](../tmp/stage1-replay-demo-150-20261010/restart-request.json)
covers reconstruction, fresh actor cloning, native verification and the new
150-collection Stage1/2/3 chain, with unchanged reward, physics and workspace.
The [seed assessment](../tmp/stage1-replay-demo-150-20261010/seed-assessment.json)
separates exact teacher reconstruction from the new cloned actor's nominal
execution and confirms unchanged critic weights, empty PPO optimizers and zero
learner/exploration counters after supervised initialization. Original protected
inputs and stopped-run files are pinned by hash. No PPO result or model promotion
is claimed by these seed checks. Training launch verification follows below.

2026-10-10 replay-demonstration restart launch verification:
the [actor-transfer preflight](../tmp/stage1-replay-demo-150-20261010/preflight.json)
exercised the actual create-run path with an immutable actor payload and
`resume=null`. Loading the seed into a fresh PPO learner left critic, RNG,
optimizers and counters unchanged and reproduced its five retained physical
state arrays exactly in a native rollout. All three settings loaded as
150 collections/19,200 episodes at128 lanes and radius0.55 m. The new
[campaign driver](../tmp/stage1-replay-demo-150-20261010/campaign.py)
opened [Stage1](../runs/ppo_stage1/20261010-221814-887854/) in the Training UI,
with subsequent Stage2/3 final-actor transfers and reviews retained in this
campaign's folder. The copied one-off [reviewer](../tmp/stage1-replay-demo-150-20261010/review.py)
now accepts the approved Stage1 actor initializer and checks the150-collection
range; the copied driver's episode label is19,200. These fixes affect the
campaign's review/reporting, not the learner or success criteria.
The [live initial checkpoint check](../tmp/stage1-replay-demo-150-20261010/initial-live-check.json)
confirmed exact seed actor weights, exact fresh seed0 critic weights, empty
actor/value Adam states, zero learner/exploration offsets and19,200-episode
entropy budget at collection0. Stage1 training is running; no completed PPO
result is claimed. The campaign retains the original per-stage reward,
curriculum, workspace, physical execution and Stage4 handoff assessment.
Stage2 and Stage3 are pending this new chain's preceding final actor.

2026-10-10 replay-demonstration initial PPO check:
the [retained-array review](../runs/ppo_stage1/20261010-221814-887854/interim-review.json)
checked the first128 stochastic episodes (117 valid,1 valid and contained),
including all failed lanes. Float32 return reconstruction error was0;
transition-sum error was9.53674e-7. The first updated actor's native evaluation
was valid with selected-event reach0.621090 m, forward tip speed5.42418 m/s and
tip-to-drone height error-0.751389 m. Its whole-episode maximum radial excess
was0.143841 m, so it was not contained; raw return0.00277292 includes workspace
cost0.0124904. Saved native reward reconstruction was exact. This first update
retains a useful stroke but has not solved height, containment or target success.
The [initial review summary](../tmp/stage1-replay-demo-150-20261010/initial-review-summary.json)
and [launch preservation check](../tmp/stage1-replay-demo-150-20261010/launch-checks.json)
retain those limits and confirm300 pinned files/71 execution hashes unchanged,
including the stopped random run. Training continues with the approved reward
and schedule; no scientific criterion was relaxed and no full regression suite
or real-flight experiment was rerun for this one-off seed reconstruction.


2026-10-10 Clean Stage1–3 150-collection restart: Stage1 stopped by the user at collection44; no later stage was launched. Existing results and checkpoints were retained; no completion is claimed.


2026-10-10 Replay-demonstration Stage1–3 150-collection restart: Stage1 launched in [20261010-221814-887854](../runs/ppo_stage1/20261010-221814-887854); its subsequent status and review are recorded below.

2026-10-10 Stage1 reward/progress review, without implementation changes:
the updated [saved-array review](../runs/ppo_stage1/20261010-221814-887854/interim-review.json)
checks113 collections/14,464 episodes and native evaluations through collection112.
Maximum float32 cohort return and native return reconstruction errors were0.
The collection112 native rollout was valid and contained, but selected-event
reach was0.370392 m and tip-to-drone height error-0.921438 m, compared with
the seed's0.639620 m and-0.737543 m. Forward tip speed remained5.85219 m/s.
The existing Gaussian height factor was0.0219331 and event credit0.00263726.
This confirms declining nominal reach/height while containment improved; weak
reward signal is a diagnosis hypothesis, not proof of a single causal failure.
The [review and proposal record](../tmp/stage1-replay-demo-150-20261010/reward-review-proposal.json)
separates these retained measurements from illustrative same-event calculations
for an unapproved alternative. Those altered coordinates were not simulated;
the alternative was not implemented, all other events were not rescored under
it, and no new policy was trained. The current reward and automatic campaign
remained unchanged until the subsequent approval recorded below.

2026-10-10 approved linear outreach/height Stage1 implementation and preparation:
the [approved request](../tmp/stage1-linear-outreach-height-150-20261010/approved-request.json)
specifies `K*(0.5r+0.5h+2rh)`, linear h, unchanged workspace/validity rules and
150 new collections at128 lanes. The preceding campaign
[stopped after121 collections](../tmp/stage1-linear-outreach-height-150-20261010/stopped-summary.json)
without launching Stage2. Its final actor was copied immutably to
[source-actor.pt](../tmp/stage1-linear-outreach-height-150-20261010/source-actor.pt);
all logs/checkpoints remain in the original run. The implementation adds a
versioned Stage1 reward and explicit weight guards to task/scorer, loader,
environment and worker dispatch, with matching existing-UI labels/components.
Earlier reward arithmetic, physical execution, source recordings and model
selection are preserved; this is actor-only transfer, not exact resume.

[Validation](../tmp/stage1-linear-outreach-height-150-20261010/validation.json)
passed129/129 tests, including8 new synthetic scoring/delivery probes for the
formula and selected-event accounting, height symmetry/clipping, rigid-motion
and speed gates, future/invalid-suffix causality, failure correction and CUDA
parity. Synthetic coordinate probes do not establish feasible drone motions.
The [native transfer preflight](../tmp/stage1-linear-outreach-height-150-20261010/preflight.json)
exercised the actual immutable initializer/create-run path, confirmed exact
source actor with fresh critic/Adam/RNG/clocks, and re-integrated one source
actor under old and new task rewards. Five physical state arrays were exactly
identical, and execution was valid and contained. The new reward selected
reach0.358478 m and height error-0.921940 m, with event credit0.0760935; larger
credit here is a scoring change, not motion improvement. Independent same-event
accounting and an old saved native return reconstruction passed. All766 checked
retained input/run files stayed exact, with no physical source changes.
The [campaign driver](../tmp/stage1-linear-outreach-height-150-20261010/campaign.py)
will review Stage1's native motion and all-lane accounting before the existing
Stage2/3 continuation, retaining their original reward and handoff definitions.
It stops the sequence on a user stop or execution failure. Preparation alone
does not establish PPO improvement, robustness, target success or real flight.

2026-10-10 linear outreach/height launch verification:
the new [Stage1 run](../runs/ppo_stage1/20261010-232154-080392/)
was launched in the Training UI with the new v12 preset, then stopped at the
user's explicit pause before any completed collection or PPO update. The
[live initial checkpoint check](../tmp/stage1-linear-outreach-height-150-20261010/initial-live-check.json)
confirmed exact actor transfer from collection121, exact fresh seed0 critic
weights (different from the trained source critic), empty actor/value Adam
states, zero learner/exploration counters and a19,200-episode entropy budget.
The immutable actor payload matches the retained source hash. This is a fresh
reward experiment; its history is separate from the preserved old-reward logs.
The new driver retains a full review before each later-stage launch. Stage2/3
remain on their already approved presets and have not yet launched in this
new campaign. No completion or improvement is claimed by initialization checks.


2026-10-10 Replay-demonstration Stage1–3 150-collection restart: Stage1 stopped by the user at collection121; no later stage was launched. Existing results and checkpoints were retained; no completion is claimed.


2026-10-10 Linear outreach/height Stage1 experiment and existing Stage2/3 continuation: Stage1 launched in [20261010-232154-080392](../runs/ppo_stage1/20261010-232154-080392); its subsequent status and review are recorded below.


2026-10-10 Linear outreach/height Stage1 experiment and existing Stage2/3 continuation: Stage1 stopped by the user at collection0; no later stage was launched. Existing results and checkpoints were retained; no completion is claimed.

2026-10-10 explicit training pause for simulation work:
the [pause check](../tmp/stage1-linear-outreach-height-150-20261010/pause-check.json)
and current run summary confirm a user stop with0 completed collections,
0 completed episodes and0 gradient updates. The interrupted first cohort was
discarded. The latest checkpoint and all logs remain available. No training
workers are active, the campaign driver has exited, and Stage2/3 have not
launched. Training will remain paused until the user requests a restart or
resume. This state check did not rerun simulations or scientific experiments.


2026-10-10 measured simulation execution and Linux CUDA conditional support:
training was already stopped before these checks; it was not restarted.
The task-owned driver now supports Linux NVRTC/driver loading, and supported
Linux Torch CUDA builds select the existing device-controlled loop by default.
The [retained scripts, source hashes, timings and numerical checks](../tmp/simulation-performance-20261011/summary.json)
separate CPU plant/prescribed-cable timings, GPU plant throughput and display
rendering. Native/simulation sources and the checked physical/actuator
configurations match their pre-change hashes. Host/conditional1- and128-lane
state comparisons are bitwise equal; the collapsed-edge rejection and driver
WHILE caps1/4/20 were checked. No physical equation, solver tolerance, controller
clock, model artifact, reward, or policy was changed by this simulation work.
No model fitting, new policy training, real-flight test, viewer redesign or ROS
integration was run. The Windows loader branch was not rerun on Windows.
Exact-resume source checks remain enforced: task execution-source changes do
not authorize bypassing old checkpoint provenance.
The [existing regression suite](../tmp/simulation-performance-20261011/validation.json)
passed129/129 tests with CUDA available and no skips. These checks do not
establish policy improvement or real-flight validity.
The [default-backend check](../tmp/simulation-performance-20261011/default-backend-check.json)
constructed the128-lane task without an explicit scheduler choice and advanced
one finite step using `cuda_conditional_v1`. The
[paused-run source comparison](../tmp/simulation-performance-20261011/paused-run-source-check.json)
finds changed hashes for the two task scheduling files in the retained zero-update
run; exact resume under its old execution provenance remains incompatible.
This work did not restart that run or weaken its source checks.

2026-10-11 independent simulation execution and flight viewer: the user approved
the worker/display-interface/viewer scope. The local simulation now runs in its
own thread, while Qt consumes immutable latest frames. The shared renderer/window
accept named robots and optional cables with explicit source/timestamp metadata;
simulation, saved replay and a read-only frame feed are implemented. The docked
UI and camera/layer/playback controls replace the former simulation sidebar.
The existing saved-outcome browser remains available; `run_replay.py --flight-view`
opens its selected arrays in the new window.

The [regression report](../tmp/viewer-rebuild-20261011/validation-final-ui.json)
passes139/139 checks with CUDA available and no skips. New checks cover30 native
trajectory steps (0.3 simulated seconds) with every runtime checkpoint bitwise
equal to direct stepping, diagnostic toggles, pause/reset/source switching,
immutable ownership, invalid-publication prefix retention, freshness indicators,
read-only feed/frame guards and saved-array replay isolation. These are numerical
and interface checks, not policy-training or real-flight results.
The [X11/OpenGL UI checks](../tmp/viewer-rebuild-20261011/ui-checks.json) cover
trajectory application, manual restoration, force/layer/camera controls,
default/small window layouts, retained actor identities, multiple robots with an
absent cable, worker shutdown and the existing packaged Stage4 replay window.
The final checkbox check exposed unsafe inline tree rebuilding. Structural tree
updates now wait until checkbox callbacks finish, and force-diagnostic changes
retain simulation checkbox items. The [50-cycle layer check](../tmp/viewer-rebuild-20261011/layer-stress.json)
passes repeated force/robot hiding and restoration with stable checkbox identity
and a joined worker; two subsequent complete UI checks passed. This is a local
GUI check, not proof of cross-platform graphics stability.
A deliberately blocked Qt consumer allows14 native output steps while it is
idle; a short24-render camera rotation loop allows2 steps. Both resulting native
checkpoints equal direct execution at the same output count. This demonstrates
execution independence in those short cases; it does not establish a sustained
interactive-frame-rate or worst-case timing guarantee. The
[entry-point check](../tmp/viewer-rebuild-20261011/entry-point-check.json) opens
the simulation with an ephemeral local controller port, single-steps and closes
with its worker joined. The read-only feed preview also displays measured pose
and an explicitly stale estimated cable without playback/trajectory controls.
[Simulation preview](../tmp/viewer-rebuild-20261011/simulation-flight.png),
[replay preview](../tmp/viewer-rebuild-20261011/replay-flight.png) and
[frame-feed preview](../tmp/viewer-rebuild-20261011/display-feed.png) are retained
display artifacts, not flight measurements. The checked126 native/simulation,
controller, task, training and configuration files match their
[pre-change hashes](../tmp/viewer-rebuild-20261011/source-hashes-before.json).
No physics, physical timing, reward, learner source, active model or recorded
data was changed, and training remains paused. Exact-resume source guards remain
intact; the earlier task-scheduler changes still require their existing provenance
checks. This viewer change did not include a ROS bridge; the subsequent streaming
adapter is recorded below. A causal real-flight cable observer and learned-policy
deployment remain proposed work. Windows-specific interactive behavior was not
exercised by these Linux checks.

2026-10-11 approved custom-simulation ROS connection: added the host launcher,
streaming-only Docker ROS adapter, Crazyswarm `--custom-sim` entry and scripted PVA
test. Real/stock simulation launch branches remain unchanged. The adapter owns
transport/frame/clock checks and recording; native physics/controller, physical
parameters, delays, rewards, learning and active artifacts remain unchanged.

The [regression report](../tmp/ros-connection-20261011/validation-final.json) passes145/145
checks with CUDA available and no skips, including6 new transport/frame rejection,
FIFO/state-preservation, socket lifecycle and native command-clock parity checks.
The direct-step check compares every complete native checkpoint over10 commanded
output steps and confirms the configured controller/delay behavior.
The [retained ROS connection check](../tmp/ros-connection-20261011/connection-checks.json)
records270 scripted commands sent/received/scheduled with zero rejection and no
missed command slots. ROS pose and full-state feedback equal the retained native
snapshots bitwise. Replaying the actual recorded request schedule through the
same native model reproduces1,502 output states and the final exact checkpoint
bitwise; all270 retained applied events match their request plus0.03 s. Header
intervals in this run range0.02–0.05 s despite nominal30-Hz scheduling; these are
observed desktop/ROS timing results, not a hard-rate guarantee. The receipt-sample
to request-boundary lag ranges0–0.01 s in this case. The
[raw PVA result](../tmp/ros-connection-20261011/pva-result.json) and
[host run](../tmp/ros-connection-20261011/ros-pva-final/metadata.json) are retained.
The checked121 physical/controller/task/training/configuration source files match
their [pre-change hashes](../tmp/ros-connection-20261011/source-hashes-before.json).
These are transport/reconstruction checks, not identification, policy-training,
firmware equivalence or real-flight evidence. Training remains paused and no
policy is promoted.
The [final X11/GPU viewer check](../tmp/ros-connection-20261011/ui-checks-final.json)
runs a shorter2-s motion/5-s stream with150 accepted commands while the viewer
is open. The named `cf_2` source, absence of local trajectory editing, force
diagnostics, pause/step/reset and native-owner shutdown passed. During a0.22-s
blocked Qt consumer the owner advances22 output steps. The
[received-feedback check](../tmp/ros-connection-20261011/feedback-checks.json)
compares535 pose/full-state/odometry/cable/clock samples and536 TF samples to the
native trace: pose, quaternion, full-state fields, TF and cable positions agree
bitwise, timestamps match, and independent `Rᵀ v_world` reconstruction differs
from published body-frame velocity by at most1.39e-17 m/s. The
[viewer capture](../tmp/ros-connection-20261011/custom-simulation-flight-final.png)
is a simulated display artifact. The checked ROS adapters terminate after normal
window/headless shutdown; no test ROS server remains running. Windows behavior,
high-level Crazyswarm services, firmware controller agreement, calibrated latency,
causal cable observation and actual policy/flight use were not tested or added
in those original streaming checks; the later API step is recorded below.

2026-10-11 custom-simulation TCP restart fix: the reported port47654 failure had
no active listener when inspected; its original cause cannot be established
retrospectively. A separate closed-connection reproduction confirmed `Errno98`
with the old listener lacking address reuse. The listener now sets
`SO_REUSEADDR`, and occupied-port errors are reported without a traceback while
normal source/evidence cleanup still runs. Default port/domain and command,
feedback, physics, controller, reward and learning behavior remain unchanged.
The [restart checks](../tmp/ros-port-restart-20261011/restart-checks.json) pass two
back-to-back Docker-backed headless0.5-s launches on47654. An occupied-port CLI
check preserves the existing listener and returns the intended concise error.
The [regression report](../tmp/ros-port-restart-20261011/validation.json) passes146/146
checks without skips, including a real accepted-TCP-connection restart and
live-listener exclusivity test. These are transport lifecycle and short native
hold checks; the PVA trajectory, GPU viewer, fitting, policy training and real
flight were not rerun for this fix.

2026-10-11 Crazyswarm airborne command compatibility: user-approved command/YAML
integration now loads the hardware robot metadata with the explicit simulation
position override, exposes stock client discovery/parameters and functional
high-level services, and keeps command planning outside the unchanged native
plant/controller. The simulation shell/client clock patch and simulated CSV
scheduler handle ordinary Python execution, hover pacing and stream-to-land
handoff. Initial API/CSV checks found a stale hover deadline and independent ROS
topic/service delivery races. Their failed traces remain under
`tmp/ros-api-20261011`; they are not counted as successful results.

The [completed CSV check](../tmp/ros-api-20261011/csv-repeat-final-verified/checks.json)
runs the user's actual M5 CSV with3 repeats,15-s pre-hover,15-s between-hover and
3-s post-hover on isolated domain144. It records2,160 received/scheduled stream
commands and5 received/scheduled API operations, with zero stream/API rejections,
8,199 native steps over81.99 s and a completed airborne landing reference before
disarming. Final physical feedback remains distinct from the landing reference;
these results do not establish a real-flight tracking acceptance threshold.
The [reconstruction](../tmp/ros-api-20261011/csv-repeat-final-verified/replay-checks.json)
replays the retained request boundaries: all8,198 retained drone/cable snapshots
and the final BDF2 checkpoint agree bitwise. This is transport/numerical evidence,
not parameter identification or calibrated controller agreement.

The [API checks](../tmp/ros-api-20261011/api-check-complete/checks.json)
exercise upstream `Crazyswarm()` discovery without a joystick, metadata,
arm/disarm, relative/absolute go-to, group filtering, upload/reverse/timescale,
stream release/landing and emergency/reset at0.5× simulation pacing. Emergency
leaves the native checkpoint unchanged while suspended and blocks resume until
reset. The [firmware planner comparison](../tmp/ros-api-20261011/firmware-planner-checks.json)
checks63 takeoff/land/go-to samples against pinned float32 firmware polynomial
evaluation; the maximum absolute component difference is1.04e-5 (mixed SI fields),
within the declared2e-4 numerical comparison tolerance. It does not establish
firmware motor-control equivalence.

The [GPU viewer check](../tmp/ros-api-20261011/ui-checks-final.json) runs ROS PVA
commands with the overridden1.1-m initial state, named `cf_2`, simulated cable,
force controls and pause/step/reset. During a0.22-s blocked Qt consumer the
native owner advances21 output steps, and normal window closure joins the owner.
The [capture](../tmp/ros-api-20261011/custom-simulation-flight-final.png) is a
simulation display artifact. The [domain checks](../tmp/ros-api-20261011/domain-checks.json)
reject a duplicate custom domain and an existing ROS server without stopping it.
The [regressions](../tmp/ros-api-20261011/validation-complete.json) pass153/153
checks with no skips. The [source check](../tmp/ros-api-20261011/protected-source-checks.json)
finds all121 protected native/simulation/controller/task/learning/config source
hashes unchanged from the retained pre-connection baseline. Ground/contact,
physical power cutoff, full firmware parameter emulation, simultaneous multiple
plants, real-flight cable observation, policy deployment and new fitting/flight
experiments remain outside this implementation. Training remains paused and no
model or policy is promoted.

2026-10-11 consistent YAML initialization: the selected robot's hardware YAML
position now initializes startup and Reset by default. The1.1-m simulation
example is used only with an explicit `--simulation-yaml`; metadata hashes only
the YAML files actually loaded. The launcher no longer treats the displayed
floor as a physical boundary. This removes a renderer-dependent restriction,
without changing native cable validation, contact assumptions or controller
behavior. `z=0` remains a free-space initial state, not a grounded/contact model.

The [initialization checks](../tmp/yaml-initialization-20261011/verified/checks.json)
run three4-s Docker-backed headless launches on isolated domains172–174:
the actual YAML at `(0,0,0)`, an explicit `(0,0,1.1)` override, and an independently
named `generic_robot` at `(2.3,-1.7,-0.4)` with explicit cable geometry and a
display plane at100 m. Each completes400 native output steps; native initial
positions, ROS position parameters and status agree with configuration. The
52/53/51 captured ROS pose samples respectively agree bitwise with retained
native states. Explicit cable nodes translate with the configured drone origin;
the attachment agrees within the declared1e-14-m floating-point reconstruction
tolerance. Owner-thread Reset restores the exact initial BDF2 checkpoint and
position, zeros time/step counters and increments generation in all three cases.
The hardware YAML bytes remain unchanged.

The [focused regressions](../tmp/yaml-initialization-20261011/verified/regressions.json)
pass14/14 without skips, including default selection, arbitrary robot identity
and explicit overrides. The [source hashes](../tmp/yaml-initialization-20261011/verified/protected-source-checks.json)
find all121 previously protected physical/controller/task/learning/configuration
files unchanged. The first checker used a different floating-point association
for its expected cable translation; its traces remain outside `verified` and
are not the reported completed check. The expected arithmetic was corrected
without relaxing native criteria. Full CSV playback, GPU viewing, physical
ground/contact, fitting, policy training and real flight were not rerun for this
initialization change.

2026-10-11 README refresh: [README](../README.md) now documents Linux setup,
standalone versus Crazyswarm-connected simulation, the Docker compose/start/shell
sequence, CSV execution, exact YAML startup/Reset, explicit overrides, and the
free-space/contact limits. Outdated Linux/connection status and launcher-preset
descriptions were replaced using the inspected entry points and this record.
Checked11 local README links/anchors, seven Bash examples with syntax-only
`bash -n`, balanced code fences and diff whitespace. This is documentation-only
work: no launch example, simulation, regression suite, fitting, training or
real-flight experiment was executed or rerun.

2026-10-11 shared policy contract: added the explicitly
[proposed state/action interface](#shared-policy-interface-proposed) and a README
pointer. Inspected the current actor encoding, velocity/PVA controller paths,
bridge/mailbox timing, installed Crazyswarm servers, FullState serialization and
firmware source. The [installed-source and no-radio packet audit](../tmp/policy-contract-20261011/source-and-codec-audit.json)
retains commits, file hashes and a mocked cflib send. Scalar packing confirms
millimetre/milliradian scaling, truncation and overflow rejection; quaternion
decode, flashed firmware, active onboard parameters and flight behavior remain
unverified. This initial draft recommended a separately versioned shared
velocity-intent to PVA generator with reference-state actor inputs and compatible
new training; that recommendation was superseded by the acceleration proposal
below. No profile, policy runner or cable observer was implemented. Existing
profiles, checkpoints, controller defaults and research definitions were not
changed. [Static checks](../tmp/policy-contract-20261011/static-checks.json)
verified 18 local README/contract links and anchors, seven Bash blocks with
syntax-only `bash -n`, balanced fences, absence of nonbreaking spaces and all 121
protected source hashes unchanged. Diff whitespace also passed. No simulation,
model fit, policy training, regression suite or real-flight experiment ran.

2026-10-11 acceleration-action proposal and Mellinger review: at the user's
request, replaced the unimplemented velocity-intent proposal with acceleration
outputs, distinguishing acceleration feedforward through integrated PVA from
direct acceleration-to-force control. The
[read-only source/configuration review](../tmp/policy-acceleration-20261011/mellinger-review.json)
retains force/attitude/output equations, exposed gains, source control rate and
requested YAML parameters. `FullState` retains absolute-XYZ feedback; zeroing
all translational P/D/I gains would remove those feedback terms in the inspected
source without necessarily changing firmware, but that configuration and its
mode transitions are not approved or implemented. The proposed PVA generator
uses exact held-acceleration reference updates; acceleration bounds, braking,
jerk handling and the deployment-route choice remain unselected. Existing
physics, controllers, presets, checkpoints, firmware and onboard parameters
were not changed. [Static checks](../tmp/policy-acceleration-20261011/static-checks.json)
verified 19 README/current-contract local links and anchors, seven Bash blocks
with syntax-only `bash -n`, balanced fences, no nonbreaking spaces and all 121
protected source hashes unchanged; diff whitespace passed. This was source and
documentation review only: no new packet probe, simulation, fitting, policy
training, regression suite or real flight ran. Flashed firmware and active
onboard parameters remain unverified.

2026-10-11 requested simulation relaunch: reopened the existing custom-simulation
flight viewer through the Crazyswarm `./docker/start.sh --custom-sim` entry point
for `cf_2` on simulation ROS domain142, with the current hardware YAML initial
position `(0,0,0)` and no position override. The new
[session](../output/custom_simulation/20261011-092036-b69e92d3/metadata.json) and
[launch check](../tmp/simulation-relaunch-20261011/checks.json) confirm a live
simulation process, connected ROS bridge and ongoing state feedback with no
reported simulation error. No training workers are running; training remains
paused. No settings, reward, model or policy changed, and no scripted flight,
regression suite or real-flight experiment was run for this relaunch.


2026-10-11 Linear outreach/height Stage1 relaunch and existing Stage2/3 continuation: Stage1 launched in [20261011-052639-788494](../runs/ppo_stage1/20261011-052639-788494); its subsequent status and review are recorded below.

2026-10-11 training relaunch verification: the user clarified that the relaunch
request meant training. The simulation opened by this task was
[closed](../tmp/simulation-relaunch-20261011/closed-for-training.json), preserving
its evidence. The [compatibility check](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/compatibility.json)
confirmed0 completed collections/episodes/updates in the paused source and exact
actor equality with its original collection121 initializer. Only the conditional
CUDA driver and scheduler-selection source differ from the paused execution
snapshot; exact-resume checks remain enforced. The already approved reward,
150-collection budget,128 lanes,0.55-m workspace and later-stage definitions
remain unchanged.

The [native preflight](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/preflight.json)
checked immutable actor initialization, fresh critic/Adam/RNG/clocks, same-event
reward accounting, old saved-return reconstruction and804 retained input/run
files. The [baseline comparison](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/native-baseline-comparison.json)
finds five native drone/cable state arrays identical to the retained pre-pause
baseline. The [initial live checkpoint](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/initial-live-check.json)
confirms exact actor transfer, fresh seed0 value weights, empty optimizers and
zero learner/exploration offsets with a19,200-episode entropy budget. At that
initial check a live worker was collecting the first128-lane cohort in the
Training UI. This launch
check does not establish learned improvement or full CUDA campaign equivalence;
no regression suite, fitting or real-flight experiment was rerun. The replacement
driver retains evaluation before each later-stage handoff and stops on user stop
or failure, without adding research thresholds or model promotion.

2026-10-11 training relaunch early accounting check:
the [saved-array review](../runs/ppo_stage1/20261011-052639-788494/interim-review.json)
checks the first2 completed collections/256 episodes and the collection1 native
evaluation. Cohort float32 return reconstruction error is0; maximum transition
sum error is5.96046e-8. The first native evaluation is valid, with raw return
0.0860904 under the unchanged approved linear Stage1 reward; native return
reconstruction error is0. The [live launch check](../tmp/stage1-linear-outreach-height-150-relaunch-20261011/launch-checks.json)
confirms the UI, campaign driver and sole training worker are running, with815
pinned files/current execution hashes unchanged. Stage2/3 have not launched.
These early checks confirm execution and accounting, not convergence or learned
motion improvement.



2026-10-11 Linear outreach/height Stage1 relaunch and existing Stage2/3 continuation: Stage1 completed to collection150 /19,200 total stage episodes. Its [retained review](../runs/ppo_stage1/20261011-052639-788494/review.json) checks 19200 newly collected episodes, saved native scoring/crossing accounting, finite checkpoints and input preservation. Current-run valid/contained/geometric/strict counts are19200/9061/2583/976. Cohort counts use the current target; Stage3 native evaluation uses final target geometry. One nominal case and adaptive cohorts do not establish robustness or real flight.


2026-10-11 Linear outreach/height Stage1 relaunch and existing Stage2/3 continuation: Stage1 native evaluation reviewed before Stage2 continuation: {'forward_reach_m': 0.606187296161614, 'tip_drone_height_error_m': -0.41458605080642297, 'forward_mps': 6.19299179610647}. Existing Stage2/3 reward definitions remain unchanged; no new handoff threshold or model promotion.


2026-10-11 Linear outreach/height Stage1 relaunch and existing Stage2/3 continuation: Execution needs attention: cannot unpack non-iterable Stage1Score object. No successful completion is claimed.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Stage2 launched in [20261011-063417-108510](../runs/ppo_stage2/20261011-063417-108510); its subsequent status and review are recorded below.

2026-10-11 completed Stage1 review and Stage2/3 continuation repair:
the [final Stage1 review](../runs/ppo_stage1/20261011-052639-788494/review.json)
checks150 collections/19,200 episodes,39 native evaluations and finite
checkpoints. All19,200 training episodes were valid;9,061 were contained,
2,583 had broad geometric crossings and976 met the strict crossing diagnostic.
These adaptive cohort totals are not independent robustness trials. Float32
cohort return reconstruction error is0; maximum transition-sum error is5.96046e-8.
At the same new-reward nominal start, final selected-event reach increased from
0.358478 to0.606187 m and tip-to-drone height error changed from-0.921940 to
-0.414586 m. Final forward tip speed is6.19299 m/s. The final native rollout
exceeded the0.55-m radius by0.0177724 m and its broad crossing entry angle was
31.2389 degrees, so it does not pass containment or the existing30-degree strict
diagnostic. No model was promoted and no research threshold changed.

The original campaign stopped before Stage2 because its handoff helper attempted
to unpack the single `Stage1Score` returned by `Stage1Scorer`. The
[continuation request/fix record](../tmp/stage23-continuation-150-20261011/continuation-request.json)
and separate driver correct only that return-value use and begin after the
completed Stage1. Original scripts, failure traces, logs and checkpoints remain
unchanged; Stage1 is not retrained. The
[Stage2 native handoff](../tmp/stage23-continuation-150-20261011/stage2-handoff.json)
retains valid execution and five drone/cable state arrays identical to the final
Stage1 replay, with exact actor transfer and fresh critic/Adam/RNG/clocks checks.
The [initial live check](../tmp/stage23-continuation-150-20261011/stage2-initial-live-check.json)
confirms a live Stage2 UI/worker, exact actor weights, a critic distinct from the
trained source critic, empty optimizers, reset counters/exploration offsets and
all1,452 pinned files/current execution hashes unchanged. Stage2/3 keep their
approved Gaussian-height rewards,150 collections each,128 lanes,0.55-m workspace
and unchanged handoff/curriculum criteria. This repair was verified through the
actual handoff and launch; no regression suite, fitting or real-flight experiment
was rerun.

2026-10-11 Stage2 early continuation check:
the [interim saved-array review](../runs/ppo_stage2/20261011-063417-108510/interim-review.json)
checks3 completed collections/384 episodes: all valid,133 contained,281 geometric
crossings and71 strict crossing diagnostics. Float32 return reconstruction error
is0, with maximum transition-sum error1.49012e-8. The first updated native actor
is valid and its saved return reconstruction is exact; this is early execution
evidence, not convergence or target success. The
[launch check](../tmp/stage23-continuation-150-20261011/stage2-launch-checks.json)
confirms the Stage2 UI, sole training worker and continuation driver remain live,
with1,452 pinned files/current execution hashes unchanged. Stage3 is pending the
completed Stage2 review and unchanged native actor-transfer checks.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Stage2 completed to collection150 /19,200 total stage episodes. Its [retained review](../runs/ppo_stage2/20261011-063417-108510/review.json) checks 19200 newly collected episodes, saved native scoring/crossing accounting, finite checkpoints and input preservation. Current-run valid/contained/geometric/strict counts are19200/5771/18741/9955. Cohort counts use the current target; Stage3 native evaluation uses final target geometry. One nominal case and adaptive cohorts do not establish robustness or real flight.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Stage3 launched in [20261011-073423-438409](../runs/ppo_stage3/20261011-073423-438409); its subsequent status and review are recorded below.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Stage3 completed to collection150 /19,200 total stage episodes. Its [retained review](../runs/ppo_stage3/20261011-073423-438409/review.json) checks 19200 newly collected episodes, saved native scoring/crossing accounting, finite checkpoints and input preservation. Current-run valid/contained/geometric/strict counts are19112/2533/5918/3626. Cohort counts use the current target; Stage3 native evaluation uses final target geometry. One nominal case and adaptive cohorts do not establish robustness or real flight.


2026-10-11 Approved Stage2/3 continuation after Stage1 handoff script repair: Existing Stage4 handoff assessment: {'final_target_level_reached': False, 'final_native_valid': True, 'final_native_contained': False, 'final_native_strict_hit': False}. This Stage1–3 restart did not train Stage4 or relax any criterion.

2026-10-13 completed-stage motion and curriculum review:
the [saved-array analysis](../tmp/stage-review-20261013/comparison.json),
[analysis script](../tmp/stage-review-20261013/analyze.py) and visually inspected
[comparison figure](../tmp/stage-review-20261013/motion-comparison.png) compare
three final native replays and Stage3's best-by-saved-reward replay. All450 retained
128-lane collection files independently reconstruct float32 returns exactly;
the maximum transition-sum error is1.19209e-7. Four native return reconstructions
are exact. All2,612 retained run files and current execution hashes remain
unchanged; no worker is active. This work rescored saved arrays only: no physics,
policy inference, training, fitting, regression suite or real-flight experiment
was rerun, and no model was promoted.

Final task-selected reach for Stages1/2/3 is0.606187/0.668553/0.638201 m;
tip-to-drone height error is-0.414586/-0.342179/-0.326884 m and selected forward
tip speed is6.19299/5.11391/4.80189 m/s. Their maximum drone sideways offsets
are0.430323/0.481688/0.545378 m, all occurring after the rewarded releases.
Full-motion peak forward tip speeds are6.36394/5.50993/6.09444 m/s, so selected
release speeds must not be described as full-motion peaks. Rescoring all three
final motions with the same approved linear Stage1 reward gives raw returns
0.403537/0.417631/0.386438; this task proxy does not define physical whip success.
Stages2/3 retain the older Gaussian-height base, so their original raw returns
cannot be compared directly with Stage1's new reward. Stage2 improved broad
delivery: all1,024 episodes in its last8 collections geometrically hit the broad
target, with64% strict diagnostics when rounded to whole percent (exact63.8672%).
Its final native broad crossing was strict-qualified, although radial excess
remained0.00361050 m. Stage3 improved aim but traded some reach/release speed and
containment; this is a mixed outcome rather than loss of all learned motion.

Stage3 advanced once, at collection4, from1.0×0.6 to0.6×0.4 m. Its next8
collections averaged1.07422% geometric hits, recovering to71.4844% in the last8,
without satisfying80% hits/all lanes valid for3 consecutive collections plus
the nominal hit. The longest post-shrink streak was2; collection141 had81.25%
hits but only127/128 valid lanes and reset the streak. None of39 saved native
evaluations hit the final0.1×0.1-m target. The final forward-plane crossing
missed its center by-0.284614 m laterally and-0.0569181 m vertically; both are
outside the approved±0.05-m per-axis tolerance. The best-by-reward collection132
also missed, with lateral error-0.294977 m. Only1.75781% of the last1,024 Stage3
episodes were contained, despite generally small weighted boundary costs.

Inspection confirms that event credit rewards a best release with broad
proximity/centering, and the travel discount stops at that release. The final
precision native center bonus was1.09634e-9 while positive base/proximity credit
remained; current-target center guidance uses the wider0.15-m sigma. These facts
support hypotheses of weak precision guidance, an abrupt first size transition
and insufficient incentive for settling after release, but do not isolate their
causal effects. Consistent outreach/height bases, stronger guidance at actual
crossings, intermediate target sizes and explicit settling would be proposed
research changes requiring approval; none was implemented in this review.
Stage4 was not trained because this campaign did not request it and its existing
handoff assessment failed. The completed Stage2 actor remains a comparison
reference, without promotion or replacement of registered models.

2026-10-13 approved custom real-flight display: added `--custom-real` and
viewer-only `--attach`, using the existing real ROS/OptiTrack stack and shared
GPU window. Drone poses use `/poses` capture in its declared `mocap` frame;
there is no onboard-estimate fallback, YAML-position stand-in or simulated cable.
The [focused checks](../tmp/custom-real-20261013/focused-tests.log) passed20/20.
Retained [synthetic ROS reconstruction](../tmp/custom-real-20261013/capture-checks.json)
and [source/frame rejection](../tmp/custom-real-20261013/source-frame-rejection-checks.json)
checks cover direct/transformed capture, multiple names, stale tracking, invalid
orientation, duplicate names, recovery and unavailable/wrong sources.
The actual NVIDIA GPU window, Qt Ctrl+C cleanup and mock owned-stack lifecycle
were checked separately. The [final check record](../tmp/custom-real-20261013/final-checks.json)
records syntax, whitespace, reader cleanup and unchanged hashes of121 protected
physics/controller/task/training/configuration files. Failed initial Jazzy
metadata attempts and display-test harness failures remain retained alongside
the successful checks; no scientific threshold was changed. No real flight,
OptiTrack hardware capture, fitting, policy training or physical-model experiment
was run for this change. Hardware capture alignment and flight behavior remain
unverified; force-controller and learned-policy deployment work is unchanged.


2026-10-13 approved audit fixes: reset real and simulated CSV hover deadlines;
record raw OptiTrack poses before TF orientation fallback; mark force-controller
gains dirty before partial writes and retain emergency cleanup on v1 restoration
failure; validate and commit simulator parameter batches atomically, terminating
the bridge on uncertain API acknowledgement. New logger recordings use schema
`optitrack_poses_pva_motor_v4` and `pose_*` source timestamp columns; the existing
v3 script filename and command/motor sidecars remain.
The [full regression record](../tmp/audit-fixes-20261013/regressions.json) passed
163/163 with CUDA available, and the adjacent
[flight-script checks](../tmp/audit-fixes-20261013/script-regressions.log) passed
3/3, covering real/simulation pacing, raw quaternion validity and injected
partial-write/restoration failures across all three force scripts.
An [isolated ROS parameter check](../tmp/audit-fixes-20261013/atomic-probe.log)
on domain221 verified rejected-batch preservation, successful batch/alias
agreement and conflicting-alias rejection. The
[synthetic ROS logger check](../tmp/audit-fixes-20261013/logger-ros-probe-verified.json)
on domain222 retained12 poses,12 commands and12 motor packets, including three
invalid raw orientations marked invalid. Its first attempt received11 poses
before DDS matching settled; that failed test-harness assertion is retained
separately. These are software, synthetic recording and connection checks.
No hardware flight, onboard gain-restoration verification, physical calibration,
new fitting or policy training was performed. Physical models, controller defaults,
rewards, checkpoints and source recordings were not changed by these fixes.
The final comparison with the prior audit also found unrelated changes in six
training/task files; these were left untouched. The
[final check record](../tmp/audit-fixes-20261013/final-checks.json) distinguishes
those changes from this task's three deployment-source edits. Regression results
refer to the recorded test execution, not an assertion that concurrently edited
training files stayed unchanged afterward.


2026-10-13 Approved Stage2 crossing-priority restart: implemented the explicitly
approved `0.2W + max_crossing(C+2G+S)` formula in a separate task/preset, with
causal endpoint delivery, complete-crossing selection, retained strict criteria
and fresh actor-only initialization. No physical plant/controller changed.
The [experiment approval](../tmp/stage2-crossing-restart-20261013/approval.json)
and [native transfer preflight](../tmp/stage2-crossing-restart-20261013/preflight.json)
retain provenance. Native tracked-drone position/velocity/rotation and cable
position/velocity arrays exactly match the preceding Stage2 final replay.
New raw return is3.256495237350464; independent NumPy crossing arithmetic and
transition-return sums have zero discrepancy. Critic/Adam/RNG/clocks remain
fresh after exact actor transfer; failure replacement/clawback was checked
on that rollout. All3,321 protected existing run/model files were unchanged.
Scoring57/57 (including7 new crossing tests and CPU/CUDA comparison), pipeline8/8,
handoff10/10 and records/UI14/14 tests passed; the new preset's UI summary and
experiment details initialized correctly offscreen. These checks establish
numerical/accounting behavior and one native nominal baseline, not trained
improvement, robustness or real-flight performance. The subsequent launch is recorded below.


2026-10-13 Stage2 crossing-priority experiment: Launched [20261013-144515-220029](../runs/ppo_stage2/20261013-144515-220029/) in the visible Training UI for150 new collections at128 lanes. The user stopped this run after52 collections/6,656 episodes; the subsequent stop record is below. No Stage3 continuation was launched.


2026-10-13 Stage2 crossing-priority launch verification: the active worker's
[snapshot/initialization check](../tmp/stage2-crossing-restart-20261013/launch-checks.json)
confirms the intended actor-only source,150-collection/128-lane budget, new
reward,0.55-m workspace and stage-local exploration. Its
[first-collection review](../runs/ppo_stage2/20261013-144515-220029/interim-review.json)
reconstructs all128 raw float32 returns exactly and verifies transition sums
within the retained1e-5 bound and additive components within1e-10. Counts were
128 valid,51 contained,128 broad geometric hits,88 strict broad hits and zero
final10×10-cm hits. One saved nominal evaluation also rescored exactly. This is
a startup/accounting check, not evidence of convergence or improved precision.
The [single-run driver status](../tmp/stage2-crossing-restart-20261013/status.json)
and run progress are the live status sources; Stage3 is not automatically
launched. Existing run/model artifacts remain preserved.


2026-10-13 User-requested Stage2 stop: the crossing-priority run
[20261013-144515-220029](../runs/ppo_stage2/20261013-144515-220029/)
stopped safely after52 completed collections/6,656 episodes using its existing
`stop.request` protocol. The worker exited and released its lock; `latest.pt`
was checked to match the stopped summary's round and episode count. Existing
checkpoints, logs and replays are retained. No Stage3 continuation was launched.
The [stop verification](../tmp/stage2-crossing-restart-20261013/user-stop.json)
records this action; stopping does not establish convergence or a new policy
performance result. No reward, physics or target definition changed.


2026-10-13 approved viewer shutdown and consecutive-flight fixes: moved GUI
shutdown out of Python signal handlers into the Qt loop; isolated the graphical
real viewer under a surviving host supervisor with ownership-checked cleanup;
completed VTK initialization before PyVistaQt import. Added per-robot display
trajectory epochs from newly discovered flight-client publishers in both ROS
adapters, plus a manual Clear trails button. Epoch changes clear displayed paths
without resetting physical state or sending flight commands. The real cable
feed remains unavailable; the custom simulation's existing cable rendering and
physical model are unchanged.
[Final regressions](../tmp/viewer-lifecycle-20261013/regressions-final.json)
passed172/172. [Eight GPU checks](../tmp/viewer-lifecycle-20261013/gpu-normal-exit.json)
passed with SIGINT delivered during VTK rendering, normal interpreter exit,
automatic/manual trail clearing and visible simulated cable. Earlier attempts
exposed native import/finalization faults and a test assertion made before the
60-Hz display refresh; those logs remain separate from the final results.
[Isolated ROS checks](../tmp/viewer-lifecycle-20261013/session-results.json)
on domains218/219 observed two consecutive client identities without restarting
either viewer source. [Mock stack ownership checks](../tmp/viewer-lifecycle-20261013/stack-lifecycle-checks.json)
verified window-close preservation, Ctrl+C cleanup and rejection of an unowned
process; [injected native worker crashes](../tmp/viewer-lifecycle-20261013/supervisor-crash.json)
verified the surviving owner's cleanup and attached-view isolation.
A [final read-only production attachment](../tmp/viewer-lifecycle-20261013/production-attach-final.json)
opened against the already running real stack and exited normally on Ctrl+C,
leaving the existing ROS processes running. No new real server, flight command,
physical calibration, fitting, training or real-flight experiment was performed
by these checks. No physical/controller parameters, research criteria or model
artifacts were changed by this work.

2026-10-13 approved Ctrl+C stack shutdown: an attached real viewer now captures
its matching local Crazyswarm/OptiTrack ROS launch and stops that tree on
Ctrl+C/SIGTERM, including supervisor fallback after a worker fails during the
stop request. This supersedes the previous attached-viewer Ctrl+C preservation
behavior above; closing only its window still leaves ROS running. The custom
simulation and standalone simulation now defer signal-driven window shutdown
through the shared Qt timer. Custom simulation isolates its Docker client and
retains session/domain/PID identities for ROS bridge cleanup; Docker remains up.
[Focused regression checks](../tmp/ctrl-c-stack-20261013/focused-tests.log) passed
41 tests, including stale-identity rejection, ambiguous-launch rejection,
preservation of another domain and an independent same-domain process, and
cleanup of remembered children after their launch parent exits.
[Docker/Qt integration checks](../tmp/ctrl-c-stack-20261013/integration-results.json)
closed the custom simulation GUI and its bridge, immediately restarted a
headless simulation on the same isolated domain224, stopped a synthetic attached
real ROS launch and its capture/server children on domain226, and closed the
standalone GPU viewer on Ctrl+C. All four launchers exited0. No test-domain
processes remained in the container. Syntax, CLI help and whitespace checks
passed. Real hardware shutdown and physical flight were not exercised; no radio
or flight-command services were used. These checks did not fit a model, train a
policy, change physical assumptions, or establish real-flight evidence.

2026-10-13 approved real-motion cable simulation and NatNet unicast support:
`--custom-real` now drives the existing cable-only native material solver from
accepted OptiTrack pose samples while preserving per-robot cable/BDF2 history.
The display labels the drone Measured and cable Simulated; its hanging, stationary,
untwisted initial condition is an explicit prior, with no ground contact or cable
measurement correction. The relay retains accepted samples between display
heartbeats and uses integer source timestamps. Policy observations/actions,
controllers, physical model files, fitting, training and research criteria were
not changed by this work.
[Focused checks](../tmp/real-cable-20261013/focused-tests-final.log) passed47 tests.
[GPU checks](../tmp/real-cable-20261013/gpu-results.json) and the
[synthetic measured-drone/simulated-cable capture](../tmp/real-cable-20261013/measured-drone-simulated-cable.png)
verified visible cable geometry, separate provenance, preserved history across
flight-display epochs, and deferred Ctrl+C shutdown. These are display checks.
[Live read-only Motive connection](../tmp/real-cable-20261013/live-refined-results.json)
on isolated domain228 accepted2805 `cf_2` samples, advanced6347 native substeps,
and ended without cable errors; no flight server or command was started.
This is capture/connection evidence, not physical cable accuracy or real flight.

Initial live attempts retained explicit failures. Accepted-prefix handling was
corrected, and a later retained jitter-driven implicit rejection motivated bounded
substep refinement. [Reconstruction of that exact rejection](../tmp/real-cable-20261013/refinement-reconstruction.json)
required one2× refinement and retained the existing native criteria; the final
live run also used one2× refinement. Maximum observed segment-length residual
was1.11e-16 m in both checks. This is a numerical constraint check, not a model-fit
or policy result. Exhausted refinement still reports failure rather than relaxing
criteria or replacing state.

The Docker patch retains the registered unicast endpoint, sends keepalives,
filters interleaved replies/frames, and preserves multicast mode selection.
The existing capture-node Ctrl+C context race was also corrected. The current
container and persistent `crazyswarm2:jazzy-pinned` image were rebuilt with the
same Crazyswarm/firmware pins; retained
[image build record](../tmp/real-cable-20261013/docker-image-results.json) and
[fresh-image source hashes](../tmp/real-cable-20261013/fresh-image-source-hashes.json)
identify them. A [fresh-image isolated unicast fixture](../tmp/real-cable-20261013/fresh-image-unicast.log)
published503 poses, retained its endpoint across10 keepalives, handled interleaved
command/frame packets, and preserved the explicit5-s missing-frame error. The
final real multicast capture node stopped with exit0 on SIGINT. Motive settings
were unchanged; that checkpoint did not include repaired live Motive unicast.
The later unicast connection check is retained below. Syntax, CLI-help and
whitespace checks passed; no test ROS processes remained at that checkpoint.

2026-10-13 real-motion cable freeze repair: the user's retained
[unicast display trace](../output/custom_real/20261013-210524-243e9f5e/connection.jsonl)
reproduced an implicit endpoint rejection at15.1855 s, after which the displayed
drone continued moving while the cable remained frozen. The old retry rule
could repeat the same single substep for very short receipt-time intervals.
Retries now multiply the actual baseline subdivision by2,4,8 and also cover
numerical position-projection and velocity-constraint rejections. Every retry
starts from the same accepted cable state/BDF2 history; native equations,
iterations, accuracy thresholds and physical configuration files are unchanged.
Invalid inputs and exhausted retries still report an error and freeze.

[Retained-capture reconstruction](../tmp/real-cable-motion-20261013/replay-results.json)
processed all9532 accepted pose samples through33.2697 s without cable errors.
The simulated attachment matched every displayed measured attachment exactly;
observed root motion ranges were0.506/0.931/0.351 m in XYZ, and the tip also moved.
There were13 refinement retries, with maximum8× subdivision. Maximum observed
segment-length residual was1.11e-16 m. These are reconstruction/numerical checks
under receipt timestamps, not physical cable accuracy, model-fit or policy results.
[GPU checks](../tmp/real-cable-motion-20261013/gpu-results.json) verified60
reconstructed frames: cable-line points, marker translations and attachment
labels followed their accepted states and the window closed normally.
The floor was hidden for the retained
[before](../tmp/real-cable-motion-20261013/moving-cable-before.png) and
[after](../tmp/real-cable-motion-20261013/moving-cable-after.png) display captures;
no ground-contact behavior was added.

A separate [live read-only Motive unicast connection](../tmp/real-cable-motion-20261013/live-results.json)
on isolated domain231 accepted2052 poses and advanced4634 native cable substeps
without cable errors; the capture node stopped with exit0 on SIGINT.
No flight server or flight commands were started, and Motive settings were
unchanged. The source/drone was near the floor during this connection check;
moving-drone behavior was verified using the retained unicast reconstruction.
The [final check record](../tmp/real-cable-motion-20261013/final-checks.json)
records49 focused tests plus syntax, CLI-help, whitespace and process-cleanup
checks. Physical cable accuracy and new live moving-drone flight remain unverified.

2026-10-13 approved experimental cable reconstruction: added optional
`--track-cable` using the user's confirmed model marker spacing, exact ROS
cloud/pose timestamp pairing, retained mixed raw candidate clouds, bounded
geometric/temporal association, explicit null/ambiguous nodes and separate
Estimated display geometry. This does not correct the existing simulated cable.
The current upstream cloud discards NatNet IDs/visibility flags; it is not an
unlabeled-only source. [Focused tests](../tmp/cable-tracking-20261013/focused-tests.log)
passed57 checks including permutation/duplicate/clutter handling, short occlusion
and reacquisition, ambiguous points, bending, initialization/gap/capacity rejection,
PointCloud2 layout decoding and generation/timestamp handling. The isolated
[synthetic ROS check](../tmp/cable-tracking-20261013/ros-results.json) retained644
clouds and reconstruction results, including632 full and12 partial frames, exact
pose/cloud timestamp pairing and test-node cleanup. No real capture node, radio
server or flight command was started. [GPU verification](../tmp/cable-tracking-20261013/gpu-results.json)
checked70 synthetic frames, separate raw/estimated/simulated geometry, missing-node
gaps, stale hiding, independent visibility and normal window closure; the
[missing-marker screenshot](../tmp/cable-tracking-20261013/tracking-gap.png) was
visually inspected. CLI help, syntax compilation and diff whitespace checks
passed. These are reconstruction/transport/rendering checks; real Motive cable
identity accuracy, occlusion robustness during whipping and physical accuracy
remain unverified. Native physics, controller, firmware, Motive settings and
training behavior were not modified by this change.

2026-10-13 approved hand-test recording and replay: added `--capture-only`,
Record/Stop/Replay/Open controls, separate clip folders, saved mixed-provenance
display states and received events, and offline `run_recording.py` playback.
[Focused checks](../tmp/record-mode-20261013/focused-tests.log) passed64 tests,
including exact saved-frame round trips, absent telemetry and missing markers,
source-clock regression on the monotonic clip timeline, multiple clips, shutdown
finalization, incomplete-prefix recovery, corruption rejection, disk-write error
reporting and capture-only command/supervisor isolation.
[Synthetic ROS recording](../tmp/record-mode-20261013/ros-results.json) retained422
display frames and1718 received/association events; all saved frames compared
exactly on offline replay, including7 partial-marker frames. Its test nodes were
stopped. A separate [network-isolated NatNet fixture](../tmp/record-mode-20261013/capture-only.log)
ran the actual capture-only launcher/receiver, received408 poses, observed no
Crazyflie flight server, and exited0 after SIGINT. The first attempt exposed an
empty-map ROS YAML parser rejection; the launcher now omits that empty mapping,
and the successful rerun is retained. No actual Motive source, radio or flight
commands were used for these tests.
[GPU verification](../tmp/record-mode-20261013/gpu-results.json) exercised60 live
synthetic frames, button-driven clip creation, saving a second clip on close,
and36 exact replay frames, with seeking, step, speed and play/pause controls.
The [recording](../tmp/record-mode-20261013/recording.png) and
[playback](../tmp/record-mode-20261013/replay.png) windows were visually inspected.
CLI help, offline inspection, syntax compilation and diff whitespace checks
passed. Real hand-movement tracking accuracy and long-duration/disk-latency behavior
remain unverified. This change adds recording/presentation and capture lifecycle;
it does not change native physics, fitting, rewards, policies or firmware.

2026-10-13 cable tracking recovery correction: partial occlusion no longer resets
all material-marker identities after0.25 s. Long-missing markers use bounded
geometric recovery, with ambiguous assignments left null; full loss still needs
the hanging prior. Crowding is handled per marker after the existing root-distance
filter. Receipt-time velocity updates use the noise/speed time scale described
above to damp bunched-frame spikes. Candidate-pair distances are reused across
beam hypotheses without changing chord bounds or assignment costs.
[Focused checks](../tmp/cable-reacquisition-20261013/focused-tests.log) passed68 tests,
including long partial occlusion while bent, ambiguous returns, impossible-clutter
filtering, closely bunched timestamps and full-loss hanging reinitialization.
[Offline reassociation](../tmp/cable-reacquisition-20261013/optimized/replay-results.json)
uses all9,228 reconstruction frames from the retained real capture
[20261013-225429-55c05031](../output/custom_real/20261013-225429-55c05031/connection.jsonl),
with its saved settings and same-frame attachment poses. The
[distance-cache comparison](../tmp/cable-reacquisition-20261013/final-checks.json)
verified identical results, including confidence, on all9,228 frames before and
after caching. Measured offline per-call median/p95 were5.77/10.89 ms for the
cached version; these do not establish sustained live100-Hz throughput. Originally
the tracker lost all assignments at36.526 s after one marker became stale, and never recovered.
The revised tracker recovers that marker and retains at least one assignment on
every subsequent frame; full10-marker frames increase from1,234 to3,036, with
partial results still present. These are assignment counts, not independently
verified cable identities: in particular the later crowded/folded configuration
still leaves most markers unavailable. The saved source is unchanged. No new
Motive capture, live-throughput test, physical cable accuracy validation or flight
was performed. Native physics, controllers, training and Motive settings remain
unchanged by this correction. Syntax compilation and diff whitespace checks passed.


### 2026-10-14 — Direct acceleration policy interface

The user approved changing policy outputs to world XYZ acceleration after
reviewing the paper's acceleration-to-thrust/attitude control structure.
Implemented `acceleration_direct_v1` in the shared tensor controller, collection
adapter, native evaluation, checkpoint/run schemas, compatible actor transfer,
replay metadata and Training UI. The new Stage1–3 launch presets retain their
previous default tasks/rewards, budgets and curriculum rules, and select50-Hz
acceleration actions bounded by(8,8,4) m/s². Existing physical equations, attitude
gains,200-Hz servo,30-ms delay, motor clipping/response and original presets are
preserved. The unimplemented PVA-reference-generator design above was replaced
with the selected direct path. No firmware parameters were changed.

[Regression evidence](../tmp/acceleration-interface-20261014/regression-checks.json)
records116 passing relevant tests, including9 direct-acceleration checks for
force/units, absence of translational feedback, action scaling/delay, native/tensor
agreement, Stage1→2→3 actor transfer, checkpoint rejection/roundtrip, and UI labels.
These are software/numerical checks, not trained-policy results.
[Nominal motion checks](../tmp/acceleration-interface-20261014/motion-checks.json)
retain four4.5-s native replays: zero command holds the initial equilibrium with
zero recorded position drift; ±1 m/s² X/Y and ±0.5 m/s² Z pulses remain numerically
valid. The horizontal cases end about0.0381 m from the start and0.0202 m lower,
illustrating that zero acceleration does not restore position. A128-lane CUDA
conditional-graph check over15 policy steps/0.3 s matches all native physical,
BDF2-history, motor and held-wrench state fields within4.11e-15 maximum absolute
difference (comparison tolerance1e-8). This is one nominal initial condition,
not hardware validation or robust tracking evidence.

[Final verification](../tmp/acceleration-interface-20261014/final-verification.json)
confirms3,583 existing run/model files are byte-identical and no training/evaluation
worker is active. No training campaign or model promotion occurred. Historical
velocity checkpoints retain their semantics and saved replays; exact resume and
fresh reevaluation still enforce the existing source-hash checks, so changed
controller sources are not silently accepted. Stage4 velocity recovery and old
velocity demonstrations cannot initialize an acceleration run. The curriculum
review, compatible recovery design, live policy runner, firmware matching and
real-flight validation remain outstanding.


### 2026-10-15 — Direct whip training from a random acceleration policy

The user requested skipping the control-learning pilot and clarified that
“zero shot” meant training the whip directly from scratch. Started the visible
Training UI with the unchanged
[acceleration Stage1 preset](../configs/learning/whip_stage1_ppo_acceleration_v1.json),
creating [this run](../runs/ppo_stage1/20261015-111358-998023/).
The immutable request has `resume=null` and `initialize_actor=null`; the run has
no source actor or demonstration. Seed0 initializes a new actor/critic and
optimizers. The existing recipe uses128 collection lanes,50-Hz acceleration
requests bounded by(8,8,4) m/s², radius0.55 m,4.5-s episodes, and an80-collection
(10,240-episode) cap. Existing reward terms, failure criteria, entropy schedule
and plateau-stopping rules were not changed. No automatic Stage2/3/4 handoff is
scheduled. This is policy training in nominal simulation, not real flight.

The [launch record](../tmp/stage1-acceleration-scratch-20261015/launch.json)
confirms no active training/evaluation worker before launch and exact source-hash
agreement with the verified acceleration implementation. Original run/model
files were inventoried before launch. The worker successfully initialized CUDA
and began collection1; no improvement or learned whipping is claimed from
startup. Subsequent initial verification is retained with this launch record.


The first launch completed collection1 but failed while embedding its evaluated
replay into the best checkpoint: the acceleration interface's scalar text labels
were passed to `torch.as_tensor`. Its logs, numerical replay and initial checkpoint
are retained in the first run; it is no longer active. Fixed only checkpoint
serialization: numeric arrays remain tensors, scalar text becomes plain strings,
and resume restores both representations to NPZ arrays. This preserves safe
`weights_only=True` loading and historical tensor-only replays. The existing
source-hash guards remain strict; the replacement run starts from scratch.
[Fix regression checks](../tmp/stage1-acceleration-scratch-20261015/checkpoint-fix-checks.json)
passed47 relevant tests, including checkpoint/replay metadata and inherited-replay
roundtrips. A [roundtrip of the actual failed evaluation replay](../tmp/stage1-acceleration-scratch-20261015/serialization-check.json)
preserved every numeric and text array exactly. No physics, reward, optimizer,
action limits, curriculum or budget was changed by this fix.

The corrected [scratch run](../runs/ppo_stage1/20261015-111659-848428/) was launched through the
Training UI with the same preset and seed; its
[restart record](../tmp/stage1-acceleration-scratch-20261015/restart-launch.json)
confirms no actor initialization/resume and matching frozen settings/sources.
Initial collection, evaluation and checkpoint verification are recorded separately
from completed-training results. No later-stage launch is scheduled.


[Initial live-run verification](../tmp/stage1-acceleration-scratch-20261015/initial-verification.json)
confirms the corrected run completed collection1 (128 episodes), logged its native
evaluation, saved loadable `latest.pt` and `best.pt` with intact acceleration
metadata, and entered collection2. All3,583 pre-existing run/model files and all12
files from the failed launch remain byte-identical. These startup checks establish
that learning/evaluation/checkpointing execute; they do not establish whip quality,
convergence or real-flight readiness. The run subsequently completed; its
performance review follows.

### 2026-10-15 — Scratch acceleration Stage1 failure review

Read all80 retained collection arrays and21 native evaluation replays from
[the completed run](../runs/ppo_stage1/20261015-111659-848428/), plus its frozen
configuration, summary and event log. The
[analysis record](../tmp/stage1-acceleration-review-20261015/analysis.json)
contains per-collection/evaluation measurements and SHA256 hashes of the input
artifacts. No simulation, evaluation or training was rerun; no reward, controller,
curriculum, model selection or physical parameter was changed by this review.

The run reached its80-collection budget with13,667 actor updates and22,825 value
updates. Only267 of10,240 episodes were valid. Collection8 first reached zero
valid episodes; collections14 and28 had isolated valid episodes, and all52
collections after28 had none. Validity means the task's numerical/envelope
criteria, not workspace containment or a successful whip. The best native
evaluation remained collection1, with raw return1.6508906810486224e-6. The other20
native evaluations ended below the0.05-m cable-height limit and reported no
numerical solver failure. The final replay lasted2.01 s: drone Z was0.994344 m,
minimum cable Z was0.042056 m and maximum tilt was30.45 degrees. This was an
envelope cutoff, not evidence of a simulated contact event.

In the first four collections, mean raw return was−15.4936 and mean workspace
cost14.9796; valid episodes' mean event credit was only0.03235. The last eight
collections had zero valid episodes, mean return−1.03348 and mean workspace
cost0.03348. [Episode scoring](../tasks/whip/sac_environment.py) gives failed
episodes `−1 − workspace_cost`, cancelling positive event credit.
[Workspace cost](../tasks/whip/workspace.py) is an uncapped squared-distance
penalty accumulated only over the observed prefix, divided by the full planned
horizon. Short failures can therefore score above longer exploratory episodes
that leave the soft workspace. All10,240 saved returns reconstruct from this
formula within6.54e-6; stored transition-reward sums agree within7.63e-6
(float32 outputs).

The descent, shrinking episode durations and lower workspace charges are
consistent with learning this cheaper failure behavior. This is a supported
diagnosis, not a causal ablation or proof that it is the sole learning problem.
The plateau detector continued because containment improved while validity and
whip quality stayed zero. The recommended next step is to correct the ordering
of failure and workspace costs before another direct-from-scratch whip run.
Any new formula/weights remain proposed and require approval; no pilot,
demonstration, restart or model promotion was performed.

2026-10-15 capture-only ROS domain default: at the user's request,
`--custom-real --capture-only` now selects43 when `--domain` is omitted. Explicit
domain overrides remain supported; normal real-flight and attach defaults remain42.
The domain is resolved before viewer supervision and independently by its worker.
Checked six CLI parsing/default/override cases with supervision mocked, Python
syntax compilation and diff whitespace. No ROS stack or hardware was launched
for these checks. README launch table and hand-capture instructions were updated.

2026-10-15 Training startup null-component fix: both reward-component plotting
paths now check for a non-null value before scaling. The saved Stage1 run
`runs/ppo_stage1/20261015-111659-848428/events.jsonl` has null plotted components
at round8; these remain gaps instead of raising `int * NoneType` during startup.
Verified offscreen Training window construction and chart processing with that
actual run, its round8 gap, and an in-memory crossing-chart case containing null,
missing and numeric-zero values. The source log hash remained unchanged and no
training worker was launched. All13 resume-history tests, Python syntax compilation
and diff whitespace checks passed. The offscreen platform does not support the
OpenGL preview; GPU preview rendering and training were not rerun. Reward
calculations, saved records, checkpoints and physical models were not changed.

2026-10-15 user-requested stock-Mellinger cf_2 acceleration circle: added the
standalone Crazyswarm script and usage instructions described above. All motion
phases use host-feedback acceleration through FullState with translation gains/
integral ranges zeroed and original values restored after confirmed disarm.
[Nine focused checks](../tmp/acceleration-circle-cf2-20261015/unit-tests.log) passed:
reference derivatives/continuity, saturation/anti-windup, normal lifecycle,
read-only preflight, partial parameter-write rollback, unknown arm outcome,
streaming failure, withheld restoration after unconfirmed disarm, and loop stalls.
The [network-isolated ROS fixture](../tmp/acceleration-circle-cf2-20261015/ros-results.json)
used actual ROS message/service types with synthetic OptiTrack/status, a fake
parameter server and an ideal acceleration actuator; it received1,665 FullState
commands, verified fixed P/zero V/zero body-rate fields, observed16 parameter
writes, and confirmed disarm and restoration. The read-only check sent no commands
or parameter writes. The disposable container had no external network or radio;
its synthetic plant is not the native cable/drone physics. CLI help in the existing
container, trajectory dry-run, syntax compilation and diff whitespace checks
passed. No real-flight execution, parameter changes on hardware, calibration,
firmware verification, native simulation validation or policy training occurred.

2026-10-15 default Docker client shell: at the user's request, plain
`./docker/shell.sh` now opens custom-simulation domain142 with
`AEROWHIP_SIMULATION=1`. Explicit `--real` and `--sim` select domain42 and disable
the custom client flag; `--custom-sim [domain]` remains supported. Existing shells
must be reopened; no running process or container-wide environment was changed.
Eight shell argument-routing/rejection checks passed with Docker mocked, along
with Bash syntax and diff whitespace checks. README real/stock examples now use
explicit shell modes. No ROS commands, simulations or real flights were executed
for this change.


2026-10-15 custom-simulation acceleration circle: added the explicit disarmed-only
ROS acceleration selector and adapted the adjacent cf_2 circle script to use
native feedback and simulation time. Default PVA behavior, the physical model,
training/reward settings and real firmware parameters were not changed. The
[22 native/connection checks](../tmp/acceleration-simulation-20261015/native-tests.log)
and [15 script checks](../tmp/acceleration-simulation-20261015/script-tests.log)
passed, including identical default-PVA native evolution, identical direct-mode
motion despite different P/V placeholders, transition-delay rejection, reset,
checkpoint incompatibility, clock/session failures and mode cleanup sequencing.
The [retained integration verification](../tmp/acceleration-simulation-20261015/verification.json)
records a complete fresh YAML-origin circle on isolated ROS domain225 with1,496
FullState commands, fixed P/zero V/zero body rates/fixed yaw, no command/API
rejections, and restored PVA. The unchanged
[waypoint script then completed](../tmp/acceleration-simulation-20261015/waypoint-after-circle.log)
in the same session. Native bridge/command records are retained in
[final-native-ros](../tmp/acceleration-simulation-20261015/final-native-ros/).
The script targets50 Hz but asynchronous ROS/native clock pacing yielded about
44 Hz average in this run; this is not a hard-real-time timing claim.
An earlier test exposed a final topic packet arriving after service disarm; the
final script drains scheduled stream receipts before disarming. An additional
circle after a waypoint maneuver failed the unchanged settling limits with
residual drone/cable oscillation and suspended the simulation; that failed run
is preserved in adjacent Crazyswarm `logs/acceleration-simulation-20261015-repeat`.
Stationary drone preflight does not certify settled cable state. No acceptance
limits were relaxed. Syntax and whitespace checks passed. These are nominal
numerical/transport/script checks, not model fitting, policy-training results,
firmware equivalence, contact validation or real-flight evidence. Only owned
isolated test sessions were closed; the user's existing simulator was untouched.


2026-10-16 GitHub source and data snapshot: preserved the two-project layout
and existing AeroWhip branch history. Source, configuration, trajectory CSVs and
packaged model artifacts are tracked in the two private repositories. At the
user's explicit request, recordings, training runs and retained test evidence
are also included as release archives under `backup-2026-10-16`; they remain
ignored by normal Git commits and require a separate download/restore. Archive
manifests identify individual files and SHA-256 hashes, and source stability and
full decompression checks verify the snapshot before upload. Successful, failed
and exploratory runs are retained without model promotion. Crazyswarm's `.env`
contains only public upstream commit pins and remains part of its source.
Virtual environments, build/cache files, publication tooling and credentials
are excluded. Both READMEs describe the checkout layout and data restoration.
This is packaging/documentation work; no simulations, fitting, training or
hardware experiments were rerun. Upload verification is retained locally under
`tmp/github-upload-20261016/` after publication.

2026-10-21 Approved offline reward review: added the standalone evaluator and
updated the implementation record with its explicit candidate, bounds, provenance,
classification uncertainty and unchanged curriculum status. Rescored198 retained
simulation traces and4 synthetic controls, without new physics execution or PPO
updates. Nine new diagnostic tests,57 existing scoring tests and10 existing
handoff tests passed (76 total); logs are under `tmp/reward-review-20261021/`.
The full suite was not rerun. No production reward, research success threshold,
curriculum gate, controller, model, default configuration or artifact was changed.
Whip-label validation, per-step credit assignment, new policy learning and
held-out/real-flight performance remain unverified.

2026-10-21 Approved separate Stage1 discovery implementation: added the explicit
50-Hz acceleration preset with0.5W credit, capped workspace cost≤0.25 and total
failure return−1; target hits remain diagnostic. Preserved defaults, physical
model, safety settings, PPO recipe and retained artifacts. Added source-transfer
guard pending motion review and a later-stage design. Updated the behavior,
configuration, accounting and evidence sections above. Ten new checks and94
existing regressions passed; three paired native probes preserve physical arrays
exactly, and198 pinned traces agree with the offline proxy/cost calculation.
Logs, scripts and results are retained in `tmp/stage1-discovery-20261021/`.
No training was launched, no model was promoted, and genuine-whip classification,
learnability, later-stage progression and held-out/real-flight success remain
unverified. The last-reviewed date remains2026-10-21.

2026-10-21 Stage1 launch authorized: started fresh discovery run
`runs/ppo_stage1/20261021-214537-277266` in the visible Training UI, with128 CUDA
lanes and the existing10,240-episode budget/plateau rule. Confirmed the frozen
discovery reward configuration, null resume/actor source, live worker and ready
CUDA physics event. Launch details are retained in
`tmp/stage1-discovery-20261021/training-launch.json`. No recipe, physics, threshold
or model selection was changed; startup is not evidence of successful learning.

2026-10-21 User-requested proxy recheck: inspected the implemented mechanics,
unfolding and direction terms; measured the first four retained collections and
their representative motions; ran prescribed traveling/standing-bend and rigid
translation controls. Recorded algebra/negative-control checks, input hashes,
limitations and a figure in `tmp/whip-proxy-review-20261021/`. Confirmed high
relative-unfolding scores for small absolute shape changes and sensitivity to
whole-cable forward translation. This read-only audit did not change training
or establish a new whip definition. Last-reviewed date remains2026-10-21.

2026-10-21 Second proxy recheck: added eight fixed native acceleration controls
and rescored three retained milestone strokes plus the earlier small-motion
example with the current reward. Recorded sources, command parameters, traces,
rankings and the comparison figure under `tmp/whip-proxy-recheck-20261021/`.
All native controls completed; their environment returns matched independent
rescoring within3e-8. Requalified the earlier interpretation: high relative U
and forward-translation sensitivity alone do not establish a faulty discovery
reward. Motion labeling and calibration remain unverified; no running-training,
physics, reward or model-promotion change was made. Last reviewed2026-10-21.

2026-10-21 Collection9 status review: inspected retained all-lane accounting,
optimizer events, round9 checkpoint and latest scheduled native evaluation8.
Confirmed clean `user_stop` after1,152 episodes, valid reward reconstruction,
finite actor weights and collection9 policy KL0.0090 below0.02. Recorded weak
containment and large nominal drift separately from physical validity. Evidence
is in `tmp/stage1-discovery-20261021/collection9-review.json`; no simulation,
training update, resume or configuration change was performed by this check.

2026-10-21 Exploration and curriculum review: inspected collections1–9 and ran
384 frozen-policy diagnostic episodes, including the analytically reconstructed
initial policy and round9 noise ablations. Recorded sampled tradeoffs, retained
arrays/replays and identity checks under `tmp/exploration-review-20261021/`.
Read Dream to Fly's primary paper to distinguish its body-rate curriculum from
our soft workspace charge and zero effort weight. No learner update, training
resume, reward/configuration change or new curriculum rule was made. Learning
benefits, robustness and genuine whipping remain unverified; no regression suite
was rerun for this diagnostic/documentation update. Last reviewed2026-10-21.

2026-10-21 Reward redesign restart: recorded the requested rethink as a proposed
task-first architecture, explicit motion/delivery/cost separation, staged penalty
introduction and offline validation scope. Earlier evaluator formulas remain
historical candidates, not the selected redesign. Read relevant implementation,
retained evidence and primary whip-control literature; no scoring implementation,
configuration, threshold, simulator or checkpoint changed and no tests or new
experiments were run for this design-only update. Last reviewed2026-10-21.

2026-10-22 Whipping-measurement literature review: examined the supplied spectral
cable-model PDF, primary whip/casting/rod sources, current simulator assumptions
and replay fields. Recorded a proposed event-based measurement contract and
model comparison. Computed and visually checked curvature/speed plots from three
retained trajectories with source hashes; no new physics simulation, training,
reward, stage rule or threshold was implemented. The new measurements are
descriptive, with classification and resolution validation still pending.

2026-10-22 Deeper whipping-measurement review: refined the proposed event family
using snapping, fly-casting and human-control literature. Distinguished material
propagation, world motion and task direction; documented timing, rigid-motion,
centroid, energy-accounting and terminal-marker limitations. These are research
qualifications, not a new physical definition or detector. Inspected the scorer,
mass configuration/native assembly and retained diagnostics. Documentation only;
no tests, simulations, training or changes to executable behavior were performed.

2026-10-22 Offline whipping-event implementation: added separate raw deformation,
speed/localization and directed-delivery measurements, conservative curvature
tracks, unreviewed candidate intervals, hash-pinned reports and annotations in
the existing replay viewer. Preserved the training scorer, reward, curriculum,
physics and recorded outcomes. Reprocessed eleven retained/synthetic cases;
75 selected unit tests passed, all eleven viewer loads and source pins checked,
and the historical plot/3D replay rendering were visually inspected. Artifacts
are retained under `tmp/whip-event-review-20261022/final-review/`. No new simulation
or training was run. Physical classification, independent labels, resolution
convergence and real-flight validation remain unverified.

2026-10-22 Event-output core: added explicitly bounded relative-tip Q in m/s,
raw baseline/signed gain, attributed mechanism judgements and release-contained
reviewed delivery links to the existing offline evaluator. Kept automatic
candidates unreviewed and v1 replay compatibility. Fourteen new core tests and29
existing measurement/replay tests passed; ten retained/synthetic cases, input
hashes and viewer loads were checked, and the annotated event-separation plot
was visually inspected. Host CUDA availability was verified. No new simulation,
training, automatic physical classifier or reward/curriculum was introduced;
Stage1's automatic qualification and scalar reward contract remain pending.

2026-10-22 Stage1 contract proposal: after the user requested exact rules before
implementation, recorded one provisional causal detector, directed speed-based
utility, failure outcome, budget and prelaunch checks above. At that point no
detector/reward code, simulation, exploration probe or training had been run for
the proposal; the implemented offline core and its43 passing tests were separate
evidence. Subsequent diagnostic implementation and the failed counterexample
are recorded in the current readiness section above.

2026-10-22 Stage1 exploration preflight: implemented the frozen candidate rules
as an offline diagnostic and ran384 fresh no-update native episodes across four
action processes and three paired seeds. Retained arrays, source hashes, scripts,
reports and plots under `tmp/stage1-exploration-20261022/`. Fifty-five selected
software tests and saved-array accounting checks passed. The separate required
wave-plus-independent-rigid-sweep acceptance check failed (score4.135850), so
the automatic training reward/preset was not integrated and no policy training
was launched. Correlated commands increased motion/event coverage and excursions;
halving independent noise increased full-horizon completion but armed no events.
No exploration winner, physical-whip validation or real-flight result is claimed.

2026-10-22 Attribution revision and milder exploration: added offline
attachment-anchored angular-motion subtraction and an explicit proposed
shape-output utility. The original rigid-sweep counterexample changes from4.135850
to0; additional rigid-motion covariance checks and a localized terminal-rotation
construction were checked without declaring a universal whip classifier. Ran384
additional no-update native episodes; the100-ms half-std condition produced20/96
positive-revised-score completed episodes versus3/96 for the repeated baseline,
with larger excursions. Sixty-two selected tests, new batch hashes and exact
baseline reconstruction checks passed; plots were visually inspected. Artifacts
are retained under `tmp/stage1-attribution-20261022/`. Original rules, presets,
physics and checkpoints were unchanged by that diagnostic step. Training
integration followed under explicit approval, as recorded in the next dated note.
Causal force/power validation remains unimplemented.


2026-10-22 Approved Stage1 integration and launch: added the separate
`directed_shape_release_v1` task/preset and correctly conditioned AR(1) PPO policy,
with explicit prior-action arguments, distribution-versioned checkpoints,
finalized event evidence and zero effort/workspace/time costs. Existing presets,
simulator, controllers, recordings and models were preserved. Focused CPU/host
checks, exact reconstruction of384 retained probe scores, and a32-episode native
optimizer preflight passed; artifacts are under `tmp/stage1-integration-20261022/`.
Launched fresh seed0 run `runs/ppo_stage1/20261022-015843-738817/` in the Training UI.
No later-stage promotion or real-flight/causal-whip validation is claimed.

First run check,2026-10-22: verified the round1 checkpoint,128-lane event records,
reward accounting, conditional KL0.006101537 and valid deterministic evaluation;
training continued into collection2. The native preflight is separate from this
fresh policy. No source code was changed after launch; only this record and
verification artifacts were updated.


2026-10-22 Overnight authorization and updated budget (original plan; run selection
and movement costs superseded by the approved restart recorded below):
The user explicitly authorized autonomous monitoring, completion and evaluation
of Stage1, evidence-driven Stage1 changes/reruns without further permission, and
implementation/training/evaluation of Stage2 broad-target delivery once Stage1
is supported by the evidence. The user subsequently requested120 collections per
stage, with further extensions if learning has not settled. This supersedes the
previous80-collection planned Stage1 budget and prior per-change approval gates
within this authorized Stage1 experimental scope. Preserve recordings/models,
scientific evaluation integrity and physical/controller separation. No Stage3
or real-hardware deployment is authorized by this overnight request.

The original zero-cost run had an immutable80-collection/10,240-episode snapshot.
[Continuation settings](../tmp/stage1-integration-20261022/stage1-120-collections.json)
were prepared and validated to add40 collections while preserving its original
entropy clock. That continuation was never launched: the user stopped the run
after13 collections, then approved the fresh movement-cost experiment below.
These old settings are retained evidence, not instructions to resume that run.
For any future exact continuation, preserve actor, critic, optimizers, RNG and
the saved entropy budget/clock; extending training must not restart exploration.
Do not edit pinned execution sources while a worker is active.
Fresh redesigned runs and new Stage2 should plan120 collections. Further
extensions may use20-collection increments, guided by frozen-policy evaluation,
mechanism evidence, failure rate and, in Stage2, event-associated actual hits.
Reward flattening alone is not convergence or scientific success. Record any
reward revision as a new experiment, preserve the previous evidence, and test it
before launch. No numeric promotion threshold is retrospectively weakened.

Recurring follow-up `finish-aerowhip-stages-1-and-2` is active in this chat every
five minutes. It must continue the work, evaluate completed runs, implement/test
Stage2 before launch, and notify only meaningful changes or actionable problems.
Use independent stochastic evaluation seeds and deterministic replay; manually
inspect representative motions, including failures and high-scoring cases.
Stage2 should retain the new shape-release core, add smooth event-local target
guidance, and separately verify actual forward target crossings associated with
the same release. The original zero-cost proposal was subsequently superseded
by the approved full-episode movement cost; retain zero additional workspace,
acceleration and time costs with unchanged physics/control. Pin exact provisional
rules and tests before training. The
old Stage2 preset is not the approved new task. Stop this follow-up after both
stages are completed and reviewed; if overnight progress is insufficient, report
that honestly rather than claim success or keep launching indefinite experiments.
This note records authorization/planning and configuration validation only;
it does not claim completion, convergence or new simulation results.


2026-10-22 06:12 UTC overnight follow-up: the Stage1 worker is active. Verified
all seven completed collections available at this check: saved transition-return
accounting,−10 failure returns and final conditional KL≤0.02. Execution source
hashes still match the run snapshot. Local-time event timestamps are four hours
behind UTC; they are not stale-log evidence. Early collections do not establish
convergence or readiness for Stage2. Retained audit:
[tmp/overnight-stage12-20261022/health-20261022T0612Z.json](../tmp/overnight-stage12-20261022/health-20261022T0612Z.json).

Under the user's delegated experimental authority, recorded prospective Stage1
promotion criteria before completed-policy evaluation in
[evaluation-plan.json](../tmp/overnight-stage12-20261022/evaluation-plan.json):
three independent policy-noise seeds,128 nominal-state episodes each; at least95%
valid and80% positive-and-valid episodes in EACH seed batch, plus a valid positive
deterministic rollout and representative motion/evidence review. These are
provisional engineering repeatability gates, not validated physical-whip labels.
Review the final and saved best-deterministic checkpoints with selection disclosed;
retain both reports and do not weaken criteria after seeing results. Inspect
median/high-scoring eligible motions, deterministic motion and available failures
for tracking/rigid-motion/numerical or unrelated-motion exploitation. Identical
nominal initial conditions cannot establish robustness to physical variation.

Prepared a standalone frozen-policy evaluator under the same temporary artifact
folder, outside hashed execution sources. It pins checkpoint bytes, source hashes,
independent seeds, all-lane histories, event evidence and a deterministic replay;
it never calls an optimizer. Only Python syntax compilation was performed on
this new script. Native evaluation is pending and should run after the active
training worker finishes, without competing GPU jobs. Stage2 remains unimplemented;
no handoff or new training run was launched in this follow-up.


2026-10-22 Human steering: excessive drone travel in current Stage1.
The user reports that the drone is going crazy and asks whether Stage1 should
penalize movement needed for excitation. Inspected current implementation and
retained collection8 motions, not the old discovery experiment. Current costs are
zero. Collection8's deterministic rollout travels13.813 m in4.5 s, reaches13.734 m
world altitude from2 m, and has zero armed events/shape-release reward. Selected
stochastic collection8 replay lanes38/121 travel8.373/9.639 m. All128 collection8
lanes leave the diagnostic workspace; mean per-lane maximum radial/vertical
violations are6.059/9.075 m. These facts establish excessive excursions under the
current objective; they do not establish divergence of the numerical solver or
convergence/failure of the final policy. A likely incentive is avoiding low-cable
failure through altitude while uncharged flight and noise produce occasional
eligible events, but that causal learning explanation remains a hypothesis.

Recommended revision: keep the event detector and shape-release utility fixed,
but subtract a modest full-episode drone path-length cost, lambda*D/L, with D
measured world tracking-point path and L rest cable length. This charges back-and-
forth motion and post-event drift; it is movement economy, not physical energy.
Do not penalize desired tip speed, impose a new hard workspace, or substitute a
reward/effort ratio. A retained32-episode preflight rescore at lambda0/.01/.025/
.05/.1 leaves7/6/6/5/4 positive-and-valid net returns; mean valid costs are0/.0925/
.2313/.4626/.9252. This is offline sensitivity, not evidence of the best learning
coefficient; retain both raw event score and travel cost as separate diagnostics.
Report: [movement-cost-tradeoff.json](../tmp/overnight-stage12-20261022/movement-cost-tradeoff.json).

No live reward/code/checkpoint was changed in this inspection. The overnight
follow-up now prioritizes this steering before blindly extending the zero-cost
objective. Existing delegated Stage1 authority covers a tested, versioned
revision if justified. Stop/checkpoint cooperatively before editing hashed
execution sources; preserve this run. If adopted, record exact coefficient and
accounting tests and revisit Stage2's earlier zero-cost assumption explicitly.


2026-10-22 User-requested stop: cooperatively stopped Stage1 run
`runs/ppo_stage1/20261022-015843-738817/` after13 completed collections /1,664
episodes. The interrupted collection14 (84 policy decisions) was discarded
from learning. The worker logged stopped; latest.pt was loaded and verified at
round13 with13 consumed PPO collections. Existing best.pt, replays, event records
and recordings are preserved. Paused the overnight follow-up
`finish-aerowhip-stages-1-and-2` so it will not automatically restart or advance
the campaign. No movement-penalty revision or Stage2 run was launched. Await
new user direction before restarting training or the overnight follow-up.


2026-10-22 Approved first movement-cost restart (completed; not promoted;
superseded by the stronger-cost fine-tuning contract below).
The user requested implementing the recommended travel penalty, restarting for
120 collections, then resuming the overnight Stage1→Stage2 plan. This supersedes
the paused zero-cost run's planned80→120 continuation. Retain that stopped run's
13-collection checkpoints and recordings for comparison; do not resume its
training or transfer its policy into this fresh experiment.

The new selectable [preset](../configs/learning/whip_stage1_ppo_shape_release_path_ar1_v2.json)
keeps the `directed_shape_release_v1` event detector, anchored shape decomposition,
0.25 m/s shape-gain condition and same-sample directed output unchanged. Add
`movement_penalty={mode:path_length_v1, weight_per_cable_length:0.05}`. For a valid
episode, return is `max_finalized S - 0.05*D/L`, with D the full100-Hz polyline
path length of the drone's WORLD tracking point and L the rest cable length
0.9525 m. This is0.0524934 reward units per metre, or0.524934 per10 m. It is a
movement-economy cost, not motor energy or cable-tip-speed punishment. This
provisional coefficient was selected under the user's delegated research
authority using the retained sensitivity check; no optimum is claimed.

[movement.py](../tasks/whip/movement.py) charges adjacent observations only in the
finite executed prefix; it includes settling, post-release drift and backtracking.
No time average, endpoint-distance substitution, clipping, free allowance or
new workspace boundary is introduced. The environment subtracts these cumulative
charges from the existing causal event potential at50 Hz. Failure still replaces
the entire return with−10; previously paid event credit AND movement charges are
reconciled in the terminal transition. Diagnostics separately retain raw
`shape_release_score`, `drone_path_length_m`, `movement_accrued_cost`, and applied
`movement_cost` (zero for replaced failed returns). Collection event JSON retains
`shape_only_episode_return` and the actual movement-penalized `episode_return`;
the shared event-measurement core remains unchanged. The UI shows raw event score
and subtracted travel cost, and derives its collection budget from settings.

This restart is fresh seed0,128 lanes,15,360 episodes/120 collections,4.5 s,
50-Hz acceleration with(8,8,4) m/s² limits, reward multiplier1, and unchanged
100-ms AR(1) exploration and PPO settings. The new entropy schedule uses the
new120-collection budget: hold through12 and reach the floor at96. No plateau
stop or automatic stage advancement is added to the trainer. Workspace,
acceleration and time costs remain zero; physical failure limits and simulator/
controller code are unchanged. A separate Stage2 implementation/review is still
required; it should explicitly retain this adopted movement cost unless a later
recorded experiment justifies changing it.

[Contract and evidence](../tmp/stage1-path-cost-20261022/contract.json):
74 CPU tests ran (71 passed,3 CUDA skips);66 focused host tests passed, including
CUDA checks. One initial straight-travel fixture unintentionally jumped from the
initial altitude to zero in its unfilled suffix; filling the intended constant
altitude corrected the fixture without relaxing tolerances. The production
measurement correctly charged that extra movement. A32-episode native preflight
with seed2026102203 independently reconstructed travel cost with NumPy, verified
net returns/failure reconciliation, and completed120 actor and120 value updates.
Eighteen episodes were valid;7 had positive raw events and5 positive net returns.
Final conditional KL was0.004547574 (<0.02); deterministic evaluation was valid
with net return−0.17996845. The pre-update drone and cable trajectories were
bit-identical to the previous same-seed preflight, documenting that the reward
change did not alter those physics/action histories; they are not independent
new learning evidence. Preflight artifacts are separate from training and are
not initialization checkpoints.

After that native preflight, only the worker's event-JSON reporting changed to
distinguish raw and net returns. The focused recording/accounting tests passed;
[final-source-check.json](../tmp/stage1-path-cost-20261022/final-source-check.json)
pins this delta. No further physics/PPO rerun is claimed for that reporting-only
change. Launch and first-collection verification are recorded below. This
cost remains an experimental tradeoff: positive raw events and small movement
must be evaluated together, and a flat reward or hover is not Stage1 success.

2026-10-22 Launch verified: fresh run
`runs/ppo_stage1/20261022-023244-251462/` started in the visible training UI at
06:32:42 UTC, with128 lanes and120 collections configured. Its request has no
resume or actor initialization; saved execution hashes match the checked sources.
The first collection completed, latest.pt was verified at round1/128 episodes
with finite agent state, and collection2 began. First-batch accounting verified
all128 returns, full-path cost normalization, failure replacement, dense reward
sums and event JSON's separate raw/net returns. Float32 reward-sum maximum error
was1.91e-6, within the existing relative/absolute tolerance. There were61 valid
episodes,20 positive raw scores and12 positive net returns. These are initial
collection diagnostics, not evidence of convergence or successful compact whipping.
The first deterministic evaluation was valid with zero raw event score and
net return−0.07726128 for1.471827 m travelled; it does not pass Stage1 promotion.
Retained checks: [launch-verification.json](../tmp/stage1-path-cost-20261022/launch-verification.json)
and [first-collection-verification.json](../tmp/stage1-path-cost-20261022/first-collection-verification.json).

The overnight follow-up `finish-aerowhip-stages-1-and-2` is ACTIVE every5 minutes,
tracking this new run. Its obsolete old-run continuation instructions were
replaced. The prospective frozen evaluator now separately reports raw/net event
returns and positive rates; that artifact-script update was syntax checked only,
with native frozen evaluation deferred until training completes. Existing
promotion criteria and independent seed plan remain unchanged. Stage2 remains
pending successful Stage1 evidence, with120 collections planned and this adopted
movement cost explicitly carried into its design. Keep the computer awake and
the Codex app running for the local overnight follow-up.

2026-10-22 06:45 UTC follow-up: eight collection artifacts passed raw-minus-cost,
failure-return, cost-normalization and saved reward-sum checks; completed-update
conditional KL stayed within0.02 and execution hashes still matched. Reconstructed
four retained collection7 motion scores on CPU using the frozen scorer and
separately computed drone travel with NumPy; their net returns matched the saved
collection accounting. This is reconstruction of existing simulated evidence,
not new simulation or independent physical validation. Travel remains substantial
and early positive-rate changes do not establish convergence. No reward, policy,
source or promotion-criteria changes were made. Retained report:
[health-20261022T064502Z.json](../tmp/stage1-path-cost-20261022/health-20261022T064502Z.json).

2026-10-22 Overnight design preparation: retained a
[Stage2 design draft](../tmp/stage1-path-cost-20261022/stage2-design-draft.json)
outside execution sources while Stage1 trains. It proposes rectangle guidance
at the same shape-maximum sample as each event's directed utility, with actual
forward crossings associated only inside that event's release interval. It
retains full-episode movement cost and failure replacement. Exact promotion
criteria, transfer entropy clock, counterexamples, software and native checks
remain pending; this is a proposal, not an implemented task or Stage1 promotion.
Removed superseded80-collection continuation instructions from the active plan;
the original configuration and stopped run remain preserved.

2026-10-22 06:56 UTC retained-motion review: reconstructed and visually inspected
static cable XZ/XY/YZ sequences, material turning-angle tracks and release velocity
plots for deterministic rounds12 and16, without running new physics or GPU
evaluation. Their selected events show changing shape and outward-tracked
deformation with forward shape-speed peaks; this is consistent with the frozen
kinematic reward and does not establish force causality or physical-whip status.
Round12 raw/net return is3.0246/2.7306 with5.6002 m travel; round16 is4.3764/3.9093
with8.8986 m travel. Round16 reaches z5.378 m and y4.381 m, so compact motion has
not been established. Continue the planned experiment; no promotion or reward
change was made. Source-pinned report and viewed plots:
[motion-review-rounds12-16.json](../tmp/stage1-path-cost-20261022/motion-review-rounds12-16.json).
These selected early examples do not replace final independent trials or
representative final-policy motion/failure review.

2026-10-22 Stage2 preparation, retained spatial-coverage probe: examined120 valid
saved motion replays containing191 positive shape-release events. These replays
were selected by training score/hit logging and are not all-lane or independent
policy trials. At the event's own q sample, median rectangle distance was1.103 m;
50/191 events received proposed Gaussian guidance above0.1 at sigma0.35 m.
Eight events had an actual broad-target qualified crossing temporally inside
their own release window under the draft diagnostic. This establishes available
examples for later counterexamples/implementation checks, not a hit-rate estimate
or Stage2 readiness. Wider guidance kernels were described for sensitivity only;
no reward setting was changed. Retained source hashes and per-event evidence:
[stage2-retained-coverage-initial.json](../tmp/stage1-path-cost-20261022/stage2-retained-coverage-initial.json).
The probe used CPU arrays only and did not compete with the training GPU.

2026-10-22 Frozen-evaluator preparation check: its raw/net event-reporting helper
exactly reconstructed all128 retained collection1 records, including67 failed
episodes, without mutating the original core records. This was a CPU helper
check only; full native frozen-policy execution remains pending completion.
Retained script hash and scope:
[evaluator-reporting-check.json](../tmp/stage1-path-cost-20261022/evaluator-reporting-check.json).
Separately measured saved deterministic edge-length errors at rounds12,16,24,28,
32 (maximum absolute errors below1.6e-15 m). These confirm those stored edges
retain rest lengths; they do not establish solver convergence, velocity/energy
consistency or discretization accuracy. No simulation, optimizer or source
changes were performed for these checks.

2026-10-22 Movement-economy timing diagnostic: in saved deterministic rounds24,
32,40,48,82–86% of full-episode travel occurs after the selected event finalizes.
Round48 has raw score5.5192, net4.8636 and12.4891 m travel, including10.3727 m
after finalization at1.27 s. These paths are charged correctly by the current
reward; compact behavior is not established. This suggests reviewing late drift
and movement-cost strength at completion, rather than accepting a positive whip
score alone as sufficient. It does not prove every later motion unnecessary,
because the policy can still seek stronger events later in the episode. No
coefficient, threshold or training change was made mid-run. Retained source-
pinned saved-trajectory calculation:
[movement-timing-rounds12-48.json](../tmp/stage1-path-cost-20261022/movement-timing-rounds12-48.json).

2026-10-22 Completion-review preparation: added a standalone
[frozen-report assessment](../tmp/overnight-stage12-20261022/review_frozen_report.py)
outside execution sources. It applies the existing prospective per-seed numeric
gates with every failure retained in the denominator, checks exact planned seeds,
and lists representative motions/failures for later inspection. It always leaves
motion review and promotion pending. Six synthetic boundary, missing-lane and
no-promotion checks passed; this is reporting validation, not native policy
evaluation. Training and the scientific criteria were unchanged.

2026-10-22 Stage1 movement-cost run completed its120 collections/15,360 episodes
at07:55:13 UTC. All120 retained collection return/cost/failure/dense-reward checks,
completed optimizer KL bounds and saved entropy schedule passed; execution
hashes still match. See
[completion-accounting.json](../tmp/stage1-path-cost-20261022/completion-accounting.json).
Final deterministic raw/net return is5.5731/4.4814 with20.7972 m drone travel;
the saved best is round84, raw/net5.9264/4.8979 with19.5915 m travel. Static event
sequences show shape evolution and a forward release, followed by excessive
ascent (approximately16 m altitude) and travel. The0.05 movement cost is applied
correctly but controlled motion remains unsatisfactory. No Stage2 promotion is
made. Independent frozen final/best evaluations and representative failure review
were completed without an optimizer or concurrent training worker, as recorded
in the completed frozen review below. Viewed
source-pinned sequences:
[motion-review-rounds84-120.json](../tmp/stage1-path-cost-20261022/motion-review-rounds84-120.json).

2026-10-22 Completed frozen review: final and saved-best84 both pass the unchanged
prospective numeric gates on three independent128-lane action-noise seeds.
Final has377/384 valid and365/384 positive-net-valid episodes; best84 has376/384
valid and354/384 positive-net-valid episodes, with failures retained. Median
valid travel is16.367/13.769 m and median maximum altitude10.052/8.703 m for
final/best respectively. Visually inspected median/high-return/failed motion
sequences from the first seed, plus deterministic replays. Their shape evolution
supports the experimental kinematic score, but extensive later excursions fail
the intended movement-economy review. Sampled failure traces exceed60° tilt;
they retain total return−10. This is nominal-state repeatability, not physical
randomization, real-flight evidence or a physical-whip proof. No Stage2 promotion
or unchanged-objective extension is made. Full evidence and decision:
[stage1-review.json](../tmp/stage1-path-cost-20261022/stage1-review.json).

2026-10-22 Stage1 v3 revision contract, under delegated overnight authority (completed and reviewed below):
use the separate
[v3 preset](../configs/learning/whip_stage1_ppo_shape_release_path_ar1_v3.json)
with valid return `max_finalized S - 0.25*D/L`, increasing movement cost fivefold
while preserving the shape-release rules, physics/control,4.5-s horizon,50-Hz
acceleration, limits and failure outcome−10. This costs2.62467 per10 m of drone
travel. No additional workspace/acceleration/time cost or hard boundary is added.
Fine-tune the complete saved-best84 actor, including learned state-dependent
scale parameters, with a fresh critic and fresh optimizers for120 NEW collections
/15,360 episodes at128 lanes and fresh RNG seed1. This is actor initialization,
not exact resume. Select best84 for its lower independent-trial travel than final
and slightly better fixed-trajectory0.25-rescored means; no training-method
comparison or optimal coefficient is claimed. The source checkpoint is preserved.

Inherit the source's15,360-episode entropy budget and10,752-episode offset:
initial coefficient0.003714286, reaching0.001 after12 new collections and staying
there. Do not restart noise or reinitialize the scale head. The independent
completion evaluation uses new seeds2026102251–53, with the same numeric criteria
and required motion/failure review. The original evaluation plan/reports remain
preserved. Record:
[v3 contract](../tmp/stage1-path-cost-v3-20261022/contract.json).

Transfer preflight qualification: complete actor copying, unchanged fresh critic,
fresh optimizer states and inherited clocks were verified. A strict re-forward
log-density check at2e-5 FAILED (maximum8.65e-4) on the trained actor; its original
script/tolerance and failure log are preserved. Stored-density algebra reproduces
exactly; captured observation prefixes are bit-identical at eight decision times,
actor weights remain unchanged, and immediate mean re-forward differences are0.
Later re-forward conditional Gaussian KL averaged2.37e-9 (maximum1.70e-8), far
below the0.02 optimizer guard. This evidence is consistent with small neural
re-forward numerical drift; it is not a passed strict test. Separate native
reward/accounting and optimizer integration checks completed: all32 episodes
were valid,26 had positive net returns, independent NumPy travel accounting and
stored-density algebra matched, and17 actor/145 value updates finished with
conditional KL0.01994382. Post-update deterministic net return was−0.7453173;
this preflight is not a training initializer or evidence of improvement. The
numerical qualification remains recorded. No execution-source or scientific-criterion changes were
made to bypass the failed check. Diagnostic files are retained under the v3
contract directory; launch status is recorded separately after integration.

2026-10-22 v3 launch: started actor-only fine-tuning in the visible Training UI
at08:19:18 UTC, run `runs/ppo_stage1/20261022-041920-426901/`, with120 new
collections configured. Verified the immutable source-copy hash, run-local
initializer request, zero-counter initialization event, inherited entropy offset/
budget, execution hashes and first saved collection reward/cost accounting.
The round0 checkpoint had already been replaced before inspection; exact initial
actor preservation and fresh critic/optimizer checks come from the native
transfer preflight, not an unobserved launch checkpoint. Monitoring now tracks
this v3 run and its new prospective holdout plan. Older runs, independent reports,
strict-check failure and numerical qualification are preserved. Stage2 remains
pending a satisfactory Stage1 review. Launch evidence:
[launch-verification.json](../tmp/stage1-path-cost-v3-20261022/launch-verification.json).

2026-10-22 v3 early motion inspection: CPU reconstructions and viewed static
sequences for deterministic rounds20/28 retain changing shape, outward tracks
and forward release. Their raw/net returns are4.9805/3.3648 and5.8080/3.1512,
with6.1560/10.1222 m travel respectively. Ascent remains (above4.5/7 m), so these
selected examples do not establish final controlled motion or promotion.
Corrected a copied diagnostic plot's net-return label to read0.25 from this run's
snapshot; corrected values match saved native evaluations. Training accounting
was already correct. Retained report:
[motion-review-rounds20-28.json](../tmp/stage1-path-cost-v3-20261022/motion-review-rounds20-28.json).
No new simulation, optimizer or execution-source changes were made for this
reconstruction review; independent completion trials remain pending.

2026-10-22 Stage2 preparation while v3 trains: an unregistered standalone NumPy
prototype passed18 synthetic same-q guidance, maximum-event selection,
forward-crossing direction/timing, observation-prefix and failure/cost
counterexamples. It cannot affect training or the simulator; the supplied shape
scores are not physically validated by these geometry tests. It uses the draft
fixed+X rectangle only. Final Stage2 contract, registration, native integration
and actor transfer remain pending satisfactory Stage1 evidence. Retained check
names and prototype hash:
[stage2-prototype-checks.json](../tmp/stage1-path-cost-v3-20261022/stage2-prototype-checks.json).
No pinned execution source, scientific criterion or active learning setting was
changed in this preparation.

2026-10-22 Stage2 prospective evaluation draft recorded before training:
three new independent128-lane seeds, per-seed at least95% valid,80% positive-net-
valid and80% actual broad geometric hits associated with the SAME reward-selected
positive-shape release; valid positive-net deterministic associated hit and
motion/failure review also required. All failures remain in denominators.
Existing3 m/s/30° strict crossings are separately reported diagnostics for this
broad geometric-delivery stage, not an implied strict-strike qualification.
These provisional engineering gates do not authorize Stage1 advancement, Stage2
launch, precision-stage success or physical validation. Retained draft:
[stage2-evaluation-plan-draft.json](../tmp/stage1-path-cost-v3-20261022/stage2-evaluation-plan-draft.json).

2026-10-22 Stage2 guidance-width draft refinement, before training: a CPU probe
of50 selected valid v3 motion replays from collection60 onward found123 positive
events with median own-q rectangle distance0.979 m. Guidance above0.1 occurs for
10/123 events at sigma0.35 m and113/123 at0.70 m. The draft now proposes0.70 m
smooth approach width to avoid nearly absent target signal; the1×0.6 m physical
target and actual-crossing evaluation gates stay unchanged. Original0.35 m draft
is preserved. These selected training replays do not estimate independent hit
rates, prove learning under either kernel, or authorize promotion. Source-pinned
probe: [stage2-retained-coverage-post60.json](../tmp/stage1-path-cost-v3-20261022/stage2-retained-coverage-post60.json).
No active reward, source or physical parameter changed.

2026-10-22 Source preservation before any later Stage2 edits: archived all current
hash-pinned execution-source bytes plus the v3 run snapshot, scene and learning
preset. Every archived execution file was rehashed against the active run, with
source hashes checked before and after the read. This does not archive external
dependencies or recorded flights and does not change active execution. Retained
manifest and archive hash:
[execution-source-archive.json](../tmp/stage1-path-cost-v3-20261022/execution-source-archive.json).

2026-10-22 09:34 UTC Stage1 v3 completion review: all120 collections/15,360 new
episodes completed with raw/net/movement/failure accounting, saved entropy clock,
source hashes and completed optimizer KL checked. Independent final120 and best112
checkpoints each used three new128-lane seeds2026102251–53. Final:382/384 valid,
357/384 positive-net-valid; best:384/384 valid,375/384 positive-net-valid. Both
passed unchanged per-seed numeric gates. Viewed deterministic and representative
median/high motions, plus final failed motion; no prolonged ascent like v2.
Valid median drone travel was4.234m final and3.924m best; best deterministic travel
2.652669m. Two final failures contain cable height below
0.05m and retain total−10. Stage1 supports advancement for nominal-state kinematic
repeatability only, not physical-whip/causal-force/robustness validation. Select
saved-best112 for Stage2; no Stage1 extension. Full frozen evidence and selection:
[stage1-review.json](../tmp/stage1-path-cost-v3-20261022/stage1-review.json).
The original strict density-check failure remains qualified as documented above.

Stage2 exact prelaunch research contract is now recorded under delegated overnight
authority: separate `directed_shape_delivery_v1`, unchanged S_e at own peak-shape q,
U_e=S_e*(1+exp(-0.5*(distance_to_rectangle(q)/0.70m)^2)); max finalized U_e minus
0.25*full-episode drone path/L, failure total−10. Actual interpolated forward
rectangle crossings in the SAME selected positive-S release [arm,end] are checked
separately, with all-eligible/global crossings retained. Same physics/control and
zero additional workspace/acceleration/time cost. Complete actor from reviewed
best112, fresh critic/optimizers/seed2, inherited entropy budget15360/offset25088
keeps coefficient0.001;120 new collections. Implementation/native checks and launch
remain pending. [Contract](../tmp/stage2-shape-delivery-20261022/contract.json) and
[prospective evaluation gates](../tmp/stage2-shape-delivery-20261022/evaluation-plan.json)
were recorded before Stage2 training; original drafts remain preserved.

2026-10-22 Stage2 implementation/preflight: registered the separate shape-delivery
scorer, fixed-target Stage2 validation, causal potential, event evidence, UI
description and narrow Stage1-shape→Stage2-shape-delivery actor handoff. Bounded
discovery and shape→legacy Stage2 remain blocked. Frozen Stage1 core implementation
is unchanged; the new scorer reuses its output, records base/selected shape scores,
guidance/delivery utility, global/all-eligible/selected-release crossings separately.
Movement and failed-return accounting remain full-horizon and−10 respectively.
Seven new event/configuration/transfer/accounting tests passed; regression runs:
shape-release8, movement17 (includes imported shape tests), AR1 six with one skip,
handoff10. Twelve synthetic Stage2 report-gate counterexamples passed. Native
32-lane preflight preserved complete actor/fresh critic/optimizers/entropy clock;
fresh core reconstruction plus separate NumPy geometry prototype agreed for all32
lanes and16 associated forward crossings. All32 valid,30 positive net, zero
selected-event broad hits: integration evidence, not target-delivery success.
Two actor/145 value updates completed at final conditional KL0.01986285.
Stored-density algebra matched exactly; the earlier Stage1 strict re-forward
failure remains preserved and is not claimed passed. No physics/control changes.
The preflight agent is not the training initializer. Evidence:
[preflight-review.json](../tmp/stage2-shape-delivery-20261022/preflight-review.json).
At this preflight milestone, Stage2 launch and final-policy evaluation were pending;
launch and completed-policy outcomes are recorded below.

2026-10-22 09:42 UTC Stage2 launched in visible Training UI: run
`runs/ppo_stage2/20261022-054241-250242`, preset
[whip_stage2_ppo_shape_delivery_path_ar1_v1.json](../configs/learning/whip_stage2_ppo_shape_delivery_path_ar1_v1.json),
120 new collections at128 lanes, seed2. Observed round0 checkpoint: complete actor
bit-identical to reviewed best112, optimizer states empty, entropy offset25088/
budget15360 and execution hashes matched. Run-local initializer copy hash matches
the source checkpoint. Initial saved collections verified U=S_selected*(1+P),
net=U−movement cost (failed entire−10), dense reward sums, raw/selected/delivery
separation and event logs. These early checks do not establish target success.
The heartbeat now monitors this Stage2 run and its prospective holdout seeds
2026102261–63, final/best evaluation and selected-release crossing/motion gates;
it must not resume older Stage1 or edit current pinned sources. Stage2 was active at this launch milestone. Its completed120-collection review
below finds the delivery goal unmet; this run is not a promoted Stage2 policy.
Retained [launch verification](../tmp/stage2-shape-delivery-20261022/launch-verification.json)
and [initial health](../tmp/stage2-shape-delivery-20261022/initial-health.json).

2026-10-22 Stage2 review preparation during training: added the retained CPU-only
[review_motions.py](../tmp/stage2-shape-delivery-20261022/review_motions.py), using
Stage2 delivery-selected event records and independently reconstructing the frozen
core plus same-q delivery selection. It retains all-lane diagnostics and plots
selected shape/output, actual associated crossings, target rectangle and full
observed drone motion. One saved collection1 lane39 trace matched reconstruction
and its plot was viewed: forward release misses the target, correctly zero
associated broad hits despite positive guidance. This is review-tool verification
on early training evidence, not independent evaluation or Stage2 success. No
active execution source, reward, physics or scientific gate changed. Retained
[motion-review-smoke.json](../tmp/stage2-shape-delivery-20261022/motion-review-smoke.json).

2026-10-22 Stage2 early CPU motion check through collection20: saved deterministic
utilities and drone paths at rounds1/4/8/12/16/20 matched reconstruction. Viewed
round20 shows outward shape evolution and forward release, but a below/negative-Y
rectangle miss. Own-q rectangle distance changed0.880m→0.474m from round1→20,
without a selected-release deterministic hit; latest values are nonmonotonic.
These evolving-actor training evaluations are neither independent trials nor
convergence evidence. Continue the120-collection contract unchanged; no reward,
source, noise or criterion edits. Retained
[early-deterministic-through20.json](../tmp/stage2-shape-delivery-20261022/early-deterministic-through20.json).

2026-10-22 Stage2 event-association spot check while training: reconstructed and
viewed saved collection32 lane24. Selected first release misses at1.0516s; a
different positive-S release produces a strict rectangle hit at3.0929s. Global/
all-eligible/selected hit flags match retained native accounting (true/true/false).
Current U correctly ranks the stronger first miss10.2970 above later hit4.8923;
retain this concrete ranking case for completed-policy review, without declaring
early learning failure or changing the reward/criteria.
[association-review-collection32-lane24.json](../tmp/stage2-shape-delivery-20261022/association-review-collection32-lane24.json).
No competing GPU evaluation or active-source edit was performed.

2026-10-22 Stage2 temporal-geometry diagnostic during active training: reconstructed
five saved deterministic traces48/56/64/72/76. In these examples q equals arm;
the first forward plane passage precedes arm by approximately4–17ms and still
misses laterally and/or vertically. There is no forward crossing inside the
selected release window. This is a concrete temporal-alignment question for
completion review alongside spatial error, not an accounting failure or proof
of impossibility. No association window, success threshold, active reward,
source or training budget was altered. These are selected training traces, not
independent completion evidence.
[temporal-geometry-through76.json](../tmp/stage2-shape-delivery-20261022/temporal-geometry-through76.json).

2026-10-22 Stage2 rare-hit verification during training: collection86 has1/128
selected-release broad hits. Its retained lane122 motion was reconstructed and
viewed; direct interpolation confirms a hit at1.044969s inside[1.04,1.24]s,
forward6.94586m/s, entry5.12279°. It also passes unchanged strict diagnostics.
The crossing is near the lower/negative-Y corner, with0.854mm lateral and0.380mm
vertical clearance. This establishes one attainable association under the current
rules, not robustness, repeatability or Stage2 completion. Full128-trial denominator
and independent prospective criteria remain unchanged.
[selected-hit-collection86-lane122.json](../tmp/stage2-shape-delivery-20261022/selected-hit-collection86-lane122.json).

2026-10-22 10:56 UTC overnight Stage2 completion and decision: run
`runs/ppo_stage2/20261022-054241-250242` completed120 collections/15,360 episodes.
All120 collection reward/cost/failure/dense-sum accounts, saved entropy clock,
completed optimizer KL and source hashes checked. Sequential frozen final120 and
saved-best112 evaluations used exactly three new128-lane seeds2026102261–63,
with unchanged actors/no optimizer. Final:384/384 valid,336/384 positive-net,
0/384 selected-release hits. Best:384/384 valid,378/384 positive-net,0/384 hits.
All-eligible-release hits were also zero; global hits were3 final and5 best,
which cannot satisfy the same-release rule. Both deterministic policies missed.
Viewed both deterministic sequences and median/high-net/first-miss motions. No
failed episodes occurred in these frozen trials and none were removed. Both
checkpoints FAIL the prospective Stage2 actual-hit/deterministic gates;
`stage2_promoted=false`. Stage1 remains reviewed and supports advancement only
under its nominal-state kinematic criteria; the goal of completing Stage2 was
not achieved tonight.

Valid median drone travel remains finite (4.214m final,3.736m best). Shape-release
evidence and high speed/proximity utility persist despite below/negative-Y misses.
Frozen event diagnostics show q at arm in327/349 final and365/381 best positive-event
trials; the nearest forward passage precedes arm in311/349 and290/381. Many
passages also miss spatially. This supports investigating objective ranking and
impact/arming alignment; it does not prove exploration irrelevant or task
impossibility. Original definitions/gates remain unchanged. The single training
seed does not identify which alternative method would learn successfully.

No20-collection extension is justified: the last20 training collections had
0/2560 selected-release hits, both independent policies had0/384, and higher
returns did not yield actual delivery. Do not continue this objective indefinitely
or mark a reward plateau as success. Future proposed work (not implemented or
retrospectively scored as success) is to review actual same-event crossing
accuracy versus shape-output reward and the temporal association, then explicitly
version/approve a revised Stage2 objective before another run. All older policies,
reports, trajectories and the known failed strict density check are preserved;
no Stage3/hardware or new reward was launched. The overnight follow-up is now PAUSED (verified in saved automation state) after
recording this reviewed unsuccessful Stage2 outcome. No further run is scheduled.

Full evidence and decision: [stage2-review.json](../tmp/stage2-shape-delivery-20261022/stage2-review.json),
[frozen final](../tmp/stage2-shape-delivery-20261022/frozen-final/report.json),
[frozen best](../tmp/stage2-shape-delivery-20261022/frozen-best/report.json),
[completed training summary](../tmp/stage2-shape-delivery-20261022/completed-training-summary.json).
All current pinned execution sources and run snapshot/scene/request were archived
and every archived source hash rechecked; external dependencies/recordings are
not included: [archive manifest](../tmp/stage2-shape-delivery-20261022/execution-source-reviewed-archive.json).
Last reviewed2026-10-22. No additional simulation, fitting or hardware result is
implied by the documentation/source-preservation step.

2026-10-22 Stage1-specific reassessment after user rejection of the prior success
claim: withdraw the earlier Stage1 advancement recommendation. Original frozen
results, checkpoints, criteria and historical stage1-review.json remain unchanged.
The new audit reads all768 retained best/final trials, retaining failures in full
denominators, and rechecks instantaneous shape decomposition around each selected
event. Best:356/380 positive-proxy-valid events select the first release-window
sample;289/380 have a larger shape-speed maximum before arming. Final:313/371
and210/371 respectively. Median pre-arm lead of the wider-window shape-speed
maximum is10ms in both. These are descriptive timing diagnostics, not new reward
or success definitions. Typical tracked evidence spans approximately0.367L→0.790L;
it does not independently establish drone-to-tip causal whipping transfer.

The best deterministic trace has actual world tip speed7.454m/s and attachment
speed1.032m/s at credited q; its forward finite-difference tip speed7.110m/s is
consistent with a real brief motion rather than a purely printed reward number.
The recorded motion shows cable deformation and fast tip output, so literal
absence of any learned motion is not established. The important correction is
that proxy-positive excitation and reduced travel are insufficient to demonstrate
the intended whipping skill. The previous visual review overinterpreted outward
shape evolution as adequate evidence for advancement.

A matched nominal deterministic descriptive source/current comparison gives
raw proxy5.926→6.235, shape-speed-at-q6.685→6.266m/s and drone path19.591→2.653m.
This is not a controlled multiseed ablation; it chiefly demonstrates the travel
change and does not establish improved whipping. Do not promote/restart Stage1
or tune Stage2 from an assumption that Stage1 succeeded. First establish an
independent motion/mechanism acceptance procedure distinguishing desired whipping
from generic cable excitation, with research rules approved before implementation.
No physics, control, reward, exploration setting or original threshold changed.

Retained [reassessment.json](../tmp/stage1-reassessment-20261022/reassessment.json),
[all-trial audit](../tmp/stage1-reassessment-20261022/audit.json),
[real-time saved motion](../tmp/stage1-reassessment-20261022/stage1-best-full-real-time.gif)
and[4x slow saved release](../tmp/stage1-reassessment-20261022/stage1-best-release-slow.gif).
GIF frame counts were checked against original sampled indices; no trajectory
interpolation, simulation, fitting or training was performed for these renderings.
Last reviewed2026-10-22.

2026-10-22 User clarification on evidence for changed rewards: old runs and
policies must not be a primary basis for interpreting expected behavior,
selecting the new reward coefficient, or judging the revised objective.
The soft-workspace re-scores above are retained as arithmetic/range checks only.
Their below-failure counts do not demonstrate early-failure behavior under the
new reward. Unbounded cost versus a fixed failure return is a structural ordering
issue, separate from whether a trained policy exploits it. Coefficient0.01
remains provisional, without empirical validation. Define the intended cost
tradeoff prospectively using excursion magnitude and duration relative to task
credit; behavioral conclusions require fresh training under the revised reward
and independent evaluation. No settings, thresholds, simulations or training
were changed by this documentation clarification. Last reviewed2026-10-22.

2026-10-22 Offline challenge after the user identified side swinging and approved
retaining it as a negative: added the original deterministic best112 states as
[a hash-pinned fixture](../evaluation/fixtures/stage1_side_swing_v1.json), with
world metre/metre-per-second units and node0 attachment. The label rejects this
recording for the intended Stage1 skill; its universal physical-whip label remains
unresolved. Original recordings, checkpoints and frozen reports are unchanged.

[whip_candidate_review.py](../evaluation/whip_candidate_review.py) implements only
an offline hypothesis, not a registered reward. It retains frozen shape-release
events and score S, then requires q strictly inside [arm,end] and an increase in
the rest-length-weighted squared shape-speed fraction at nodes s>=0.75L between
baseline and q (beyond64 float64 eps). Candidate output is S for supported events,
zero otherwise; incomplete/failed episodes cannot receive episode support.
This share is a kinematic localization diagnostic, not kinetic energy or causal
energy flux. Boundary rejection can expose timing censoring rather than absence
of whipping. Neither extra condition is an established necessary or sufficient
physical-whip rule. There is no lateral-excursion cutoff, case-name blacklist or
automatic upgrade of physical/goal acceptance.

The exact rejected recording's old score6.234533 becomes candidate output0:
q equals arm and distal share decreases0.681906→0.629633. One analytic outward
turn packet remains supported (0.837300), as do nine width/speed variants; these
are constant-edge-length synthetic kinematics, not dynamically validated positive
whips. Seven constructed controls receive0. These calibration outcomes do not
independently validate the hypothesis. On all768 retained best/final trials,
including failures in denominators,73/384 best and105/384 final episodes still
have candidate support. Original old scores reconstruct within1e-9. The cohorts
are related development evidence without complete independent motion labels;
these counts are not classifier accuracy or false-positive estimates.

First surviving examples were selected in fixed seed/lane order, not by visual
quality. Best lane2 credits a later event with baseline4.01s/arm4.22s/q4.23s/end4.42s,
output0.690044; final lane4 credits1.35/1.75/1.83/1.95s, output2.370567, with only
0.019865 distal-share increase. Their physical and goal labels remain unreviewed.
The full saved motions and exact candidate-event snapshots were rendered with
fixed world axes and checked GIF frame counts. These survivors prevent treating
rejection of the calibration recording as a solved whipping measure. No extra
threshold was fitted to remove them after the challenge.

Retained [prospective hypothesis contract](../tmp/whip-side-swing-validation-20261022/contract.json),
[all-trial challenge report](../tmp/whip-side-swing-validation-20261022/report.json),
[candidate-supported motions](../tmp/whip-side-swing-validation-20261022/surviving-candidates-full.gif)
and [selected-event evidence](../tmp/whip-side-swing-validation-20261022/surviving-candidates-event-evidence.png).
Eight new tests passed for the exact negative, analytic packet, transforms,
rigid/backward controls, independent candidate checks and failure/prefix handling.
Retained regressions passed:8 shape-release,7 decomposition,22 measurement tests.
All pinned training/simulator/controller sources remained hash-identical through
the challenge. No simulation, fitting, training, reward integration or policy
promotion was performed; an independently reviewed positive/negative motion set
and further mechanism validation remain required. Last reviewed2026-10-22.

2026-10-22 Fresh Stage1 restart requested: prepared a separate120-collection,
128-lane/15,360-episode plan with new seed3, random actor/critic, fresh optimizers
and RNG, no checkpoint source, and a fresh entropy clock (0.02 initially, hold
through12 collections,0.001 floor at96). Retain100-ms AR(1) exploration,50-Hz
acceleration control, current physics/limits and0.25*path/L movement cost as the
proposed unchanged settings. These are preparation settings, not an implemented
new reward. Asked the user whether the desired event is specifically an outward
traveling fold opening at the tip or also includes transverse traveling pulses;
this materially changes the experimental objective. Preserve all earlier runs.
The host RTX4080 was available and no training worker was found; default sandbox
GPU visibility failed, so the host check was performed with approved escalation.
Archived pre-change execution sources and hashes. No new simulation, optimizer
update, reward integration or training launch occurred in this preparation.
No automatic Stage2 advancement is planned. Last reviewed2026-10-22.

2026-10-22 Fold-goal prototype after the user's explicit selection of an outward
fold opening toward a far strike: added
[fold_release.py](../tasks/whip/fold_release.py), a standalone experimental scorer,
and a dispatch hook in ShapeReleaseScorer allowing its existing batch adapter to
test this scorer. The original shape-release scoring function, task/presets,
default selection and worker registration are unchanged. No new reward mode is
enabled, no actor checkpoint is transferred and no training run has started.

The [prospective contract](../tmp/stage1-fold-restart-20261022/contract.json)
freezes these provisional choices before the new native probes: a fold is a
local>90-degree tangent reversal across up to two edges each side of its tracked
apex; its identifiable curvature maximum must start with a fold at s<=0.35L,
show a fold again at s>=0.65L and reach the last interior node within0.60s,
using consecutive unique100-Hz associations without backward steps. At arrival,
freeze the last three interior turning angles and require their sum to fall by
at least50% at arrival+0.20s. The complete opening window must be observed; it
cannot be clipped to the final sample to manufacture completion. Choose earliest
maximum tip shape speed over[max(baseline,arrival-0.10s),arrival+0.20s]; require
gain>=0.25m/s from the first root-side fold. Event score is the minimum of its
positive world-forward tip speed, positive anchored shape-forward tip speed and
positive shape-speed gain, divided by1m/s, all associated with the same event.
Valid return retains max finalized event score minus0.25*full-episode world drone
tracking-point path/L; failure replaces the entire return with-10. Partial fold
progress is diagnostic, not rewarded. These choices do not establish a universal
physical-whip definition, tension-mediated causality or energy transfer.

Nine new tests passed: the exact user-rejected recording receives0; a constructed
larger outward fold oriented toward+X is supported; backward/small/rigid/unrelated
controls, distal-only folds, unfinished windows and folds held without opening
receive0; failure retains event evidence but totals-10; pose covariance and input
checks pass; the CPU batch adapter preserves path charges and terminal clawback.
The positive fixture is analytic constant-edge-length geometry, not a native
flight or independently reviewed physical whip. The initial accounting fixture
used float32-generated offsets against a float64 exact-path expectation; it was
corrected to generate float64 positions while preserving tolerance and its failed
log. Retained regressions passed:8 shape-release,7 decomposition,22 measurement
tests. These checks establish software/kinematic properties, not task success.

Coverage remains a failed prelaunch gate. Retained32 random-policy episodes and
128 earlier exploration episodes provided no qualifying valid release under the
prototype (development evidence, not new simulations). Then256 new no-update
native episodes ran under unchanged physical/control sources:54 pull/reverse
strokes,72 cosine excitation profiles,112 paired AR diagnostic controls and18
hover controls. Overall134/256 completed validly and0/256 had qualifying valid
releases; failures remain in denominators. Two tracked arrivals occurred in
failed pull/reverse episodes, neither passing the opening check. The only valid
root-fold-progress AR example had very large drone travel/ascent; progress alone
does not establish a useful whip. An additional offline association hypothesis
using tangent-reversal-angle maxima also produced0 qualifying valid releases
on the same development batches and was not substituted into the scorer.

The first native reporting script failed after saving the physical states because
of a duplicate failed keyword. Reporting was completed from that exact batch,
without re-simulation; a subsequent NumPy-array serialization error was corrected
in the reporting helper. Both raw batches, original failure log and corrected
reports are retained. No native optimizer preflight or training update was run.
Simulation/controller/native source hashes match the pre-restart archive.
No threshold was lowered after the coverage failure. Zero observed releases may
reflect sparse discovery, restrictive detector assumptions or maneuver feasibility;
it does not prove that the simulator cannot whip.

Evidence: [readiness and checks](../tmp/stage1-fold-restart-20261022/readiness.json),
[pull/reverse and AR batch review](../tmp/stage1-fold-restart-20261022/native-coverage.json),
[bounded sinusoid batch review](../tmp/stage1-fold-restart-20261022/sinusoidal/native-coverage.json),
[alternative association diagnostic](../tmp/stage1-fold-restart-20261022/association-challenge-v2.json).
The [curriculum proposal](../tmp/stage1-fold-restart-20261022/curriculum-proposal.json)
is pending user approval: up to30 of the total120 collections would first guide
root-side fold formation with maximum local reversal angle in radians minus the
current path cost, failures-10; independently reviewed formation gates would
precede the remaining release-learning collections. It would preserve the fresh
initial actor/critic/RNG and inherit the original15360-episode entropy clock across
the internal phase change. This is proposed research, not an installed reward,
scheduled campaign, tested learning method or accepted Stage1 skill. Last
reviewed2026-10-22.

2026-10-22 Fresh approved formation launch: started
[runs/ppo_stage1/20261022-105933-962042](../runs/ppo_stage1/20261022-105933-962042/)
from the new formation preset, seed3, without resume or actor initialization.
The launch checkpoint observed at round1/128episodes has the current source
hashes and15360-episode entropy budget/zero inherited offset. Its optimizer
states already contain the first update; round0 empty main-run optimizers were
not separately captured and are not claimed as an observed artifact. The native
preflight was a separate seed and its weights were not used for this run.
Initial completed collection accounting, event records and bounded conditional
KL were checked; early guide scores are not evidence of successful formation
or release. Raw formation angle, persistent folds, aligned full-release
diagnostic, travel cost and net return are retained separately.

The earlier duplicate-process launch check initially matched the launcher's
own shell command, so no process was started by that attempt. It was corrected
to inspect actual Python process names and argument entries; one UI-owned
training worker was then launched. This was a local guard error, not an approval
rejection or a failed training run. The selected CPU tests passed; the sandbox
AR suite had one CUDA skip and all six AR tests subsequently passed on the host.
No source edits were made to the pinned execution after launch.

The existing heartbeat was updated and reactivated as
"Review AeroWhip Stage1 fold curriculum", replacing its stale overnight Stage2
instructions. It monitors this formation run, waits for the30-collection review
stop, preserves full denominators and prospective numeric/motion criteria, and
may proceed to the approved remaining Stage1 release budget only after review
and preflight. No Stage2/hardware or indefinite reruns are authorized by this
updated monitor. It is quiet when state is unchanged/non-actionable and stops
after reviewed completion or a failed review needing user input.

Prepared [evaluate_formation.py](../tmp/stage1-fold-curriculum-20261022/evaluate_formation.py)
for frozen final/best evaluation AFTER the worker stops, with seeds2026102291–93
and native deterministic replay. It refuses active-source-worker evaluation,
keeps failures in128-lane denominators, records immutable checkpoint/batch hashes
and never writes automatic motion approval. Syntax and CLI parsing were checked;
it has not yet evaluated the active run. Warm-up-to-release handoff uses the
final stopped actor, preserving actual spent budget and entropy; saved-best
evaluation is diagnostic and cannot rewind the campaign's elapsed clock.

Retained [launch](../tmp/stage1-fold-curriculum-20261022/launch.json),
[verification](../tmp/stage1-fold-curriculum-20261022/launch-verification.json),
[initial health](../tmp/stage1-fold-curriculum-20261022/initial-health.json)
and [pinned launch sources](../tmp/stage1-fold-curriculum-20261022/launch-readiness.json).
Last reviewed2026-10-22. No training-completion or learned-whip claim is made.

2026-10-22 Approved fold curriculum implementation and preflight: the user chose
the fold-formation warm-up then release option within the120-collection budget,
and requested X-directed output/alignment rather than rewarding Y output.
Added [fold_formation.py](../tasks/whip/fold_formation.py) with two separate modes
and [formation preset](../configs/learning/whip_stage1_ppo_fold_formation_ar1_v1.json)
and [release preset](../configs/learning/whip_stage1_ppo_fold_release_ar1_v1.json).
Default Stage1 now selects formation; earlier presets and run artifacts remain.

Formation reward is max observed root-side local tangent reversal angle/1rad
at interior nodes s<=0.35L, minus0.25*full-episode world drone tracking-point
path/L; failure replaces the entire return with-10. Local reversal is measured
between tangents up to two edges either side of the node, using atan2(cross norm,
dot). A persistent formed-fold diagnostic requires the SAME root-side node's
angle>90degrees for at least two consecutive100-Hz observations. Continuous
angle guide credit is not a completed fold, directed release or Stage1 success.
The full fold-release diagnostic is recorded separately without adding it to
warm-up reward. The initial angle, if nonzero, is paid on the first transition
so the zero-start causal potential preserves the exact total guide reward.

Aligned release retains the prospective tracked-fold/opening/gain rules above,
then multiplies the SAME event/sample output by
`min(max(cos(u_shape_tip,d),0),max(cos(v_world_tip,d),0))^2`; zero speed gives0.
Pure sideways or backward output earns0 and mixed directions are downweighted
in full3D. There is no separate positive Y term. Formation itself does not require
forward preparation velocity: backward/vertical preparation remains available.
This is an experimental goal-specific kinematic utility, not an energy-transfer
proof or a validated universal whipping definition.

The fresh formation run plans seed3, random actor AND critic, empty optimizers,
no initialization/resume source,128 lanes and a full15,360-episode entropy budget.
The worker stops at collection30 for review, rather than consuming all120 in
formation. Summary explicitly states `fold_formation_motion_review_required`,
`full_stage1_complete=false` and `release_phase_started=false`. Entropy retains
weight0.02 through collection12 and reaches0.001 at96 of the total campaign.
No actor weights, RNG state or critic are taken from the native preflight.

Release launch requires an explicitly reviewed final stopped formation actor:
numeric AND motion approval must match its checkpoint hash; its episodes/round
must match the completed warm-up summary. Earlier saved-best weights cannot
rewind the used campaign budget or entropy clock. Remaining training episodes
must be15360 minus those actually used by formation. Transfer copies the complete
actor, including learned scale, into a fresh critic/optimizers with inherited
15360-episode entropy budget and used-episode offset. There is no automatic
reward switch or Stage2 launch. Prospective warm-up review uses three new seeds
2026102291–93 with128 trials each:>=95%valid and>=80%valid persistent root folds
in EACH full128 denominator, deterministic validity/fold, and representative
motion review. Passing formation is not full Stage1 completion.

Six new curriculum/alignment/handoff/accounting tests passed; existing selected
regressions passed, including shape release/delivery, fold release, movement and
AR policy checks. The first curriculum tests caught an initial-angle accounting
error and a temporary-directory Path fixture issue; both were corrected without
relaxing tolerances and the original failure log remains retained. The32-lane
native fresh-policy preflight completed:9valid,9positive formation guide,0positive
net and0persistent folds, with independently reconstructed NumPy angle/path/
persistence and exact stored-density algebra. Optimizer integration and bounded
conditional KL passed; deterministic native replay was valid. These are software
and execution checks for starting learning, not learned behavior or skill success.
The earlier strict2e-5 re-forward density failure remains preserved and was not
relabeled or repeated. A later startup-only handoff guard refinement was covered
by CPU tests; numerical preflight and physical/scorer/PPO sources are unchanged.

Retained [native preflight](../tmp/stage1-fold-curriculum-20261022/native-preflight.json),
[initial test failures](../tmp/stage1-fold-curriculum-20261022/cpu-tests.log),
[final curriculum checks](../tmp/stage1-fold-curriculum-20261022/cpu-tests-handoff-final.log)
and [startup qualification](../tmp/stage1-fold-curriculum-20261022/startup-guard-qualification.json).
No physics, controller, action limits, cost coefficient or source recordings
changed. Launch verification is recorded separately once a run exists. Last
reviewed2026-10-22.


2026-10-22 User-approved offline trajectory-search pilot: after the PPO/NMPC
comparison, the user requested trying optimization of a complete preparation and
strike with the existing simulator. Prepared a separate, bounded, gradient-free
search in [the retained pilot folder](../tmp/whip-trajectory-search-20261022/contract.json).
This is offline action-program optimization, not online NMPC, policy training,
or a change to the approved fold curriculum. It uses the current full3D plant,
50Hz acceleration interface, existing delay/limits and validity rules. Eleven
parameters describe four smooth X/Z acceleration pulses; Y command is zero,
while the plant remains unconstrained in3D. The fourth pulse cancels sampled
requested acceleration impulse; this does not imply actual stopping or position
return. Eight128-lane generations (1024total attempts, including hover controls)
use separate fold-release and delivery archives with registered lexicographic
rankings. Existing aligned-release criteria and movement costs are unchanged;
partial-progress/distance guides select search candidates only and do not
establish whipping success. All candidates/failures are retained.
CPU checks covered1000 bounded programs with zero sampled requested impulse
and ranking counterexamples; scripts compiled. The formation worker stopped normally at30collections/3840episodes with
`fold_formation_motion_review_required`; the offline GPU search then started
without a concurrent training worker. Scalar-native replay and motion review
were subsequently completed as recorded below. No online timing, robustness or
successful whip is implied by the preparation checks. Last reviewed2026-10-22.


2026-10-22 Offline trajectory-search result: the approved bounded pilot completed
1024evaluations in232.22s, with984valid and40failed. Counts include repeated
elites/controls and adaptively selected attempts, not independent success rates.
152valid evaluations crossed the broad rectangle;24also met the existing
>=3m/s forward-speed and<=30degree angle diagnostics. One valid candidate had a
persistent root fold; none passed the unchanged full aligned-fold-release rule,
and none had a target hit associated with such a release. All-batch accounting,
independent NumPy path/crossing comparisons, and execution-source hashes passed.
Both archive selections were replayed with the scalar native engine; maximum
cable-position differences from their batched traces were below7e-14m, with the
same validity/event outcomes. Saved X/Z motions and velocity histories were
visually inspected. The fold candidate retreats and oscillates with no tracked
outward release; the delivery candidate translates/ascends toward and over the
target and swings its cable through it. Neither is accepted as the requested
far-reaching whip. Real-time sampled motion GIFs are retained alongside their
native arrays. No policy or later-stage promotion occurred. The experiment
shows that bounded offline search can find target-crossing actions here; it does
not show that online NMPC is feasible/faster than PPO or that the desired whip
has been solved. Restricted action family, provisional detector and nominal-only
evaluation remain limitations. Evidence:
[pilot review](../tmp/whip-trajectory-search-20261022/review-decision.json),
[all-generation report](../tmp/whip-trajectory-search-20261022/report.json),
[native replay evidence](../tmp/whip-trajectory-search-20261022/native-review.json).
The formation checkpoint remains stopped at its scheduled review boundary;
its frozen evaluation was pending when the pilot ended and is now complete
as recorded below. Last reviewed2026-10-22.


2026-10-22 Frozen formation review failed; no release-phase launch. Evaluated
final30 and saved-best20 sequentially on new policy-noise seeds2026102291–93,
128lanes each, plus one scalar-native deterministic replay per checkpoint.
Final:376/384valid,297/384valid persistent root folds; per-seed folds100,96,101
out of128, each below the prospective80%gate. Best:363/384valid,235/384valid
persistent root folds; per-seed folds72,80,83; two seeds also miss95%validity.
Both deterministic trials are valid and form a persistent root fold. These do
not override the stochastic gates. All failures remain in denominators with
return-10. Neither frozen stochastic set has an aligned full-fold release;
full release remains a separate diagnostic, not an added formation pass rule.

The new retained CPU motion helper independently reconstructed tangent angles,
same-node persistence, full-episode drone path and angle-minus-cost/failure
returns for all768stochastic and2deterministic episodes. Actor and production
source hashes were unchanged during evaluation. Reviewed all12representative
plots (median/high/fold/miss/failure/deterministic for each checkpoint), covering
full-horizon motion traces and sampled cable shapes. Root folds occur, but
ascending and lateral excursions persist: final valid median travel6.921m and
median maximum altitude4.532m; best6.827m and4.522m, from initial altitude2m.
This is not accepted as economical preparation for the desired forward release.
Numeric criteria already fail without inventing an additional motion threshold.
No release worker, extension, earlier-best handoff, Stage2 or hardware action
was launched. Findings do not establish that PPO or the model cannot whip.
Evidence and exact final checkpoint hash:
[formation-review.json](../tmp/stage1-fold-curriculum-20261022/formation-review.json).
The automatic campaign ended at this failed review as instructed; its heartbeat
automation was deleted. Original training/evaluation artifacts remain preserved. Last reviewed2026-10-22.

2026-10-22 Soft-workspace formation preset, approved replacement of travel cost.
The new explicit v2 preset retains the formation guide,30-collection review stop,
120-collection campaign budget, fresh-run exploration settings, physics and
control. Drone workspace: radius0.55m around initial worldXY; worldZ1.5–2.6m.
Cable positions are not confined. At each100Hz right endpoint, let radial and
vertical excess be r and z, in metres. Cost is the full-horizon average of
0.01*((r/0.1)^2+(z/0.1)^2), including the post-hold. It is zero inside/on the
workspace and unbounded outside; no force, clipping, action filter or new
termination is introduced. Full path remains a diagnostic with zero movement
charge. Valid return is formation angle minus workspace cost; failures still
replace the whole return with−10. Applied and accrued workspace cost, raw guide,
net return and path are retained separately. A handoff guard rejects switching
this formation actor into the historical path-cost release preset. No new
release preset, evaluation criterion, automatic monitor or worker was started.

Offline counterfactual audit of768saved stochastic episodes and CPU accounting
replay of2saved native deterministic trajectories agreed with independent NumPy
workspace calculations. These are re-scores, not new simulated trials. At old
weight0.1, valid median boundary costs are39.269/27.153(final/best), versus raw
guide medians2.001/1.762. Provisional weight0.01 gives3.927/2.715. Even then,
29/739valid stochastic recordings score below−10: an early-failure incentive
remains possible with unbounded cost and fixed failure return. No rule was
changed to hide this. Training remains unlaunched pending the broader reward
review; do not reuse the old path-assuming frozen evaluator for this preset.
Evidence: [calibration and replay](../tmp/stage1-soft-workspace-20261022/calibration-and-replay.json).
Checks:19fold tests passed, including4new soft-workspace tests. Broader
regressions:41passed and1failed because an older discovery test expects the
retired compact-whip default; the existing formation default was not changed.
That assertion remains unchanged. Results are retained with the contract.
Last reviewed2026-10-22.


2026-10-22 Approved unified Stage1 restart implemented. New task/scorer:
[tasks/whip/unified_fold.py](../tasks/whip/unified_fold.py), version
`directed_unified_fold_v1`. Valid episode utility is the maximum across individual
attempts of0.5F+0.5P+S, minus full-horizon quadratic workspace cost. F is capped
root-side(s<=0.35L) reversal angle/(pi/2). Each root observation can supply
formation-only credit; P is zero until an actual>90degree root fold starts the
existing unique outward curvature track. P is material progress to the last
interior node, normalized from that attempt's start. Ambiguity, backward-step,
apex-turn and0.60s limits remain. S is the preserved aligned-fold release score
from the SAME segment, baseline and arrival: distal fold, opening and gain
requirements, same-q output, +X alignment and frozen timing remain unchanged.
Preparation contributes at most1 and never counts as completed release. Causal
increments pay only improvements in the best available attempt, never sums
across repetitions or independent maxima from unrelated attempts.

Workspace retains radius0.55m about initial drone tracking-point worldXY and
worldZ1.5–2.6m. Full4.5s horizon average of0.1*((radial_excess/0.1m)^2+
(vertical_excess/0.1m)^2) applies at100Hz right endpoints, including post-hold.
No cap, path/acceleration/time penalty, confinement force or altered physics.
The user explicitly approved0.1 by prospective excursion/time tradeoffs;
coefficient0.01 re-scores are not new-policy performance evidence. Failure removes
all positive utility and returns−10 minus accrued workspace cost; it does not
refund cost and makes no guarantee against every early-termination incentive.
Raw utility, selected F/P/S, completed-release fraction with failures in the
all-episode denominator, path diagnostic, applied/accrued boundary costs and net
return are retained separately. The training UI labels preparation versus release.

Fresh campaign: seed4,128lanes,15360episodes/120collections; no actor source,
resume, warm-up cap, phase transfer or automatic Stage2. Existing50Hz world
acceleration,(8,8,4)m/s2 limits and100msAR1 remain. The original launch used
linear entropy0.02 held12collections, reaching0.001at96. The approved schedule
change after81collections now uses beta=0.001+0.019*exp(-E/2560), where E is
completed training episodes plus any inherited episode offset. Thus2560episodes
is20 original128-lane collections, independent of later lane count or total budget.
There is no hold;0.001 is an asymptotic floor. At handoff E=10368 and beta=0.001331025118.
The original15360-episode budget and elapsed state are preserved.
A prospective frozen final/best evaluation plan is registered with3new stochastic
seeds2026102311–13,128lanes each, and scalar-native deterministic replay. Per-seed
gates remain>=95%valid and>=80%valid completed aligned releases over ALL128;
deterministic valid release and full representative motion review are required.
No target-hit reward or inferred physical validation is added.

Checked58targeted tests, including same-attempt arithmetic, incomplete-prefix
causality, lateral/backward/rigid controls, free contained motion, fixed-horizon
failed-prefix cost, failure credit removal and reporting. An initial new JSON
serialization assertion incorrectly expected exact decimal2.5; corrected it to
check exact equality with the computed component, retaining independent numerical
checks and the original failed log. The32lane native preflight had18valid,
18positive preparation guides and0completed aligned releases. Independent NumPy
angle/progress/utility/workspace accounting and stored-density algebra passed;
a PPO update and deterministic native replay completed with sources unchanged.
These checks establish integration only, not skill learning or classifier validity.
Historical strict re-forward failure remains preserved. Evidence, exact contract
and launch status: [unified campaign](../tmp/stage1-unified-20261022/contract.json).
Last reviewed2026-10-22.

2026-10-22 Unified Stage1 launched in the visible Training UI:
[runs/ppo_stage1/20261022-121145-434399](../runs/ppo_stage1/20261022-121145-434399/).
Captured round0 checkpoint: seed4,0episodes/steps, empty actor/value optimizer
states, entropy budget15360 and offset0, no resume or actor source, execution
hashes match. Worker observed collecting round1 of120. First launcher attempt
resolved the virtualenv symlink to systemPython and exited before any run or
worker; corrected interpreter path, preserved failed launch log, and started
exactly one worker. Native preflight policy was not reused. No skill conclusion
is drawn from startup. [Launch verification](../tmp/stage1-unified-20261022/launch-verification.json).
Last reviewed2026-10-22.

2026-10-22 Approved exponential exploration continuation. Only the entropy
schedule changes: beta(n)=0.001+0.019*exp(-n/20), n=completed episodes/128
(including inherited offsets). The original worker completed collection81 and
stopped without discarding a cohort. Original sources/checkpoint are archived
under [schedule evidence](../tmp/stage1-exponential-20261022/contract.json).
An explicit, provenance-bearing derived checkpoint preserves all learner and
CPU/CUDA RNG state, including learned scale and both nonempty Adam states; it
changes the entropy configuration, execution hashes and identity with recorded
authorization. This is a schedule migration followed by full-state resume, not
ordinary exact resume of the old experiment or actor-only transfer. The general
source/identity resume guards remain unchanged and were checked to reject the
unmodified original checkpoint after source changes. No reward, physics, action
noise process, success gate or120-collection total changed. Original linear
preset/artifacts remain available; the new exponential preset is the default.

Checked42targeted scheduler, legacy clock, AR1, unified reward and UI tests.
The first test invocation omitted the repository tests helper path and had3import
errors; retained the log and reran with PYTHONPATH=.:tests, passing all42.
A real-architecture CPU checkpoint restore was bit-identical for every learner
state field; all saved payload/RNG state and serialized derived checkpoint were
compared exactly. Verified81-to82collection clock progression and rejection of
unapproved source changes by ordinary resume. No new physical performance
evaluation was run for this schedule-only change. Lower entropy incentive does
not directly set action standard deviation or establish better whipping.
Last reviewed2026-10-22.

2026-10-22 Exponential continuation launched in the visible Training UI: [20261022-131442-557003](../runs/ppo_stage1/20261022-131442-557003/). Production initial checkpoint at81collections/10368episodes matched the derived checkpoint exactly for policy, critic, both optimizers, all clocks and CPU/CUDA RNG. Execution hashes and identity matched; resumed event and collection82 observed. Original total remains120. See [launch verification](../tmp/stage1-exponential-20261022/launch-verification.json). No new skill claim.

2026-10-22 User-requested prescribed-point cable demonstration, separate from
training. [Reproducible script](../tmp/point-whip-demo-20261022/demo.py) uses the
existing scalar-native prescribed cable solver and unchanged configs/simulation.json:
11nodes/10edges,0.9525m cable, original masses, DER bending/twist, damping,
gravity, fixed downward material clamp,600Hz physics and100Hz saved observations.
The world attachment starts at(0,0,2)m with a stationary hanging cable. A
hand-designed quintic point trajectory moves backward/up from0.30–0.85s, pauses
until1.10s, then moves forward/slightly upward until1.40s and holds until2.70s.
This replaces drone dynamics by exact boundary position/velocity/acceleration;
there is no thrust limit, workspace reward or policy. It does not change the
production model, task, evaluator, training sources or the running PPO worker.

The rendered original trial completed with finite states and maximum edge-length
error5.0e-16m. Point peak speed5.010m/s, point peak acceleration51.30m/s2,
tip peak speed6.2785m/s at1.46s and peak+Xtip speed5.6997m/s. Speed peaks generally
progress outward, with coincident samples and a delayed tip peak. Visual inspection
shows a broad arc/near-root bend, upward motion followed by forward motion; this
was subsequently explicitly rejected by the user and is not a positive physical
classifier reference.
A second hand-designed preparation pause/downward-stroke variant is retained
separately; no optimizer was used and the displayed result is the original trial.
Timestep halving to1200Hz for the original changed maximum node position by
0.597mm and peak tip speed by0.000866m/s; both integrations completed. This is
one numerical sensitivity check, not mesh convergence, model-fit validation or
drone feasibility. The same clamp orientation is held fixed; human wrist rotation
is not reproduced. [Animation](../tmp/point-whip-demo-20261022/point-whip.gif)
shows real time then0.4x playback of the saved100Hz samples. [Evidence and review](../tmp/point-whip-demo-20261022/review.json).
Last reviewed2026-10-22.

2026-10-22 Paper-informed offline measurement comparison, explicitly approved
after rejecting the point-driven demonstration. Retrieved the authors'
[canonical repository](https://github.com/alex-krotov/WhipTask_PerformanceWhipHand)
at commit17106972a303217e4e3336dfe63acd3953c6b801. The archived MAT contains32
participant/style mean speed profiles and5227 single-time cable states at peak
hand speed, with a matching5227-row trial-statistics CSV. These are processed
aggregates and snapshots, not full individual marker trajectories. Source hashes,
reproducible extraction, schema/units and preserved missing coordinates are in
[retained evidence](../tmp/whip-paper-validation-20261022/source/source_manifest.json).
The42 snapshots with missing coordinates remain in the exported data. The human
snapshot target direction/translation and exact12-marker mapping are not fully
specified in the release; no forward-axis alignment was inferred. The source
comment for extension units conflicts with the numeric values, so no automatic
length rescaling was applied. Author supplementary movie links were located;
their contents were not downloaded or reviewed in this comparison.

Added the pure offline peak-timing helper and compared eight cases: the exact
user-rejected side swing, rejected point demo, rigid rotation/translation,
small outward/backward packets, separated stationary bends, and an early wave
followed by a later rigid crossing. Analytic cases are kinematic constructions,
not dynamically feasible positive whip labels; historical policies are not used
as a fair learning comparison or reward calibration. Full records, all nodes,
explicit exploratory intervals, raw amplitudes, three velocity frames and
same-sample tip output are retained in the [report](../tmp/whip-paper-validation-20261022/comparison/report.json).
The side swing's selected sample at1.06s has relative-tip speed7.9645m/s,
shape-residual speed6.4119m/s, and world/shape forward cosines0.9388/0.9109;
nodes3–10 have progressively later relative-speed maxima. The initial review
treated this as a counterexample to the then-requested motion family; the user
subsequently included side swing, as recorded below. The rejected point demo also has
outward speed timing but substantial upward output. These counterexamples do
not establish a universal negative physical-whip label.

Replotted human means and separately displayed hit/miss preparation snapshots.
Eleven of16 discrete and three of16 rhythmic mean profiles have at least one
reversal in adjacent sampled peak ordering. This is an aggregate diagnostic,
not an individual human rejection rate, and motivates preserving timing
exceptions rather than imposing a universal strict-order gate. No new score,
threshold, curriculum or training run was introduced. Checked10 new timing
tests and29 existing measurement/decomposition tests; all passed. Verified81
pinned execution-source hashes unchanged. Data extraction checks include finite
profiles, increasing time grids, complete snapshot/CSV error crosswalk and exact
numeric NPZ round trip. Figures were visually inspected. No physics, fitting,
policy training, original MATLAB statistics or hardware experiment was rerun.
Last reviewed2026-10-22.

2026-10-22 User-approved motion-family revision: include side swing/sidearm
preparation. Updated the current offline review and fixture goal annotation;
the prior rejection, review outputs and plots are archived under
[previous interpretation](../tmp/whip-paper-validation-20261022/archive-before-side-swing-inclusion/).
The fixture retains its stable identifier, original trajectory hash and old
candidate score0.0. That score reflects the historical candidate's release-boundary
and distal-motion diagnostics, not a prohibition on lateral movement. Inclusion
permits review; it does not certify this trajectory, validate a new reward or
reverse the failed policy/delivery evaluations. Preparation plane, cable
deformation/progression and target-directed release are separate questions.
Only review annotations, presentation and this record changed. Existing data
were replotted; no physics/training rerun, scoring change or threshold relaxation.
Last reviewed2026-10-22.

2026-10-22 Four-stage restoration requested: inspected the retained curriculum,
Stage1–4 task and learning presets, reward comments and acceleration/recovery
compatibility guard. Recorded the user's selected excitation → broad delivery →
precision → recovery sequence and the remaining reward/interface decisions.
Historical Stage1 was a mechanics/outreach proxy, not a physical energy reward;
the retained outreach campaign did not reach Stage3's final target and did not
launch Stage4. No presets, reward code, training defaults, thresholds or policies
were changed. Documentation only; no tests, simulations or training rerun.
Last reviewed2026-10-22.

2026-10-22 Approved whole-attempt evaluation implemented: added separate
excitation, distal-response and geometric-delivery measurements with explicit
common attempt bounds, optional attributed release bounds and complete saved
traces. Corrected the current overview's superseded side-swing negative label;
historical evidence and candidate rules remain preserved. The
[dated plan and checks](../tmp/whip-attempt-evaluation-20261022/verification.json)
retain15 new passing analytic tests,53 passing regressions, source/hash checks
and the ten-view comparison. Visually inspected side-swing and disjoint-event
figures. Numerical measurement behavior is checked; physical-whip validity,
reward effectiveness and new-policy learning are unverified. No reward weights,
thresholds, physics, training presets, workers or policies changed.
Last reviewed2026-10-22.

2026-10-22 Fresh flexible-excitation implementation for the user-authorized Stage1
launch: added the direction-free scorer, explicit fresh preset, accounting and
recording integration, UI labels and later-stage handoff restriction. Seventy-five
focused CPU checks passed, including10 scorer and9 integration tests and retained
reward/PPO/exploration regressions. Prior sources/configuration were archived in
[execution-before.zip](../tmp/stage1-flexible-excitation-20261022/execution-before.zip).
Native execution and launch are pending at this checkpoint; these checks do not
establish policy performance or physical-whip validity. Last reviewed2026-10-22.

2026-10-22 Flexible-excitation native preflight completed before launch. The
fresh32-lane seed2026102311 cohort had17valid lanes and1positive-net lane; these
are coverage measurements, not success rates for a learned skill. Independent
pseudoinverse projection agreed within8.89e-15m/s; guide arithmetic within4.42e-15
and workspace arithmetic within1e-12. The original float32 stored-reward sum check
at absolute1e-6 FAILED (max3.0517578e-5, one terminal float32 ULP). It remains
failed with its original script, threshold and artifacts preserved. The environment
checks conservation before casting both transitions and return separately to
float32. A separate retained diagnostic reproduced the complete cohort/actions
byte-identically, verified float64 telescoping within2.85e-14 at the unchanged
1e-12 check, and verified exact equality of the reconstructed cast reward vector
to the saved vector. No production objective, tolerance or source was changed to
address the reporting-roundoff discrepancy.

The separate diagnostic completed90actor/120value updates, with final conditional
KL0.0164508038 below0.02, exact stored-density algebra, and a valid native
deterministic replay (return0.00442318). This preflight actor is never the training
initializer. Source hashes remained unchanged. The historical2e-5 strict
re-forward density failure is also still retained and was not repeated or
reclassified. [Native evidence](../tmp/stage1-flexible-excitation-20261022/roundoff-diagnostic/native-preflight.json)
and [original failed check](../tmp/stage1-flexible-excitation-20261022/native-preflight-failure.json)
separate those conclusions. Last reviewed2026-10-22.

2026-10-22 Fresh flexible-excitation Stage1 launched in the visible Training UI:
[runs/ppo_stage1/20261022-152747-014823](../runs/ppo_stage1/20261022-152747-014823/).
Budget120collections/15360episodes at128lanes, seed5. Captured round0 actor AND
critic match a separately reconstructed seed5 initialization bit-for-bit;
optimizers, completed episodes and entropy offset are zero. Entropy starts0.02
with original15360budget; no resume or initializer is present. One worker was
observed collecting, with immutable execution hashes matching preflight.
[Launch verification](../tmp/stage1-flexible-excitation-20261022/launch-verification.json)
and archived launch sources are retained. This is a training start, not a
successful whipping skill or automatic later-stage authorization.
Last reviewed2026-10-22.

2026-10-22 Fresh launch health: the first128-lane collection completed and
PPO optimization was observed running. All returns were finite, recorded path
charges were exactly zero, and pinned execution hashes were unchanged. The
first cohort had61valid lanes and1positive-net valid lane; it is an initial
random-policy sample, not evidence of learned whipping or a basis for retuning.
[First-collection evidence](../tmp/stage1-flexible-excitation-20261022/first-collection-health.json).
Training remains active within the120-collection Stage1 budget; no Stage2 starts
automatically. Last reviewed2026-10-22.

2026-10-22 User-requested Stage2 design recorded as proposed only: explicit
confirmed broad-hit priority with crossing-local flexible output, bounded
approach/quality terms, inherited exploration and separate geometric/motion
review. Inspected scorer, crossing, offline evaluator and actor-transfer code;
two independent read-only design reviews identified touch-versus-passage and
rare-fast-hit pitfalls. No production source/settings changed, no tests or
simulation were run for this proposed reward, and active Stage1 was not altered.
Exact values and gate thresholds remain subject to user approval.
Last reviewed2026-10-22.

2026-10-22 Stage2 curriculum design expanded at the user's request while current
Stage1 trains. Preserved the original base proposal and recorded curriculum-v2:
one optional finite target-extent bridge, unchanged broad-delivery objective,
120initial collections with evidence-based20-collection extensions and a240
research-review boundary, inherited exponential exploration, and separate
checkpoints/seeds for development, candidate selection and final acceptance.
Inspected current code/observations and retained early training logs; independent
read-only reviews informed the proposal. A new Stage1 health snapshot documents
poor full-horizon validity through collection16 without declaring permanent
failure. This is design/documentation work only: no reward implementation,
training-source/settings edit, GPU evaluation, simulation rerun, promotion or
Stage2 launch. JSON, phase-budget/seed consistency and82unchanged production source hashes
checked; proposed numerical
curriculum choices remain unvalidated. Last reviewed2026-10-22.

2026-10-22 Stage1 restart preparation at user request: stopped the flexible-
excitation worker cooperatively at39collections and preserved all artifacts.
Audited reward/termination code and current-run evidence; independently checked
bounded-workspace counterfactual costs on73saved motions. Recorded an exact
research proposal that removes the valid/failure total-return inversion while
retaining the original near-boundary quadratic. Explicit approval requested
under the repository research-decision rule before changing the reward/preset.
No production edit, new training or physical-simulation result is claimed.
Last reviewed2026-10-22.

2026-10-22 Approved bounded-workspace Stage1 restart implemented: a separate v2
learning preset selects existing bounded-density arithmetic, preserving original
near-boundary curvature and capping full-episode workspace cost at5. A narrow
configuration guard admits only the approved excitation profile; the launcher
and Training UI identify it. Original v1 sources, preset, run and checkpoints
are retained. All92focused CPU/regression tests pass after pinning a historical
discovery test to its original baseline; no tolerance was relaxed. Native
preflight is in progress and production restart is not yet launched. No policy
success or physical-whip validity is claimed. Last reviewed2026-10-22.

2026-10-22 Bounded-cost native preflight completed: retained32-lane rollout,
paired initial physical/action arrays, independent projection/utility/workspace
and exact reward-vector accounting, conditional PPO update and deterministic
native replay. All non-reward initial arrays matched the earlier same-seed
preflight bit-for-bit. Strict float32 summation remained a recorded1.90735e−6
failure against unchanged1e−6; separate exact-cast and float64 accounting checks
passed. These are integration results, not policy performance or whipping
validation. See retained prelaunch review. Last reviewed2026-10-22.

2026-10-22 Approved bounded-workspace fresh Stage1 launched in the Training UI:
[20261022-160251-474373](../runs/ppo_stage1/20261022-160251-474373/).
Captured round0matched independent seed5actor/critic initialization bit-for-bit;
optimizers and all elapsed clocks were empty/zero, with no resumed or transferred
policy. Verified the exact bounded-cost preset,128lanes,120collections, exponential
entropy start0.02and execution hashes. Original39-collection stopped run remains
preserved. No automatic Stage2 promotion or policy success is claimed.
Last reviewed2026-10-22.

2026-10-22 Bounded restart first-collection health:128episodes,61valid and67
invalid, matching the original seed5first cohort exactly for validity, lengths,
raw excitation, sampled tip-speed components and drone-path diagnostics. The
reward change now ranks every valid outcome above every failed outcome:
valid returns−3.08999to0.63106(mean−1.81947),failed−13.39787to−10.05972
(mean−11.48097). Workspace charges0.05972to3.92190are within the approved bound;
terminal return vectors exactly match independent float32-formula reconstruction,
movement cost is zero and all execution hashes remain unchanged. See
[first-collection evidence](../tmp/stage1-excitation-restart-20261022/first-collection-health.json).
This initial pre-update cohort verifies cost isolation/accounting; it is not
learning progress, independent evaluation or a whipping-success result. Training
continues with the approved120-collection budget. Last reviewed2026-10-22.

2026-10-22 User requested starting again and selected exact resume of the stopped
bounded-workspace run at11collections/1,408episodes. Source hashes still match
the tested implementation. The original latest checkpoint and all run artifacts
are preserved; the continuation has been dispatched through the visible Training
UI for the remaining109collections (13,952episodes), retaining the original
120-collection total. Checkpoint-state verification is pending initialization;
no new scientific tests or reward changes are implied. Last reviewed2026-10-22.

2026-10-22 Exact Stage1 continuation verified and running:
[20261022-195928-651339](../runs/ppo_stage1/20261022-195928-651339/) resumes
[20261022-160251-474373/latest.pt](../runs/ppo_stage1/20261022-160251-474373/latest.pt)
at11collections/1,408episodes, leaving109collections/13,952episodes within the
original120-collection total. Actor, critic, optimizer states, CPU/CUDA RNG and
training/evaluation clocks matched the source checkpoint bit-for-bit; settings
and execution hashes matched. The entropy coefficient is0.0119620464without a
reset. The initial90-second verification window expired during compilation;
a read-only follow-up captured and verified the initial checkpoint without
starting another worker. Original artifacts are preserved. No code, reward,
physics or acceptance criteria changed, and tests were not rerun because the
execution sources are unchanged. Launch-state verification is not policy success.
Last reviewed2026-10-22.

### 2026-10-22: completed excitation run, post-excitation climb audit

The exact continuation completed120total collections. Offline reconstruction of
retained deterministic best108and final120shows peak excitation at1.72/1.73s,
with drone height1.673/1.977m. The remaining motion earns no additional
excitation-record credit, yet finishes at14.384/14.647m world height.
Workspace density saturates at2.65/2.56s; cost after the excitation peak is
2.149/2.272. Returning inside still reduces cost, but further excursion beyond
saturation has no additional distance-dependent charge. This is consistent
with poor post-excitation control under the capped objective, not evidence that
the later climb supplies useful excitation. Deterministic commands confirm
continued upward acceleration; the precise learning cause remains unverified.
See [retained audit](../tmp/stage1-excitation-climb-audit-20261022/audit.json).
No new simulation, independent robustness evaluation, reward change, restart or
model promotion was performed for this audit.

### 2026-10-22: distance-sensitive Stage1 reward proposal, pending approval

The user requested analysis of a final reward and a Stage1 relaunch. The exact
new research formula has been submitted for approval; it is **not implemented
in training and no new run has launched**. The proposed raw cost restores the
original uncapped quadratic workspace density, accumulated over the fixed450
sample denominator. Excitation Q remains unchanged. With x=Q-C, use F(x)=x
for x>=0 and F(x)=5x/(5-x) otherwise. Proposed valid return is F(Q-C);
failed return is -10+F(-C_accrued), removing all excitation credit. Causal
rewards would be differences of F(running Q-running C), with terminal correction.

For finite quantities in real arithmetic, increasing cost at fixed excitation
strictly reduces return, positive net scores are unchanged, valid returns exceed
-5 and failed returns are at most -10. Very negative returns are compressed;
this weakens learning sensitivity to extreme excursions and does not preserve
expected-return rankings across stochastic policies. It is not a convergence or
containment guarantee. Existing physics, workspace, control and exponential
exploration would remain unchanged for the proposed fresh120-collection run.

[Offline proposal checks](../tmp/stage1-distance-sensitive-proposal-20261022/proposal-and-checks.json)
passed a monotonicity grid, distance/hover/failure counterexamples, float64 causal
accounting at1e-12 tolerance and valid/failure ordering on73retained diagnostic
motions. Float32 synthetic sum discrepancy was9.54e-7, recorded separately.
Rescoring deterministic best108/final120changes their returns from7.927/7.847
to-4.883/-4.882. These are counterfactual arithmetic results, not new-policy
evaluations. No native simulation, optimizer check or training was rerun for this
proposal; production integration remains contingent on approval and testing.

### 2026-10-22: approved distance-sensitive return implemented, preflight pending

The user explicitly approved the exact formula and fresh restart. The v3 preset
adds only stage1_return_transform=negative_net_rational_v1 to the historical
uncapped excitation v1 settings. Configuration registration is restricted to the
approved excitation task/cost; earlier presets keep their existing arithmetic.
The net transformation is applied before causal differences, and failure
claws back excitation with no unexecuted-suffix cost. UI, snapshot metadata and
collection components distinguish raw cost from transformed return.

99CPU tests passed, including seven new analytic-history cases for monotonic
distance sensitivity, free contained excitation, zero hover, failed-prefix
accounting and rejecting unsupported combinations. An initial six-decimal-place
comparison of two independently rounded float32 quantities failed by5.27e-7;
that failure is retained. The corrected test compares the stored reward sum to
the independent real-valued objective at the same six-decimal-place precision,
and separately checks the exact terminal float32 cast. No tolerance was enlarged.
Native execution and optimizer preflight remain pending; no training launch yet.

### 2026-10-22: v3 native preflight reviewed, fresh launch authorized

The32-lane native preflight completed with17valid lanes and1positive-net lane.
Before updates, physical histories and action/noise fields were bit-identical to
the retained same-seed baseline; this isolates the reward change. Independent
NumPy geometry, uncapped cost, transformed return and exact float32 reward-vector
reconstruction agreed. Float64 telescoping error3.55e-15passed1e-12. The strict
float32 reward-sum check **failed** at1.91e-6against the unchanged1e-6criterion
(two terminal ULPs). This reduction/cast discrepancy is retained and was not
reclassified as a pass. Independent vector equality and accounting supported
launch; see [prelaunch review](../tmp/stage1-distance-sensitive-restart-20261022/prelaunch-review.json).
PPO performed103actor/120critic updates with final conditional KL0.0190049<0.02.
The subsequent deterministic native replay was valid, return-0.658608, and passed
independent accounting; these are integration checks, not training success.
The100unique CPU checks include the added JSON/NPZ reporting test.

The first v3 production launch (20261022-221345-356970) failed at round0 before
checkpoint creation or any episode: the worker still required the legacy reward
delivery label. The paired metadata guard was corrected to require the transformed
label only for the approved return-transform setting; source/schema and task
validation remain enforced. That failed run is preserved. All100CPU tests then
passed together, and the native preflight was repeated under the corrected source
hashes with the same32-lane results and accounting evidence, including the
unchanged failed float32 reduction check. No preflight agent is an initializer.

### 2026-10-22: distance-sensitive v3 fresh training launched and verified

Run [20261022-221619-402166](../runs/ppo_stage1/20261022-221619-402166/)
started at22:16local time. Captured round0 checkpoint SHA256
`79fe147b1dc618e8295df72fb73c5552e582f6ab395e9338fba5a766f4616b45`
matched independent seed5random actor AND critic initialization, with empty Adam
states, no resume/initializer and zero training clocks. Approved uncapped cost,
negative-net rational transform,128lanes/15360episodes, source hashes and initial
entropy0.02were verified. One worker (PID48660) is collecting. The earlier
metadata-rejected launch and all older runs remain preserved. This verifies
launch identity only; learned excitation, containment and useful motion remain
unverified. No automatic later-stage training or new monitoring automation.

The first128-episode collection subsequently completed and PPO optimization
began. It contained61valid and67failed episodes; independently reconstructed
transformed returns matched all recorded float32 returns exactly. Raw workspace
cost ranged0.0597–520.341, confirming the per-sample cap is absent. Source hashes
remained unchanged. These initial random-policy outcomes do not establish
learning progress; [first-collection accounting](../tmp/stage1-distance-sensitive-restart-20261022/verified-startup/first-collection-health.json)
is retained.

### 2026-10-24: exact v3 Stage1 continuation resumed at collection24

The user requested restarting after the stopped24/120collection status was
reported. Resumed [20261024-160428-146323](../runs/ppo_stage1/20261024-160428-146323/)
from the preserved v3 latest checkpoint (3,072episodes). Initial continuation
checkpoint matched actor, critic, both optimizer states, CPU/CUDA RNG, identity,
training/evaluation clocks and settings bit-for-bit. Execution hashes matched
before and after launch. Entropy resumes at0.0067226900 with3,072completed
episodes, zero offset, original15,360budget and unchanged exponential schedule.
The remaining96collections add12,288episodes, not a fresh120collections. One
worker (PID14415) was verified. See
[resume verification](../tmp/stage1-rational-resume-20261024/launch-verification.json).
No reward, source, physics, control or evaluation-method change was made, and
no implementation tests or independent policy evaluations were rerun. This
verifies continuation identity, not learned motion quality or Stage1 success.

### 2026-10-24: completed v3 Stage1 frozen evaluation and motion review

The exact continuation completed120total collections/15,360episodes. Final and
saved-best actors are identical at collection120, so frozen behavior was evaluated
once with three new action-noise seeds2026102471–73,128attempts each, and native
deterministic replay. This varies policy noise at one nominal physical initial
state; no initial-state, model-parameter or real-flight robustness is established.
Each stochastic batch had127/128valid positive-return attempts (381/384overall,
failures retained). Valid median excitation Q=9.82; deterministic world/shape tip
speed at peak was10.64/10.19m/s at2.46s, attachment speed1.51m/s. Motion and
shape residuals support a repeatable excitation burst, not a validated physical
whip or causal energy-transfer conclusion.

Deterministic drone height1.46–2.46m avoids the old extreme ascent, but horizontal
radius reaches0.83m versus the0.55m soft radius. Across all384attempts only one
valid attempt stayed entirely within the soft workspace; maximum height4.75m
and radius3.55m occurred under noise. Median-return and largest-cost motion
examples still climb after peak excitation. The largest-cost example has Q9.55,
raw C7.32 and positive return2.23: the cost is active and this return is on the
linear branch, not hidden by the negative-return compression. Direction remains
unrewarded:382/384peak tip velocities point negative X; global broad geometric
hits were8/384. These diagnostics do not establish aimed delivery.

Independent frozen geometry, cost and terminal formula checks passed. Strict
exact-vector and float32-reduction checks did not all pass; original failures
remain preserved with unchanged criteria. Two independently reconstructed
near-zero vector entries differed by at most1.78e-15; production-potential cast
vectors match exactly. Float32 sum error reached1.91e-6against1e-6. Sources and
actor remained unchanged, no optimizer was used. Representative deterministic,
median/high/low-return, largest-cost and failed whole-attempt plots were reviewed.
See [review](../tmp/stage1-rational-review-20261024/review.json) and
[complete analysis](../tmp/stage1-rational-review-20261024/analysis.json).
Recommendation is to retain an experimental Stage2 initialization candidate,
while explicitly carrying incomplete containment and recovery. No new success
threshold, reward change, automatic promotion or Stage2 launch was made.

### 2026-10-24: Stage2 readiness and proposed-reward audit

At the user's request, rescored the384retained frozen v3 Stage1 trials on the
intended broad rectangle and the earlier larger extent bridge. Valid attempts
with any forward x=1.1m plane passage:26/384; intended hits8/384; larger-rectangle
hits22/384. Median maximum tip X0.639m indicates a forward-reach/delivery gap,
not just target extent. Hit-local median forward/shape speeds were1.875/1.144m/s;
median hit time was0.608s after the Stage1 excitation peak. This does not establish
causal connection to that peak. Retained recordings now serve as development
evidence; they are not future Stage2 held-out acceptance trials.

Auditing the earlier proposed5+q/(1+q) hit utility revealed that even zero
shape output earns the base5. Recommendation is to revisit crossing-quality
dependence before implementation, so geometric passage alone does not dominate
flexible delivery. No revised formula or threshold is approved by this analysis.
Recommend one broad-target objective with reach/hit/quality diagnostic milestones;
the larger-target bridge is not supported by current forward coverage. All
recommendations remain proposed. Production sources, policies and settings were
unchanged; no new simulation, training or Stage2 launch occurred. CPU prefix and
same-crossing interpolation checks completed; no exact-zero contacts occurred,
so that counterexample remains for implementation testing. See
[findings and limitations](../tmp/stage2-readiness-analysis-20261024/findings.md)
and [all-attempt analysis](../tmp/stage2-readiness-analysis-20261024/analysis.json).

### 2026-10-24: research-informed Stage2 curriculum proposal, not implementation

Reviewed original GoalGAN, self-paced deep RL, reverse curriculum and reward-
shaping papers; their methods/results do not establish AeroWhip success rules.
The revised proposal keeps one reward and target extent, changing only distance
0.6->0.8->1.1m based on measured competence. Offline rescoring of384saved trials
yielded69/28/8valid geometric hits, or43/9/3with both same-crossing forward and
shape speeds>=1m/s. These are counterfactual geometries on traces generated with
the1.1m goal input, not new goal-conditioned simulations. Native probes are
required before launch. The1m/s quality floor is a proposed engineering choice,
not a physical whip definition or retrospective Stage1 acceptance rule.

Proposed hit utility5*q/(1+q) replaces the earlier5+q/(1+q) geometric base bonus.
Keep bounded proximity guide, best-only utility, current raw workspace cost and
rational return/failure handling.18algebra checks passed, explicitly retaining
weak-hit/near-miss, expected-quality/reliability and workspace ranking tradeoffs.
This does not certify physical correctness or policy-invariant shaping.

Proposed progression: independent128-trial reviews every10collections, minimum
20collections per intermediate level, two checks with>=122valid and>=103valid
q>=1hits plus deterministic hit and motion review. Final proposed holdout requires
>=122valid and>=116quality hits per128, three seeds, after primary selection.
Initial budget120collections, minimum40at final distance, evidence-based20-block
extensions and mandatory research review at240; no run or extension authorized
by merely writing this proposal. Existing width/height curriculum cannot be
silently reused as distance curriculum. Actor and critic goal input was inspected;
82production execution hashes remain unchanged. No production tests, native
rollouts or training were run for this proposal. See [exact proposal](../tmp/stage2-curriculum-redesign-20261024/proposal-v3.json),
[research notes](../tmp/stage2-curriculum-redesign-20261024/research-notes.md) and
[design checks](../tmp/stage2-curriculum-redesign-20261024/design-checks.json).

### 2026-10-24: Stage2 implementation held for curriculum/exploration discussion

After approving distance-curriculum implementation/preflight only, the user
requested discussing lower exploration and continuous reach encouragement
instead of discrete plane-distance changes. No production source had been
changed; all82execution hashes still matched the completed Stage1 review.
Recorded the hold in [discussion status](../tmp/stage2-curriculum-redesign-20261024/discussion-hold.json).
A CPU re-forward audit on reconstructed histories for32valid saved lanes at9
decision times estimated median local marginal acceleration-noise standard
deviations1.51/0.84/0.91m/s² in X/Y/Z. These are state-local distribution
estimates, not trajectory errors or a reduced-noise performance experiment.
Inherited entropy coefficient is0.0010471, but learned action scale remains
separate. See [exploration audit](../tmp/stage2-curriculum-redesign-20261024/exploration-discussion-audit.json).
A modest action-scale comparison and a fixed-plane continuous guidance objective
are discussion options only. No new noise scale, reward, curriculum, gate or
training run was implemented, and no native simulations were run in this check.

### 2026-10-24: user clarified target position AND full desired tip velocity

The final command must specify target position and a full desired cable-tip
velocity vector (direction and magnitude). A faster crossing is not automatically
better when the requested speed is lower. Position and velocity must be evaluated
at the same delivery instant. The earlier fixed+X, monotone-speed Stage2 proposal
is therefore insufficient and superseded in scope before production coding.

Confirmed implementation limits by source inspection: actor/critic encoders
currently receive three target-center coordinates, not an independent desired
velocity vector; existing surface-task validation permits only horizontal normals
and minimum-speed mode; legacy multi-target training is explicitly launch-blocked.
No guard was bypassed. Proposed direction is to teach varied goals in Stage2 with
broad position/velocity tolerances, expand a feasible goal distribution based on
competence, tighten tolerances in Stage3, then train recovery/repetition in Stage4.
Goal ranges, reward, velocity tolerance and promotion criteria remain unresolved
research choices, not implemented functionality. No production code, policy,
training configuration or physics was changed. Updated only this record and the
[discussion/scope record](../tmp/stage2-curriculum-redesign-20261024/discussion-hold.json).

### 2026-10-24: simplified fixed-direction Stage2 proposal

The user endorsed one-direction Stage2 and moving varied directions to Stage3,
then approved simplifying the proposal. Recorded the concrete
[fixed-direction contract](../tmp/stage2-curriculum-redesign-20261024/proposal-fixed-direction-v4.json)
and [readable design](../tmp/stage2-curriculum-redesign-20261024/fixed-direction-v4.md).
One fixed+X rectangle at[1.1,0,2]m,1x0.6m;120collections; retain the previously
reviewed same-crossing flexible quality/guide and current rational workspace
return. No internal curriculum, new goal input, reduced exploration or production
implementation. Full velocity matching remains a Stage3 design requirement.
Exact fixed-task/evaluation approval and native preflight remain pending.
Reran18offline algebra checks, including known adverse reward rankings; all passed.
All82execution hashes match the reviewed Stage1 snapshot. No new simulation,
PPO training or physical validation was performed. Prior failed strict numerical
checks remain retained. Original proposals and policy artifacts are preserved.

### 2026-10-24: fixed-direction Stage2 implementation and preflight

User explicitly requested Stage2 after reviewing the exact fixed-task proposal.
[Contract](../tmp/stage2-fixed-delivery-20261024/contract.json): +X target rectangle
center[1.1,0,2]m,width1m,height0.6m;120collections x128lanes. No task transitions.
[Scorer](../tasks/whip/flexible_delivery.py) uses q=min(positive world-X tip speed,
attachment-anchored shape-residual tip speed)/(1m/s), same-sample approach
q/(1+q)*exp(-0.5*(rectangle_distance/0.7m)^2), and same-interpolated-state crossing
utility5q/(1+q). Episode utility is a running maximum, not accumulated hits.
Strict positive-side confirmation is required; exact-zero plateaus use last-zero
departure. Initial guide is unpaid. Selected utility records are not physical
release events. q>=1 is an engineering quality diagnostic, not a whip definition.

The environment reuses the approved uncapped full-episode workspace cost and
rational map via explicit stage2_return_transform. Valid return F(U-C); failure
-10+F(-C_accrued). Causal differences and failure correction retain costs after a
hit; no path/acceleration/time cost. Raw utility, crossing quality/vector/position,
workspace cost, raw net, compression and final return remain separate.
[Preset](../configs/learning/whip_stage2_ppo_flexible_delivery_v1.json) is the new
Stage2 default; legacy presets remain unchanged. Stage1 final actor is transferred
with architecture and learned scale intact; critic/optimizers/RNG are fresh.
Exponential entropy inherits15,360episodes: start0.00104709629,100ms AR1 retained.
A narrow flexible-excitation-to-flexible-delivery transfer path is registered;
physical/control compatibility and exact-resume checks remain active.

Worker saves raw all-lane accounting, crossing records, typical/weak/high responses
and failures. Every10collections it runs128frozen development trials with a
separate seed, restores training RNG, leaves training/entropy clocks unchanged,
and retains development checkpoints and full trajectories. Deterministic checks
also run at these boundaries. Deterministic best.pt remains a reward-selected
legacy convenience; final primary selection must use development hit rates AND
motion review, not that filename alone. No automatic Stage3 or budget extension.

57targeted CPU checks passed, including10new geometry/accounting/registration
checks and47retained excitation/shape-delivery/AR1/entropy checks. The first test
attempt had two incorrect hand-calculated expectations (rigid-fit weighting);
corrected analytically from weights[1/4,1/2,1/4], without scorer or tolerance change.
Original failure log retained. Native preflight and training launch are pending;
these tests do not establish learned delivery or physical validity. See
[implementation diff](../tmp/stage2-fixed-delivery-20261024/implementation.diff)
and [test log](../tmp/stage2-fixed-delivery-20261024/regressions.txt).

2026-10-24 Stage2 native preflight completed:32-lane accounting/optimizer test,
then two independent128-lane baseline batches (seeds2026102601/02) using the
original transferred actor, and native deterministic replay.254/256baseline
trials valid,4geometric hits,3quality(q>=1)hits; deterministic valid but miss.
These are source-policy baselines, not trained Stage2 success. Independent NumPy
projection/passage/utility/cost/return comparisons passed stated tolerances;
strict float32 sum1e-6 passed. Exact reconstructed float32 vectors FAILED in
4/28,800and6/28,800entries,max3.55e-15, retained with cancellation diagnosis and
unchanged tolerances. Historical strict density failure is not relabeled.
Optimizer test:8actor/145critic updates, conditional KL0.01939233<=0.02. Full actor
transfer bit-identical, fresh critic unchanged by transfer, empty optimizers,
entropy offset/budget15,360 and start0.00104709629 verified. Typical/miss/hit/failure
motion plots reviewed; side excitation and post-hit wandering/ascent remain
baseline weaknesses. No trained success or containment claim. The
[prelaunch review](../tmp/stage2-fixed-delivery-20261024/prelaunch-review.json)
supports starting the authorized bounded experiment from the original Stage1
checkpoint, never the preflight-updated policy. UI smoke passed; initial check
looked for detailed reward text in the summary rather than its existing tooltip,
then corrected the inspection without changing UI or acceptance criteria.

2026-10-24 Stage2 training launched and round0 verified in
[20261024-205138-342045](../runs/ppo_stage2/20261024-205138-342045/).
Source actor, full learned scale, fresh seed6critic, empty optimizers, zero new
training counters, inherited episode offset/budget15,360, and matching pinned
execution hashes were verified. Exactly one worker observed. Original initializer
used; optimizer preflight weights were not used. Initial collection is running;
no learning outcome claimed. See [startup verification](../tmp/stage2-fixed-delivery-20261024/launch-verification.json)
and the [file-by-file implementation/why report](../tmp/stage2-fixed-delivery-20261024/implementation-report.md).

### 2026-10-24: approved first-hit Stage2 revision, implemented

User explicitly requested stopping the current run and applying the reviewed
[first-hit proposal](../tmp/stage2-fixed-delivery-20261024/first-hit-timeout-proposal.json).
The old run stopped at60collections/7,680episodes; artifacts and original execution
sources are preserved. New [contract](../tmp/stage2-first-hit-20261024/contract.json)
and [preset](../configs/learning/whip_stage2_ppo_first_hit_delivery_v2.json) use
remaining60collections/7,680new episodes, latest old Stage2 actor, fresh critic,
optimizers and RNG. No automatic Stage3 or budget extension.

The first confirmed forward rectangle crossing with q>=1 ends that lane at its
100Hz confirmation endpoint. q=min(positive world-X tip velocity, shape-residual
speed)/(1m/s). This is a useful-delivery engineering threshold, not a physical
whip classifier or full desired velocity match. Weak crossings remain diagnostic.
Safety failure wins on the same endpoint; a qualifying crossing confirmed at5s
wins over timeout. No post-hit tail. Unchanged plane/physics/50Hz acceleration.
Valid hit return F(B-C-0.1T), B=5q/(1+q) at that first crossing; timeout return
-2+F(A-C-0.5), with unpaid-initial maximum same-sample approach A<1; failure return
-10+F(-C-0.1T), removing all positive task credit. F(x)=x for nonnegative x and
5x/(5-x) otherwise. C retains uncapped quadratic workspace cost10/450 per executed
finite100Hz observation, not a new500denominator. T counts executed endpoints,
including a failing attempted endpoint; interpolated crossing time is separate.
Causal transformed-potential differences and terminal correction telescope to
these returns. Time, workspace, timeout penalty, utility and final return remain
separate in records; ended lanes have no padded future rewards or bootstrap.

The 5s horizon uses501observations, but preserves the old actor's451sample input
window and index/450time feature. Prefix inputs/distributions through4.5s remain
identical; the extra0.5s uses a rolling451sample window and is not claimed to be
behaviorally identical to old training. Actor-only transfer retains learned scale
and inherited15,360episode entropy budget with23,040offset, start0.001002344786;
100msAR1 and exponential schedule unchanged. Critic resets for the changed return.

68CPU tests passed (11first-hit checks plus57retained checks). Initial errors are
retained: NumPy-bool assignment to a Torch bool tensor was corrected explicitly;
the GAE test initially unpacked two values instead of its actual three-value API.
No scientific tolerance changed. Native preflight and launch still pending at
this entry; tests alone establish neither learned delivery nor physical validity.

Native first-hit preflight complete:256/256independent source-policy attempts valid,
234qualifying hits,22timeouts; deterministic hit at2.70s. Independent NumPy
geometry/projection/utility/workspace/time/return checks passed, exact independent
float32 reward vectors matched, reward sums within1e-6.32-lane optimizer performed
3actor/90critic updates with conditional KL0.01999423<=0.02. Actor transfer,
fresh critic, empty optimizers and inherited offset23,040/budget15,360 verified.
Median/high/deterministic hit and timeout plots reviewed; timeout still includes
large lateral excursion. This is initialization evidence, not revised learning
success or recovery validation. Native batches had no physical failures; unit
counterexamples cover failure priority/accounting. See
[prelaunch review](../tmp/stage2-first-hit-20261024/prelaunch-review.json),
[native report](../tmp/stage2-first-hit-20261024/preflight-report.json) and
[file-by-file changes and reasons](../tmp/stage2-first-hit-20261024/implementation-report.md).

First-hit Stage2 launched in [20261024-214318-825971](../runs/ppo_stage2/20261024-214318-825971/).
Round0 retained and verified: actor bit-identical to stopped Stage2 latest,
fresh seed6critic bit-identical to independent initialization, empty optimizers,
zero new counters, offset23,040/budget15,360 and matching execution hashes.
Exactly one training worker observed. Remaining60collections, not120additional.
See [launch verification](../tmp/stage2-first-hit-20261024/launch-verification.json).

First production collection completed:128valid attempts,120qualifying hits,
8timeouts, no physical failures; return mean2.85457, mean executed duration2.860625s,
reward-sum error2.12e-7, accepted conditional KL0.01910994. Movement/acceleration
costs remain zero. All128crossed the soft workspace boundary at some point,
mean workspace cost0.16824; do not claim containment. Collection2 observed active.
This is startup health evidence only, not convergence or improved skill.
[First-collection record](../tmp/stage2-first-hit-20261024/first-collection.json).

2026-10-24 23:40 EDT: user requested restart after first-hit run stopped at6/60
collections. Exact-resumed latest checkpoint into
[20261024-233827-625394](../runs/ppo_stage2/20261024-233827-625394/).
Worker confirms round6/768episodes restored, no learning/evaluation migration,
and collection7 active toward60total (54remaining). Source hashes match;
actor, critic, optimizer and exploration state retained. No code/reward change.

### 2026-10-25: standalone C++ physical-plant baseline

The user selected the existing C++ simulation rather than a new NumPy
implementation. The initial sibling `AeroWhip_simulation` extraction contained six
copied native source/header/include files, `main.cpp`, and `Makefile` as its
source/build files; the subsequent approved viewer addition is recorded below.
Existing IDE/environment directories remain. The native
files are byte-identical to this checkout; retained
[source hashes](../tmp/standalone-cpp-20261025/source-hashes.json) establish their
provenance. No physical equation, projection, solver tolerance, history handling,
or CPU linear algebra was changed. Existing optional actuator/residual code stays
in the copied engine; no fitting implementation, learned weights, tensor/GPU
backend, training system, ROS adapter, or viewer was copied. The example uses the
current 0.145 kg drone, 0.017 kg marked cable, 0.9525 m/10-edge mesh, original
inertia/offsets/material coefficients/drag, and maximum 1/600 s physics step.

The example is an open-loop plant experiment: a held applied collective/body
torque every 0.01 s for 1 s, starting stationary at tracking position (0,0,2) m.
It has no flight-controller or motor-response execution and no delayed command
queue; `ModelInput.delay=0` is explicitly metadata for this direct-input example,
not a change to the original application's 0.03 s command delay. The Makefile
retains the original Linux `-O3 -fno-fast-math -ffp-contract=off` build flags and
C++17. These existing compiler settings are not a new algorithmic optimization.

Built and ran with GCC 13.3.0. The example's 101 samples plus four additional
61-sample comparisons (three-axis excitation, variable timesteps, initial twist,
and checkpoint continuation) matched the original native extension bit-for-bit
for all 177 physical-state/history/checkpoint values at every sample. The
checkpoint-continuation case restored the extracted solver at its midpoint and
compared against uninterrupted original execution. Additional-case maximum
edge-length error was 2.082e-16 m and axial relative-velocity error was
1.293e-16 m/s. Both implementations rejected a collapsed edge, negative history
step, and negative collective thrust while preserving their prior checkpoint.
See [validation](../tmp/standalone-cpp-20261025/validation.json) and retained
comparison arrays in that folder. These are extraction/numerical checks on mild
synthetic cases, not physical validation, aggressive-whipping robustness,
gradient/GPU validation, or a controlled performance comparison. The one smoke
run took 0.0901 s for 1 s simulated; it is not a benchmark or a speedup claim.
Original runtime, physics, configurations, recordings, policies, and training
sources were not modified. No optimization campaign or external sharing was
performed. Last reviewed 2026-10-25.

### 2026-10-25: native viewer with the existing visual assembly

The user approved adding a professional standalone viewer while retaining the
existing drone/cable appearance. Added
[viewer.cpp](../../AeroWhip_simulation/src/app/viewer.cpp) and the single
[drone.glb](../../AeroWhip_simulation/assets/drone.glb) visual asset; updated only the
sibling project's `main.cpp` and `Makefile`. The six copied native physical
sources remain byte-identical to the initial extraction and this checkout. No
equations, solver tolerances, cable parameters, controller, reward, fitting,
training, or execution sources in this checkout were changed.

The GLB was generated once using the existing `viewer/drone_model.py` assembly,
original `body_v2.stl`/`guard_v5.stl`, and default viewer settings, at the tracking
origin with identity attitude and attachment (0,0,-0.051) m. It retains all 82
parts/36,196 triangles, colors, metallic/roughness settings, and four separate
rotor groups: CAD plates/guards, propellers, motors, electronics, battery, leads,
zip tie, legs, tracking markers, and `cf_2` visual identity. glTF's Y-up root
conversion is reversed on import to retain the simulation's Z-up tracking frame;
CAD millimetre scaling, +90-degree X rotation, measured clearances, and visual
accessory estimates are inherited from the original assembly. Float32 mesh
storage changes coordinates by at most 1.840e-9 m; the loaded C++ part bounds
differ from source bounds by at most 8.301e-9 m in the retained check. This is
visual geometry preservation, not new hardware calibration. The physics remains
float64. See [asset provenance and geometry checks](../tmp/standalone-viewer-20261025/asset-validation.json).

The new viewer uses Qt6 Widgets/OpenGL 3.3, four-sample antialiasing, material-based
lighting, orbit/pan/zoom, system/detail/front/side/top views, Run/Pause/Step/Reset,
replay, 0.25/0.5/1/2-times playback, visual marker/grid toggles, and world-position,
attitude and tip-speed telemetry. It draws the original green cable at radius
0.0018 m and light markers at radius 0.006 m from native observations; marker 0
is not drawn as a free cable marker. Rotor phase uses the existing visual-only
90 rpm and never drives thrust. The floor is visual only and adds no contact.
Lighting/shader implementation is new; pixel-identical VTK rendering is not
claimed. The inspector scrolls at small window sizes. A numerical failure stops
playback, displays a banner, disables further stepping/running until reset, and
preserves the native solver's last accepted state/history.

At this initial viewer milestone, superseded by the continuous-flight addition
below, the viewer paused at the unchanged example's initial state. Run
executed the same 100 applied-wrench samples/1 s trajectory as `--headless` and
`--trace`; it does not add a flight controller, motor-response experiment, or a
new maneuver. Pausing discards paused wall time. Slow rendering retains pending
simulation steps; a bounded number is advanced per UI tick without skipping
physical states. Playback completion pauses at 1 s; Replay explicitly resets
state/history. The displayed per-step physics time excludes rendering but is
not a controlled benchmark. No speedup or optimization result is claimed.

Installed Ubuntu `qt6-base-dev` 6.4.2 (including its OpenGL/header dependencies).
The application and asset load from the executable's directory independently of
the working directory and need no Python or IDE. `make` builds the viewer;
`make headless` builds the Qt-free terminal executable. Both executables' traces
match all 101 retained original-native checkpoints bit-for-bit, including 177
state/history values per sample, and run headlessly with DISPLAY unset from
`/tmp`. See [numerical checks](../tmp/standalone-viewer-20261025/physics-validation.json).

Native Qt integration checks exercised initial rendering, all 82 parts, exact
single steps, pause, startup/history reset, five view presets, visual-only
toggles, 1000x680 scrolling layout, fast/slow playback, replay, and agreement of
the GUI's completed checkpoint with uninterrupted native integration. A synthetic
extreme-velocity case triggered the existing projection failure; the UI froze
state/history and recovered on Reset. Screenshots were visually inspected for
system, close-up, small-window, and error states. OpenGL reported NVIDIA GeForce
RTX4080. See [viewer checks](../tmp/standalone-viewer-20261025/viewer-validation.json).
These checks do not establish physical accuracy, aggressive-whipping robustness,
newly calibrated geometry, Windows/macOS support, GUI latency guarantees, or a
fair physics performance comparison. Last reviewed 2026-10-25.

### 2026-10-25: continuous controlled flight and repeatable CPU benchmark

The user approved replacing the one-second viewer cap with continuous Hover and
Back-and-forth flight using the original controller/motors, plus a repeatable
benchmark. Added `flight.cpp`/`flight.hpp` in the sibling `AeroWhip_simulation`
project and copied this checkout's `native/runtime.cpp`/`runtime.hpp` unchanged.
Updated only that project's `main.cpp`, `viewer.cpp`, and `Makefile`; the six
existing physical sources and `drone.glb` remain unchanged. Changes were first
staged and checked under [continuous-flight evidence](../tmp/continuous-flight-20261025/),
with pre-change files/hashes retained. Delivery refused to overwrite intervening
source edits. Both delivered executables are byte-identical to the checked
staging builds; see [delivery verification](../tmp/continuous-flight-20261025/delivery-validation.json).

`Flight` is an application layer outside the reusable plant. Its geometric
controller is a C++ port of the existing Python implementation, including
tracking-origin position/velocity feedback, static cable-weight trim, nominal
desired-body-rate preparation from analytic jerk, attitude error, rate transport,
and gyroscopic compensation. Translation gains remain [4,4,6.25] s^-2 and
[4,4,5] s^-1; attitude frequencies [20,20,15] rad/s and damping ratios [1,1,1].
The unchanged native Runtime retains exact 0.005 s controller updates, 0.03 s
command delay, 0.01 s output, and original motor geometry, bounds, initial thrust,
and 0.015 s first-order response. Motor values remain uncalibrated assumptions.
The setup's delay metadata is now 0.03 s to match this controlled path; the
earlier direct-wrench example had no command queue. Physical defaults, maximum
1/600 s integration step, float64 arithmetic, numerical gates, and solver
algorithms are unchanged.

Hover requests tracking position (0,0,2) m and zero yaw. Back-and-forth repeats
the original quintic 0.25 m world-X displacement over 4 s, with a 2 s outward
and 2 s return leg. References are sampled at 30 Hz. The original right-hand
jerk convention at turnaround is retained; this is C2 motion with jerk/body-rate
feedforward jumps, not a C3 claim. Each cycle repeats the original nonterminal
120 packets instead of imposing the original terminal stationary hold. Neither
physical state, BDF2 history, motors, nor pending commands are reset at cycle
boundaries. The source sample is reconstructed from absolute simulation time,
so pause/continuation needs no hidden mutable trajectory cursor. No trajectory
position is written into the plant state.

The viewer now runs without a configured duration cap. Hover/Back-and-forth are
selected in one dropdown; changing preset explicitly pauses and resets the full
runtime. Reset restores integration history, motor thrust, command queues, and
all clocks. A cyan reference path/target is display-only and can be hidden. The
native CAD/cable appearance remains unchanged. Run/Pause/Step and pacing retain
all physical updates, and solver/controller failures stop playback atomically.
The displayed step time now includes reference preparation, controller, motors,
and physical integration. It still excludes rendering and is not itself a
controlled timing result. The updated application was reopened with
Back-and-forth selected and initially paused.

Terminal interface: `--headless` or `--trace` runs continuously until Ctrl+C;
`--trajectory hover|back-and-forth` selects the source, and `--duration SECONDS`
explicitly bounds terminal operation in whole 0.01 s steps. Trace rows contain
simulation time and the original self-describing runtime checkpoint wire values;
row lengths can change with queue contents. `--benchmark` defaults to both
trajectories, 30 s each, one full warm-up and three measured fresh-initial-state
runs per trajectory, emitting JSON. `--raw --trace` retains the original 101-row,
one-second direct-wrench trace exactly. `make headless` remains Qt-free. No
Python, IDE, training, policy, fitting, or GPU physics is required by the app.

Verification: 512 seeded controller fixtures differed from the original Python
wrench outputs by at most 4.441e-16 in the encoded wrench components, against
1e-12 absolute comparison tolerance. All 720 prepared reference packets (three
cycles per preset) differed by at most 2.665e-15, also within 1e-12. Independent
original-preset/Python-controller/native-runtime comparisons covered 801 full
checkpoints per preset over 8 s. Maximum absolute checkpoint differences were
8.882e-16 for Hover and 5.552e-14 for Back-and-forth, within the retained 1e-9
numerical comparison tolerance. Differences mix the checkpoint fields' own SI
units; no physical fit is inferred. Exact C++ checkpoint continuation for a
further 365 outputs passed, including delayed queues/motors/clocks. A throwing
controller rolled back the complete runtime and recovered on the next valid
step. Continuous CLI operation continued beyond 1 s and stopped on SIGINT with
exit 130. Invalid duration/mode inputs were rejected. The legacy raw trace is
bitwise unchanged. See [reference checks](../tmp/continuous-flight-20261025/reference-validation.json).

Qt integration checks confirmed continuous timed playback past 1 s while still
running; 500 output steps/5 s crossed the repeat boundary without resetting and
matched an uninterrupted C++ runtime checkpoint bit-for-bit. Step, pause,
reset, preset changes, reference-display isolation, fast/slow pacing, all 82
visual parts, smaller-window scrolling, synthetic numerical failure rollback,
and recovery passed. The screenshot was visually inspected. See
[viewer checks](../tmp/continuous-flight-20261025/viewer-validation.json).

The retained [30 s benchmark](../tmp/continuous-flight-20261025/benchmark.json)
completed all eight warm-up/measured runs with zero solver/constraint failures.
The median of three measured advance-only real-time factors was 34.7635 for
Hover and 8.00949 for Back-and-forth. Median within-run p95 0.01 s advance times
were 0.290514 ms and 1.27689 ms, respectively. Hover tracking error was zero in
this nominal equilibrium fixture. Back-and-forth position RMSE was 0.01060995 m
and maximum error 0.02028583 m, measured against the requested analytic
tracking-origin path at output timestamps, including command delay. Maximum
length error was 2.22045e-16 m and axial velocity error 1.44405e-16 m/s; repeated
measured runs ended with bitwise-identical full runtime checkpoints.

Timing covers `Flight::step` (reference generation, controller, motors, and
coupled CPU physics); initialization, observation diagnostics, rendering, and
output are excluded. JSON separately records wall time including diagnostics.
The CPU was an AMD Ryzen9 9950X; compiler GCC13.3, unchanged -O3/strict-FP flags.
The workstation was not isolated and CPU frequency was not pinned; see
[benchmark context and source hashes](../tmp/continuous-flight-20261025/benchmark-context.json).
These are local gentle-motion baseline measurements, not an optimization
speedup, throughput guarantee, aggressive-whipping result, hardware model-fit
validation, or policy-training evidence. No scientific acceptance threshold,
physical assumption, reward, or original project's execution source changed.
Last reviewed 2026-10-25.

### 2026-10-25: combined Stage1 mechanism selected, implementation pending

Recorded the user's selected outward-deformation/associated-distal-acceleration
mechanism and its unresolved event/reward rules in the existing research note.
Rechecked eight existing measurement tests with unittest: all passed. pytest
was unavailable. Checks cover synthetic kinematics and tracking ambiguity only;
no physical rollout, trained policy, new success criterion or reward was validated.
Documentation/research-note edits only; production sources, presets and training
were not changed by this work.

2026-10-25 Simple event proposal v0: documented exact candidate factors and a
causal completion-rule proposal in the existing research note. Nine algebra
checks passed; retained kinematic checks also expose an unresolved association
ambiguity, including an interior speed peak. Preserved the initial analysis
mistake excluding boundary peaks and its correction. These are design checks,
not native physics, causal-prefix verification, accepted whip evidence or approval
to train. No production code, reward preset, worker or saved policy was changed.

2026-10-25 Retained native bend-tracking review: added a CPU-only analysis script,
local review page and inert-animation renderer under
`tmp/bend-tracking-review-20261025/`. Compared four preselected historical
coupled-plant recordings with source-specific snapshots; did not rerun physics
or compare training success. Recomputed arrival evidence from observed prefixes,
excluded 229 padded Stage2 frames, checked preserved hashes/exact copies and
independent curvature/gain arithmetic, and inspected static figures. The proposed
first-arrival stopping rule can interrupt ongoing excitation and is not supported
for launch. HTML syntax passed; browser interaction was blocked by file-URL policy
and remains untested. Full changes, checks, limits and media are linked in the
retained review. Production sources, presets, policies and original data unchanged.

2026-10-25 Continuous bending-region experiment: added an offline natural cubic
centreline baseline and a normalized measured-turning field, with local region
centres/widths and conservative consecutive-frame association, under
`tmp/continuous-bend-review-20261025/`. Compared four pinned saved recordings and
controlled geometry. Nine new checks and 22 existing measurement checks passed;
source hashes, numerical quadrature and visual outputs were checked. The isolated
travelling-bend construction has a smooth centre, but saved-trace associations
remain ambiguous and bandwidth-sensitive. U-turn and S-bend constructions have
identical unsigned turning profiles despite different end-tangent directions;
a gently curved arc also produces weak regions. Region detection is explicitly
not fold classification. No new fold/reward/termination rule, physics change,
policy promotion or training was implemented. Detailed evidence and per-file
changes are in the retained review; physical and real-flight validity remain
unverified.

2026-10-25 Original fold/unfolding source audit: rechecked the supplied human-study
PDF methods, results and figures and retained author speed-plot code; inspected
indexed primary passages from the 2002/2003 mechanics papers (full PDF retrieval
failed, explicitly recorded). Corrected attribution: human speed-cascade evidence
does not provide a fold classifier, and an ideal complete loop can have parallel
outer tangents. Recorded the preferred synchronized animation format and the
historical status of its red observation window. Documentation only; no code,
measurement criterion, reward, simulation or training changed.

### 2026-10-25: approved dynamic oscillation and figure-eight references

The user approved two stronger trajectory presets with fixed amplitudes/periods
and a 4 s smooth startup. Updated only `flight.cpp`, `flight.hpp`, `main.cpp`, and
`viewer.cpp` in the sibling `AeroWhip_simulation` project; no additional files
were added to that project's root. The eight native engine/material/actuator/
scheduler files, original controller function/gains, physical coefficients,
numerical tolerances, clocks, motor bounds/response, and `drone.glb` are unchanged.
Pre-change files, staged sources, hashes and checks are retained under
[tmp/dynamic-trajectories-20261025](../tmp/dynamic-trajectories-20261025/).

Oscillation requests x=0.5 sin(theta) m, y=0; Figure eight requests
x=0.5 sin(theta) m, y=0.3 sin(2 theta) m. Both request z=2 m and yaw=0.
Fundamental periods are 2 s and 3 s, respectively; steady requested peak speeds
are approximately 1.57080 and 1.63578 m/s, and maximum requested acceleration
norms approximately 4.93480 and 5.49225 m/s^2. These are reference kinematics,
not measured hardware or guaranteed tracking limits.

For startup duration R=4 s, u=t/R, phase is
`theta=omega*R*(2.5*u^4-3*u^5+u^6)` before R and
`theta=omega*(t-R/2)` thereafter, with omega=2*pi/period. This integrates the
quintic smoothstep frequency ramp; initial position/velocity/acceleration/jerk
are at rest and join continuously to the periodic reference. Position, velocity,
acceleration and jerk use analytic chain-rule derivatives, including the second
harmonic in Y. The existing nominal desired-body-rate preparation consumes jerk.
After startup the phase angle is bounded modulo its own period for long runs;
the source/plant clock and all physical/history/motor/queue states continue.
The startup is not repeated each cycle. Original Hover/Back-and-forth packets and
the legacy raw trace remain bitwise unchanged.

Both presets are available in the GUI and as `--trajectory oscillation` and
`--trajectory figure-eight`. New reference overlays use the same source geometry;
the oscillation overlay is one line segment, and the figure eight uses 360
segments. New presets receive a wider System camera; changing preset resets
flight and System view, while the existing Reset-view control remains display
only. All 11 cable nodes of both retained 30 s rollouts fit inside the default
1440x900 System viewport in a projection check (maximum normalized coordinates
0.83770/0.86819 versus visible limit 1). This is observed framing, not a guarantee
for arbitrary future motion. The existing CAD assembly and physical scale remain
unchanged. See [framing checks](../tmp/dynamic-trajectories-20261025/framing-validation.json).

The old benchmark defaults/metrics are retained. Dynamic cases are selected
explicitly, each using the same 30 s duration, one fresh-state warm-up and three
fresh-state measured runs. JSON additionally records trajectory period and
startup duration. All eight new warm-up/measured runs completed with zero solver
or constraint failures and bitwise-identical final checkpoints across measured
repeats. Median advance-only real-time factors were 6.91196 for Oscillation and
6.07177 for Figure eight. Median within-run p95 0.01 s advance times were
1.60228 ms and 1.91669 ms. Position RMSE was 0.07496398 m and 0.11483275 m;
maximum errors were 0.12468932 m and 0.21513205 m. Tracking compares actual
tracking-origin position with the requested analytic path at output timestamps,
including startup and command delay. These tracking deviations are retained;
controller/motor settings were not tuned to improve them. Timing excludes
initialization, rendering, diagnostic observations and output, as before.
See [oscillation benchmark](../tmp/dynamic-trajectories-20261025/oscillation-benchmark.json),
[figure-eight benchmark](../tmp/dynamic-trajectories-20261025/figure-eight-benchmark.json),
and [context/source hashes](../tmp/dynamic-trajectories-20261025/benchmark-context.json).
The workstation was not isolated and CPU frequency was not pinned; these are
local baseline timings, not optimization speedups or throughput guarantees.

Independent 30 s diagnostic traces retained 3001 output samples per case. Sampled
peak drone speeds were 1.66961/1.85433 m/s and cable-tip speeds 9.62383/9.61516 m/s
for Oscillation/Figure eight. Peaks are at the 100 Hz output grid, not asserted
continuous-time maxima. Actual altitude varied approximately 1.8821--2.0000 m
and 1.8631--2.0249 m despite the 2 m reference. No allocation clipping was
observed in the last-request snapshots at those output times; this does not
constitute a full 200 Hz saturation-history measurement. Maximum cable-length
errors were 2.499e-16/2.915e-16 m, and benchmark axial-velocity errors remained
below 3.132e-15 m/s. These are simulation predictions, not a physical-whip
classification or hardware validation. See
[rollout diagnostics](../tmp/dynamic-trajectories-20261025/rollout-validation.json).

Verification used 55-digit mpmath differentiation of position and the original
Python body-rate preparation, independently checking 254 complete PVA/body-rate
packets including startup joins, steady period boundaries and long-time inputs.
Maximum absolute packet difference was 1.444e-14 against 1e-12 tolerance; startup
is at rest and steady periodic pairs agreed exactly in the tested samples.
Original controlled Hover/Back-and-forth traces and the raw extraction trace
remain bitwise identical. See [reference checks](../tmp/dynamic-trajectories-20261025/reference-validation.json).
An independent original Python-controller/native-runtime rollout supplied the
first 1201 full checkpoints/12 s for each new reference. Maximum checkpoint-field
differences were 4.663e-13/2.092e-11, within the existing 1e-9 absolute numerical
comparison tolerance. No acceptance tolerance was changed.

GUI checks covered all four selections/resets, reference-display isolation,
single-step, 850 continuous outputs/8.5 s for each dynamic reference, pause,
2x pacing, numerical-failure rollback/recovery and smaller-window scrolling.
Dynamic GUI checkpoints matched uninterrupted C++ execution bit-for-bit. All 82
visual parts remained present; final previews were inspected. One visible-desktop
attempt failed a pause check: its retained screenshot shows Running before the
harness's first Run action, with an unprogrammed camera orientation. This is
consistent with uncontrolled desktop input, but the exact input source was not
recorded. A rerun isolated external keyboard/mouse/shortcut events in the test
harness only and passed all unchanged checks; no input events were filtered on
that rerun and no application pause code or numerical criterion was altered.
The failed attempt and screenshot remain in the
[GUI attempt record](../tmp/dynamic-trajectories-20261025/gui-attempts.json);
see [final GUI checks](../tmp/dynamic-trajectories-20261025/viewer-validation.json).

Deployment preserved intervening edits and rebuilt the actual standalone targets.
[Delivery checks](../tmp/dynamic-trajectories-20261025/delivery-validation.json)
pin source hashes and byte-identical staged/installed executables. No controller
optimization, physics/model promotion, reward/training change, real flight, or
claim of physical fidelity was introduced. Last reviewed 2026-10-25.

### 2026-10-25 — Offline tangent evolution and whole-cable speed review

Implemented the approved [visual diagnostic extension](../tmp/tangent-evolution-review-20261025/review.md): actual segment tangent paths, gap-aware signed planar projections, unchanged bending-region associations with local raw-angle summaries, and synchronized speeds at all cable nodes. Seven constructed controls and four preserved simulation recordings distinguish unsigned bending, coherent planar turning, rigid motion and 3D projection ambiguity. Ten new diagnostic checks, nine region checks and 22 existing measurement checks passed. Original recording/snapshot hashes were verified. No physical model, reward, termination, training or accepted-whip status changed; physical fold identification and event completion remain unvalidated research decisions. Last reviewed 2026-10-25.


### 2026-10-25: standalone 20-edge cable mesh

Approved scope implemented in the sibling C++ project: default `make_model()` and
all CLI execution modes now use 20 edges / 21 nodes; `--links 10` retains the
10-edge / 11-node baseline. `--links 20` explicitly selects the default. The
factory accepts only these two meshes. Each of the ten nonuniform measured
marker intervals is subdivided equally in two. The ten moving point-marker
masses stay at even node indices 2,4,...,20; inserted odd nodes receive only the
existing bare-cable lumped mass. Cable length is 0.9525 m, total cable/marker mass
0.017 kg, and initial common marker positions agree within 4.45e-16 m. Floating
sums were 0.95250000000000024 m and 0.016999999999999994 kg. State width changes
from 88 to 158; engine checkpoint width changes from 177 to 317. See
[mesh construction checks](../tmp/mesh20-20261025/mesh-validation.json).

Only `flight.hpp`, `flight.cpp`, `main.cpp`, and `viewer.cpp` change in the sibling
project. The factory chooses subdivisions, the CLI forwards mesh selection to
viewer/headless/trace/raw/benchmark paths, benchmark JSON includes links/nodes/
physical-marker count/length/mass, and the viewer derives its mesh description
from the actual model. The eight copied native source files remain byte-identical
to this checkout. Original CAD, material constants, damping, two-way attachment,
controller and motor parameters, 600 Hz maximum physics step, 200 Hz control,
30 Hz references, 30 ms delay, 100 Hz output, convergence checks, integration
history, and constraint criteria remain unchanged. This checkout's active
physics configuration remains ten edges. No solver optimization is introduced.
The shareable project still has its existing fourteen root files; temporary QA
code and results are retained under this checkout's `tmp/` directory.

Both meshes passed Hover, Back-and-forth, Oscillation, and Figure-eight with
30 simulated seconds per run, one warm-up plus three measured fresh-state runs
per case: 32 completed runs in total. Measured final checkpoints are bitwise
identical within each mesh/preset. For 20 edges, maximum output-sampled length
error across the measured cases is 3.61e-16 m and axial-velocity residual is
4.84e-15 m/s, below the existing 1e-10 m / 1e-9 m/s criteria. No tolerance,
iteration cap, timestep, gain, or motion condition was relaxed. Median advance
real-time factors on the Ryzen 9950X workstation are:

| Preset | 10 edges | 20 edges |
| --- | ---: | ---: |
| Hover | 33.97x | 9.56x |
| Back-and-forth | 7.83x | 2.07x |
| Oscillation | 6.74x | 1.77x |
| Figure-eight | 5.89x | 1.52x |

These are `Flight::step` timings including source/controller/motors/physics;
initialization, rendering, diagnostics, and output are excluded. Cases were run
serially with the same compiler/options and protocol. The workstation was not
isolated or frequency pinned; timing is local evidence, not a general performance
guarantee. Refinement increases mean step cost by about 3.55–3.87x here. See
[per-case benchmarks and summary](../tmp/mesh20-20261025/benchmark-summary.json)
and [source hashes and measurement context](../tmp/mesh20-20261025/benchmark-context.json).

A separate 12 s / 1201-sample comparison uses corresponding physical markers
and equal output timestamps, because the two state vectors have different
sizes. Oscillation tip-position RMS / maximum difference is 0.02570 / 0.08471 m;
Figure-eight is 0.48802 / 1.19950 m. Maximum drone-position difference is 0.01850 m
and 0.10765 m respectively. Thus the strong Figure-eight response is substantially
mesh-sensitive despite both simulations satisfying the constraint checks.
These two meshes do not establish spatial or temporal convergence, parameter
calibration at 20 edges, improved physical fidelity, or agreement with hardware.
The retained 10-edge dynamic physical states remain bitwise identical to the
previous implementation over 12 s; the initial 10-edge raw trace is also bitwise
unchanged. A 1 s / 101-sample 20-edge raw-wrench run matches the original native
binding's full checkpoints exactly, with the same subdivision supplied in memory;
no deployment configuration was edited. Missing, unsupported, fractional, and
nonnumeric link options were rejected. See
[regression and mesh-response evidence](../tmp/mesh20-20261025/validation.json).

The 20-edge Qt/OpenGL viewer passed all four preset selections, the 21-node /
10-marker label, original 82-part CAD rendering, reference visibility isolation,
single steps, 850 steps for each dynamic preset, checkpoint agreement with an
independent uninterrupted C++ flight, continuous playback beyond one second,
pause/reset, 2x pacing controls, small-window access, and numerical-failure
rollback/reset recovery. The test harness filtered desktop input events only
for isolation (zero events filtered); production controls retain their behavior.
Renderer was NVIDIA RTX4080. See
[viewer validation](../tmp/mesh20-20261025/viewer-validation.json) and
[Figure-eight preview](../tmp/mesh20-20261025/figure-eight-viewer.png).
The deployed-source/build verification is retained in
[deployment record](../tmp/mesh20-20261025/deployment.json).
Last reviewed 2026-10-25.

### 2026-10-25 — Historical Stage4 tangent review

Applied the unchanged offline tangent/bending-region/speed diagnostics to the packaged final Stage4 replay from `20261007-034135-720530`, retaining full recovery and a presentation crop through saved recovery start. The [review](../tmp/stage4-tangent-review-20261025/review.md) records the near-strike doubling-back/opening sequence and association limitations, original source/configuration hashes and historical workspace breaches. An initial independently recomputed density-prefix equality check failed at 8.88e-16; that failure is retained. Presentation now slices the full computed arrays and retains exact equality checks rather than relaxing a tolerance. No production diagnostic formulas, physics, reward, training or accepted-policy status changed. Last reviewed 2026-10-25.

### 2026-10-25 — User-selected visual fold reference

Retained the user-selected old Stage4 interval (2.20–2.58 s) in a [seven-case diagnostic reference set](../tmp/fold-reference-set-20261025/reference-set.json), with explicit visual/constructed/unresolved labels and source hashes. Added one offline comparison script and generated a figure/local media page using unchanged diagnostic arrays. Control identities and source preservation were checked; the figure was inspected. No detector accuracy, reward, threshold, training launch or policy acceptance is claimed. The formation → propagation → opening objective remains a proposal awaiting a validated measurement and approved implementation contract. Last reviewed 2026-10-25.


### 2026-10-25: organized standalone source and slower figure-eight

Implemented the user-approved sibling-project layout:

```text
AeroWhip_simulation/
  src/simulation/    engine, material frames, motors, runtime, model defaults
  src/controller/    geometric controller and nominal attitude-rate preparation
  src/trajectories/  trajectory references and sampling
  src/app/           main program, flight composition, viewer
  assets/drone.glb
  Makefile
  run.sh
  build/             generated executables and copied display asset
```

The eight native files moved byte-for-byte into `src/simulation/`. Physical/model
and actuator defaults moved from the old combined `flight.cpp` into
`src/simulation/model.cpp`. The existing controller law moved unchanged into
`src/controller/geometric.cpp`; nominal reference body-rate preparation shares
its existing attitude calculation without adding a new controller. Trajectory
references moved into `src/trajectories/trajectory.cpp`; `src/app/flight.cpp` now
only composes initial conditions, source events, controller and runtime. Main and
viewer moved into `src/app/`, and the unchanged CAD asset moved into `assets/`.
Old flat source copies were removed after preserving their hashes and a complete
snapshot under [reorganization evidence](../tmp/organized-simulation-20261025/).
Existing IDE/virtual-environment files were preserved. No ROS adapter, framework,
training code, or new documentation system was added to the standalone project.

The executable root [run.sh](../../AeroWhip_simulation/run.sh) resolves its own
location, runs Make incrementally, and uses `exec` to forward arguments/signals.
Viewer is the default; headless, trace, benchmark, raw and help options select the
Qt-free binary. Build output goes to stderr so trace and JSON stdout remain
parseable. Make builds all relocated translation units with the original
`-O3 -fno-fast-math -ffp-contract=off` flags and copies the original asset next to
the viewer executable. Direct `make`, `make headless`, and executable invocations
remain available.

Before changing motion speed, the refactored executable retained the original
three-second Figure-eight. For each of four presets and both 10/20-edge meshes,
12 s / 1201-sample full length-prefixed runtime checkpoint traces matched the
previous executable bit-for-bit (eight comparisons). This checks physical state,
integrator history, motors, command queues, and clocks for those runs. See
[refactor validation](../tmp/organized-simulation-20261025/refactor-validation.json).

The separate approved trajectory change increases Figure-eight's steady period
from 3 to 6 s; X/Y amplitudes remain +/-0.5 / +/-0.3 m, altitude 2 m, yaw zero,
and the original 4 s frequency startup is retained. After startup, the reference
velocity amplitude halves and acceleration amplitude becomes one quarter; the
analytic body-rate feedforward is recalculated consistently. Viewer tooltip and
reference-path generation use the new period. Oscillation remains 2 s,
Back-and-forth 4 s, and Hover stationary. The 20-edge default, ten markers,
physical parameters, controller gains, motors, all clocks, convergence gates and
constraint criteria remain unchanged. Earlier three-second Figure-eight timing
and mesh-sensitivity results remain historical evidence for that earlier preset;
they are not measurements of the six-second version.

The new references passed 254 packets against 55-digit independent position
differentiation plus the original controller's body-rate preparation, using the
20-edge model: maximum full-packet difference 1.45e-14 at the existing 1e-12
absolute criterion. Both dynamic presets start at rest and steady references
repeat after their declared period. Controller function text and all eight
native source files are unchanged. See
[reference validation](../tmp/organized-simulation-20261025/reference-validation.json).

The launcher passed Bash syntax/help checks, clean JSON benchmark stdout,
bitwise trace forwarding with `--links 10`, and a clean headless build from
`/tmp` with spaces in its project path while `pkg-config` was deliberately
unavailable. Both 10/20-edge six-second Figure-eight flights completed 30 simulated
seconds through the launcher with the existing numerical diagnostics enabled.
Output-sampled maximum cable-length error was 2.23e-16 m or less; tracking RMSE
was 0.03958 / 0.03879 m respectively. These are numerical checks, not hardware or
mesh-convergence validation. No new controlled performance benchmark was run;
wall times in the smoke logs include concurrent QA activity. See
[launcher checks](../tmp/organized-simulation-20261025/launcher-validation.json).

Qt/OpenGL validation passed original 82-part asset rendering, all four preset
selections, the six-second tooltip, reference visibility isolation, single step,
8.5 s Oscillation and 17 s Figure-eight (including two full steady cycles), exact
GUI/uninterrupted-flight checkpoints, continuous timed playback, pause/reset,
2x pacing, small-window controls, and numerical-failure rollback/recovery. Desktop
input filtering was confined to the test harness. See
[viewer checks](../tmp/organized-simulation-20261025/viewer-validation.json).
[Deployment verification](../tmp/organized-simulation-20261025/deployment.json)
records the published sources/binaries and both builds.

Read-only inspection confirmed the existing Crazyswarm `docker/start.sh
--custom-sim` dispatches to `AeroWhip_deploy/run_custom_simulation.py`, whose
command/feedback adapter is separate from these standalone C++ sources. The
standalone has no ROS/CRTP/radio connection and does not run the onboard firmware
controller. Reusing its physics as that backend is possible through an explicit
adapter; no such adapter or firmware/hardware test was implemented here.
Last reviewed 2026-10-25.

### 2026-10-25 — Agreed Stage1 horizontal-outreach objective

Recorded the user's approved qualitative refinement: form, propagate outward and open a fold into horizontal tip-to-attachment outreach, with no preferred azimuth. Preparatory motion remains unrestricted in direction; outreach is associated with the same opening event. Clarified that the old Stage4 replay is a visual fold reference rather than an adequate-outreach reference. Replaced the superseded top-level combined-stage description and kept numerical rewards, event rules, containment and training status explicitly unapproved/unimplemented. Documentation only; checked the edited text and whitespace. No code, presets, simulation, evaluation runs or training changed.

### 2026-10-25 — Proposed horizontal-fold reward and termination contract

Saved [proposal v1](../tmp/stage1-horizontal-fold-contract-20261025/proposal.md) at the user's request, with explicit experimental event rules, same-event horizontal reach, terminal precedence, seven-second deadline, time cost, proposed hard containment and terminal-zero shaping. Arithmetic examples were checked and retained separately; no full detector, new physical simulation, training code or PPO run was executed. Numerical research decisions and the containment change await explicit approval. This is a design record, not an implementation claim. Last reviewed 2026-10-25.

### 2026-10-25 — Horizontal-fold contract and root-bend audit

Added [offline audit and evidence](../tmp/stage1-contract-root-audit-20261025/review.md) at the user's request. Distinguished the prescribed attachment tangent/first-edge angle from the first interior joint angle; measured straight hover and nonzero moving-root bends from archived physical records, and inspected the sibling standalone's point-clamp model and node-based renderer. Compared four retained ten/twenty-edge rollouts without rerunning physics. An offline interpretation of the proposed event rule exposes opening confirmation before increasing horizontal reach; the old Stage4 geometry is not task success because it breaches the proposed workspace earlier. Five controls do not complete. Event-prefix, coherence, reward-algebra and recorded-coordinate checks passed as detailed in the report; native task integration, contained feasibility, root-condition calibration and learnability remain untested. Original v1 formulas are preserved with audit status. No production physics/reward/configuration, sibling source, worker or training launch changed. Last reviewed 2026-10-25.

### 2026-10-25 — Clarified project versus standalone mesh scope

At the user's correction, explicitly retained 10 links / 11 nodes for this project's physics, training and fold evaluation. The 20-link simulation is a separate standalone project and is outside this task's scope. Corrected the audit interpretation without deleting its evidence. Read and asserted the active physics configuration has ten marker intervals with one segment each; no configuration, code, physics or training changed. Documentation only; no simulation rerun.

### 2026-10-25 — Outward velocity versus maximum separation

Analysed saved ten-link Stage4 node velocities at 2.57–2.63 s in a new offline script and retained source hash/results. Distinguished actual world outward tip motion from increasing attachment-relative reach; at 2.62 s the former is inward despite increasing separation. Recorded the user's requested outward-motion qualification and proposed checking both radial components during the same opening event. Positive components alone do not prove predominantly horizontal motion or adequate strike speed. No reward formula, tolerance, event boundary, production code or training changed. The script ran successfully in the project virtual environment; the initial system-Python attempt lacked NumPy and performed no analysis. No new simulation was run.


### 2026-10-25: editable preset trajectory preview and Apply workflow

Implemented the approved first stage of the trajectory editor in the standalone
C++ viewer. The custom spline editor and polynomial export remain future work.
No new production files, dependencies, configuration-file format, or ROS adapter
were added. Six existing files changed: `src/trajectories/trajectory.hpp/.cpp`,
`src/app/flight.hpp/.cpp`, `src/app/main.cpp`, and `src/app/viewer.cpp`. Physical
model/defaults, eight native engine/runtime files, geometric controller, original
CAD asset, Makefile and launcher script are unchanged. The standalone default
remains twenty edges; deployment/training stays at ten.

`TrajectorySettings` is the common C++ configuration for preset, amplitudes,
horizontal center, altitude, period, and startup. Existing preset-only sampler
calls are retained as wrappers around the unchanged defaults. The analytic
sampler still computes P/V/A and nominal body-rate feedforward from jerk; no
finite-difference derivative or new control law is introduced. Preview geometry,
command estimates, scheduled 30 Hz references, live targets, and tracking
metrics all use this sampler. The 200 Hz controller, 30 ms delay, 100 Hz output,
600 Hz maximum physics step, integrator history and convergence/constraint gates
are unchanged. Back-and-forth retains its existing C2 joins and one-sided jerk
convention; dynamic presets retain the quintic frequency startup with editable
positive duration.

The existing inspector now contains seven numeric fields. Controls that do not
apply to a preset are disabled. Changes remain a draft while the active flight
can continue. The active path/target stay blue; unapplied draft geometry is amber
with a visible legend and status. Frame paths fits both paths. Degenerate zero-
amplitude cases render as a point or a single line. Sampled speed/acceleration
estimates cover 361 startup samples plus 721 samples in a steady cycle and are
labeled as estimates, not certified bounds. The editor uses simulation seconds;
the existing playback selector affects wall-clock pacing only.

Apply & restart validates and constructs a fresh native runtime at the new
trajectory's first position, with zero initial velocity/rotation rate, identity
attitude (fixed zero yaw), and an initial stationary PVA hold. It clears command
queues, delayed holds, motors, scheduler clocks and BDF2 history by using the
original constructor. Construction occurs before replacing the accepted runtime,
so a rejected reset leaves it intact. The new flight is paused at zero; Run starts
it. Revert restores the accepted settings; Defaults fills the selected preset
without applying it. Reset restarts the accepted configuration and keeps any
draft. The app-level C++ runtime is now accessed through `Flight::runtime()`;
`Flight::settings()` exposes the accepted settings by const reference. The native
`Runtime` implementation and checkpoint format did not change.

Copy command produces a compact, round-trip numeric command with the selected
preset, relevant fields, and link count. The same options work in GUI, headless,
trace and benchmark modes, including option ordering independent of preset
selection. Example:

```bash
./run.sh --trajectory figure-eight --amplitude-x 0.65 --amplitude-y 0.4 \
  --center-x 0.15 --center-y -0.2 --altitude 2.3 --period 8 --startup 3
```

Benchmark workload metadata now retains amplitude, center, altitude, period and
startup. Raw mode remains a separate direct-wrench diagnostic and rejects
trajectory overrides. No editable yaw, new preset, controller retuning, spline
construction, save/load file format, or hardware connection was included.

Validation retained in [editor evidence](../tmp/trajectory-editor-20261025/):

- An initial default Back-and-forth comparison failed because the first version
  initialized its stationary delayed hold with nonzero jerk-derived body rates.
  The original zero-rate hold was restored; no tolerance or criterion changed.
  The [failure summary](../tmp/trajectory-editor-20261025/initial-hold-regression-failure.json)
  is retained. Final full checkpoint traces for all four presets at both meshes
  match the preceding executable bit-for-bit over 12 s / 1201 samples per case:
  [default regression](../tmp/trajectory-editor-20261025/refactor-validation.json).
- Independent 55-digit position differentiation plus the original controller's
  rate preparation passed 869 packets across custom offsets, altitudes, sizes,
  periods, startups, and one-axis/zero-amplitude cases, with maximum full-packet
  error 6.67e-15 at the existing 1e-12 absolute criterion. Classical third-
  derivative comparisons exclude Back-and-forth's C2 join instants; unchanged
  default traces cover its implemented one-sided convention. See
  [reference checks](../tmp/trajectory-editor-20261025/reference-validation.json).
- Both meshes completed a 30 s configured Figure-eight with amplitudes 0.65/0.4 m,
  center (0.15,-0.2) m, altitude 2.3 m, period 8 s and startup 3 s. Tracking RMSE
  was 0.03847 / 0.03798 m for 10/20 edges; output-sampled maximum length error was
  below 4.45e-16 m. Existing length and axial-velocity criteria remained enabled.
  Raw traces at both meshes remained bitwise identical. Nine invalid CLI cases
  were rejected; parameterized benchmark metadata/repeat checks and launcher
  trace stdout from `/tmp` passed. These runs are numerical checks, not new
  controlled performance measurements or hardware/mesh-convergence evidence. See
  [CLI checks](../tmp/trajectory-editor-20261025/cli-validation.json).
- Native Qt/OpenGL tests passed all seven fields, active/draft separation,
  unchanged paused checkpoints during edits, unchanged active flight while
  drafting during playback, exact fresh-runtime reset on Apply, 1200 configured
  GUI steps matching an uninterrupted Flight, and 21 full trace samples matching
  the copied command's terminal execution. Revert, Defaults, invalid drafts,
  all presets, disabled irrelevant controls, zero-amplitude rendering, framing,
  small-window scrolling, and numerical-failure rollback/recovery passed. The
  original 82-part CAD rendered on NVIDIA RTX4080. Desktop input filtering was
  limited to the harness. See
  [viewer checks](../tmp/trajectory-editor-20261025/viewer-validation.json) and
  [draft preview](../tmp/trajectory-editor-20261025/trajectory-editor-draft.png).

The [deployment record](../tmp/trajectory-editor-20261025/deployment.json) retains
published source/binary hashes and both build results. No real-flight, firmware,
new model calibration, training, or physical-accuracy validation was performed.
Last reviewed 2026-10-25.

### 2026-10-25 — Complete proposed Stage1 outward-fold contract v2

Prepared the [complete numerical contract and preflight scope](../tmp/stage1-horizontal-fold-contract-v2-20261025/contract.md): buffered opening observations, same-sample reach and both-frame outward velocity,45-degree output cones,0.05m/s radial floors, three-observation phase closure, unchanged proposed reward scale/time/failure values, explicit hard containment and independent evaluation. The project remains ten links. Added an offline output-phase interpreter;21 named descriptor/arithmetic/replay checks passed, including eight causal prefixes. Source inspection identified legacy timing/history assumptions requiring explicit integration work. No physical simulation, production task/configuration change, native integration, PPO run or training launch occurred. Numerical research decisions remain awaiting approval; implementation/preflight is the next proposed scope, with training held until its review. Last reviewed2026-10-25.

### 2026-10-25: approved horizontal-fold v2 implementation and preflight

Added an isolated streaming geometry/event task, ten-link seven-second adapter,
first-substep containment/failure telemetry, explicit701-frame history and16
critic event features, task configuration checks, fresh-actor protection and
all-outcome fold-specific reporting. Added budget-normalized exponential entropy
with exponent8 and the user-authorized200-collection preset. UI/snapshot labels
distinguish the new objective from target-hit and legacy reward accounting.
CPU/regression and native CPU/GPU evidence are summarized at the top; bounded
contained feasibility and worker lifecycle checks remain in progress. No
production campaign or successful policy is claimed by this implementation note.

### 2026-10-25: explicit direct-launch authorization

The user directed immediate200-collection training, waived further preflight and
the proposed contained-qualified-example launch gate, and requested autonomous
progress without repeated permission questions when going to sleep. This does
not change event formulas, success thresholds, physics or post-training review.
The two completed prescribed-motion searches are retained:128trials each,104/84
valid,69/89formed,46/71propagated,23/43opened,0/0qualified. These are feasibility
searches, not training policies or proof the task is impossible. The first search's
best selected reach was0.63833L; its retained motion plot was inspected. No further
search is running. Training initializes neither from these motions nor from the
preflight PPO agent. Added all-lane cable/velocity/drone traces and reward ledgers
to collection evidence so failures and event accounting can be reconstructed.

### 2026-10-25: fresh200-collection horizontal-fold run launched

Active run: [20261025-122524-250328](../runs/ppo_stage1/20261025-122524-250328),
seed20261025,128lanes,25,600episodes,200collections. Explicit preset
`configs/learning/whip_stage1_ppo_horizontal_fold_v2.json`; global legacy defaults
remain unchanged. Observed collection1 and the round0 checkpoint: no resume or
actor initializer, both optimizer states empty, finite actor, zero completed
episodes/offset, saved entropy budget25,600, execution hashes matching current
sources. Verification: [initial-checkpoint-verification.json](../tmp/stage1-horizontal-fold-implementation-20261025/initial-checkpoint-verification.json).
The first process launch resolved the virtualenv Python symlink to system Python
and failed on importing numpy before initialization. It was corrected by retaining
the virtualenv executable path; the original log is preserved. The successful
worker is PID230775. No duplicate training worker or active feasibility search
was observed at launch. Compilation and `git diff --check` passed. No completed
collection or learned success is asserted by the round0 verification.

A10-minute thread heartbeat `monitor-stage1-horizontal-fold` was created ACTIVE
after the environment permission change, for monitoring and final/best independent
review. It must remain quiet for unchanged state, preserve pinned execution sources,
stop after review, and never automatically launchStage2 or extend the budget.
The earlier automation attempt failed under the former permission policy; that
failure is superseded by the confirmed successful creation, not hidden.

Exact implementation changes and rationale:

- `tasks/whip/fold_geometry.py`: ported the reviewed Gaussian-density basin/core
  extraction into production, eliminating temporary-script imports. It treats
  bending as material-coordinate regions rather than a jumping single-node peak.
- `tasks/whip/horizontal_fold.py`: added causal identity tracking, coherent-fold
  formation, fixed-support propagation/opening, buffered same-sample outward
  reach selection, first-output termination, qualification and a telescoping
  reward ledger. This prevents unrelated geometry/speed maxima, later swings or
  temporary guidance from adding event credit. Thresholds remain the approved
  experimental contract; no feasibility-search result changed them.
- `simulation/tensor_model.py`: added an optional read-only native-substep
  observer, including actuator/controller state at the substep timestamp. This
  exposes the first constraint violation without changing the solver, forces or
  step sizes. CPU observer/no-observer state equality was tested bit-for-bit.
- `tasks/whip/horizontal_fold_environment.py`: added the isolated7s adapter with
  first-failure latching, per-lane terminal masks, actual terminal times,701-state
  replay histories, explicit event rewards and fold-specific diagnostics. This
  removes the legacy4/4.5s scoring assumptions without changing old tasks.
- `tasks/whip/sac_environment.py`: added factory dispatch,701-frame policy history
  and16 extra critic event-state features only for the new task. Direct use of
  the legacy base adapter for this task is rejected.
- `tasks/whip/learned_recovery.py`: registers non-scoring task metadata for the
  new adapter; attempting legacy full-horizon scoring raises instead of silently
  returning an unrelated reward. No recovery behavior was added.
- `tasks/whip/stage1_configuration.py` and `training/configuration.py`: validate
  the exact new contract, ten links,7s/zero posthold, fixed control/safety settings
  and no old return transform or extra costs. Snapshots label the new reward
  delivery `terminal_event_zero_potential_v2`. Old configurations remain readable.
- `configs/tasks/whip_stage1_horizontal_fold_v2.json`: explicit task contract.
  `configs/learning/whip_stage1_ppo_horizontal_fold_v2.json`: separate fresh seed,
  128lanes/25,600episodes,701-frame task and requested entropy schedule.
- `training/ppo.py`: accepts the explicit701-frame layout and adds
  `exponential_budget_v1`, normalized to the saved planned episode budget. Exact
  resume preserves both budget and elapsed clock. Existing fixed-episode and
  linear entropy modes remain unchanged.
- `training/artifacts.py`: rejects actor initialization for this fresh Stage1
  mode so an old policy cannot silently enter the new campaign.
- `training/worker.py`: dispatches both stochastic collection and deterministic
  evaluation to the new adapter; validates its distinct reward snapshot marker;
  retains all-lane physical cable/velocity/drone traces, masks, terminal times,
  event records and reward ledgers, plus representative replay files. Reporting
  calls the metric qualified outreach, not target-hit success.
- `training/ui/window.py`: explains the actual horizontal-fold objective, hard
  containment, return formulas and budget-scaled entropy when this preset is
  selected, avoiding the legacy target-hit/soft-workspace labels. GUI behavior
  itself was not interactively tested in this turn.
- `tests/test_horizontal_fold.py` and `tests/test_horizontal_fold_events.py`:
  added geometry/coherence, output-frame, ledger, timeout, native observer,
  first-substep boundary, per-lane freezing, history/clock and event-state
  counterexamples. Synthetic descriptor sequences test logic, not physical
  attainability. Combined with retained first-hit/entropy/AR1 regressions,
  42tests passed and1CUDA-only test was skipped; retained logs include the earlier
  test-harness failures and their corrections. GPU8-lane PPO execution was a
  separate check, not a pass of the historical strict re-forward density test.

The new run is an authorized learning experiment despite unproven qualified-task
feasibility. Neither code checks, finite optimization nor completed training can
replace the unchanged independent event/motion evaluation.

Launch follow-through: collection1 completed its PPO update and deterministic
evaluation, and collection2 was observed collecting with128episodes completed.
The confirmed round0 initialization evidence remains separate from subsequent
learning. No scientific success conclusion is drawn from this early progress.

### 2026-10-25 16:36UTC: horizontal-fold monitoring

Worker230775 is active;10/200collections completed. The execution-source hashes
match and entropy0.013736080874677148 equals the approved formula at1,280episodes.
Recent collections6–10 have finite optimizer KL below0.02 and reward-sum errors
below9e−7 after float32 storage. All640 recent attempts ended at workspace
violations; none formed a fold or completed an event. These early failures are
retained, not interpreted as proof of eventual success or impossibility. No
source, reward, physics, criteria, budget or exploration change was made and no
competing GPU evaluation was started. [Monitoring evidence](../tmp/stage1-horizontal-fold-implementation-20261025/monitor-20261025-1636Z.json).

### 2026-10-25 16:46UTC: monitoring and post-training evaluator preparation

34/200collections completed; source hashes still match and entropy
0.005876554762117562 exactly matches the saved25,600-episode formula. Recent
collections30–34 retain640workspace failures, with zero formed/propagated/opened
folds and zero qualified outputs. Conditional KL remains finite below0.02 and
float32 reward-ledger errors below9.4e−7. Mean return around−10.059 reflects
earlier failure times, not improved whipping. Conditional on failing, the
approved−10−0.1T formula ranks shorter failures higher; this is a potential
learning failure mode, not evidence of an implementation mismatch. The authorized
200-collection run continues unchanged; no success or convergence claim is made.
[Monitoring evidence](../tmp/stage1-horizontal-fold-implementation-20261025/monitor-20261025-1646Z.json).

Prepared the standalone post-training helper
[evaluate_frozen.py](../tmp/stage1-horizontal-fold-implementation-20261025/evaluate_frozen.py).
It refuses an active/incomplete campaign, verifies checkpoint source hashes,
evaluates the registered three128-lane seeds and one deterministic attempt,
retains all-lane traces and all-started denominators, checks the actor and entropy
clock remain frozen, and separates numeric gates from required motion review.
It never declares approval automatically. This uses the same native tensor
substep adapter with independent action noise, not a second physical solver.
Five pure reporting counterexamples passed, including failure denominators,
deterministic invalidity and the unchanged0.75reach gate; a mixed timeout/failure
cohort also passed. No GPU evaluation was launched. Production execution sources
remain untouched. After completion, invoke this helper sequentially for
`latest.pt` and `best.pt` with separate new output folders, then inspect retained
representative motions and event evidence before recording the final review.

### 2026-10-25 16:56UTC: material learning failure pattern identified

At collection70, all8,960training attempts so far were invalid; zero qualified
outreach attempts were recorded. Recent collections66–70 each had zero formed,
propagated or opened folds and mean terminal times approximately0.57s. Numerical
execution remains finite, source hashes unchanged, and entropy0.0021553911898791415
matches the approved clock. [Monitoring record](../tmp/stage1-horizontal-fold-implementation-20261025/monitor-20261025-1656Z.json).

Read-only review of saved deterministic trajectories confirms strong downward
commands: around−3.86m/s² mean Z acceleration, crossing the lower1.5m workspace
boundary at about0.56s in the recent replays. The first deterministic replay
failed at the upper boundary after2.1683s. Thus the improved negative return is
not fold progress: the approved failure formula gives−10.056 for the later
short failures versus−10.2168 for the initial slower failure. This behavior is
consistent with a local incentive to fail sooner when all sampled attempts fail;
it does not prove that time cost alone caused the policy collapse. A valid
timeout would still outrank either failure under the approved objective.
[Saved-motion diagnosis](../tmp/stage1-horizontal-fold-implementation-20261025/early-failure-diagnosis-1656Z.json).

The user-authorized200-collection experiment continues without changing its
reward, criteria, exploration schedule or physics. This material failure pattern
is being reported once rather than silently describing numerical execution as
learning success. Frozen final/best review remains scheduled; no automatic
extension, policy promotion or Stage2 transition is permitted.

### 2026-10-25: live UI launch attempted for the existing worker

The user requested the training UI. Added the isolated launcher
`tmp/stage1-horizontal-fold-implementation-20261025/open_live_ui.py`, which opens
the existing TrainingWindow on the current run, polls its saved live events and
provides the existing motion/replay views. A monitor-only subclass maps the new
formation/propagation/opening/qualified-outreach and reward fields into existing
plots. Launch/configuration/worker-control buttons are disabled to prevent a
second run or falsely implying this window owns the detached worker. Closing the
monitor does not stop training. Production/pinned sources, training state and
reward definitions were not edited. The original worker remains PID230775.

### 2026-10-25: user-requested stop

Created the cooperative stop.request for the current run. Confirmed summary
status stopped and worker-lock removal; the latest completed checkpoint and all
prior evidence remain preserved. Deleted heartbeat monitor-stage1-horizontal-fold.
No further training, automatic resume, or GPU evaluation was launched. The UI
launch attempted immediately beforehand was not verified visible before the
user requested stopping; it must not be reported as a successfully opened UI.

### 2026-10-25: retained-progress Stage2 design, no production change

Saved the [proposed reward and termination rules](../tmp/stage2-retained-progress-design-20261025/proposal.md)
and a standalone arithmetic reference. The design retains useful partial progress
through unsuccessful endings while bounding repeat/retry credit, removes explicit
time pressure, and uses the same unsuccessful-outcome adjustment for completed
misses and timeouts. Its propagation guide reaches full credit only when both
existing propagation requirements hold; its opening guide uses fixed150/60degree
references instead of an inflatable running peak. Proposed target proximity,
horizontal reach and direction factors are evaluated at one eligible sample of
the same opening. Useful hit verification, including a proposed1m/sforward speed
floor, is separate and has not been adopted in code.

[36 constructed checks](../tmp/stage2-retained-progress-design-20261025/arithmetic-checks.json)
passed on hand-supplied descriptors and exact reward algebra. An initial
floating-point zero assertion failed from cancellation and is preserved; exact
rational arithmetic now checks the reference ledger without weakening that
assertion. Streaming detection, native execution, production float accounting,
learning and physical feasibility were not tested in this design step. The
proposal documents partial-progress plateaus and the distinction between
per-trajectory outcome ordering and maximizing expected success probability.
Only design documentation and its standalone arithmetic script changed; no
production reward, source, preset, physics, worker or experimental criterion was
modified. Training remains stopped at the user's request.


### 2026-10-25: aggressive oscillation self-overlap and folding failure

Investigated the reported pinched cable while the drone and simulation clock
continue moving. Reproduction uses the standalone executable, 20 edges, default
0.5 m X amplitude/2 m altitude/4 s startup and a changed 1.5 s Oscillation cycle.
No production source, model parameter, timestep, tolerance, or controller setting
was changed. UI redesign remains a separate proposed task; no UI refresh or
self-contact fix is claimed here.

The requested 30 s trace exits nonzero after the last accepted 20.85 s with
`Projected implicit endpoint iteration failed after 20 iterations`. It retains
2086 output samples. The minimum distance between nonadjacent finite segments
is approximately 1.98e-15 m at 4.77 s (segments 0/17), confirming a centerline
intersection. Some nonadjacent segment pair remains below the nominal 3.5 mm
cable diameter at every output sample from 4.77 s through termination. Maximum
adjacent turning angle reaches 154.17 degrees at node 7; the minimum interior
`1 + tangent dot tangent` is 0.09994, above the explicit antiparallel rejection
threshold. This is an iteration-convergence failure, not a demonstrated trigger
of the antiparallel exception. Output-sampled length error remains below
4.03e-16 m, illustrating that inextensibility acceptance does not ensure
self-contact validity. See [reproduction](../tmp/cable-fold-review-20261025/oscillation-1p5-reproduction.json),
[observations](../tmp/cable-fold-review-20261025/oscillation-1p5-20links.npz),
and [attachment-aligned XZ projection](../tmp/cable-fold-review-20261025/fold-reproduction.png).
The projection does not itself establish 3D contact; the reported distance uses
3D segments. World Y displacement reaches about 8.85 mm, so exact planarity is
not assumed.

A diagnostic harness linked the unchanged production sources and changed only
its local model's maximum step from 1/600 to 1/1200 s. It completes 30 s, while
still self-intersecting/overlapping: first separation below 3.5 mm is again at
4.77 s, with 2103 such output samples. Maximum adjacent turn is 155.96 degrees.
The two step sizes' tip predictions differ by up to 0.56 mm through 4 s, 19.1 mm
through 5 s, 79.8 mm through 6 s and 1.31 m over their shared later horizon.
These two runs do not establish convergence or distinguish truncation error from
amplification by sensitive folded dynamics. Smaller steps in this trial improve
completion but do not supply self-contact. See
[step-size diagnostic](../tmp/cable-fold-review-20261025/timestep-1200-summary.json).

Source inspection confirms no nonadjacent segment collision detection, contact
force, friction, or adhesion model. The rendered cable radius is 1.8 mm and
marker radius 6 mm, while physical markers enter as point masses; display
spheres are not collision bodies. Two moving marker centers reach 3.75 mm
separation at 12.77 s, so the 12 mm display spheres overlap. Their display size
is not a verified physical marker contact radius. See
[display overlap](../tmp/cable-fold-review-20261025/marker-display-overlap.json).
The renderer's averaged tube tangent also degenerates at a fully reversing
joint; that code-level possibility is not established as the cause of this
observed pinching. Likewise the evidence does not isolate whether persistent
folding is dominated by material damping, elastic curvature, or integration.

Production source hashes remain identical to the deployed trajectory-editor
snapshot: [verification](../tmp/cable-fold-review-20261025/production-source-verification.json).
Retained earlier standard presets did not reproduce the same close fold in their
saved 12 s windows. Adding finite-thickness self-contact or modifying the solver
requires a separately approved physical/numerical change; neither is implemented
by this diagnostic. Parameter refitting alone cannot add missing contact forces.
No real-world contact validation was performed. Last reviewed 2026-10-25.

### 2026-10-25 17:38UTC: approved retained-progress Stage2 implemented and launched

Implemented the preceding approved contract and started a fresh200×128 CUDA PPO
campaign at [runs/ppo_stage2/20261025-133812-161285](../runs/ppo_stage2/20261025-133812-161285).
The [detailed report](../tmp/stage2-retained-progress-implementation-20261025/implementation-report.md)
lists every changed file, purpose, exact outcome rules, launch details and limits.
The new task retains partial progress, removes time cost, bounds retry credit,
separately verifies same-opening crossings and preserves safety precedence.
Ten-link physics/control are unchanged;44physical/controller/native source hashes
match the stopped source. Its87matching execution files are archived.

Combined regressions:80passed,1CUDA-only test skipped in the sandbox;8constructed
UI-label cases passed. No additional GPU preflight was performed. The actual CUDA
worker completed collection1and its first update with final conditional
KL0.00247258297228404, then continued to collection2. Initial round0checkpoint
confirmed random initialization with no source/resume, empty optimizers,
0completed episodes and original entropy budget25,600. The visible UI is exposed
on the desktop, attached to this worker, with Stop/Save and correctly labelled
replay. Closing the monitor leaves the worker running.

The first128attempts all failed workspace containment, with no formed folds or
hits; mean retained progress0.03852/return−9.80740 are accounting observations,
not evidence of learned success. Float32 stored return-sum error was below9.54e−7.
All started attempts/failures and execution hashes are retained. Historical
strict re-forward-density failure remains a failure. No later-stage transition,
extension, hardware operation or new recurring automation was started.

### 2026-10-25: stopped run and read-only whipping diagnostics page

The user stopped retained-progress Stage2 at53collections/6784episodes. The saved
summary reports `stopped`; latest/best checkpoints and all recorded evidence are
preserved. The latest deterministic evaluation is collection52, not collection53.
The subsequent discussion of continuous horizontal-whipping reward is not an
implemented or approved numerical replacement.

Added the authorized **Whipping diagnostics** tab in the training UI. It displays
individual saved evaluations and sampled training lanes (including failures),
two attachment-centred motion views, existing FoldGeometry bend-density/tracks,
reach/height and world/relative outward-speed plots, recorded event milestones,
and synchronized playback controls. Current reconstructed geometry is explicitly
separated from saved detector outcomes; missing fold records remain unavailable,
and no reward or task rescoring is performed. Normalization uses the selected
run's saved rest lengths. Invalid/padded suffixes are excluded from playback.

Nine focused tests and full-window offscreen integration passed. Desktop opening
and exposure were verified on the new tab, and screenshots inspected. The
collection52 source hash remained unchanged. A failed collection53 lane was also
checked. No worker, simulation, evaluation campaign, reward change or policy
promotion was launched. Detailed file changes, evidence, limits and screenshots:
[implementation report](../tmp/whipping-diagnostics-ui-20261025/implementation-report.md).
Last reviewed2026-10-25.

### 2026-10-25: standalone experimental cable self-contact

Following the reproduced pinched/overlapping cable, the user approved addressing
folding/contact first and confirmed the nominal cable diameter as **3.5 mm**.
This change is confined to the sibling standalone C++ simulator. The original
deployment/training/fitting physics, model artifacts, controller, trajectories,
motor parameters, EI/Cb/GJ/Ct, drag, masses and marker locations were not retuned.
The earlier aggressive-oscillation diagnostic above is the pre-change baseline;
its no-contact/fixed-substep failure is superseded for the standalone executable.
The cosmetic UI redesign and data-based refitting remain deferred.

The new [self_contact.inc](../../AeroWhip_simulation/src/simulation/self_contact.inc)
checks closest points between all nonadjacent finite cable segments. Adjacent
segments sharing a node are excluded. The effective tube radius is 1.75 mm;
the numerical barrier activates within another 0.35 mm, with scale 100,000 N/m
and 1 nm safety clearance. These last three values are numerical choices, not
identified cable material constants. For positive surface gap g below activation
gh, energy is `-k*(g-gh)^2*log(g/gh)`; forces are its conservative gradient.
The model is frictionless and strictly repulsive, with no adhesion. Endpoint
barycentric forces include the attachment's force/body-moment reaction. It is a
custom logarithmic-barrier implementation, not a complete IPC implementation or
a certified contact solver. Physical marker masses remain; their 6 mm rendered
spheres are display geometry, not calibrated collision bodies. Drone-body,
floor/wall contact and contact friction are absent.

Closest-segment calculations use long-double scalar/cross-product determinants
to reduce near-parallel cancellation. A swept-AABB broad phase and conservative
advancement reject tunnelling along linear nodal motion; unresolved sweeps reject
rather than pass. Segment zero includes a conservative chord-deviation allowance
for the attachment's SO(3) rotation. Near contact, relative endpoint travel is
limited to one quarter of the activation width per accepted physical substep,
preventing a collision-free endpoint from silently skipping the contact-force
band. These checks describe the implemented discrete path; they do not prove
accuracy of the unknown continuous physical motion between samples.

[engine.cpp](../../AeroWhip_simulation/src/simulation/engine.cpp) retains the
projected Lie-BDF2 endpoint formulation, 20 iteration cap, 2e-11 relative-change,
1e-10 m length and 1e-9 m/s velocity-constraint tolerances and four position
projections. Positive normal Gauss-Newton stiffness preconditions contact;
acceptance additionally checks the original unpreconditioned force balance so a
stiff preconditioner cannot hide its residual. Feasibility backtracking is bounded
at 24 trials and does not use the damped update as a relaxed convergence metric.
Numerical failures or unresolved contact split a nominal substep into halves,
bounded by depth 12 and 2047 attempts per nominal substep. The original maximum
step remains 1/600 s. Accepted variable-step history is preserved; original
backward-Euler startup/growth-above-two restarts remain. State/history commit
only after the whole advance call succeeds. Controller callback/wrench failures
propagate once without retry, and motor responses use the original local elapsed
time through retries. Controller, reference, delay and output event clocks remain.

[engine.hpp](../../AeroWhip_simulation/src/simulation/engine.hpp) adds three
contact parameters and a surface-gap diagnostic; generic parameters default to
contact disabled. [model.cpp](../../AeroWhip_simulation/src/simulation/model.cpp)
enables the nominal contact model for the standalone 10/20-link examples.
[prescribed_material.inc](../../AeroWhip_simulation/src/simulation/prescribed_material.inc)
explicitly rejects contact-enabled prescribed-boundary advance/acceleration;
that solver has not been extended to contact. Explicit contact-disabled
prescribed-boundary use still works. This is a coupled-example feature, not a
silently promoted replacement for the original fitting solver.

[main.cpp](../../AeroWhip_simulation/src/app/main.cpp) adds `--no-self-contact`,
contact metadata and minimum output-sampled surface gap to benchmark JSON.
Disabled contact reports null minimum gap. The viewer adds an experimental
diameter/frictionless label and geometry explanation; Copy command preserves
the disabled-contact option. No general UI redesign, framework or dependency was
added. Six existing production files changed and one simulation include was added;
the other 15 original project files and CAD asset remain unchanged. The root
launcher and Makefile are unchanged.

Retained validation is in [folding-contact-20261025](../tmp/folding-contact-20261025/):

- [Repeated 30 s benchmarks](../tmp/folding-contact-20261025/benchmark-summary.json):
  Oscillation, 0.5 m X amplitude, 1.5 s period, 4 s startup, 10 and 20 links;
  one warm-up plus three measured runs each completed, with identical repeated
  endpoint checkpoints. Median terminal real-time factors were 5.118 and 1.134
  respectively on this workstation. These are observed wall times, not an
  isolated speedup comparison. The old 20-link case stopped after 20.85 s.
- [Internal-substep instrumentation](../tmp/folding-contact-20261025/substep-validation.json)
  for the 20-link 30 s case checked 22,748 accepted substeps: minimum physical
  cable-surface gap 0.0939035 mm, smallest accepted step 13.0208 microseconds,
  4748 splits/27,496 attempts. The maximum numerical pair force was 137.233 N;
  no measured impact-force validation is implied. Output-only minimum gap was
  0.281313 mm. Output length/rate errors stayed below 4.03e-16 m/5.02e-15 m/s.
- [Contact unit checks](../tmp/folding-contact-20261025/contact-unit-validation.json)
  passed energy/force finite differences, force/moment balance, rotation and
  tangential-force checks, closest-point cases, tunnelling/sliding/translation,
  root-rotation and contact-band guards. Of 1000 seeded random sweeps, 810
  accepted sweeps were independently densely sampled and 135 conservatively
  rejected; the remaining cases started invalid. This finite sample is not proof.
- [Resume and motor checks](../tmp/folding-contact-20261025/resume-validation.json)
  passed 500 bit-identical resumed contact steps, exact motor endpoint response,
  callback failure without retry/partial commit, explicit prescribed-contact
  rejection and a contact-disabled prescribed step.
- [Default regression](../tmp/folding-contact-20261025/refactor-validation.json):
  all four default presets at 10 and 20 links retained bit-identical full
  checkpoint traces against the pre-change executable over 12 s/1201 samples.
  Raw one-second 10/20-link traces also remained bit-identical. These cases do
  not activate the aggressive contact regime.
- [Viewer checks](../tmp/folding-contact-20261025/viewer-validation.json) passed
  3000 aggressive output steps/30 s with exact terminal-flight state, plus
  1200 custom-trajectory steps, all seven editor fields, draft isolation,
  Apply/Reset/Revert/Defaults, clipboard/CLI parity, small-window operation and
  failure recovery. Original 82-part CAD loaded on NVIDIA RTX4080. Desktop input
  was filtered only in the test harness; production input is unchanged.
- [Independent gap/sensitivity checks](../tmp/folding-contact-20261025/contact-rollout-validation.json)
  passed positive output gaps and original length/velocity gates for six 12 s
  cases spanning 10/20 links, maximum steps 1/600, 1/1200 and 1/2400 s, and
  barrier scale factors 0.1/1/10. Nevertheless nominal-versus-smaller-step tip
  differences reach 1.22–1.26 m over 12 s; barrier-scale variants reach
  1.14–1.45 m. Through 5 s the corresponding differences are about 1.5–2.0 cm
  and 3.5 mm. **Long-horizon timestep/regularization independence is unresolved.**

Earlier failed barrier/solver prototypes and the adaptive-only no-contact
comparison remain in the evidence folder; they were not installed. Passing
nonpenetration, repeatability or solver acceptance does not establish physical
accuracy, mesh convergence, calibrated impact response or real-flight validity.
No new data fit, training, GPU implementation or hardware flight was performed.
The [installation record](../tmp/folding-contact-20261025/deployment.json) verifies
all 22 installed source/asset files against the tested stage and both rebuilt
executables byte-for-byte. An installed-launcher benchmark smoke check passed
from `/tmp`; the actual viewer opened paused at Oscillation/1.5 s on RTX4080.
The [original native-source verification](../tmp/folding-contact-20261025/original-native-verification.json)
confirms all eight original deployment physics/runtime files still match the
pre-change standalone snapshot. No production IDE or environment files changed.
Last reviewed: 2026-10-25.


### 2026-10-25 — continuous horizontal Stage1 restart

Implemented `continuous_horizontal_whip_v1` after the user delegated finalizing
the reward and restarting. Replaced milestone credit/early completion with a
causal integral of recent fold travel and current horizontal, outward,
end-to-end-extending output. Safe horizon7s; failure entire return−10; no event
counts or time/effort/path penalties. The exact
[contract and limitations](../tmp/stage1-continuous-horizontal-20261025/contract.md)
and [file-by-file report](../tmp/stage1-continuous-horizontal-20261025/implementation-report.md)
are retained. Fresh actor/critic run20261025-153330-415723 starts200collections
with128lanes and unchanged exponential entropy. The stopped Stage2 run is preserved.
All47 focused CPU/UI/checkpoint checks passed after fixing a reproduced Qt test-only
teardown crash. Retained recordings were reconstructed without mutation or
success relabeling. Launch verified44 unchanged physics/controller/native hashes,
zero starting clocks/empty optimizers and visible UI. No extra GPU preflight,
new physical validation or claim of learned task success. Last reviewed2026-10-25.

First launch20261025-152903-205562 stopped on a checkpoint text-array serialization
bug after its first collection/update. Its artifacts remain preserved. The
serializer now round-trips Unicode metadata arrays through weights-only checkpoints;
actual failed replay24-array roundtrip passed. Replacement fresh run
20261025-153330-415723 uses the identical reward/physics/preset/seed with a new
zero-offset200collection budget; no learned first-launch state is resumed.

Post-repair execution check: latest checkpoint2 preserves256completed episodes,
275actor/275value updates and original25,600episode entropy budget. Best-checkpoint
Unicode diagnostics restore successfully; pinned source hashes still match.
First collection0/128valid, all failure returns exactly−10, ledger error0. This is
execution/accounting evidence only. Worker and exposed UI remain live at
collection3/200. The user cancelled the proposed64-lane switch before any edit;
128parallel lanes remain unchanged. Evidence:
[tmp/stage1-continuous-horizontal-20261025/serialization-fix-launch/first-collection-check.json](../tmp/stage1-continuous-horizontal-20261025/serialization-fix-launch/first-collection-check.json).

### 2026-10-25 — replace workspace termination with distance cost

User explicitly approved the correction after inspecting the1.028s upper-height
boundary failure. Implemented versioned continuous soft-workspace reward; no
physical model change. All safe lanes can run7s outside the preferred cylinder,
while original physical/numerical guards remain. Final stopped actor and entropy
clock carry forward, critic/optimizers refresh,178collections remain. Exact
formula and tradeoffs are retained in the v2contract above. Software checks and
launch evidence are recorded under tmp/stage1-continuous-soft-workspace-20261025.
Last reviewed2026-10-25.

Soft-workspace continuation20261025-161103-645210 launched with128lanes and178new
collections.52focused tests passed, including full7s constructed boundary crossing
and preserved physical failure checks. Initial production checkpoint confirms
bit-identical actor weights, fresh optimizers/219critic inputs and entropy offset
2816/originalN25,600. All44physical source hashes unchanged. No learned-success
claim. Detailed changed files/reasons and actual tests are retained in the report.

First v2native collection:5/128reached7s,123physical failures,0workspace
terminations. Raw cost/return/ledger reconstruction passed at float32 storage
precision; source hashes unchanged. Surviving motions have severe boundary
excursions (C377.68–4015.02), not successful whipping. Negative-map compression
reduces their return separation; no convergence/behavior approval is claimed.
See tmp/stage1-continuous-soft-workspace-20261025/first-collection-check.json.

### 2026-10-25 — user requested stop of soft-workspace training

Confirmed worker lock removed and summary phase stopped at3new completed
collections/384episodes. Fourth collection interrupted. Existing latest/best
checkpoints and recordings remain. No reward, physics or threshold edits and no
restart. Stop evidence: tmp/stage1-continuous-soft-workspace-20261025/user-stop.json.
Last reviewed2026-10-25.

### 2026-10-25 — direct boundary cost and remaining-time failure correction

User approved replacing compressed/fixed-total accounting with direct Q−C and
a cost proportional to unused horizon on physical/numerical failure. Implemented
v3 with rate1reward unit per remaining second, preserved historical modes and
source runs. Full formula/tradeoffs and file-by-file changes are retained in
tmp/stage1-continuous-distance-20261025.58focused tests passed;12modified
production modules compiled. Native launch/accounting checks follow separately.
No physical validation or learned-success claim. Last reviewed2026-10-25.

V3 launched20261025-163957-732295 with128lanes/175remainingcollections. Initial
production checkpoint confirms exact actor transfer, zero optimizer/update
states, entropy offset3200/N25600; all44physical-source hashes unchanged.
Visible/exposed UI confirmed on Whipping diagnostics. Evidence: launch.json,
initial-checkpoint-check.json and ui-visible.json in the v3 evidence directory.

First v3native collection retained all128attempts:23full7s horizons,105physical
failures,0workspace terminations. Independent distance-cost reconstruction error
<=2.73e−12; plain return/remaining-time failure/float32 ledger checks passed.
Raw whip output remains tiny (max0.011828529), with large boundary costs up to
5351.478729226; this is not successful learned behavior. First PPO cycle made
10actor/625value updates, preserving the completed-episode entropy clock.
See tmp/stage1-continuous-distance-20261025/first-collection-check.json.

First post-update deterministic replay also failed physically at5.546667s,
Q=0, accrued reward boundary C=744.796888075, remaining-time cost1.453333333,
total−746.250221408 before float32 storage. The saved checkpoint has finite
actor/critic parameters,10actor/625critic updates and15replay diagnostic fields.
No behavior pass is claimed. The initial ad hoc replay inspection addressed the
empty PPO replay-buffer field; corrected inspection of best_rollout succeeded
without changing code or tests.

Reporting caveat: the legacy round-log field `acceleration_cost` carries the
environment-returned cost tensor (boundary cost here); no acceleration penalty
is applied. The separate generic workspace diagnostic includes the saved
failed endpoint and can differ from reward-accrued C, which excludes failed
endpoints. Use `components.workspace_cost` for reward accounting. The UI reward
breakdown uses this component. These existing generic labels were not rewritten
while execution sources are pinned by an active worker.

### 2026-10-25 — stop after plateau/failure audit

User approved stopping the v3run and resolving the failure incentive before
relaunch. Verified stopped summary22collections/2816episodes, absent worker/lock,
retained latest/best hashes. Interrupted collection23is not counted as completed.
Recorded observed reward ranking and the finite-failure-cost/unbounded-boundary
cost limitation, distinguishing it from a proof of globally optimal failure.
No reward/source/physics changes or new training/evaluation runs. Existing
recordings/logs inspected; no new simulation/test claim. Last reviewed2026-10-25.

### 2026-10-25 — physical stops removed in versioned Stage1 v4

Implemented user-requested7s continuation through finite tilt/cable-height
violations, with numerical emergency aborts retained and unchanged Q−C. Preserved
physical-invalid diagnostic flags rather than reporting completion as safety.
Added v4preset, explicit replay/UI descriptions and narrow continuation guard.
63software tests passed;44physical-source hashes unchanged; no actual native
physics rollout, training launch or physical-validation claim. Initial test
fixture AttributeError and corrected loader use are retained in the report.
Last reviewed2026-10-25.

### 2026-10-25 — full-horizon v4 continuation launched

User asked to keep going. Launched20261025-173944-562924 with128lanes and153new
collections/19584episodes remaining in the original200collection campaign.
All13actor tensors match stopped v3final22 exactly; fresh critic/optimizers/RNG,
original completed-episode entropy budget25600/offset6016 retained. No reward,
physics or threshold edits during launch. Visible/exposed Whipping diagnostics
UI verified; all44physical source hashes unchanged. Initial checkpoint and
launch evidence retained under tmp/stage1-continuous-full-horizon-20261025.
Native rollout checks follow; no learned-success claim. Last reviewed2026-10-25.

## First production collection under v4

All128started attempts completed7seconds;0numerical aborts and0physical early
terminations. All128violated the old physical envelope and remained labeled
physically invalid. Lowest recorded cable height across lanes was−40.404066m
(no ground-contact response is modeled); maximum tilt57.536200degrees. These
are failed motions, not successful whipping, and the negative-height portion
must not be interpreted as realistic ground-contact dynamics.

Raw output ranged0–0.192816662; boundary costs236.472386–5150.986881; returns
−5150.986816 to−236.471817. Independent saved-tracking-point cost reconstruction
agreed to3.64e−12. Q−C return and float32 reward-ledger checks passed at existing
rtol1e−6/atol1e−6, with maximum absolute errors0.000185648 and0.000488281
respectively. All source hashes remain pinned and unchanged. First-collection
evidence verifies execution/accounting only, not convergence or physical success.
See first-collection-check.json and check_first_collection.py.

### 2026-10-25 —64-lane full-state continuation and shorter budget

User explicitly requested64parallel and keeping151remaining collections to
reduce work. Updated preset to64lanes/9920v4lineage episodes including256already
completed; active run20261025-175422-750718. State-preserving checkpoint migration
records the batching change without relaxing exact-resume guards. Only production
code edit is dynamic UI lane/budget wording. Full actor/critic/optimizers/counters/
RNG checked equal on production resume;13UI/replay tests passed;44physical-source
hashes unchanged. Entropy budget/clock retained rather than reset. Native
progress64active; no behavior-validation claim. Evidence under
tmp/stage1-parallel64-20261025/reduced-budget. Last reviewed2026-10-25.

First resized collection3contains64attempts;64completed7s,physical violations64. Reward formula/ledger checks passed;execution hashes unchanged. Recorded collect-to-evidence wall time102.35s is one observation,not a controlled performance benchmark. Evidence:tmp/stage1-parallel64-20261025/reduced-budget/first64-collection-check.json.


### 2026-10-25 — Resume64-lane Stage1 on user request

Verified previous worker/UI absent and resumed latest saved collection5 using existing exact-resume source checks. No production code, reward, physics, or budget changes. A local monitor wrapper corrects the title to the remaining collection count. Startup/runtime checks are retained with launch evidence; no policy-success claim.


### 2026-10-25 — Approved directed +X reward and fresh restart

Implemented versioned v5 output projections, registered native task/PPO/history/replay routing, exact contract validation, fresh-only initialization guard and direction-correct UI diagnostics. Kept v1–v4 formulas/configurations and all stopped artifacts.56focused scoring/accounting/termination/entropy/UI regressions passed.44physical source hashes unchanged. Fresh64-lane200collection worker/UI launched; runtime startup evidence retained under tmp/stage1-directed-x-v5-20261025. No extra GPU preflight, physical validation or trained-success claim.

V5production startup verified collection1/64active lanes. Saved round0has zero episodes/updates, empty optimizers, null resume/initializer and entropy budget12800/offset0. UI visible/exposed with correct run. First full collection and policy quality not yet evaluated at this startup check.


### 2026-10-25 — Four-second directed Stage1 restart

User approved4s/200collections/restart. Added versioned v6 preset/contract, consistent400sample mean workspace cost,4sremaining numerical failure cost,200decisions/401state replay and encoder layout, horizon-normalized critic features and correct UI labels. Old presets/physical sources preserved.57focused tests passed;14UI/replaychecks additionally passed after final label edit. Fresh64-lane worker/UI launched with12800episode entropy budget; startup verification retained under tmp/stage1-directed-four-second-v6-20261025. No physical validation or policy-success claim.

V6startup verified collection1with64active lanes; retained round0checkpoint has fresh learner state,12800episode entropy budget/zero offset and401sample policy history. UI visible/exposed and correct run. Full first-collection timing and learned performance remain unverified at startup.


### 2026-10-25 — Cleanup, source commit and other-computer handoff

Stopped-state verified; current v6has0completed collections. Corrected default to approved4s preset and pinned legacy test fixtures to intended presets/original handoff budget. Initial570test run had5failures/15errors; corrected run passed570tests with10skips under CPU-only verification.2.66GiB verified duplicate backup/cache cleanup; unique artifacts retained in checksummed release archive. Tracked handoff instructions preserve active contract and source/test change rationale. No training or physical validation rerun during cleanup.
