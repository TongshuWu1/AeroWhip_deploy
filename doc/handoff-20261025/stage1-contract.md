> Historical pre-cleanup snapshot. For the supported implementation, see [current Stage 1](../stage1.md). Legacy-mode compatibility mentioned below was subsequently removed with approval.

# Stage1 directed +X four-second v6 — approved2026-10-25

User requested four seconds,200collections,restart. Fresh actor and critic;
64lanes*200=12800episodes. Same seed20261028 as the preceding fresh experiment;
no checkpoint transfer. Entropy coefficient .001+.019exp(-8n/12800), offset0,
near floor at150collections;100msAR1 unchanged. No automatic Stage2.

The +X reward rate is unchanged:10 F H² A² U/L. F is existing causal fold
transport support. H=clip(max(0,tipX−attachmentX)/L,0,1), A is the clipped
minimum of world/attachment-relative tip velocity cosines with world+X,
U=max(0,min(world tipX speed,relative tipX speed,end-to-end extension speed)).
Integrate at100Hz for4seconds. No count limit or per-whip bonus.

C=sum[10*(radial_excess²+vertical_excess²)/400] at the tracking point.
This keeps C=10*mean squared workspace excess over the new fixed horizon.
Workspace unchanged: radius.55m around resetXY, worldZ1.5–2.6m.
Valid numerical full-horizon return=Q−C; no cap/nonlinear transform.
No early completion, time cost, effort/path cost, or tilt/cable-height early abort.
Physical violations remain diagnostics/invalidity; numerical corruption still
aborts with Q_prefix−C_prefix−(4−tau). No fixed−10 or contact physics added.

Ten links, native600Hzphysics,50Hzacceleration control unchanged.200control
steps and401states includingreset. Actor history401/time reference400 and
critic time/support/output durations normalize by4s. Old7s modes retain their
701state/350decision schedules and original formulas.57focusedtests passed;
14UI/replaychecks rerun after label correction. No extraGPUpreflight.
These are implementation checks, not evidence of learned successful whipping.
