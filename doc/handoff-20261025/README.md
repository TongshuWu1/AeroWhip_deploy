# Continue on another computer — 2026-10-25

Training is STOPPED. No worker or monitor automation is active. The latest experiment
is Stage1 directed+X v6,4seconds,64parallel lanes,200collections/12800episodes,
10links,exponential entropy. No automatic Stage2. The last run stopped in its first
collection: **0completed collections/episodes,0optimizerupdates**. Its checkpoint
is an untrained initialization, not a learned policy. Start fresh tomorrow.

## Source and setup

```bash
git clone https://github.com/TongshuWu1/AeroWhip_deploy.git
cd AeroWhip_deploy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[build,review]"
python build_engine.py
python run_training.py --settings configs/learning/whip_stage1_ppo_continuous_directional_four_second_v6.json
```

The last command opens setup without starting. Add `--start` when ready to launch.
The default Stage1 preset also now selects this four-second configuration.
Recreate dependencies/native extensions on the new computer; do not copy `.venv`
or OS-specific `.so` binaries. CUDA training requires a compatible NVIDIA GPU/Torch
installation. The checked host uses Python3.12 and Torch2.11.0; other-host execution
has not been tested. See the main README for Windows build instructions.

## Restore historical runs and evidence only when needed

The source commit contains all configurations,task/learning/evaluation/UI code,
tests and historical packaged Stage1–4models. Large runs and `tmp` evidence are
separate release assets, not part of `git clone`:

[Download the handoff data release](https://github.com/TongshuWu1/AeroWhip_deploy/releases/tag/handoff-2026-10-25).

```bash
gh release download handoff-2026-10-25 --repo TongshuWu1/AeroWhip_deploy --dir ../aerowhip-handoff-data
(cd ../aerowhip-handoff-data && sha256sum -c SHA256SUMS)
for archive in ../aerowhip-handoff-data/aerowhip-data-*.tar.gz; do
    tar -xzf "$archive" -C .
done
```

Extract into a fresh checkout to avoid overwriting local work. Each numbered part
is an independent tar.gz, so do not concatenate parts. `manifest.json` lists each
file's original relative path,size,SHA256 and containing archive. Restore all parts
to recover the complete unique current run/evidence/output set. A few old scratch
snapshot copies were removed only after verification against the existing
`backup-2026-10-16` release; that release remains available if needed.
Third-party downloaded papers were kept locally outside the upload tree.

The GitHub repository is currently PUBLIC (verified during handoff); older README
claims that it is private were outdated. No credentials or local environment files
are part of this source commit/archive. A targeted credential-pattern scan passed;
this is not a comprehensive security audit.

## Current task and verification

Read [the contract](stage1-contract.md) and [implementation report](stage1-implementation-report.md).
The +Xreward rate and workspace cost are experimental. Physical violations remain
diagnostics; no contact model or validated whipping-success threshold was added.
Full duration is not proof of physical validity. None of these failed/unfinished
training experiments is promoted by this backup.

The handoff test run initially found20stale-fixture/default failures/errors. Original
log retained. Legacy acceleration/reward-review fixtures now explicitly select their
intended historical presets, rather than the mutable default. Path tests assert the
current default; the v4handoff test reconstructs its original128lane budget rather
than silently using its later64lane continuation preset. No scientific tolerances
or acceptance criteria were weakened. The subsequent CPU-only suite passed:
**570tests,10skipped**. CUDA-specific cases were not exercised in that check.

Exact checkpoint continuation is source/binary-hash guarded and may be unavailable
on another OS/build; those guards were not bypassed. In particular the default
configuration correction changed a tracked execution source after the last saved
checkpoint. Since v6has0learning updates, the recommended path is a fresh run.

## Cleanup

Removed2.66GiB of old-release-verified duplicate downloads/snapshots,caches and
stale locks. One snapshot file differing from the old manifest was preserved.
Historical run/evidence folders are archived before removal from the local working
checkout. The latest stopped v6run and its implementation evidence remain local.
Release manifest/checksums retain the original paths for restoration. Generated
build/virtualenv/native binaries remain ignored and are not pushed into Git history.
