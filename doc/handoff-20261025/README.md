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

The current checkout contains one supported Stage 1 training implementation.
Retired rewards, presets, SAC, stage transitions and bundled historical models
were removed with approval. They remain available at source-preservation commit
`3c9a49e`. Use a separate checkout of that commit for historical code/model work;
old checkpoints are not guaranteed to work with the pruned code.
Large runs and `tmp` evidence are separate release assets, not part of `git clone`:

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

Read [the current Stage 1 specification](../stage1.md) and
[the detailed cleanup report](cleanup-report.md). The dated contract and original
implementation report in this directory are preserved historical snapshots.
The +X reward and workspace cost remain experimental. Physical violations remain
diagnostics; no contact model or validated whipping-success threshold was added.
Full duration is not proof of physical validity. No failed or unfinished experiment
is promoted by this backup.

Before pruning, the historical suite passed 570 tests with 10 skips; its original
logs remain here. After retiring obsolete features and their dedicated tests,
the current CPU-only suite ran 229 tests: 227 passed, 2 skipped. These are different
suite scopes, not a claim of equivalent coverage. Current counterexamples and
numerical tolerances were preserved. See the cleanup report for integration and
equivalence evidence, including their limits.

Exact checkpoint continuation is source/binary-hash guarded and may be unavailable
on another OS/build; those guards were not bypassed. In particular the default
configuration correction changed a tracked execution source after the last saved
checkpoint. Since v6has0learning updates, the recommended path is a fresh run.

## Cleanup

Removed 2.66 GiB of verified duplicate downloads/snapshots, caches and stale locks.
Then removed 10,676 archived files (14.00 GiB) from old local `runs/` and `tmp/`
folders only after matching every file to its archive manifest and verifying all
22 GitHub assets by SHA-256 and byte count. No changed file was deleted.
The latest stopped v6 run and its implementation evidence remain local, as do
source recordings and reusable simulation/fitting/deployment tools. A local copy
of the compressed archives remains outside the checkout.

The release contains 18 independent archive parts and four manifests/checksum
files (8,545,338,190 bytes in total). Build products, environments and native
binaries remain ignored; rebuild them on the new computer.
