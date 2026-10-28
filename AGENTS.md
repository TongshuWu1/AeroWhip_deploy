# Repository collaboration

Start here: `internal/handoff_2026-10-28.md` (current state, open items, working rules; notes in `internal/agent_notes/`).

- Inspect the relevant implementation and evidence before explaining behavior or proposing changes. Distinguish confirmed facts, hypotheses, and unverified claims.
- Explain the scope of a proposed implementation and obtain the user's approval before coding. An approved step does not authorize unrelated features, physics changes, reward definitions, or model promotion.
- Keep the reusable simulation separate from controllers, fitting, task definitions, and learning. Do not add unrelated frameworks, tests, documentation, or UI.
- Preserve user changes, source recordings, and existing model artifacts. Report only checks actually performed; never relax scientific criteria silently.

## Documentation and the implementation record

- `docs/` is the public documentation (also on the `publication` branch). Keep it accurate for outside readers: no development history, no links into `internal/`.
- `internal/implementation_record.md` is the maintained detailed record of the current pipeline (development branch only), not a future design presented as working code. Read the relevant sections before changing the pipeline.
- `internal/`, `AGENTS.md` and `CLAUDE.md` are removed on the `publication` branch; see `internal/PUBLICATION.md`.

Whenever a change affects pipeline behavior, interfaces, configuration defaults, model selection, physical assumptions, fitting/data preparation, controller timing, spline parameterization, training, reward, or evaluation:

1. Update the affected sections of `internal/implementation_record.md` and of `docs/` in the same change, before handing the work back to the user.
2. Keep source links, units, frame conventions, active artifacts, and implemented/proposed status accurate. Replace superseded descriptions rather than leaving contradictory versions.
3. Update its last-reviewed date and append a short dated change note. State what was checked and what remains unverified; documentation-only work must not imply simulations or experiments were rerun.
4. Treat whip definitions, success thresholds, reward terms/weights, and curriculum rules as research decisions requiring explicit approval. Do not turn a discussion, proxy, or demonstration match into an established physical definition.
5. Separate reconstruction checks, numerical checks, model-fit results, policy-training results, and real-flight evidence. Cite retained run artifacts for scientific results; conversational benchmarks alone are not paper-ready evidence.

This is an editing requirement, not an automatic synchronization mechanism. Do not introduce additional documentation systems or automation without approval.
