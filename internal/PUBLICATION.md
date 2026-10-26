# Publication branch

Development happens on the main branch, which also carries this `internal/`
folder (detailed implementation record, handoff notes) and the agent
instructions `AGENTS.md` and `CLAUDE.md`. The `publication` branch is the same
code and public documentation without those files. It is produced by hand with
the steps below; nothing synchronizes it automatically.

## Preparing a release

1. On the development branch, make sure the public material is current:
   `README.md` and `docs/` describe the code as it is, contain no development
   history and do not link into `internal/`.
2. Check the tree:

   ```bash
   python -m pytest
   python scripts/equivalence_check.py   # only if a local reference exists
   ```

3. Update the publication branch from the development branch and drop the
   internal files:

   ```bash
   git switch publication            # first time: git switch -c publication
   git merge --no-ff main            # or the development branch name
   git rm -r --quiet --ignore-unmatch internal AGENTS.md CLAUDE.md
   git commit -m "Publication snapshot of <commit>"
   ```

   A merge re-adds files deleted on `publication` only if they changed on the
   development branch; the `git rm` line removes them again each time.
4. Before pushing, check that no internal paths or personal data remain:

   ```bash
   git grep -n -I -E "internal/|HANDOFF|/home/" publication
   ```

## Decisions to make before the first public release

- Licence (add `LICENSE`) and citation metadata (`CITATION.cff`) for the paper.
- Whether the public repository starts from fresh history (an orphan branch
  with one snapshot commit) rather than this repository's full history, which
  includes internal notes and earlier experiment files.
- Which release archives (recordings, runs, evidence) accompany the paper.
