"""Launch the main-checkout editable policy simulation tester. No automatic run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.
from aerowhip.ui.policy_tester import main

if __name__ == '__main__':
    raise SystemExit(main())
