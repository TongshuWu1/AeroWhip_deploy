 """Launch saved motions or --measurement-report review; never train or simulate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.
from aerowhip.viewer.replay_browser import main


if __name__ == '__main__':
    raise SystemExit(main())
