import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.
from aerowhip.paths import ROOT
import argparse


def main():
    parser = argparse.ArgumentParser(description="Take collection and non-destructive processing")
    parser.add_argument("--data", type=Path, help="Recording collection; otherwise use the saved folder or choose one")
    parser.add_argument("--library", type=Path, default=ROOT / "data/take_processing/catalog.json")
    args = parser.parse_args()
    from PySide6.QtWidgets import QApplication
    from aerowhip.ui.takes.manager_window import TakeManager
    from aerowhip.takes.paths import select_recording_root
    from aerowhip.paths import project_path
    app = QApplication(sys.argv[:1])
    folder = select_recording_root(args.data)
    if folder is None:
        return 0
    window = TakeManager(folder, project_path(args.library))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
