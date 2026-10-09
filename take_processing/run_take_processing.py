from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Take collection and non-destructive processing")
    parser.add_argument("--data", type=Path, help="Recording collection; otherwise use the saved folder or choose one")
    parser.add_argument("--library", type=Path, default=ROOT / "data/take_processing/catalog.json")
    args = parser.parse_args()
    from PySide6.QtWidgets import QApplication
    from take_processing.manager_window import TakeManager
    from take_processing.paths import select_recording_root
    from project_paths import project_path
    app = QApplication(sys.argv[:1])
    folder = select_recording_root(args.data)
    if folder is None:
        return 0
    window = TakeManager(folder, project_path(args.library))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
