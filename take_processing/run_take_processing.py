from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Take collection and non-destructive processing")
    parser.add_argument("--data", type=Path, default=ROOT.parent / "particle_filter_cable_project")
    parser.add_argument("--library", type=Path, default=ROOT / "data/take_processing/catalog.json")
    args = parser.parse_args()
    from PySide6.QtWidgets import QApplication
    from take_processing.manager_window import TakeManager
    app = QApplication(sys.argv[:1])
    window = TakeManager(args.data, args.library)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
