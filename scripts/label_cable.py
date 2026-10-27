import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.
import argparse


def main():
    parser = argparse.ArgumentParser(description='Label the cable markers of calibration flights from their recorded '
                                     'marker cloud (cloud.jsonl): automatic labelling, then review and correction.')
    parser.add_argument('folders', nargs='+', type=Path, help='Flight folders with cloud.jsonl and poses.jsonl')
    parser.add_argument('--auto', action='store_true', help='Only auto-label and save; no window')
    args = parser.parse_args()
    from aerowhip.takes import cable_labels as cl
    if args.auto:
        for folder in args.folders:
            flight = cl.load_flight(folder)
            labeller = cl.AutoLabeller(flight, cl.load_labels(flight))
            summary = cl.save(flight, labeller.run())
            print(f"{folder.name}: {summary['complete_frames']}/{summary['frames']} frames complete, "
                  f"{summary['unreviewed_flagged_frames']} flagged frames unchecked, "
                  f"per marker {summary['per_marker_labelled_percent']} %")
        return 0
    from PySide6.QtWidgets import QApplication
    from aerowhip.ui.takes.cable_review import CableReview
    app = QApplication(sys.argv[:1])
    windows = [CableReview(folder) for folder in args.folders]
    for window in windows:
        window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
