"""Label the cable markers of calibration flights from the whole recorded cloud at once
(`aerowhip.takes.sequence_labels`). The previous labels are kept as cable_labels_filter.npz
(the live filter's output, for scoring the live filter); the new ones become cable_labels.npz.

    python scripts/label_sequence.py <flight folders...>
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs='+', type=Path)
    args = parser.parse_args()
    from aerowhip.takes import cable_labels as cl
    from aerowhip.takes import sequence_labels as sl
    for folder in args.folders:
        t0 = time.time()
        flight = cl.load_flight(folder)
        labels, report = sl.label(flight)
        previous = folder/cl.LABELS_FILE
        if previous.exists() and not (folder/'cable_labels_filter.npz').exists():
            shutil.copy(previous, folder/'cable_labels_filter.npz')
            if previous.with_suffix('.json').exists():
                shutil.copy(previous.with_suffix('.json'), folder/'cable_labels_filter.json')
        summary = cl.save(flight, labels)
        report.update(method='sequence_labels', seconds=round(time.time()-t0, 1))
        (folder/'cable_labels_report.json').write_text(json.dumps(report, indent=1)+'\n')
        print(f"{folder.name}: {report['labelled_per_frame']:.2f} markers/frame, {summary['complete_frames']}/{summary['frames']} complete, "
              f"{report['uncertain_marker_frames']} uncertain marker-frames left unlabelled, {summary['unreviewed_flagged_frames']} flagged; {report['seconds']} s", flush=True)


if __name__ == '__main__':
    main()
