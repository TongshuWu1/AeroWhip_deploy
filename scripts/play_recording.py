"""Open a saved hand-test/display recording without ROS, hardware or physics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.
import argparse
from dataclasses import replace
import json
import signal
from threading import Event

from aerowhip.viewer.recording import RecordingSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', nargs='?', type=Path, help='Recording folder or recording.json; omit for a file picker')
    parser.add_argument('--inspect', action='store_true', help='Validate saved frames and print a summary without opening a window')
    args = parser.parse_args()
    if args.inspect and args.recording is None:
        parser.error('--inspect requires a recording path')
    if args.recording is None:
        from aerowhip.viewer.simulation_window import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open saved recording',
            str(Path(__file__).resolve().parent/'output/custom_real'), 'Display recording (recording.json)')
        if not selected:
            return
        args.recording = Path(selected)
    source = None
    handlers = {}
    try:
        source = RecordingSource(args.recording)
        if args.inspect:
            print(json.dumps(dict(path=str(source.replay.path), frames=len(source.replay.time),
                duration_s=source.replay.duration_s, complete=source.replay.metadata.get('complete'),
                frame_id=source.latest.frame.frame_id, warning=source.replay.warning)))
            return
        from aerowhip.viewer.display_state import WorldDisplay
        from aerowhip.viewer.pyvista_viewer import PyVistaViewer
        stop = Event()
        handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        viewer = PyVistaViewer(source.configuration, world=replace(
            WorldDisplay.from_configuration(source.configuration.viewer), frame_id=source.latest.frame.frame_id))
        viewer.stop_requested = stop
        viewer.show_source(source)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'Recording: {exc}\n')
    finally:
        if source is not None:
            source.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
