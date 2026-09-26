"""UI-owned experiment worker. Stdin Stop or parent EOF requests a safe stop."""

import json
from pathlib import Path
import sys
from threading import Event, Thread
import traceback


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit('This worker is started by the Training window.')
    stop = Event()

    def listen():
        try:
            for line in sys.stdin:
                if line.strip() == 'stop':
                    stop.set()
                    return
        finally:
            # Closing/crashing the UI must not leave the experiment running.
            stop.set()

    def progress(event):
        try:
            print(json.dumps(event, allow_nan=False), flush=True)
        except (BrokenPipeError, OSError):
            stop.set()

    try:
        # Load native libraries before blocking on stdin: NumPy's Windows
        # extension initialization can contend with that pipe read.
        from .experiment import run_experiment
        options = json.loads(sys.argv[3]) if len(sys.argv) == 4 else {}
        if not isinstance(options, dict) or set(options)-{'total_updates', 'resume_checkpoint', 'expected_resume_sha256'}:
            raise ValueError('Invalid experiment launch options')
        Thread(target=listen, daemon=True).start()
        run_experiment(Path(sys.argv[1]), progress=progress, stop_requested=stop.is_set,
                       expected_configuration_sha256=sys.argv[2], **options)
    except Exception as exc:
        progress(dict(event='error', message=str(exc)))
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
