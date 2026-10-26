"""Open visible Training; optionally start fresh or resume a compatible checkpoint."""
import argparse
from training_history import main
from experiment import DEFAULT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true',
                        help='Open the Training UI and start new training with the current defaults')
    parser.add_argument('--settings', help='Optional learning-settings JSON; the run remains UI-owned')
    parser.add_argument('--resume', help='Select a compatible exact-resume checkpoint in the visible Training UI')
    args = parser.parse_args()
    settings = args.settings or (None if args.resume else DEFAULT)
    raise SystemExit(main(auto_start=args.start, resume_checkpoint=args.resume, settings_path=settings))
