"""Open visible Training; optionally start fresh or resume a compatible checkpoint."""
import argparse
from training.ui.window import main
from training.configuration import FROZEN_PRESETS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true',
                        help='Open the Training UI and start new training with the current defaults')
    parser.add_argument('--settings', help='Optional learning-settings JSON; the run remains UI-owned')
    parser.add_argument('--stage', type=int, choices=(1,2,3,4), help='Select current cone25-v4 stage (default: 1): whip, broad delivery, nominal precision, hybrid settling; 80-collection cap each, plateau stopping never before40 collections')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--resume', help='Select a compatible exact-resume checkpoint in the visible Training UI')
    source.add_argument('--initialize-actor', help='Fine-tune a saved PPO actor with new task/settings and a fresh critic, not resume')
    args = parser.parse_args()
    if args.stage and args.settings:
        parser.error('Select --stage OR --settings, not both')
    raise SystemExit(main(auto_start=args.start,resume_checkpoint=args.resume,
                         initialize_actor=args.initialize_actor,
                         **({'settings_path': FROZEN_PRESETS[args.stage]} if args.stage else
                            {'settings_path': args.settings} if args.settings else
                            {'settings_path': None if args.resume else FROZEN_PRESETS[1]})))
