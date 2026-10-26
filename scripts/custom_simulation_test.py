"""Run the PVA connection test inside ROS Docker against a custom simulation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.
from aerowhip.paths import ROOT
import argparse
from datetime import datetime, timezone

from aerowhip.deployment.custom_simulation import launch_container_script
from aerowhip.deployment.ros_bridge import DEFAULT_DOMAIN
from custom_simulation import DEFAULT_CONTAINER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default=DEFAULT_CONTAINER)
    parser.add_argument('--domain', type=int, default=DEFAULT_DOMAIN)
    parser.add_argument('--robot', default='cf_2')
    parser.add_argument('--rate', type=float, default=30.)
    parser.add_argument('--motion-duration', type=float, default=6.)
    args = parser.parse_args()
    if args.domain == 42 or not 0 <= args.domain <= 232:
        parser.error('Use a simulation ROS domain separate from domain 42')
    # Existing logs mount keeps the result on the host without docker cp.
    filename = f'custom-simulation-pva-{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}.json'
    process = launch_container_script(args.container, ROOT/'aerowhip/deployment/ros_test.py',
        ['--robot',args.robot,'--rate',args.rate,'--motion-duration',args.motion_duration,
         '--output',f'/workspace/logs/{filename}'], domain=args.domain)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait(5.)
        raise
    if code:
        raise SystemExit(code)
    print(f'Result saved in the Crazyswarm project: logs/{filename}')


if __name__ == '__main__':
    main()
