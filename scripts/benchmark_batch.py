"""Throughput of the native batch runtime (coupled drone + cable, flight controller per lane).

    python scripts/benchmark_batch.py --lanes 24 96 --seconds 2
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting.flight_drone import firmware_config
from aerowhip.physics.model import Model, native_module
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lanes', type=int, nargs='+', default=[24, 96])
    parser.add_argument('--seconds', type=float, default=2.)
    args = parser.parse_args()
    config = load_model_configuration(DEFAULT_MODEL)
    native = native_module()
    model = Model(config)
    state = model.native.initial_state(np.array([0., 0., 1.5]), np.eye(3), np.zeros(3), np.zeros(3), None, None)
    hold = np.zeros(13)
    firmware = firmware_config()
    rng = np.random.default_rng(0)
    print(f'OpenMP threads: {native.BatchRuntime.threads()}')
    for lanes in args.lanes:
        controllers = [native.FlightController(firmware, config, 7.2) for _ in range(lanes)]
        batch = native.BatchRuntime(model.native, state, hold, config['controller']['controller_period_s'],
                                    config['controller']['command_delay_s'], native.Actuator(config['actuator']), controllers)
        steps = int(args.seconds/.01)
        # Smooth random accelerations, 3 m/s^2 scale, plus the hover trim the thrust map needs.
        phase = rng.uniform(0, 2*np.pi, (lanes, 3))
        t0 = time.perf_counter()
        for k in range(steps):
            command = np.zeros((lanes, 13))
            command[:, 6:9] = 3*np.sin(2*np.pi*.7*k*.01+phase)*[1, 1, .3]
            command[:, 8] += 1.1
            batch.step(command)
        elapsed = time.perf_counter()-t0
        failed = sum(batch.failed)
        print(f'{lanes:4d} lanes: {lanes*steps/elapsed:8.0f} lane-steps/s (10 ms each) = {lanes*steps*.01/elapsed:6.1f} simulated s per s; '
              f'{failed} lanes failed', flush=True)
        if failed:
            print('  first error:', next(e for e in batch.errors if e))


if __name__ == '__main__':
    main()
