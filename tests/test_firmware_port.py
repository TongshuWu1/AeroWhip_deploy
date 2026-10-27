"""The ported Mellinger controller (native/firmware.hpp) against the vendored firmware C sources.

The C files in native/third_party/crazyflie-firmware are compiled with the extension's floating-point
flags into a harness; both are driven with the same random setpoints, states and gyro rates, including
saturating commands and a stateful sequence (integrators, derivative memory), and must agree to float
rounding."""
import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aerowhip.paths import ROOT
from aerowhip.physics.model import native_module

VENDORED = ROOT/'native/third_party/crazyflie-firmware'
PARAMS = dict(mass=.144, massThrust=17700., kp_xy=0., kd_xy=0., ki_xy=0., i_range_xy=0., kp_z=0., kd_z=0., ki_z=0., i_range_z=0.,
              kR_xy=70000., kw_xy=20000., ki_m_xy=0., i_range_m_xy=1., kR_z=60000., kw_z=12000., ki_m_z=500., i_range_m_z=1500., kd_omega_rp=200.)
ORDER = ('mass', 'massThrust', 'kp_xy', 'kd_xy', 'ki_xy', 'i_range_xy', 'kp_z', 'kd_z', 'ki_z', 'i_range_z', 'kR_xy', 'kw_xy',
         'ki_m_xy', 'i_range_m_xy', 'kR_z', 'kw_z', 'ki_m_z', 'i_range_m_z', 'kd_omega_rp')


@pytest.fixture(scope='module')
def harness(tmp_path_factory):
    if shutil.which('gcc') is None:
        pytest.skip('gcc is needed to compile the vendored firmware')
    out = tmp_path_factory.mktemp('firmware')/'libharness.so'
    sources = [VENDORED/f for f in ('harness.c', 'controller_mellinger.c', 'power_distribution_quadrotor.c')]
    subprocess.run(['gcc', '-O3', '-fno-fast-math', '-ffp-contract=off', '-shared', '-fPIC', f'-I{VENDORED/"stubs"}', f'-I{VENDORED}',
                    *map(str, sources), '-lm', '-o', str(out)], check=True)
    lib = ctypes.CDLL(str(out))
    lib.harness_init.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.harness_step.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    return lib


def reference(lib, params, inputs):
    p = np.array([params[k] for k in ORDER], np.float32)
    lib.harness_init(p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    out = np.empty((len(inputs), 9), np.float32)
    for row, x in zip(out, np.asarray(inputs, np.float32)):
        lib.harness_step(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return out


def port(params, inputs):
    c = native_module().Mellinger(params)
    return np.array([c.step(np.asarray(x, np.float32)) for x in inputs], np.float32)


def random_inputs(rng, n, *, wild=False):
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1
    if not wild:                       # mostly upright: small tilt from identity
        small = rng.normal(scale=.15, size=(n, 3)); angle = np.linalg.norm(small, axis=1, keepdims=True)
        q = np.concatenate([small/angle*np.sin(angle/2), np.cos(angle/2)], 1)
    scale = 4. if wild else 1.
    acc = rng.uniform(-8, 8, (n, 3))*scale; acc[:, 2] = rng.uniform(-6, 12, n)*scale
    return np.column_stack([rng.normal(scale=.5, size=(n, 6)), acc, rng.uniform(-180, 180, n), rng.normal(scale=.5, size=(n, 6)),
                            q, rng.normal(scale=200*scale, size=(n, 3))]).astype(np.float32)


@pytest.mark.parametrize('wild', [False, True])
def test_port_matches_firmware_sequence(harness, wild):
    rng = np.random.default_rng(1 if wild else 0)
    x = random_inputs(rng, 800, wild=wild)
    a, b = reference(harness, PARAMS, x), port(PARAMS, x)
    thrust_scale = max(1., float(np.abs(a[:, 0]).max()))
    np.testing.assert_allclose(b[:, 0], a[:, 0], rtol=0, atol=2e-6*thrust_scale)      # float thrust
    np.testing.assert_array_equal(b[:, 1:4], a[:, 1:4])                                 # int16 legacy outputs
    np.testing.assert_array_equal(b[:, 4:8], a[:, 4:8])                                 # uint16 PWM after capping
    np.testing.assert_array_equal(b[:, 8], a[:, 8])                                     # capped flag
    assert (a[:, 8] > 0).any() == wild or not wild                                       # wild inputs exercise the cap


def test_flight_parameters_from_yaml_are_used(harness):
    params = dict(PARAMS, kR_xy=50000., kw_xy=15000., massThrust=20000., ki_m_z=0.)
    rng = np.random.default_rng(2)
    x = random_inputs(rng, 200)
    np.testing.assert_array_equal(port(params, x)[:, 1:8], reference(harness, params, x)[:, 1:8])
    assert not np.array_equal(port(params, x)[:, 1:4], port(PARAMS, x)[:, 1:4])
