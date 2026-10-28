"""Closed-loop simulation of one planned part with one model: the drone with the cable (the
configured model, or an ensemble member), the firmware attitude law at the flown gains, and the
flight script's host PID at 50 Hz following the same reference the real flight would, from the
hover point through the part and a hover after it. Returns per 10 ms step what the planner scores
and constrains: the drone's position error and tilt, the cable's highest node above the
attachment, the tip position and speed, and the smallest motor force as a share of its hover share."""
from copy import deepcopy
import math
import sys
from types import SimpleNamespace

import numpy as np

from aerowhip.paths import ROOT
from aerowhip.physics.model import Model, native_module, settled_cable
from aerowhip.planning.measures import cable_rise

FLIGHT_STACK = ROOT.parent/'Crazyflie'/'crazyswarm2_jazzy_whipping'
COMMAND_WIDTH = 13
HOVER_AFTER_S = 2.
TILT_FAIL_DEG = 80.


def flight_script(stack=FLIGHT_STACK):
    if str(stack/'scripts') not in sys.path:
        sys.path.insert(0, str(stack/'scripts'))
    import acceleration_circle_cf2 as flight
    import flight_trajectories as trajectories
    return flight, trajectories


def part_args(spec, height=1.3, stroke_length=1., stack=FLIGHT_STACK):
    """A flight-script argument namespace whose motion is the single part `spec` (as in
    config/calibration_mixes.yaml), so the script's own reference() flies it."""
    flight, trajectories = flight_script(stack)
    motion = trajectories.mix('part', [dict(spec)], float(spec.get('stroke_length', stroke_length)))
    lift = max(0., -min(motion.state(t)[0][2] for t in np.arange(0, motion.duration+.005, .005)))
    return SimpleNamespace(height=height, hover=None, motion=motion, lift=lift, tune_sets=None, battery_run=None)


def motion_args(motion, height=1.3):
    lift = max(0., -min(motion.state(t)[0][2] for t in np.arange(0, motion.duration+.005, .005)))
    return SimpleNamespace(height=height, hover=None, motion=motion, lift=lift, tune_sets=None, battery_run=None)


def simulate_part(config, firmware, spec, voltage=7.6, stack=FLIGHT_STACK):
    """Fly one part (a spec as in calibration_mixes.yaml) closed loop; see simulate_motion."""
    return simulate_motion(config, firmware, part_args(spec, stack=stack).motion, voltage, stack)


def simulate_motion(config, firmware, motion, voltage=7.6, stack=FLIGHT_STACK, markers_every=10, nodes=False):
    """Fly a motion closed loop. Returns dict of arrays over the steps from the motion's start to
    HOVER_AFTER_S after it ends: t, phase (list), error (m), tilt (deg), rise (m), tip (3), tip_speed,
    motor (min force / hover share), command (3), position (3); every `markers_every` steps also
    markers (M, 10, 3) and rotation (M, 3, 3) for the sensitivities, with `nodes=True` also every cable
    node (M, n, 3) and the reference (M, 3) at the same steps, for display; and `failed` (None or a reason)."""
    flight, _ = flight_script(stack)
    args = motion_args(motion)
    native = native_module()
    model = Model(config)
    weight = (model.mass+float(np.sum(model.node_mass)))*flight.GRAVITY
    controller = native.FlightController(firmware, config, voltage)
    hover = np.array([0., 0., args.height+args.lift])
    root = hover+np.asarray(model.attachment_tracking)
    cable = settled_cable(model, root)
    state = model.native.initial_state(hover, np.eye(3), np.zeros(3), np.zeros(3), cable, np.zeros_like(cable))

    def command(a):
        c = np.zeros(COMMAND_WIDTH); c[6:9] = a; return c
    low, high = -5., 12.
    for _ in range(60):
        mid = .5*(low+high)
        if controller(state, command([0., 0., mid]))[0] < weight:
            low = mid
        else:
            high = mid
    trim = np.array([0., 0., .5*(low+high)])
    actuator = deepcopy(config['actuator'])
    thrust = controller(state, command(trim))[0]
    actuator['initial_thrust_n'] = [float(np.clip(thrust/4, actuator['minimum_thrust_n'][i], actuator['maximum_thrust_n'][i])) for i in range(4)]
    runtime = native.Runtime(model.native, state, command(trim), config['controller']['controller_period_s'],
                             config['controller']['command_delay_s'], native.Actuator(actuator))
    host = flight.AccelerationController()
    start = flight.motion_start(args)
    end = start+args.motion.duration+HOVER_AFTER_S
    origin = np.zeros(3)
    n = len(model.rest)+1
    out = dict(t=[], phase=[], error=[], tilt=[], rise=[], tip=[], tip_speed=[], motor=[], command=[], position=[])
    markers, rotations, cable_nodes, references = [], [], [], []
    index = np.asarray(model.marker_indices)[1:]
    sent = command(trim)
    hover_share = weight/4
    failed = None
    for i, t in enumerate(np.arange(start-1., end, .01)):
        s = runtime.state()
        obs = model.native.observation(s)
        p, v, R = obs[:3], obs[3:6], obs[6:15].reshape(3, 3)
        q = obs[18:18+3*n].reshape(n, 3)
        qv = obs[18+3*n:18+6*n].reshape(n, 3)
        phase, ref, dref, aref = flight.reference(max(t, start-1.), origin, args)
        events = []
        if i % 2 == 0:
            a, _ = host.command(p, v, ref, dref, aref, .02, trim)
            sent = command(a)
            events = [(runtime.time, sent)]
        try:
            runtime.step(events, controller)
        except Exception as exc:
            failed = f'engine: {exc}'
            break
        if t < start:
            continue
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1., 1.))))
        out['t'].append(t-start); out['phase'].append(phase)
        out['error'].append(float(np.linalg.norm(ref-p))); out['tilt'].append(tilt)
        out['rise'].append(float((q[1:, 2]-q[0, 2]).max()))
        out['tip'].append(q[-1].copy()); out['tip_speed'].append(float(np.linalg.norm(qv[-1])))
        out['motor'].append(float(min(controller.motors(s, sent))/hover_share))
        out['command'].append(sent[6:9].copy()); out['position'].append(p.copy())
        if (len(out['t'])-1) % markers_every == 0:     # the first step, then every markers_every (1 = all)
            markers.append(q[index].copy()); rotations.append(R.copy())
            if nodes:
                cable_nodes.append(q.copy()); references.append(np.asarray(ref, float).copy())
        if tilt > TILT_FAIL_DEG:
            failed = f'drone tilt {tilt:.0f} deg'
            break
    result = {k: (np.array(v) if k != 'phase' else v) for k, v in out.items()}
    result['markers'] = np.array(markers); result['rotation'] = np.array(rotations)
    result['trim'] = trim
    if nodes:
        result['nodes'] = np.array(cable_nodes); result['reference'] = np.array(references)
        result['marker_t'] = result['t'][::markers_every][:len(markers)]
    result['failed'] = failed
    return result
