"""Local display sources. Only the simulation worker touches its native runtime.

Qt/VTK consume a latest-frame mailbox; slow rendering drops display frames, never
physics steps. Commands execute between output steps. This is desktop pacing,
not a real-time scheduler or a ROS flight bridge.
"""
from concurrent.futures import Future
from dataclasses import dataclass, replace
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
from time import perf_counter
from types import SimpleNamespace

from aerowhip.physics.controller_interface import ControlInput
from aerowhip.viewer.display_state import DisplayFrame, immutable, simulation_frame


@dataclass(frozen=True)
class SourceSnapshot:
    revision: int
    state: object
    frame: object
    running: bool
    error: str | None
    step_count: int
    speed: float
    published_monotonic_s: float
    playback_available: bool = True

    def status(self, now=None):
        age = max(0., (perf_counter() if now is None else now)-self.published_monotonic_s)
        if self.error:
            return 'Error', age
        if not self.frame.robots:
            return 'No data', age
        if not self.running:
            return 'Paused', age
        stale = age > .25 or any(self.frame.time_s-r.time_s > .25 for r in self.frame.robots)
        return ('Stale' if stale else 'Live' if not self.playback_available else
                'Playing' if self.frame.recorded else 'Running'), age


class SimulationSource:
    """Compatibility facade for trajectory controls, backed by one owner thread."""
    is_replay = False
    has_playback = has_trajectory = True

    def __init__(self, simulation, *, control_source=ControlInput):
        self.configuration = simulation.configuration
        self.timestep_s = simulation.timestep_s
        self._simulation = simulation
        self._control_source = control_source
        # prepare_reference is pure and reads fixed model constants only. Do not
        # expose the mutable native runtime to reference export or the viewer.
        self.dynamics = SimpleNamespace(controller=SimpleNamespace(
            prepare_reference=simulation.dynamics.controller.prepare_reference))
        self._commands = Queue(maxsize=64)
        self._wake, self._stop = Event(), Event()
        self._lock = Lock()
        self._error = None
        self._speed = 1.
        self._revision = 0
        self._publish()
        self._thread = Thread(target=self._run, name='AeroWhip simulation', daemon=True)
        self._thread.start()

    def _publish(self):
        sim = self._simulation
        try:
            state = immutable(sim.state)
            frame = simulation_frame(state)
        except Exception as exc:
            if not hasattr(self, '_latest'):
                raise
            # Retain the accepted prefix visibly as an error, never publish a
            # malformed pose or let validation silently kill the owner thread.
            sim.running = False
            self._error = 'Display publication failed: '+str(exc)
            state, frame = self._latest.state, self._latest.frame
        self._revision += 1
        snapshot = SourceSnapshot(self._revision, state, frame,
                                  sim.running, self._error, sim.step_count,
                                  self._speed, perf_counter())
        with self._lock:
            self._latest = snapshot
            self._source, self._reset = sim.command_source, sim.command_reset

    @property
    def latest(self):
        with self._lock:
            return self._latest

    @property
    def state(self):
        return self.latest.state

    @property
    def step_count(self):
        return self.latest.step_count

    @property
    def running(self):
        return self.latest.running

    @running.setter
    def running(self, value):
        self._call('running', bool(value))

    @property
    def command_source(self):
        with self._lock:
            return self._source

    @command_source.setter
    def command_source(self, value):
        self._call('source', value)

    @property
    def command_reset(self):
        with self._lock:
            return self._reset

    @command_reset.setter
    def command_reset(self, value):
        self._call('reset_callback', value)

    def _call(self, action, value=None):
        if self._stop.is_set() or not self._thread.is_alive():
            raise RuntimeError('Simulation source is closed')
        future = Future()
        try:
            self._commands.put_nowait((action, value, future))
        except Full as exc:
            raise RuntimeError('Simulation command queue is full') from exc
        self._wake.set()
        # This wait is only for user actions; render/telemetry reads never wait
        # for integration. A command is acknowledged after its frame publishes.
        try:
            return future.result(timeout=5.)
        except TimeoutError:
            future.cancel()
            raise RuntimeError('Simulation command did not respond within 5 s') from None

    def _execute(self, action, value):
        sim = self._simulation
        if action in ('toggle', 'step') and self._error:
            raise RuntimeError(self._error)
        if action == 'toggle':
            sim.toggle_running()
        elif action == 'running':
            if value and self._error:
                raise RuntimeError(self._error)
            if sim.running != value:
                sim.toggle_running()
        elif action == 'step':
            sim.step(value)
        elif action == 'reset':
            sim.reset()
            self._error = None
        elif action in ('source', 'reset_callback'):
            if sim.running:
                raise ValueError('Pause before changing the command source')
            if action == 'source' and hasattr(value, 'validate_controller'):
                value.validate_controller()
            setattr(sim, 'command_source' if action == 'source' else 'command_reset', value)
        elif action == 'forces':
            sim.set_force_telemetry(value)
        elif action == 'speed':
            if value not in (.25, .5, 1.):
                raise ValueError('Playback speed must be 0.25, 0.5 or 1')
            self._speed = float(value)
        elif action == 'checkpoint':
            return sim.dynamics.checkpoint()
        else:
            raise ValueError('Unknown simulation action: '+action)

    def _run(self):
        last = perf_counter()
        try:
            while not self._stop.is_set():
                self._wake.clear()
                clock_reset = False
                for _ in range(64):
                    try:
                        action, value, future = self._commands.get_nowait()
                    except Empty:
                        break
                    if not future.set_running_or_notify_cancel():
                        continue
                    try:
                        result = self._execute(action, value)
                    except Exception as exc:
                        if action in ('step', 'reset', 'forces'):
                            self._simulation.running = False
                            self._error = str(exc)
                        self._publish()
                        future.set_exception(exc)
                    else:
                        self._publish()
                        future.set_result(result)
                    clock_reset |= action in ('running', 'toggle', 'step', 'reset', 'speed')
                now = perf_counter()
                elapsed, last = (0. if clock_reset else now-last), now
                try:
                    controls = self._control_source()
                    if controls.reset:
                        self._simulation.reset()
                        self._error = None
                        self._publish()
                        last = perf_counter()
                    elif controls.toggle:
                        if not self._error:
                            self._simulation.toggle_running()
                            self._publish()
                        last = perf_counter()
                    elif self._simulation.advance(elapsed*self._speed):
                        self._publish()
                except Exception as exc:
                    self._simulation.running = False
                    self._error = str(exc)
                    self._publish()
                self._wake.wait(.005)
        finally:
            self._simulation.running = False
            self._publish()
            while True:
                try:
                    _, _, future = self._commands.get_nowait()
                except Empty:
                    break
                if not future.done():
                    future.set_exception(RuntimeError('Simulation source closed'))

    def toggle_running(self):
        self._call('toggle')

    def step(self, command=None):
        self._call('step', command)
        return self.state

    def reset(self):
        self._call('reset')

    def set_force_telemetry(self, enabled):
        self._call('forces', bool(enabled))

    def set_speed(self, speed):
        self._call('speed', speed)

    def checkpoint(self):
        return self._call('checkpoint')

    def tick(self, elapsed_s):
        """Rendering has no influence on the simulation clock."""
        return self.latest

    def close(self):
        self._stop.set()
        self._wake.set()
        if current_thread() is not self._thread:
            self._thread.join(timeout=5.)
            if self._thread.is_alive():
                raise RuntimeError('Simulation worker did not stop within 5 s')


class ReplaySource:
    """Saved arrays through the same display contract; never creates a plant."""
    is_replay = True
    has_playback = True
    has_trajectory = False
    command_source = command_reset = None

    def __init__(self, replay, configuration):
        self.replay = replay
        self.configuration = configuration
        self.timestep_s = configuration.timestep_s
        self._revision = 0
        self._publish()

    def _publish(self):
        self._revision += 1
        state = immutable(self.replay.state)
        self.latest = SourceSnapshot(self._revision, state,
            simulation_frame(state, recorded=True), self.replay.running,
            None, self.replay.index, self.replay.speed, perf_counter())

    @property
    def state(self):
        return self.latest.state

    @property
    def running(self):
        return self.replay.running

    @running.setter
    def running(self, value):
        self.replay.running = bool(value)
        self._publish()

    @property
    def step_count(self):
        return self.replay.index

    def toggle_running(self):
        self.replay.toggle_running()
        self._publish()

    def step(self):
        self.replay.step()
        self._publish()
        return self.state

    def reset(self):
        self.replay.reset()
        self._publish()

    def seek(self, index):
        self.replay.running = False
        self.replay.seek(index)
        self._publish()

    def set_speed(self, speed):
        if speed not in (.25, .5, 1.):
            raise ValueError('Playback speed must be 0.25, 0.5 or 1')
        self.replay.speed = float(speed)
        self._publish()

    def tick(self, elapsed_s):
        previous_running = self.replay.running
        if self.replay.advance(elapsed_s) or self.replay.running != previous_running:
            self._publish()
        return self.latest

    def close(self):
        self.running = False


class DisplayFeed:
    """Read-only frame mailbox for external producers; no flight commands.

    The caller supplies already transformed, timestamped DisplayFrames. This
    class is not a ROS connector, telemetry estimator or physical simulation.
    """
    is_replay = has_playback = has_trajectory = False
    command_source = command_reset = None

    def __init__(self, frame: DisplayFrame, configuration):
        self.configuration = configuration
        self._lock = Lock()
        self._closed = False
        self._revision = 0
        self._frame_id = frame.frame_id if isinstance(frame, DisplayFrame) else None
        self.publish(frame)

    def publish(self, frame, *, error=None):
        if not isinstance(frame, DisplayFrame):
            raise TypeError('DisplayFeed accepts DisplayFrame objects only')
        with self._lock:
            if self._closed:
                raise RuntimeError('Display feed is closed')
            if frame.frame_id != self._frame_id:
                raise ValueError('Transform frames before publication; a display feed has one world frame')
            self._revision += 1
            self._latest = SourceSnapshot(self._revision, None, frame, error is None,
                error, 0, 1., perf_counter(), playback_available=False)

    @property
    def latest(self):
        with self._lock:
            return self._latest

    def tick(self, elapsed_s):
        return self.latest

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._latest = replace(self._latest, revision=self._latest.revision+1,
                                       running=False, error='Display source closed')
