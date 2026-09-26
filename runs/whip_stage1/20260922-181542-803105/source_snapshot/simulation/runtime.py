"""Fixed-step coupled simulation, independent of rendering and input hardware."""

import math
from time import perf_counter

from simulation.configuration import SimulationConfiguration
from simulation.dynamics.coupled_runtime import CoupledRuntime


class Simulation:
    def __init__(self, configuration: SimulationConfiguration, *, model_config, device=None,
                 compile_physics=True, command_source=None, command_reset=None) -> None:
        """Create a simulation with an explicit model path or configuration dict.

        command_source(step_index) supplies P/V/A/yaw at physics-step boundaries.
        Stateful sources can supply command_reset(), called on initialization and
        reset. Without a source, step(command) supplies commands directly; omitted
        commands hold the last command (initially the configured position/yaw).
        """
        self.configuration = configuration
        self.timestep_s = configuration.timestep_s
        # Optional source: command_source(step_index) -> P/V/A/yaw.
        # Indexed by physics steps, never wall time or renderer callbacks.
        self.command_source = command_source
        self.command_reset = command_reset
        self.dynamics = CoupledRuntime(configuration, model_config=model_config, device=device,
                                       compile_physics=compile_physics)
        self.reset()

    def reset(self) -> None:
        self.running = False
        self.dynamics.reset()
        if self.command_reset is not None:
            self.command_reset()
        self.state = self.dynamics.snapshot(0.0)
        self.step_count = 0
        self._remaining_s = 0.0

    def toggle_running(self) -> None:
        self.running = not self.running
        self._remaining_s = 0.0

    def step(self, command=None):
        """Pause and advance one fixed step; an explicit command overrides the source."""
        self.running = False
        self._remaining_s = 0.0
        self._advance_steps(1, command=command)
        return self.state

    def advance(self, elapsed_s: float) -> int:
        """Consume wall time in whole timesteps; paused time is discarded."""
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("elapsed_s must be finite and non-negative")
        if not self.running:
            return 0
        # Bound wall-time debt to keep camera/input responsive under overload.
        # Never enlarge the physics timestep or skip physics states: instead
        # simulation time progresses slower than wall time when overloaded.
        self._remaining_s = min(self._remaining_s + elapsed_s, 5 * self.timestep_s)
        steps = int((self._remaining_s + self.timestep_s * 1e-10) / self.timestep_s)
        advanced = 0
        started = perf_counter()
        while advanced < steps:
            self._advance_steps(1, snapshot=False)
            advanced += 1
            self._remaining_s = max(0.0, self._remaining_s - self.timestep_s)
            # Return to the UI between expensive steps; never coarsen physics.
            if perf_counter() - started >= .015:
                break
        if advanced:
            try:
                self.state = self.dynamics.snapshot(self.step_count * self.timestep_s)
            except Exception:
                self.running = False
                raise
        return advanced

    def _advance_steps(self, steps: int, *, snapshot=True, command=None) -> None:
        try:
            for _ in range(steps):
                next_command = command
                if next_command is None and self.command_source is not None:
                    next_command = self.command_source(self.step_count)
                self.dynamics.step(next_command)
                self.step_count += 1
            if snapshot:
                self.state = self.dynamics.snapshot(self.step_count * self.timestep_s)
        except Exception:
            self.running = False
            raise
