"""Interface between a batched environment and a task's reward.

The environment steps the physics and calls the task at every 100 Hz sample of
each active lane. The task returns that sample's reward, decides when a lane's
episode ends, and supplies critic features and recorded diagnostics. Rewards
are returned as differences of the task's running return, so the per-decision
rewards of an episode always sum to its final return.
"""


class Task:
    """Base class for one batch of episodes. Subclasses implement every method.

    Attributes set by subclasses:
        name: identifier of the reward definition (recorded with every run).
        critic_feature_size: length of critic_features() rows.
        trace_fields: names of the per-sample diagnostic columns in trace().
        score_component: component name used as the episode's headline score.
    """
    name = None
    critic_feature_size = 0
    trace_fields = ()
    score_component = None

    def reset(self, observation):
        """Start new episodes. observation holds batched world-frame numpy arrays:
        cable_positions (B,N,3) [m], cable_velocities (B,N,3) [m/s], drone_positions (B,3) [m]
        and workspace_origin (B,3) [m], the reset drone position that centres the workspace."""
        raise NotImplementedError

    def observe(self, lane, time_s, observation):
        """Score one 100 Hz sample of one lane (arrays without the batch axis); return its reward."""
        raise NotImplementedError

    def fail(self, lane, time_s, reason='numerical_failure'):
        """The lane ended early at time_s (numerical failure, or an envelope violation when the
        environment terminates on it); return the reward correction."""
        raise NotImplementedError

    def done(self, lane):
        raise NotImplementedError

    def end_time(self, lane):
        """Time [s] at which the lane's episode ended (valid once done)."""
        raise NotImplementedError

    def episode_return(self, lane):
        raise NotImplementedError

    def critic_features(self):
        """(B, critic_feature_size) float32 task state for the value function only."""
        raise NotImplementedError

    def components(self):
        """Per-lane result components: name -> list of floats (one per lane)."""
        raise NotImplementedError

    def record(self, lane):
        """JSON-serializable summary of one lane's episode."""
        raise NotImplementedError

    def trace(self, lane):
        """(samples, len(trace_fields)) per-sample diagnostics, including the reset sample."""
        raise NotImplementedError
