"""Task-only scene, episode, workspace and Stage 1 reward settings."""

from dataclasses import dataclass

from .configuration import EpisodeTiming, WhipReturnConfiguration
from .workspace import CylinderWorkspace


@dataclass(frozen=True)
class Stage1Configuration:
    scene: str
    model: str
    episode: EpisodeTiming
    workspace: CylinderWorkspace
    returns: WhipReturnConfiguration
    continuous_whip: dict

    def __post_init__(self):
        from .continuous_whip import SHORT_CONTRACT
        if (self.continuous_whip != SHORT_CONTRACT or self.episode.duration_s != 4
                or self.returns.failure_cost_per_remaining_s != 1.):
            raise ValueError('Stage 1 differs from the approved four-second contract')
        if not all(isinstance(p, str) and p for p in (self.scene, self.model)):
            raise ValueError('Scene/model paths are required')

    @property
    def reward_mode(self):
        return self.continuous_whip['version']
