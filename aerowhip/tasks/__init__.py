"""Task registry: the experiment file's task.name selects the reward definition."""
from aerowhip.tasks.strike.configuration import StrikeConfig
from aerowhip.tasks.strike.strike import StrikeTask

TASKS = {'strike_x_v1': (StrikeConfig, StrikeTask)}


def task_config(raw):
    """Typed task settings for the named task."""
    name = raw.get('name') if isinstance(raw, dict) else None
    if name not in TASKS:
        raise ValueError(f'Unknown task {name!r}; implemented: {sorted(TASKS)}')
    return TASKS[name][0].from_dict(raw)


def make_task(config, environment, rest):
    """Task instance for one batch: config from task_config(), rest = cable segment lengths [m]."""
    return TASKS[config.name][1](config, environment, rest)
