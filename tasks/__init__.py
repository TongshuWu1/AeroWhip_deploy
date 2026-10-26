"""Task registry: the experiment file's task.name selects the reward definition."""
from tasks.whip.configuration import FoldTransportConfig
from tasks.whip.fold_transport import FoldTransportTask

TASKS = {'continuous_directional_whip_four_second_v6': (FoldTransportConfig, FoldTransportTask)}


def task_config(raw):
    """Typed task settings for the named task."""
    name = raw.get('name') if isinstance(raw, dict) else None
    if name not in TASKS:
        raise ValueError(f'Unknown task {name!r}; implemented: {sorted(TASKS)}')
    return TASKS[name][0].from_dict(raw)


def make_task(config, environment, rest):
    """Task instance for one batch: config from task_config(), rest = cable segment lengths [m]."""
    return TASKS[config.name][1](config, environment, rest)
