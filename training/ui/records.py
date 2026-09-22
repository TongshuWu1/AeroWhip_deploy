"""Read experiment records for display without loading models or checkpoints."""

from dataclasses import dataclass, field
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@dataclass
class RunHistory:
    path: Path | None = None
    configuration: dict = field(default_factory=dict)
    updates: dict = field(default_factory=dict)
    evaluations: dict = field(default_factory=dict)
    status: str = 'Ready'
    elapsed_seconds: float | None = None
    initialization: dict = field(default_factory=dict)

    def add_update(self, record):
        self.updates.setdefault(record['seed'], {})[record['update']] = record

    def add_evaluation(self, seed, update, metrics):
        self.evaluations.setdefault(seed, {})[update] = metrics

    def add_prefix(self, prefix):
        """Earlier records are a retained curve prefix, not new session work."""
        for record in prefix['updates']:
            self.add_update(record)
        for record in prefix['evaluations']:
            self.add_evaluation(prefix['seed'], record['update'], record['metrics'])

    @property
    def seeds(self):
        return sorted(set(self.updates) | set(self.evaluations))

    @property
    def completed_updates(self):
        return sum(len(rows) for rows in self.updates.values())

    @property
    def episodes(self):
        return self.initialization.get('episodes', 0)+sum(
            row['samples']['episodes'] for rows in self.updates.values() for row in rows.values())

    def policy_files(self):
        if self.path is None:
            return []
        rows = []
        for folder in sorted(self.path.glob('seed_*')):
            if not folder.is_dir() or not folder.name[5:].isdigit():
                continue
            seed = int(folder.name[5:])
            names = {p.stem for p in folder.glob('policy_*.*') if p.suffix in ('.pt', '.csv', '.json')}
            for name in sorted(names):
                if not name[7:].isdigit():
                    continue
                update = int(name[7:])
                metrics = self.evaluations.get(seed, {}).get(update, {})
                rows.append((seed, update, metrics.get('return_mean'),
                    folder/(name+'.pt') if (folder/(name+'.pt')).is_file() else None,
                    folder/(name+'.csv') if (folder/(name+'.csv')).is_file() else None))
        if (self.path/'initial_policy.pt').exists():
            seed = self.configuration.get('experiment', {}).get('seeds', [0])[0]
            metrics = self.evaluations.get(seed, {}).get(0, {})
            csv = self.path/'initial_policy.csv'
            rows.append((seed, 0, metrics.get('return_mean'), self.path/'initial_policy.pt', csv if csv.exists() else None))
        return rows


def load_history(path):
    path = Path(path).resolve()
    protocol = read_json(path/'protocol.json')
    config = protocol['experiment']
    history = RunHistory(path, config, status='Incomplete')
    if (path/'resume_history.json').exists():
        history.add_prefix(read_json(path/'resume_history.json'))
    for folder in sorted(path.glob('seed_*')):
        if not folder.is_dir() or not folder.name[5:].isdigit():
            continue
        seed = int(folder.name[5:])
        for record in sorted(folder.glob('update_*.json')):
            value = read_json(record)
            if value['seed'] != seed:
                raise ValueError(f'Seed mismatch in {record.name}')
            history.add_update(value)
        for record in sorted(folder.glob('policy_*.json')):
            history.add_evaluation(seed, int(record.stem[7:]), read_json(record))
    if (path/'initial_policy.json').exists():
        history.add_evaluation(config['experiment']['seeds'][0], 0, read_json(path/'initial_policy.json'))
    if (path/'summary.json').exists():
        summary = read_json(path/'summary.json')
        history.status = summary.get('status', 'completed').capitalize()
        history.elapsed_seconds = summary.get('elapsed_seconds')
        if summary.get('mode') == 'initialization':
            history.initialization = summary
    if (path/'cancellation.json').exists():
        history.status = 'Stopped'
    if (path/'failure.json').exists():
        history.status = 'Failed'
    return history
