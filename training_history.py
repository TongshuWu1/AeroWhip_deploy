"""Read-only resume history for the Training entry point.

This display adapter reuses the existing window and charts. It does not change
the pinned trainer sources, append events, or migrate checkpoints.
"""
from dataclasses import dataclass
import pickle

from project_paths import project_path, project_reference
from training.artifacts import events, identity, read_json
from training.ui import window as ui


@dataclass
class History:
    records: list
    signature: tuple
    directories: tuple
    warnings: tuple


class ResumeHistory:
    def __init__(self):
        self._cache = {}

    def _read(self, path, reader, signatures):
        try:
            stat = path.stat()
            signature = (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except FileNotFoundError:
            signatures.append((str(path), None))
            raise
        signatures.append(signature)
        cached = self._cache.get(path)
        if cached is None or cached[0] != signature:
            cached = (signature, reader())
            self._cache[path] = cached
        return cached[1]

    def load(self, directory):
        directory = project_path(directory)
        signatures, warnings, directories = [], [], []

        def json_file(path):
            return self._read(path, lambda: read_json(path), signatures)

        def evaluation_key(event):
            return tuple(event.get(k) for k in
                         ('phase', 'round', 'lane', 'evaluation_return', 'valid'))

        def walk(current, cutoff=None, seen=()):
            if current in seen:
                raise ValueError('Resume history contains a cycle')
            try:
                local = self._read(current/'events.jsonl', lambda: events(current), signatures)
            except FileNotFoundError:
                if current != directory:
                    raise
                local = []
            try:
                request = json_file(current/'request.json')
            except FileNotFoundError:
                request = {}
            inherited = []
            if request.get('resume'):
                try:
                    checkpoint = project_path(request['resume'])
                    parent = checkpoint.parent
                    current_config = json_file(current/'configuration.json')
                    parent_config = json_file(parent/'configuration.json')
                    if (identity(current_config) != current_config.get('identity')
                            or identity(parent_config) != parent_config.get('identity')
                            or current_config['identity'] != parent_config['identity']):
                        raise ValueError('Resume history has incompatible run identities')
                    resumed = next((e for e in local if e.get('phase') == 'resumed'), None)
                    if resumed is not None:
                        boundary = resumed['round']
                    else:
                        # Before the worker's resumed event, use the selected
                        # checkpoint, never the parent's last logged collection.
                        import torch
                        saved = self._read(checkpoint, lambda: torch.load(
                            checkpoint, map_location='cpu', weights_only=True), signatures)
                        if saved['identity'] != current_config['identity']:
                            raise ValueError('Resume checkpoint identity differs from its run')
                        boundary = saved['round']
                    if type(boundary) is not int or boundary < 0:
                        raise ValueError('Invalid resume collection boundary')
                    inherited = walk(parent, min(cutoff, boundary) if cutoff is not None else boundary,
                                     seen+(current,))
                except (OSError, ValueError, KeyError, TypeError, RuntimeError,
                        EOFError, pickle.UnpicklingError) as exc:
                    warnings.append(f'Earlier history unavailable for {current.name}: {exc}')
            directories.append(current)
            known = {evaluation_key(e) for e in inherited if e.get('phase') == 'evaluation'}
            result = list(inherited)
            for event in local:
                number = event.get('round')
                if cutoff is not None and isinstance(number, (int, float)) and number > cutoff:
                    continue
                if (event.get('phase') == 'evaluation' and event.get('inherited')
                        and evaluation_key(event) in known):
                    continue
                # Source ownership exists only in memory; original JSONL and
                # run-local replay references remain untouched.
                row = dict(event, _history_run=project_reference(current))
                result.append(row)
            return result

        records = walk(directory)
        for row in records:
            if row.get('phase') == 'evaluation' and project_path(row['_history_run']) != directory:
                row['inherited'] = True
        return History(records, tuple(signatures), tuple(directories), tuple(warnings))


class HistoryTrainingWindow(ui.TrainingWindow):
    def __init__(self, *args, **kwargs):
        self.resume_history = ResumeHistory()
        self.history_signature = None
        self.history_warning = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def event_directory(directory, event):
        return project_path(event['_history_run']) if event and event.get('_history_run') else directory

    def poll(self):
        directory = self.runs.currentData()
        if directory is None:
            return super().poll()
        try:
            history = self.resume_history.load(directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.live_detail.setText(f'History unavailable: {exc}')
            return
        signature = (directory, history.signature)
        if signature != self.history_signature:
            self.history_signature = signature
            self.last_event_count = -1
            self._event_signature = None
        # The existing chart renderer takes its reader from this UI module.
        # Inject only for this synchronous UI call; trainer event readers are
        # unchanged, and no historical events are copied into the current run.
        reader = ui.events
        try:
            ui.events = lambda selected: history.records
            super().poll()
        finally:
            ui.events = reader
        lineage = ' → '.join(p.name for p in history.directories)
        details = f'History: {lineage} (exact resumes only)'
        if history.warnings:
            details += '\n'+'\n'.join(history.warnings)
        self.live_detail.setText(self.live_detail.text().split('\nHistory:', 1)[0]+'\n'+details)
        for plot in (self.return_plot, self.reward_plot, self.loss_plot,
                     self.mechanics_plot, self.valid_plot, self.entropy_plot, self.temperature_plot):
            plot.setToolTip(plot.toolTip().split('\nHistory:', 1)[0]+'\n'+details)
        for index, event in enumerate(getattr(self, 'evaluation_rows', [])):
            item = self.evaluations.item(index, 0)
            if item is not None:
                item.setToolTip('Recorded in '+event.get('_history_run', project_reference(directory)))
        warning = '\n'.join(history.warnings)
        if warning and warning != self.history_warning:
            self.log.appendPlainText(warning)
        self.history_warning = warning

    def update_overview(self, directory, history, settings):
        latest = next((e for e in reversed(history) if e['phase'] == 'evaluation'), None)
        super().update_overview(self.event_directory(directory, latest), history, settings)
        if self.process is None:
            phase = history[-1]['phase'] if history else 'no events'
            self.live_detail.setText(f'Viewing saved run: {directory.name} · '
                                     f'{phase.replace("_", " ")} · no worker owned by this window')

    def open_replay(self, selection):
        try:
            rows = getattr(self, 'evaluation_rows', [])
            if not rows:
                raise ValueError('No completed evaluation is saved yet')
            deterministic = [e for e in rows if e['phase'] == 'evaluation']
            if selection == 'best':
                event = max(deterministic, key=lambda e: e['evaluation_return'])
            elif selection == 'latest':
                event = deterministic[-1]
            else:
                index = self.evaluations.currentRow()
                if index < 0:
                    raise ValueError('Select an evaluation row')
                event = rows[index]
            if not event.get('replay'):
                raise ValueError('This evaluation has no valid numerical replay prefix')
            from training.ui.replay_window import ReplayWindow
            owner = self.event_directory(self.runs.currentData(), event)
            window = ReplayWindow(owner, event['replay'], event, self)
            self.replays.append(window)
            window.show()
        except Exception as exc:
            self.error(exc)


def main(**kwargs):
    # Reuse the existing start/resume entry point and all worker ownership.
    factory = ui.TrainingWindow
    ui.TrainingWindow = HistoryTrainingWindow
    try:
        return ui.main(**kwargs)
    finally:
        ui.TrainingWindow = factory
