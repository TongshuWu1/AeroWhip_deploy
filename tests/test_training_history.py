"""Resume display boundaries and replay ownership; no workers or physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from helpers import project_temporary

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets
from project_paths import ROOT, project_reference
from training.artifacts import identity, save_json
from training.configuration import DEFAULT, load
from training.ui import window as ui
from training_history import ResumeHistory, HistoryTrainingWindow


def round_event(number, value=None):
    return dict(phase='round', round=number, episodes=128*number,
                steps=128*45*number, gradient_updates=number,
                return_mean=float(number if value is None else value),
                valid_fraction=1., components={'continuous_whip_credit': .5})


def evaluation(number):
    return dict(phase='evaluation', round=number, evaluation_return=float(number),
                best_return=float(number), valid=True, replay=f'round-{number:05d}.npz',
                components={'continuous_whip_credit': .5, 'completed_horizon': 1.})


class HistoryFixture:
    def setUp(self):
        self.temp = project_temporary()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = deepcopy(load(DEFAULT)[1])
        self.snapshot['execution_hashes'] = {'training/ppo.py': 'synthetic'}
        self.snapshot['identity'] = identity(self.snapshot)
        self.reader = ResumeHistory()

    def run_folder(self, name, numbers, *, parent=None, cutoff=None, extra=(), actor=None):
        directory = self.root/name
        directory.mkdir()
        save_json(directory/'configuration.json', self.snapshot)
        save_json(directory/'request.json', dict(
            resume=project_reference(parent/'latest.pt') if parent else None,
            initialize_actor=actor))
        records = list(extra)
        if parent:
            records.append(dict(phase='resumed', round=cutoff, episodes=cutoff*128))
        records.extend(round_event(n) for n in numbers)
        self.write_events(directory, records)
        torch.save(dict(identity=self.snapshot['identity'], round=max(numbers, default=0)),
                   directory/'latest.pt')
        return directory

    @staticmethod
    def write_events(directory, records):
        (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in records))

class HistoryTests(HistoryFixture, unittest.TestCase):
    def test_older_checkpoint_excludes_later_parent_collections(self):
        parent = self.run_folder('parent', range(1, 11))
        child = self.run_folder('child', [6, 7], parent=parent, cutoff=5)
        originals = {p:p.read_bytes() for p in (parent/'events.jsonl', child/'events.jsonl')}
        history = self.reader.load(child)
        self.assertEqual([e['round'] for e in history.records if e['phase']=='round'], list(range(1, 8)))
        self.assertEqual(history.directories, (parent, child))
        self.assertFalse(history.warnings)
        self.assertTrue(all(p.read_bytes()==b for p,b in originals.items()))

    def test_multiple_resumes_follow_each_checkpoint_boundary(self):
        first = self.run_folder('first', range(1, 11))
        second = self.run_folder('second', range(6, 13), parent=first, cutoff=5)
        third = self.run_folder('third', [9, 10], parent=second, cutoff=8)
        history = self.reader.load(third)
        self.assertEqual([e['round'] for e in history.records if e['phase']=='round'], list(range(1, 11)))
        self.assertEqual(history.directories, (first, second, third))

    def test_inherited_best_is_not_a_duplicate_or_a_new_trial(self):
        original = evaluation(2)
        parent = self.run_folder('parent', [1, 2, 3], extra=[original])
        copied = dict(original, inherited=True, replay='inherited-best.npz')
        child = self.run_folder('child', [4], parent=parent, cutoff=3, extra=[copied])
        rows = [e for e in self.reader.load(child).records if e['phase']=='evaluation']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['_history_run'], project_reference(parent))
        self.assertEqual(rows[0]['replay'], original['replay'])
        self.assertTrue(rows[0]['inherited'])

    def test_actor_transfer_keeps_a_separate_training_history(self):
        parent = self.run_folder('parent', range(1, 9))
        child = self.run_folder('child', [1, 2], actor=project_reference(parent/'latest.pt'))
        history = self.reader.load(child)
        self.assertEqual([e['round'] for e in history.records], [1, 2])
        self.assertEqual(history.directories, (child,))

    def test_missing_parent_warns_and_restores_when_files_return(self):
        parent = self.run_folder('parent', [1, 2])
        child = self.run_folder('child', [3], parent=parent, cutoff=2)
        path = parent/'configuration.json'
        payload = path.read_bytes()
        path.unlink()
        history = self.reader.load(child)
        self.assertTrue(history.warnings)
        self.assertEqual([e['round'] for e in history.records if e['phase']=='round'], [3])
        path.write_bytes(payload)
        self.assertEqual([e['round'] for e in self.reader.load(child).records if e['phase']=='round'], [1, 2, 3])

    def test_outside_parent_and_incompatible_identity_are_not_joined(self):
        parent = self.run_folder('parent', [1, 2])
        child = self.run_folder('child', [3], parent=parent, cutoff=2)
        changed = deepcopy(self.snapshot)
        changed['task']['request'][2] += .1
        changed['identity'] = identity(changed)
        save_json(parent/'configuration.json', changed)
        history = self.reader.load(child)
        self.assertTrue(history.warnings)
        self.assertEqual([e['round'] for e in history.records if e['phase']=='round'], [3])
        save_json(child/'request.json', dict(resume=str(ROOT.parent/'outside/latest.pt')))
        history = self.reader.load(child)
        self.assertIn('inside this project', history.warnings[0])
        self.assertEqual([e['round'] for e in history.records if e['phase']=='round'], [3])

    def test_missing_parent_log_warns_without_clearing_current_metrics(self):
        parent = self.run_folder('parent', [1, 2])
        child = self.run_folder('child', [3], parent=parent, cutoff=2)
        (parent/'events.jsonl').unlink()
        history = self.reader.load(child)
        self.assertTrue(history.warnings)
        self.assertEqual([e['round'] for e in history.records if e['phase']=='round'], [3])

    def test_cycle_does_not_recurse_forever(self):
        first = self.run_folder('first', [1])
        second = self.run_folder('second', [2], parent=first, cutoff=1)
        save_json(first/'request.json', dict(resume=project_reference(second/'latest.pt')))
        self.write_events(first, [dict(phase='resumed', round=0), round_event(1)])
        history = self.reader.load(second)
        self.assertTrue(any('cycle' in message for message in history.warnings))
        self.assertLess(len(history.records), 10)

    def test_before_resume_event_uses_checkpoint_not_parent_log_end(self):
        parent = self.run_folder('parent', range(1, 11))
        torch.save(dict(identity=self.snapshot['identity'], round=4), parent/'latest.pt')
        child = self.run_folder('child', [], parent=parent, cutoff=4)
        self.write_events(child, [dict(phase='initializing')])
        self.assertEqual([e['round'] for e in self.reader.load(child).records if e['phase']=='round'], [1, 2, 3, 4])

    def test_partial_json_line_and_later_append_refresh_the_cache(self):
        parent = self.run_folder('parent', [1])
        with (parent/'events.jsonl').open('a') as stream:
            stream.write('{"phase":"round"')
        self.assertEqual(len(self.reader.load(parent).records), 1)
        self.write_events(parent, [round_event(1), round_event(2)])
        self.assertEqual(len(self.reader.load(parent).records), 2)


class HistoryInterfaceTests(HistoryFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.parent = self.run_folder('parent', [1, 2, 3], extra=[evaluation(3)])
        self.child = self.run_folder('child', [4], parent=self.parent, cutoff=3)
        for name in ('PPO_RUNS',):
            replacement = self.root if name == 'PPO_RUNS' else self.root/'absent'
            patcher = patch.object(ui, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.window = HistoryTrainingWindow()
        self.window.timer.stop()
        self.addCleanup(self.window.close)
        self.window.refresh_runs(self.child)

    def test_plots_join_earlier_collections_and_preview_has_original_owner(self):
        with patch.object(self.window.preview, 'set_result') as result:
            self.window.select_run()
        self.assertEqual([n for n,_ in self.window.return_plot.series[0][1]], [1, 2, 3, 4])
        self.assertEqual(result.call_args.args[0], self.parent)
        self.assertTrue(result.call_args.args[1]['inherited'])
        self.assertIn(self.child.name, self.window.live_detail.text())
        self.assertIsNone(self.window.process)

    def test_parent_replay_opens_from_its_owner_and_fresh_failure_stays_latest(self):
        with patch('training.ui.replay_window.ReplayWindow') as replay:
            self.window.open_replay('latest')
            self.assertEqual(replay.call_args.args[0], self.parent)
        failure = dict(evaluation(4), valid=False, replay=None)
        records = [dict(phase='resumed', round=3), round_event(4), failure]
        self.write_events(self.child, records)
        with patch.object(self.window.preview, 'set_result') as result:
            self.window.poll()
        self.assertIsNone(result.call_args.args[1])
        self.assertIn('INVALID', self.window.latest_summary.text())

    def test_same_length_log_correction_refreshes_plot_values(self):
        self.write_events(self.child, [dict(phase='resumed', round=3), round_event(4, 9.)])
        self.window.poll()
        self.assertEqual(self.window.return_plot.series[0][1][-1][1],
                         9*self.snapshot['settings'].get('reward_scale', 1.))


if __name__ == '__main__':
    unittest.main()
