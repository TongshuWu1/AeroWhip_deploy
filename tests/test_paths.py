"""Small containment/packaging checks; no saved experiments or live renderer."""
import hashlib
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from helpers import project_temporary
from project_paths import ROOT, project_path, project_reference, contained_child
from model_catalog import packaged_stages, verified_file
from training.configuration import DEFAULT, CURRENT_PRESETS


class PathTests(unittest.TestCase):
    def test_references_are_project_relative_independent_of_cwd(self):
        before = Path.cwd()
        try:
            os.chdir(ROOT.parent)
            self.assertEqual(project_path('configs\\simulation.json'), ROOT/'configs/simulation.json')
            self.assertEqual(project_reference(ROOT/'configs/simulation.json'), 'configs/simulation.json')
        finally:
            os.chdir(before)
        self.assertEqual(DEFAULT, CURRENT_PRESETS[1])

    def test_outside_home_foreign_and_owner_escape_are_rejected(self):
        for path in (ROOT.parent/'outside.json', '../outside.json', '~/file.json', 'Z:relative.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                project_path(path)
        with self.assertRaises(ValueError):
            contained_child(ROOT/'models', '../README.md')
        with self.assertRaises(ValueError):
            contained_child(ROOT, ROOT/'README.md')

    def test_symlink_escape_is_rejected_if_platform_permits_links(self):
        with project_temporary() as folder:
            link = Path(folder)/'escape'
            try:
                link.symlink_to(ROOT.parent, target_is_directory=True)
            except OSError as exc:
                self.skipTest('Creating symlinks is unavailable: '+str(exc))
            try:
                with self.assertRaises(ValueError):
                    project_path(link/'outside.json')
            finally:
                link.unlink()

    def test_manifest_discovery_and_hash_guard_use_only_local_synthetic_files(self):
        with project_temporary() as folder:
            root = Path(folder)
            package = root/'models/sample'
            package.mkdir(parents=True)
            checkpoint = package/'stage1.pt'
            checkpoint.write_bytes(b'synthetic checkpoint')
            reference = 'stage1.pt'
            manifest = dict(schema='aerowhip_simulation_milestone_models_v1',
                stages=[dict(stage=1, checkpoint=reference,
                    evidence={'configuration_and_snapshot': 'evidence/stage1/configuration.json'})],
                files={reference: dict(bytes=checkpoint.stat().st_size,
                    sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())})
            (package/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            entries = packaged_stages(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(verified_file(entries[0], reference), checkpoint)
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verified_file(entries[0], reference)

    def test_take_replay_without_initial_folder_can_open_picker(self):
        from PySide6 import QtWidgets
        from take_processing.replay_window import ReplayWindow
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with project_temporary() as folder, \
             patch('take_processing.replay_window.QtInteractor', return_value=MagicMock(interactor=QtWidgets.QWidget())):
            window = ReplayWindow()
            try:
                with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=folder):
                    window.open_folder()
                self.assertEqual(window.folder, Path(folder))
            finally:
                window.close()


if __name__ == '__main__':
    unittest.main()
