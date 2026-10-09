"""Recording locations, separate from immutable source provenance."""

import json
import os
from pathlib import Path, PureWindowsPath

from project_paths import ROOT, project_path, project_reference, contained_child


SETTINGS_PATH = ROOT/'configs/local_paths.json'
DEFAULT_RECORDINGS = ROOT/'data/recordings'
PREFIX = 'recordings://'


def collection_root(folder):
    """Accept a collection, its flight_batches directory, or a folder within it."""
    path = project_path(folder)
    for parent in (path, *path.parents):
        if not parent.is_relative_to(ROOT):
            break
        if (parent/'data/flight_batches').is_dir():
            return parent
    return path


def recording_root():
    """Use a project-local choice; never infer a root from historical metadata."""
    settings_path = project_path(SETTINGS_PATH)
    if settings_path.is_file():
        try:
            settings = json.loads(settings_path.read_text(encoding='utf-8'))
            return project_path(settings['recordings'])
        except (ValueError, KeyError, TypeError):
            pass  # An old outside-project or malformed preference is not usable.
    return project_path(DEFAULT_RECORDINGS)


def remember_recording_root(folder):
    root = collection_root(folder)
    if not root.is_dir():
        raise ValueError(f'Recording folder not found: {root}')
    value = project_reference(root)
    settings_path = project_path(SETTINGS_PATH)
    settings = json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.is_file() else {}
    settings['recordings'] = value
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = project_path(settings_path.with_suffix('.json.tmp'))
    temporary.write_text(json.dumps(settings, indent=2)+'\n', encoding='utf-8')
    temporary.replace(settings_path)
    return root


def select_recording_root(value=None, *, parent=None, for_fitting=False):
    """No-argument PyCharm launches ask only when the collection is unavailable."""
    root = collection_root(value) if value is not None else recording_root()
    def available(candidate):
        if not candidate.is_dir():
            return False
        if for_fitting:
            try:
                flight_batches_path(candidate)
            except FileNotFoundError:
                return False
        return True

    if available(root):
        return root
    if value is not None:
        if for_fitting:
            flight_batches_path(root)
        raise FileNotFoundError(f'Recording folder not found: {root}')
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
    app = QApplication.instance() or QApplication([])
    while True:
        selected = QFileDialog.getExistingDirectory(parent, 'Recordings folder', str(ROOT))
        if not selected:
            return None
        try:
            candidate = collection_root(selected)
        except ValueError as exc:
            QMessageBox.warning(parent, 'Recordings folder', str(exc))
            continue
        if available(candidate):
            return remember_recording_root(candidate)
        QMessageBox.warning(parent, 'Recordings folder',
                            'Select the collection containing data/flight_batches.')


def flight_batches_path(root):
    root = project_path(root)
    path = project_path(root if root.name == 'flight_batches' else root/'data/flight_batches')
    if not path.is_dir():
        raise FileNotFoundError(f'Flight batches not found: {path}. Select the collection containing data/flight_batches.')
    return path


def _under(root, relative):
    parts = relative.replace('\\', '/').split('/')
    if not relative or any(p in ('', '..') for p in parts) or PureWindowsPath(relative).drive:
        raise ValueError(f'Invalid recording-relative reference: {relative}')
    return contained_child(root, relative)


def resolve_recording_path(value, *, root=None):
    """Resolve exact layout references; never search by basename or change hashes.

    An explicit root rebases recognized legacy data/flight_batches and
    runs/rehearsals_pva references even if the old collection still exists.
    Native inputs must stay inside this project. Legacy metadata may be rebased
    only through its exact recognized layout and an explicit contained root.
    """
    text = os.fspath(value)
    if text.startswith(PREFIX):
        chosen = project_path(root) if root is not None else recording_root()
        return _under(chosen, text[len(PREFIX):])
    windows = PureWindowsPath(text)
    native = None
    try:
        native = project_path(text)
    except ValueError:
        pass
    absolute = windows.is_absolute() or Path(text).is_absolute()
    if not absolute:
        return project_path(text)
    if root is None:
        if native is not None:
            return native
        raise FileNotFoundError(f'Cannot use foreign absolute path without a recording root: {text}')
    chosen = project_path(root)
    parts = text.replace('\\', '/').split('/')
    matches = [i for i in range(len(parts)-1)
               if tuple(p.lower() for p in parts[i:i+2]) in
               (('data', 'flight_batches'), ('runs', 'rehearsals_pva'))]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous recording layout reference: {text}')
    if matches:
        return _under(chosen, '/'.join(parts[matches[0]:]))
    if native is not None:
        return native
    raise FileNotFoundError(f'Cannot relocate {text}; select the recording collection with its original relative layout')


def recording_reference(path, *, root=None):
    """Portable references for new catalog entries; historical entries stay intact."""
    path = project_path(path)
    chosen = project_path(root) if root is not None else recording_root()
    if path.is_relative_to(chosen):
        return PREFIX+path.relative_to(chosen).as_posix()
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    return project_reference(path)


def rehearsal_archive(reference, take_path):
    """Resolve the protocol's exact rehearsal directory alongside relocated takes."""
    root = collection_root(project_path(take_path).parent)
    text = os.fspath(reference).replace('\\', '/').rstrip('/')+'/rehearsal.npz'
    if not (PureWindowsPath(text).is_absolute() or Path(text).is_absolute() or text.startswith(PREFIX)):
        return _under(root, text)
    return resolve_recording_path(text, root=root)
