"""Project-owned files are relative to this checkout, never the process cwd."""

import os
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parent


def project_path(value):
    """Resolve a project-relative or native absolute path, including spaces.

    Accept either separator for relative paths. Foreign absolute paths require
    explicit relocation by the recording loader, not a guessed local filename.
    """
    text = os.fspath(value)
    windows = PureWindowsPath(text)
    if windows.drive and (os.name != 'nt' or not windows.is_absolute()):
        raise ValueError(f'Path is not absolute on this computer: {text}')
    path = Path(text.replace('\\', '/')).expanduser()
    if not path.is_absolute():
        path = ROOT/path
    return path.resolve()
