"""Project-owned files are relative to this checkout, never the process cwd."""

import os
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parent


def project_path(value):
    """Resolve a path inside this checkout, including existing symlink targets.

    Native absolute paths from file dialogs are accepted only inside ROOT.
    Relative inputs always use ROOT, never cwd or a user's home directory.
    This is application-level validation, not an OS filesystem sandbox.
    """
    text = os.fspath(value)
    windows = PureWindowsPath(text)
    if windows.drive and (os.name != 'nt' or not windows.is_absolute()):
        raise ValueError(f'Path is not absolute on this computer: {text}')
    if text.startswith('~'):
        raise ValueError('Use a path inside this project, not a home-directory path')
    path = Path(text.replace('\\', '/'))
    if not path.is_absolute():
        path = ROOT/path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Path must stay inside this project: '+text)
    return path


def project_reference(value):
    """Portable reference for newly written metadata; never rewrite old evidence."""
    return project_path(value).relative_to(ROOT.resolve()).as_posix()


def contained_child(directory, relative):
    """Resolve a relative artifact path beneath its owning folder."""
    directory = project_path(directory)
    text = os.fspath(relative).replace('\\', '/')
    if not text or Path(text).is_absolute() or PureWindowsPath(text).drive:
        raise ValueError('Expected a relative file inside its owning folder')
    path = project_path(directory/text)
    if not path.is_relative_to(directory) or path == directory:
        raise ValueError('Artifact reference escapes its owning folder')
    return path
