"""Atomic artifact writes with bounded retries for file-sharing conflicts."""

from pathlib import Path
from time import sleep


def atomic_save(path, writer):
    """Write through a sibling temporary file; exhausted retries remain fatal.

    ``writer`` serializes to the supplied temporary Path. This shared boundary
    covers JSON and Torch artifacts without changing their formats.
    """
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    for attempt in range(6):
        try:
            writer(temporary)
            temporary.replace(path)
            return
        except OSError as exc:
            sharing=isinstance(exc,PermissionError) or getattr(exc,'winerror',None) in (5,32,33)
            if not sharing or attempt==5:
                raise
            sleep(min(.05*2**attempt,.4))
