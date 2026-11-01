"""Cross-platform, non-blocking exclusive file locks (fcntl on macOS / Linux, msvcrt on Windows).
The operating system releases a lock automatically when the process holding it dies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO


def acquire(path: Path) -> IO | None:
    """Lock ``path``; returns the open handle (keep it; pass it to ``release``) or None if another
    process holds the lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def release(handle: IO | None) -> None:
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        handle.close()
