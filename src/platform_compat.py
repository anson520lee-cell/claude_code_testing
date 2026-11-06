"""Small helpers so the same code runs on macOS, Linux and Windows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_utf8() -> None:
    """On Windows, Python reads and writes text files in the local code page (cp950 for Traditional Chinese,
    cp1252 ...) unless UTF-8 mode is on - every file here is UTF-8. Re-run the script in UTF-8 mode if needed
    (child processes inherit PYTHONUTF8)."""
    os.environ.setdefault("PYTHONUTF8", "1")
    if sys.platform == "win32" and not sys.flags.utf8_mode:
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))


def redirect_output(log_file: str | None) -> None:
    """``--log FILE``: append stdout and stderr to a file (the Windows background task and launcher use
    pythonw.exe, which has no console)."""
    if not log_file:
        return
    target = Path(log_file)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(target, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = handle


def load_dotenv(file: Path | None = None) -> None:
    """Read KEY=VALUE lines from the project's .env (e.g. SEC_USER_AGENT) into the environment, without
    overriding variables that are already set. The macOS scripts source .env themselves; on Windows this
    is how the settings get there."""
    file = file or PROJECT_ROOT / ".env"
    try:
        text = file.read_text(encoding="utf-8-sig")       # -sig: PowerShell 5 writes a BOM
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
