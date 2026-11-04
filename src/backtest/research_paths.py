"""Portable location for research inputs; never points at the live terminal's private data."""
import os
from pathlib import Path


def research_path(*parts):
    override = os.environ.get("TRADING_RESEARCH_ROOT")
    legacy = Path("/home/user/research_data")
    root = Path(override).expanduser() if override else legacy if legacy.is_dir() else Path(__file__).resolve().parents[2] / "research_data"
    return str(root.joinpath(*parts))
