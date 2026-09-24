"""Load and validate settings from config/settings.yaml.

Settings are kept as plain nested dictionaries, e.g.::

    settings["events"]["return_zscore_threshold"]   # -> 2.5

DEFAULT_SETTINGS below is used for any value missing from the YAML file,
so an incomplete settings file still works.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

DEFAULT_SETTINGS: dict[str, Any] = {
    "data": {
        "period": "5y",
        "benchmark": "SPY",
        "download_attempts": 3,
        "retry_wait_seconds": 2,
    },
    "statistics": {
        "zscore_window": 60,
        "volume_window": 20,
        "volatility_window": 20,
        "trading_days_per_year": 252,
    },
    "events": {
        "return_zscore_threshold": 2.5,
        "volume_ratio_threshold": 2.0,
        "volume_zscore_threshold": None,
        "abs_return_threshold": None,
        "earnings_match_window_days": 1,
    },
    "paths": {
        "database": "data/stock_research.db",
        "reports_dir": "reports",
    },
    "report": {
        "max_events_in_summary": 30,
    },
}

VALID_PERIODS = {"1y", "2y", "5y", "10y", "ytd", "max"}

# The event rules that can be switched on/off in settings.yaml.
EVENT_RULE_KEYS = (
    "return_zscore_threshold",
    "volume_ratio_threshold",
    "volume_zscore_threshold",
    "abs_return_threshold",
)


class ConfigError(ValueError):
    """Raised when settings are missing or invalid."""


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` where values from ``override`` replace matching keys."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Read settings.yaml, fill in defaults and validate the result.

    Args:
        path: Path to a YAML settings file. Defaults to ``config/settings.yaml``.

    Raises:
        ConfigError: if the file cannot be parsed or contains invalid values.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    user_settings: dict = {}
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as handle:
                user_settings = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse {config_path}: {exc}") from exc
        if not isinstance(user_settings, dict):
            raise ConfigError(f"{config_path} must contain a YAML mapping at the top level.")
    elif path is not None:
        # The user explicitly asked for a file that does not exist.
        raise ConfigError(f"Settings file not found: {config_path}")

    settings = _deep_merge(DEFAULT_SETTINGS, user_settings)
    validate_settings(settings)
    return settings


def validate_settings(settings: dict[str, Any]) -> None:
    """Check that numbers are in sensible ranges. Raises ConfigError otherwise."""
    stats = settings["statistics"]
    for key in ("zscore_window", "volume_window", "volatility_window"):
        value = stats.get(key)
        if not isinstance(value, int) or value < 5:
            raise ConfigError(f"statistics.{key} must be a whole number >= 5 (got {value!r}).")
    if not isinstance(stats.get("trading_days_per_year"), int) or stats["trading_days_per_year"] <= 0:
        raise ConfigError("statistics.trading_days_per_year must be a positive whole number.")

    events = settings["events"]
    enabled = 0
    for key in EVENT_RULE_KEYS:
        value = events.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"events.{key} must be a positive number or null (got {value!r}).")
        enabled += 1
    if enabled == 0:
        raise ConfigError("At least one event rule in the 'events' section must be enabled.")
    window = events.get("earnings_match_window_days")
    if not isinstance(window, int) or window < 0:
        raise ConfigError("events.earnings_match_window_days must be a whole number >= 0.")

    data = settings["data"]
    if str(data.get("period")) not in VALID_PERIODS:
        raise ConfigError(
            f"data.period must be one of {sorted(VALID_PERIODS)} (got {data.get('period')!r})."
        )
    if not isinstance(data.get("download_attempts"), int) or data["download_attempts"] < 1:
        raise ConfigError("data.download_attempts must be a whole number >= 1.")

    max_events = settings["report"].get("max_events_in_summary")
    if not isinstance(max_events, int) or max_events < 1:
        raise ConfigError("report.max_events_in_summary must be a whole number >= 1.")


def apply_overrides(settings: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a copy of ``settings`` with command-line overrides applied.

    Only overrides that are not None are applied. Supported keywords:
    ``period``, ``benchmark``, ``sigma``, ``volume_ratio``, ``database``, ``reports_dir``.
    Pass ``benchmark=""`` to disable the benchmark.
    """
    updated = copy.deepcopy(settings)
    if overrides.get("period") is not None:
        updated["data"]["period"] = overrides["period"]
    if overrides.get("benchmark") is not None:
        updated["data"]["benchmark"] = overrides["benchmark"] or None
    if overrides.get("sigma") is not None:
        updated["events"]["return_zscore_threshold"] = overrides["sigma"]
    if overrides.get("volume_ratio") is not None:
        updated["events"]["volume_ratio_threshold"] = overrides["volume_ratio"]
    if overrides.get("database") is not None:
        updated["paths"]["database"] = str(overrides["database"])
    if overrides.get("reports_dir") is not None:
        updated["paths"]["reports_dir"] = str(overrides["reports_dir"])
    validate_settings(updated)
    return updated


def resolve_path(path_value: str | Path) -> Path:
    """Turn a (possibly relative) path from settings into an absolute path.

    Relative paths are interpreted relative to the project root, so the
    program behaves the same no matter which folder it is started from.
    """
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
