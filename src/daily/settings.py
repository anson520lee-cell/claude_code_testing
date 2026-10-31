"""Loads and validates config/terminal.yaml (the research terminal's settings)."""

from __future__ import annotations

import copy
import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.config import ConfigError, resolve_path
from src.market import calendar

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "terminal.yaml"


def load_terminal_settings(path: str | Path | None = None) -> dict[str, Any]:
    file = Path(path) if path else DEFAULT_PATH
    if not file.exists():
        raise ConfigError(f"Settings file not found: {file}")
    with file.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    settings = copy.deepcopy(settings)
    validate(settings)
    agent = os.environ.get("SEC_USER_AGENT")
    if agent:
        settings["data"]["sec_user_agent"] = agent
    settings["core_universe"] = dict(settings["universe"])
    merge_auto_universe(settings)
    for day, name in (settings.get("calendar", {}).get("extra_closures") or {}).items():
        calendar.SPECIAL_CLOSURES[date.fromisoformat(str(day))] = str(name)
        calendar.holidays.cache_clear()
    return settings


def validate(settings: dict[str, Any]) -> None:
    for section in ("paths", "data", "market", "universe", "liquidity", "scanner", "weights", "penalties", "tiers"):
        if section not in settings:
            raise ConfigError(f"terminal.yaml: missing section '{section}'")
    weights = settings["weights"]
    total = sum(float(v) for v in weights.values())
    if abs(total - 100) > 1e-6:
        raise ConfigError(f"terminal.yaml: weights must sum to 100 (they sum to {total:g})")
    if any(float(v) < 0 for v in weights.values()):
        raise ConfigError("terminal.yaml: weights must not be negative")
    if any(float(v) < 0 for v in settings["penalties"].values()):
        raise ConfigError("terminal.yaml: penalties must not be negative")
    universe = settings["universe"]
    if not isinstance(universe, dict) or not universe:
        raise ConfigError("terminal.yaml: 'universe' must map tickers to a sector / industry ETF")
    odd = [repr(k) for k in universe if not isinstance(k, str)]
    if odd:
        raise ConfigError(f"terminal.yaml: universe keys {odd} are not text - quote tickers such as "
                          "\"ON\", \"YES\", \"NO\" (YAML reads them as booleans)")
    known = set(settings["market"]["sector_etfs"]) | set(settings["market"].get("industry_etfs", {}))
    unknown = sorted({etf for etf in universe.values() if etf not in known})
    if unknown:
        raise ConfigError(f"terminal.yaml: universe uses ETFs not listed under market: {unknown}")
    tiers = settings["tiers"]
    if not (tiers["A"] > tiers["B"] > tiers["C"]):
        raise ConfigError("terminal.yaml: tiers must satisfy A > B > C")


def path(settings: dict[str, Any], key: str) -> Path:
    return resolve_path(settings["paths"][key])


def market_symbols(settings: dict[str, Any]) -> list[str]:
    m = settings["market"]
    symbols = [*m["indices"], *m["volatility"], *m["credit_rates"], *m["sector_etfs"], *m.get("industry_etfs", {})]
    return list(dict.fromkeys(symbols))


def all_symbols(settings: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys([*market_symbols(settings), *settings["universe"]]))


def merge_auto_universe(settings: dict[str, Any]) -> None:
    """Append the auto-built universe (config/universe_auto.yaml) to the core list, up to max_size."""
    from src.market.universe_builder import load_auto

    builder = settings.get("universe_builder") or {}
    if not builder.get("enabled", True):
        return
    auto = load_auto()
    if not auto:
        return
    max_size = int(builder.get("max_size", 500))
    universe = settings["universe"]
    known = set(settings["market"]["sector_etfs"]) | set(settings["market"].get("industry_etfs", {}))
    for symbol, etf in (auto.get("symbols") or {}).items():
        if len(universe) >= max_size:
            break
        if isinstance(symbol, str) and symbol not in universe and etf in known:
            universe[symbol] = etf
    settings["universe_auto_info"] = {k: auto.get(k) for k in ("generated_at", "rules", "screened", "passed")}
