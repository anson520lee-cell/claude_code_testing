"""Regression checks for restart safety, missing-data hindsight, and drawdown from initial capital."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.optimize.checkpoint import Checkpoint
from src.backtest.optimize.checkpoint import require_complete
from src.backtest.optimize.engine import Config, Pool, run_book, stats
from src.backtest.optimize.final import perf as final_perf
from src.backtest.optimize.overfit import perf as overfit_perf


def test_committed_trials_survive_restart_and_changed_inputs_are_rejected(tmp_path):
    path = tmp_path / "checkpoint.sqlite3"
    idx = pd.date_range("2010-01-01", periods=100, freq="W-FRI")
    a = np.linspace(-0.1, 0.1, len(idx)).astype(np.float32)
    cp = Checkpoint(path, "same-inputs")
    cp.put("c00000", Config(), {"score": 2.0}, {0.8: a, 0.3: a / 3, "base": a / 8})
    cp.close()
    cp = Checkpoint(path, "same-inputs")
    assert cp.completed() == {"c00000"} and cp.count() == 1
    out = cp.export(tmp_path, idx)
    cp.close()
    assert out.loc["c00000", "score"] == 2.0
    assert np.array_equal(pd.read_parquet(tmp_path / "returns_tv80.parquet")["c00000"].to_numpy(), a)
    with pytest.raises(ValueError, match="changed"):
        Checkpoint(path, "different-inputs")
    cp = Checkpoint(path, "same-inputs")
    assert cp.count() == 1
    cp.close()


def test_missing_return_cannot_reallocate_a_position_using_hindsight():
    rng = np.random.default_rng(71)
    idx = pd.date_range("2010-01-01", periods=180, freq="W-FRI")
    x = pd.DataFrame(rng.normal(0.001, 0.01, (len(idx), 2)), index=idx, columns=["a", "b"])
    cash = pd.Series(0.0, index=idx)
    m = pd.DataFrame({"family": ["macro", "macro"], "crypto": [False, False],
                      "unfunded": [False, False], "spread": [0.01, 0.01]}, index=x.columns)
    cfg = Config(universe="all", method="ew")
    run_book(Pool(x, cash, m, {"all": list(x.columns)}), cfg)
    x.iloc[150, 0] = np.nan
    with pytest.raises(ValueError, match="Missing return for allocated"):
        run_book(Pool(x, cash, m, {"all": list(x.columns)}), cfg)


def test_first_period_loss_counts_toward_drawdown():
    idx = pd.date_range("2010-01-01", periods=104, freq="W-FRI")
    r = np.zeros(len(idx))
    r[0] = -0.5
    cash = pd.Series(0.0, index=idx)
    assert stats(r, cash.to_numpy(), idx, "2010-01-01", "2013-01-01")["max_dd"] == -0.5
    for perf in (final_perf, overfit_perf):
        assert perf(pd.Series(r, index=idx), cash)["max_dd"] == -0.5


def test_partial_search_exports_cannot_enter_candidate_selection(tmp_path):
    import json
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"complete": False, "completed": 25, "total": 100}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        require_complete(tmp_path)
    progress.write_text(json.dumps({"complete": True, "completed": 100, "total": 100}), encoding="utf-8")
    require_complete(tmp_path)


def test_search_restarts_without_recomputing_committed_trials(tmp_path, monkeypatch):
    import json
    import sys
    from dataclasses import replace
    from src.backtest.optimize import search
    rng = np.random.default_rng(88)
    idx = pd.date_range("2010-01-01", periods=760, freq="W-FRI")
    x = pd.DataFrame(rng.normal(0.002, 0.02, (len(idx), 2)), index=idx, columns=["a", "b"])
    cash = pd.Series(0.0, index=idx)
    m = pd.DataFrame({"family": ["macro", "macro"], "crypto": [False, False],
                      "unfunded": [False, False], "spread": [0.01, 0.01]}, index=x.columns)
    pool = Pool(x, cash, m, {"all": list(x.columns)})
    choices = [Config(universe="all", method="erc")]
    choices += [replace(choices[0], method=name) for name in ("ivol", "ew")]
    monkeypatch.setattr(search, "configs", lambda *_: choices)
    monkeypatch.setattr(search, "_init", lambda *_: setattr(search, "_POOL", pool))

    class LocalWorkers:
        def __init__(self, *args, initializer, initargs):
            initializer(*initargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def imap_unordered(self, fn, todo, chunksize):
            return map(fn, reversed(todo))

    monkeypatch.setattr(search.mp, "Pool", LocalWorkers)
    one, called = search._one, []
    def counted(cfg):
        called.append(cfg.key())
        return one(cfg)
    monkeypatch.setattr(search, "_one", counted)
    cmd = ["search", "unused_pool", "unused_cash", str(tmp_path), "0", "--workers", "1"]
    monkeypatch.setattr(sys, "argv", cmd + ["--max-new", "1"])
    search.main()
    assert len(called) == 1
    assert not json.loads((tmp_path / "progress.json").read_text())["complete"]
    assert not (tmp_path / "metrics.csv").exists()
    monkeypatch.setattr(sys, "argv", cmd)
    search.main()
    assert len(called) == len(set(called)) == 3
    assert json.loads((tmp_path / "progress.json").read_text())["complete"]
    assert pd.read_csv(tmp_path / "metrics.csv")["id"].tolist() == ["c00000", "c00001", "c00002"]


def test_downloaded_option_index_is_cut_at_the_frozen_asof_date():
    from scripts.research.collect import freeze_index_history
    index = pd.date_range("2024-01-01", "2025-01-05")
    content = pd.DataFrame({"DATE": index.strftime("%m/%d/%Y"), "PUT": np.arange(len(index)) + 100}).to_csv(index=False).encode()
    frame = freeze_index_history(content, "2025-01-01")
    assert pd.to_datetime(frame["DATE"]).max() == pd.Timestamp("2024-12-31")
    assert len(frame) == 366
