"""Transactional search checkpoints. Only the parent process writes; every completed trial survives a restart."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


def require_complete(directory):
    """A resumable search is not ready for selection until all exports were published."""
    progress = Path(directory) / "progress.json"
    if progress.exists() and not json.loads(progress.read_text(encoding="utf-8")).get("complete"):
        raise ValueError("Search is incomplete; resume it before selecting or evaluating candidates")


def fingerprint(pool, configurations) -> str:
    """Hash the actual PRE-holdout inputs, trial list and implementation, never a path or file mtime."""
    h = hashlib.sha256()
    for data in (pool.w_ex, pd.Series(pool.cash, index=pool.idx)):
        h.update(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
    for a in (pool.unf, pool.spread, pool.is_c):
        h.update(a.tobytes())
    detail = {"schema": 1, "columns": pool.cols, "families": pool.fam.tolist(),
              "universes": {k: v.tolist() for k, v in pool.universes.items()},
              "configurations": [asdict(c) for c in configurations]}
    h.update(json.dumps(detail, sort_keys=True).encode())
    for name in ("engine.py", "search.py", "pool.py", "checkpoint.py", "__init__.py"):
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


class Checkpoint:
    def __init__(self, path, signature: str):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS trials (id TEXT PRIMARY KEY, config TEXT NOT NULL, metrics TEXT NOT NULL, returns BLOB NOT NULL)")
        row = self.db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        if row and row[0] != signature:
            self.db.close()
            raise ValueError("Checkpoint inputs/code/trial list changed; use a new output directory. Existing trials are preserved.")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('fingerprint', ?)", (signature,))

    def completed(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT id FROM trials")}

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM trials").fetchone()[0]

    def put(self, key, cfg, metrics, returns):
        arrays = np.stack([returns[0.8], returns[0.3], returns["base"]]).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arrays, allow_pickle=False)
        with self.db:
            self.db.execute("INSERT INTO trials VALUES (?, ?, ?, ?)",
                            (key, json.dumps(asdict(cfg), sort_keys=True), json.dumps(metrics),
                             zlib.compress(buf.getvalue())))

    def export(self, out, idx):
        from dataclasses import fields
        from .engine import Config
        rows, r80, r30, base = [], {}, {}, {}
        for key, cfg, metrics, blob in self.db.execute("SELECT id,config,metrics,returns FROM trials ORDER BY id"):
            a = np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False)
            if a.shape != (3, len(idx)):
                raise ValueError(f"Invalid checkpoint return shape for {key}: {a.shape}")
            rows.append({"id": key, **json.loads(cfg), **json.loads(metrics)})
            r80[key], r30[key], base[key] = a
        met = pd.DataFrame(rows).set_index("id")
        dest = Path(out)
        def atomic_csv(obj, name):
            temp = dest / (name + ".tmp")
            obj.to_csv(temp)
            temp.replace(dest / name)
        atomic_csv(met, "metrics.csv")
        atomic_csv(met[[f.name for f in fields(Config)]], "configs.csv")
        for values, name in ((r80, "returns_tv80.parquet"), (r30, "returns_tv30.parquet"), (base, "books.parquet")):
            temp = dest / (name + ".tmp")
            pd.DataFrame(values, index=idx).to_parquet(temp)
            temp.replace(dest / name)
        return met

    def close(self):
        self.db.close()
