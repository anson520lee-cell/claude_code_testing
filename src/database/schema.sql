-- Schema for the stock research database (SQLite).
-- Dates are stored as ISO text 'YYYY-MM-DD'; timestamps as 'YYYY-MM-DDTHH:MM:SS' (UTC).
-- Every statement uses IF NOT EXISTS, so running this file again is safe.
-- Columns added in later versions are added to older databases by db._add_missing_event_columns().

-- One row per ticker ever analysed.
CREATE TABLE IF NOT EXISTS stocks (
    ticker          TEXT PRIMARY KEY,
    name            TEXT,
    exchange        TEXT,
    sector          TEXT,
    industry        TEXT,
    currency        TEXT,
    country         TEXT,
    quote_type      TEXT,
    first_analyzed  TEXT,
    last_analyzed   TEXT
);

-- Daily prices. (ticker, date) is unique, so re-downloading updates rows instead of duplicating them.
CREATE TABLE IF NOT EXISTS price_data (
    ticker      TEXT NOT NULL REFERENCES stocks(ticker),
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL NOT NULL,
    adj_close   REAL,
    volume      REAL,
    source      TEXT,
    updated_at  TEXT,
    PRIMARY KEY (ticker, date)
);

-- Abnormal price/volume events. One row per (ticker, event_date).
-- Quantitative columns are refreshed on every run (forward returns fill in as time passes).
-- Research columns (event_category ... notes) are protected once you edit them - see README.
CREATE TABLE IF NOT EXISTS events (
    event_id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                        TEXT NOT NULL REFERENCES stocks(ticker),
    event_date                    TEXT NOT NULL,
    close                         REAL,
    daily_return                  REAL,
    return_zscore                 REAL,
    volume                        REAL,
    avg_volume                    REAL,
    volume_ratio                  REAL,
    volume_zscore                 REAL,
    benchmark                     TEXT,
    benchmark_return              REAL,
    abnormal_return               REAL,
    fwd_return_1d                 REAL,     -- forward returns: price h trading days later / event close - 1
    fwd_return_2d                 REAL,
    fwd_return_3d                 REAL,
    fwd_return_5d                 REAL,
    fwd_return_7d                 REAL,
    fwd_return_20d                REAL,
    fwd_abnormal_1d               REAL,     -- forward return minus the benchmark's over the same days
    fwd_abnormal_2d               REAL,
    fwd_abnormal_3d               REAL,
    fwd_abnormal_5d               REAL,
    fwd_abnormal_7d               REAL,
    fwd_abnormal_20d              REAL,
    mfe_3d                        REAL,     -- max favourable excursion: max High next h days / close - 1
    mfe_5d                        REAL,
    mfe_7d                        REAL,
    mae_3d                        REAL,     -- max adverse excursion: min Low next h days / close - 1
    mae_5d                        REAL,
    mae_7d                        REAL,
    direction                     TEXT,     -- 'up', 'down' or 'flat'
    is_price_event                INTEGER,  -- 1 if a price rule fired
    is_volume_event               INTEGER,  -- 1 if a volume rule fired
    trigger_rules                 TEXT,     -- e.g. 'return_zscore,volume_ratio'
    anomaly_type                  TEXT,     -- data-derived description, never a cause
    trading_days_since_prev_event INTEGER,
    detection_settings            TEXT,     -- JSON: thresholds/windows used when detected
    event_category                TEXT NOT NULL DEFAULT 'Unknown',
    event_description             TEXT,
    source                        TEXT,
    verification_status           TEXT NOT NULL DEFAULT 'Unverified',
    notes                         TEXT,
    first_run_id                  INTEGER REFERENCES analysis_runs(run_id),
    last_run_id                   INTEGER REFERENCES analysis_runs(run_id),
    created_at                    TEXT,
    updated_at                    TEXT,
    UNIQUE (ticker, event_date)
);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_zscore ON events(return_zscore);

-- Fundamental data in "long" format: one row per metric per period.
--   period_type = 'snapshot' : provider's current value (period_end = retrieval date)
--   period_type = 'annual'   : fiscal-year statement value (period_end = fiscal year end)
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker        TEXT NOT NULL REFERENCES stocks(ticker),
    period_type   TEXT NOT NULL,
    period_end    TEXT NOT NULL,
    metric        TEXT NOT NULL,
    value         REAL,
    unit          TEXT,
    source        TEXT,
    retrieved_at  TEXT,
    PRIMARY KEY (ticker, period_type, period_end, metric)
);

-- One row per execution of analyze.py.
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL,     -- 'running', 'success', 'success_with_warnings', 'failed'
    period          TEXT,
    benchmark       TEXT,
    price_source    TEXT,
    start_date      TEXT,
    end_date        TEXT,
    trading_days    INTEGER,
    n_events        INTEGER,
    parameters      TEXT,              -- JSON copy of the settings used
    warnings        TEXT,              -- JSON list of warnings
    error           TEXT,
    report_path     TEXT
);
