"""Regression tests for chart labels (found during the acceptance test's visual review).

The chart functions normally save and close their figure; here ``_save`` is
replaced so the finished figure can be inspected instead.
"""

import pytest

from src.analysis.statistics import add_statistics
from src.data.prices import standardize_price_frame
from src.events.detector import detect_events
from src.visualization import charts
from tests.conftest import make_yf_prices

RULES = {"return_zscore_threshold": 2.5, "volume_ratio_threshold": 2.0,
         "volume_zscore_threshold": None, "abs_return_threshold": None}


@pytest.fixture
def captured(monkeypatch):
    figures = []

    def keep(fig, path):
        fig.canvas.draw()
        figures.append(fig)
        return path

    monkeypatch.setattr(charts, "_save", keep)
    yield figures
    for fig in figures:
        charts.plt.close(fig)


def stats(n, **kwargs):
    prices = standardize_price_frame(make_yf_prices(n=n, **kwargs))
    return add_statistics(prices, "adj_close", zscore_window=60, volume_window=20, volatility_window=20)


def labels(axis):
    return [t.get_text() for t in axis.get_ticklabels() if t.get_text()]


def test_percent_axis_labels_are_distinct_for_low_volatility(captured, tmp_path):
    df = stats(300, daily_vol=0.0005)  # e.g. a T-bill ETF: daily moves of a few hundredths of a percent
    charts.returns_distribution_chart(df, "LOWVOL", tmp_path / "r.png")
    x_labels = labels(captured[0].axes[0].xaxis)
    assert len(x_labels) >= 3
    assert len(set(x_labels)) == len(x_labels)  # no repeated "0%, 0%, 0%"


def test_few_days_show_dates_not_clock_times(captured, tmp_path):
    df = stats(2)
    charts.price_chart(df, detect_events(df, "TWO", RULES), "TWO", tmp_path / "p.png")
    x_labels = labels(captured[0].axes[0].xaxis)
    assert x_labels and not any(":" in label for label in x_labels)


def test_undefined_statistics_show_na_not_nan(captured, tmp_path):
    df = stats(2)  # a single daily return: standard deviation is undefined
    charts.returns_distribution_chart(df, "TWO", tmp_path / "r.png")
    ax = captured[0].axes[0]
    text = " ".join(t.get_text() for t in ax.texts)
    legend = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert "n/a" in text
    assert "nan" not in text.lower().replace("n/a", "") and "nan" not in legend.lower()


def test_volume_axis_is_labelled_up_to_the_threshold(captured, tmp_path):
    df = stats(200)
    events = detect_events(df, "T", RULES)
    charts.event_chart(df, events, "T", tmp_path / "e.png", zscore_threshold=2.5, volume_ratio_threshold=50)
    scatter = captured[0].axes[1]
    assert max(scatter.get_yticks()) >= 50
