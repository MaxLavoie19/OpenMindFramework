import re

from openmind.dashboard.constant.dashboard_constant import CHART_HEIGHT, CHART_PAD
from openmind.dashboard.mapper.svg_chart_mapper import SvgChartMapper

BARS = re.compile(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"')


def heights(chart: str) -> list[float]:
    return [float(bar[3]) for bar in BARS.findall(chart)]


def test_a_bar_twice_another_s_value_is_twice_as_tall() -> None:
    chart = SvgChartMapper().stacked("Games", ["round 1", "round 2"], (("decisive", [10, 20]),))

    first, second = heights(chart)
    assert round(second / first, 6) == 2.0
    assert second == CHART_HEIGHT - 2 * CHART_PAD  # The largest value fills the chart.


def test_a_stacked_bar_puts_its_parts_one_above_the_other_and_names_them() -> None:
    chart = SvgChartMapper().stacked("Games", ["round 1"], (("decisive", [3]), ("drawn", [1])))

    bars = BARS.findall(chart)
    assert len(bars) == 2
    decisive, drawn = bars
    assert float(drawn[1]) < float(decisive[1])  # The second part sits on the first.
    assert round(float(decisive[3]) / float(drawn[3])) == 3
    assert "round 1: decisive 3" in chart and "round 1: drawn 1" in chart
    assert "decisive" in chart and "drawn" in chart


def test_a_column_with_nothing_in_it_draws_no_bar() -> None:
    chart = SvgChartMapper().stacked("Games", ["round 1", "round 2"], (("decisive", [0, 5]),))

    assert len(BARS.findall(chart)) == 1


def test_a_line_runs_through_every_value_with_a_band_between_the_low_and_the_high() -> None:
    chart = SvgChartMapper().line("Plies", ["round 1", "round 2"], [30.0, 60.0], [(10, 40), (20, 80)])

    (line,) = re.findall(r'<polyline points="([^"]+)"', chart)
    (band,) = re.findall(r'<polygon points="([^"]+)"', chart)
    assert len(line.split()) == 2
    assert len(band.split()) == 4  # Two highs, then the two lows back the other way.
    first, second = (float(point.split(",")[1]) for point in line.split())
    assert second < first  # The larger value is drawn higher up.
    assert "round 2: 60" in chart


def test_the_largest_value_and_the_first_and_last_columns_are_written_on_the_chart() -> None:
    chart = SvgChartMapper().stacked("Games", ["round 1", "round 2", "round 3"], (("decisive", [1, 2, 7]),))

    assert ">7<" in chart
    assert ">round 1<" in chart and ">round 3<" in chart and ">round 2<" not in chart


def test_a_chart_of_nothing_is_still_a_chart() -> None:
    chart = SvgChartMapper().stacked("Games", [], ())

    assert "<svg" in chart and BARS.findall(chart) == []


def test_what_a_chart_says_is_escaped() -> None:
    chart = SvgChartMapper().stacked("Games & <b>", ["<round>"], (("won & lost", [1]),))

    assert "<b>" not in chart.replace("<b>", "", 0) or "&amp;" in chart
    assert "&lt;round&gt;" in chart and "&amp;" in chart
