from collections.abc import Sequence
from html import escape

from openmind.dashboard.constant.dashboard_constant import (
    CHART_BAND,
    CHART_COLOURS,
    CHART_HEIGHT,
    CHART_PAD,
    CHART_WIDTH,
)


class SvgChartMapper:
    """Draws a training's numbers as SVG, written into the page itself.

    Inline rather than a chart library: the page is served on a tailnet and left open for days, so it fetches nothing.
    Every chart is the same shape — a band per column along the bottom, values growing upwards, the largest value
    labelled — so they can be read side by side."""

    def stacked(self, title: str, columns: Sequence[str], parts: Sequence[tuple[str, Sequence[float]]]) -> str:
        """Bars, one per column, each stacked from the parts in order: decisive and drawn games, or how games ended."""
        totals = [sum(part[1][index] for part in parts if index < len(part[1])) for index in range(len(columns))]
        highest = max(totals, default=0.0)
        bars: list[str] = []
        for index, column in enumerate(columns):
            bottom = 0.0
            for place, (name, values) in enumerate(parts):
                value = values[index] if index < len(values) else 0.0
                if value <= 0:
                    continue
                bars.append(self._bar(index, len(columns), bottom, value, highest, CHART_COLOURS[place % len(CHART_COLOURS)], f"{column}: {name} {self._number(value)}"))
                bottom += value
        return self._chart(title, columns, bars, highest, tuple(name for name, _ in parts))

    def line(self, title: str, columns: Sequence[str], values: Sequence[float], band: Sequence[tuple[float, float]] = ()) -> str:
        """A line through one value per column, with a band between a low and a high where one is given: the mean plies
        of a round's games, between its shortest game and its longest."""
        highest = max([*values, *(high for _, high in band)], default=0.0)
        drawn: list[str] = []
        if band:
            over = " ".join(self._point(index, len(columns), high, highest) for index, (_, high) in enumerate(band))
            under = " ".join(
                self._point(index, len(columns), low, highest) for index, (low, _) in reversed(list(enumerate(band)))
            )
            drawn.append(f'<polygon points="{over} {under}" fill="{CHART_BAND}" />')
        points = " ".join(self._point(index, len(columns), value, highest) for index, value in enumerate(values))
        drawn.append(f'<polyline points="{points}" fill="none" stroke="{CHART_COLOURS[0]}" stroke-width="2" />')
        for index, value in enumerate(values):
            place = self._point(index, len(columns), value, highest).split(",")
            drawn.append(
                f'<circle cx="{place[0]}" cy="{place[1]}" r="3" fill="{CHART_COLOURS[0]}">'
                f"<title>{escape(f'{columns[index]}: {self._number(value)}')}</title></circle>"
            )
        return self._chart(title, columns, drawn, highest, ())

    def _chart(self, title: str, columns: Sequence[str], drawn: Sequence[str], highest: float, names: Sequence[str]) -> str:
        legend = " ".join(
            f'<span class="key"><i style="background:{CHART_COLOURS[place % len(CHART_COLOURS)]}"></i>{escape(name)}</span>'
            for place, name in enumerate(names)
        )
        labels = self._labels(columns)
        return (
            f'<figure class="chart"><figcaption>{escape(title)}<span class="legend">{legend}</span></figcaption>'
            f'<svg viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" role="img" aria-label="{escape(title)}">'
            f'<line x1="{CHART_PAD}" y1="{CHART_HEIGHT - CHART_PAD}" x2="{CHART_WIDTH - CHART_PAD}" '
            f'y2="{CHART_HEIGHT - CHART_PAD}" stroke="#999" />'
            f'<text x="{CHART_PAD}" y="{CHART_PAD - 4}" class="highest">{escape(self._number(highest))}</text>'
            f"{''.join(drawn)}{labels}</svg></figure>"
        )

    def _labels(self, columns: Sequence[str]) -> str:
        """The first and last column named under the axis; naming every one would crowd a round a day."""
        if not columns:
            return ""
        shown = {0, len(columns) - 1}
        return "".join(
            f'<text x="{self._middle(index, len(columns)):.1f}" y="{CHART_HEIGHT - 4}" class="column" '
            f'text-anchor="middle">{escape(columns[index])}</text>'
            for index in sorted(shown)
        )

    def _bar(self, index: int, columns: int, bottom: float, value: float, highest: float, colour: str, hint: str) -> str:
        width = self._width(columns)
        left = self._middle(index, columns) - width / 2
        top = self._height(bottom + value, highest)
        height = self._height(bottom, highest) - top
        return (
            f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{max(height, 0.5):.1f}" fill="{colour}">'
            f"<title>{escape(hint)}</title></rect>"
        )

    def _point(self, index: int, columns: int, value: float, highest: float) -> str:
        return f"{self._middle(index, columns):.1f},{self._height(value, highest):.1f}"

    def _middle(self, index: int, columns: int) -> float:
        room = CHART_WIDTH - 2 * CHART_PAD
        return CHART_PAD + room * (index + 0.5) / max(columns, 1)

    def _width(self, columns: int) -> float:
        return max((CHART_WIDTH - 2 * CHART_PAD) / max(columns, 1) * 0.7, 1.0)

    def _height(self, value: float, highest: float) -> float:
        room = CHART_HEIGHT - 2 * CHART_PAD
        return CHART_HEIGHT - CHART_PAD - (room * value / highest if highest > 0 else 0.0)

    def _number(self, value: float) -> str:
        """A number as the page shows it: whole where it is whole, two decimals where it isn't."""
        return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"
