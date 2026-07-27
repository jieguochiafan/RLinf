from __future__ import annotations

from pathlib import Path

import numpy as np

from tools import plot_cpu_core_util_timeline as timeline


class _FakeSpine:
    def set_visible(self, visible: bool) -> None:
        self.visible = visible


class _FakeAxis:
    def set_major_locator(self, locator) -> None:
        self.locator = locator

    def set_major_formatter(self, formatter) -> None:
        self.formatter = formatter


class _FakeFrame:
    def set_edgecolor(self, color: str) -> None:
        self.edgecolor = color

    def set_linewidth(self, linewidth: float) -> None:
        self.linewidth = linewidth


class _FakeLegend:
    def __init__(self) -> None:
        self.frame = _FakeFrame()

    def get_frame(self) -> _FakeFrame:
        return self.frame


class _FakeAxes:
    def __init__(self) -> None:
        self.fills = []
        self.plot_labels = []
        self.spines = {"top": _FakeSpine(), "right": _FakeSpine()}
        self.xaxis = _FakeAxis()

    def fill_between(self, *args, **kwargs) -> None:
        self.fills.append((args, kwargs))

    def plot(self, *args, **kwargs) -> None:
        self.plot_labels.append(kwargs.get("label"))

    def axvline(self, *args, **kwargs) -> None:
        pass

    def text(self, *args, **kwargs) -> None:
        pass

    def set_ylim(self, *args, **kwargs) -> None:
        pass

    def set_ylabel(self, *args, **kwargs) -> None:
        pass

    def grid(self, *args, **kwargs) -> None:
        pass

    def legend(self, *args, **kwargs) -> _FakeLegend:
        return _FakeLegend()

    def set_xlim(self, *args, **kwargs) -> None:
        pass

    def tick_params(self, *args, **kwargs) -> None:
        pass


class _FakeFigure:
    def __init__(self) -> None:
        self.saved_paths = []

    def autofmt_xdate(self, *args, **kwargs) -> None:
        pass

    def tight_layout(self) -> None:
        pass

    def savefig(self, path: Path, *args, **kwargs) -> None:
        self.saved_paths.append(path)


def test_plot_cpu_core_util_draws_only_mean_line(monkeypatch, tmp_path) -> None:
    axes = _FakeAxes()
    figure = _FakeFigure()

    monkeypatch.setattr(timeline.plt, "subplots", lambda *args, **kwargs: (figure, axes))
    monkeypatch.setattr(timeline.plt, "close", lambda fig: None)

    times = np.array([1_700_000_000.0, 1_700_000_001.0], dtype=float)
    timeline.plot_cpu_core_util(
        times=times,
        mean_util=np.array([10.0, 20.0], dtype=float),
        p90_util=np.array([40.0, 50.0], dtype=float),
        max_util=np.array([90.0, 100.0], dtype=float),
        phases=[],
        output_prefix=tmp_path / "cpu_core_util_timeline",
        smooth_window_s=0.0,
        x_pad_s=0.0,
    )

    assert axes.plot_labels == ["Mean"]
    assert axes.fills == []
