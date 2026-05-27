"""Mini chart matplotlib pour affichage live des metrics observer."""
from __future__ import annotations
from collections import defaultdict, deque
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ark.ui import design_tokens as T


class MetricChart(FigureCanvasQTAgg):
    """Chart compact 4-row pour observer (mean_abs, eff_rank, intrinsic_dim, power_slope).

    Garde rolling buffer de N points (défaut 60), redessine sur set_metrics.
    Style sombre matchant design ARK.
    """

    DEFAULT_WIDTH = 240
    DEFAULT_HEIGHT = 140
    BUFFER_SIZE = 60

    METRIC_ORDER = ["eff_rank", "intrinsic_dim", "power_slope", "mean_abs"]
    METRIC_COLORS = {
        "eff_rank": T.ACCENT,
        "intrinsic_dim": T.SUCCESS,
        "power_slope": T.WARN,
        "mean_abs": T.FG_SECONDARY,
    }

    def __init__(self, parent=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        fig = Figure(figsize=(width / 100, height / 100), dpi=100,
                       facecolor=T.BG_PANEL)
        super().__init__(fig)
        self.setParent(parent)
        self.fig = fig
        self.setFixedSize(width, height)
        self._buffer: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.BUFFER_SIZE))
        self._setup_axes()

    def _setup_axes(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(T.BG_DEEP)
        for spine in self.ax.spines.values():
            spine.set_color(T.BORDER_DEFAULT)
            spine.set_linewidth(0.5)
        self.ax.tick_params(colors=T.FG_TERTIARY, labelsize=6, length=2)
        self.ax.grid(True, color=T.BORDER_SUBTLE, linewidth=0.3, alpha=0.5)
        self.fig.subplots_adjust(left=0.18, right=0.96, top=0.95, bottom=0.18)

    def push_metrics(self, metrics: dict):
        """Ajoute nouveau point + redraw."""
        for name in self.METRIC_ORDER:
            if name in metrics:
                v = metrics[name]
                if v != v:  # NaN
                    continue
                self._buffer[name].append(float(v))
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.set_facecolor(T.BG_DEEP)
        for spine in self.ax.spines.values():
            spine.set_color(T.BORDER_DEFAULT)
            spine.set_linewidth(0.5)
        self.ax.tick_params(colors=T.FG_TERTIARY, labelsize=6, length=2)
        self.ax.grid(True, color=T.BORDER_SUBTLE, linewidth=0.3, alpha=0.5)
        any_data = False
        for name in self.METRIC_ORDER:
            buf = self._buffer.get(name)
            if not buf or len(buf) < 1:
                continue
            xs = list(range(len(buf)))
            ys = list(buf)
            self.ax.plot(xs, ys, color=self.METRIC_COLORS[name],
                          linewidth=1.2, label=name)
            any_data = True
        if any_data:
            self.ax.legend(loc="upper left", fontsize=6,
                            facecolor=T.BG_PANEL, edgecolor=T.BORDER_DEFAULT,
                            labelcolor=T.FG_SECONDARY, framealpha=0.85,
                            handlelength=1.5, borderpad=0.3)
        else:
            self.ax.text(0.5, 0.5, "no data",
                          ha="center", va="center", color=T.FG_TERTIARY,
                          fontsize=8, transform=self.ax.transAxes)
        self.draw()

    def clear_buffer(self):
        self._buffer.clear()
        self._redraw()
