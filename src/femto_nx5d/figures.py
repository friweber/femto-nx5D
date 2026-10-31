"""
Paper figures for the FemtoMAX rare-earth analysis (matplotlib).

Every function draws one panel (optionally into a given `ax`) and returns the
figure, so panels can be combined freely or saved standalone (`save`).
Input: the datasets of femtomax_nx5d.analyze_fluence_series /
analyze_temperature_series (dims series, pump|temperature, delay[, q_rel]).

Colour conventions:
  fluence      -> one-hue blue ramp, light (low) to dark (high)
  temperature  -> one-hue orange ramp, light (cold) to dark (hot)
  strain maps  -> diverging blue (contraction) - grey (0) - red (expansion)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# --- style ---------------------------------------------------------------------
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
ORANGE_RAMP = ["#f5b896", "#f0925f", "#eb6834", "#c9521f", "#a03f15", "#7a2f0e"]
DIVERGING = LinearSegmentedColormap.from_list(
    "contract_expand", ["#0d366b", "#3987e5", "#f0efec", "#e66767", "#a51f1f"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ACCENT = "#eb6834"            # fitted peak position on top of blue maps
INK, INK2 = "#0b0b0b", "#52514e"

WINDOWS = {   # delay windows: the big picture and the magnetoelastic short-time range
    "lang": {"range": (-20, 4000), "xscale": "symlog", "linthresh": 20},
    "kurz": {"range": (-5, 60), "xscale": "linear"},
}

PAPER_RC = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "lines.linewidth": 1.2,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
}
COL1, COL2 = 3.4, 7.0        # single / double column width [inch]


def use_style():
    mpl.rcParams.update(PAPER_RC)


def ramp(colors, n):
    """n colours sampled along a one-hue ramp (light -> dark)."""
    cmap = LinearSegmentedColormap.from_list("r", colors)
    return [cmap(x) for x in (np.linspace(0, 1, n) if n > 1 else [0.7])]


def save(fig, name, outdir="results/figures", formats=("pdf", "png")):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for f in formats:
        fig.savefig(out / f"{name}.{f}")
    return [str(out / f"{name}.{f}") for f in formats]


def _ax(ax, size):
    if ax is None:
        fig, ax = plt.subplots(figsize=size, constrained_layout=True)
        return fig, ax
    return ax.figure, ax


def _xaxis(ax, window):
    w = WINDOWS[window] if isinstance(window, str) else window
    if w.get("xscale") == "symlog":
        ax.set_xscale("symlog", linthresh=w.get("linthresh", 20), linscale=0.6)
    ax.set_xlim(*w["range"])
    ax.set_xlabel("Delay (ps)")
    ax.axvline(0, color=INK2, lw=0.5, ls=":")


def _sel_delay(d, window):
    w = WINDOWS[window] if isinstance(window, str) else window
    return d.sel(delay=slice(*w["range"]))


def _empty(ax, msg="keine Daten in diesem Zeitfenster"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=INK2)
    ax.set_axis_off()
    return ax.figure


def _inner(ds):
    return ds.attrs.get("inner", "pump")


VAR_LABEL = {"strain": "Strain (‰)", "sigma": "Peak width σ (Å$^{-1}$)", "center": "q$_z$ (Å$^{-1}$)",
             "amplitude": "Peak amplitude (arb. u.)"}


# --- line plots ----------------------------------------------------------------
def fig_lines(ds, series, window="lang", var="strain", values=None, ax=None, band=True, title=None,
              shade_short=True):
    """var(delay) per pump (or temperature), one-hue ramp, with error band."""
    inner = _inner(ds)
    d = _sel_delay(ds.sel(series=series), window)
    vals = [v for v in d[inner].values if np.isfinite(d[var].sel({inner: v})).any()]
    if values is not None:
        vals = [v for v in vals if v in values]
    cols = ramp(BLUE_RAMP if inner == "pump" else ORANGE_RAMP, len(vals))
    fig, ax = _ax(ax, (COL1, 2.4))
    if not vals:
        ax.text(0.5, 0.5, "keine Daten in diesem Zeitfenster", ha="center", va="center",
                transform=ax.transAxes, color=INK2)
        ax.set_axis_off()
        return fig
    for v, c in zip(vals, cols):
        y = d[var].sel({inner: v})
        ok = np.isfinite(y)
        x = d.delay.values[ok]
        ax.plot(x, y.values[ok], color=c, label=f"{v:g}" + (" K" if inner == "temperature" else ""))
        if band and var in ("strain", "center") and "center_err" in d:
            err = d.center_err.sel({inner: v}).values[ok]
            if var == "strain":
                err = err / float(ds.sel(series=series).center.sel({inner: v}).mean()) * 1e3
            ax.fill_between(x, y.values[ok] - err, y.values[ok] + err, color=c, alpha=0.18, lw=0)
    if shade_short and window == "lang":
        ax.axvspan(*WINDOWS["kurz"]["range"], color="#f0efec", zorder=0, lw=0)
    ax.axhline(0, color=INK2, lw=0.5)
    _xaxis(ax, window)
    ax.set_ylabel(VAR_LABEL.get(var, var))
    label = "Pump (arb. u.)" if inner == "pump" else "T$_0$ (K)"
    if len(vals) > 4:
        # many curves: a discrete colour bar instead of a legend that hides the data
        cmap = mpl.colors.ListedColormap(cols)
        bounds = np.arange(len(vals) + 1)
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.BoundaryNorm(bounds, cmap.N), cmap), ax=ax,
                          pad=0.02, ticks=bounds[:-1] + 0.5, aspect=25)
        cb.ax.set_yticklabels([f"{v:g}" for v in vals])
        cb.ax.tick_params(length=0)
        cb.set_label(label)
    else:
        ax.legend(title=label, loc="best", handlelength=1.2)
    ax.set_title(title if title is not None else str(series), loc="left")
    return fig


# --- colour maps (with or without contour lines) -----------------------------
def fig_map(ds, series, window="lang", var="strain", contours=True, ax=None, title=None, levels=None):
    """var as colour over delay (x) and pump/temperature (y); optional contour lines."""
    inner = _inner(ds)
    d = _sel_delay(ds.sel(series=series), window)
    vals = np.array([v for v in d[inner].values if np.isfinite(d[var].sel({inner: v})).any()])
    fig, ax = _ax(ax, (COL1, 2.4))
    if len(vals) < 2:          # e.g. while the ingest is still running
        ax.text(0.5, 0.5, "zu wenige Kurven für eine Karte", ha="center", va="center",
                transform=ax.transAxes, color=INK2)
        ax.set_axis_off()
        return fig
    Z = np.vstack([d[var].sel({inner: v}).values for v in vals])
    x = d.delay.values
    keep = np.isfinite(Z).sum(axis=0) >= max(1, len(vals) // 2)
    x, Z = x[keep], Z[:, keep]
    # fill single missing bins along delay for a clean map
    for i in range(Z.shape[0]):
        ok = np.isfinite(Z[i])
        if ok.sum() > 1:
            Z[i] = np.interp(x, x[ok], Z[i][ok])
    ye = np.concatenate([[vals[0] - (vals[1] - vals[0]) / 2], (vals[1:] + vals[:-1]) / 2,
                         [vals[-1] + (vals[-1] - vals[-2]) / 2]]) if len(vals) > 1 else [vals[0] - .5, vals[0] + .5]
    xe = np.concatenate([[x[0]], (x[1:] + x[:-1]) / 2, [x[-1]]])
    if var == "strain":
        lim = np.nanpercentile(np.abs(Z), 99.5) or 1.0
        norm, cmap = TwoSlopeNorm(0, -lim, lim), DIVERGING
    else:
        norm, cmap = None, SEQUENTIAL
    pc = ax.pcolormesh(xe, ye, Z, cmap=cmap, norm=norm, shading="flat", rasterized=True)
    if contours and len(vals) > 1:
        # contours from lightly smoothed data (running mean over 9 delay bins) at round values
        from scipy.ndimage import uniform_filter1d
        Zs = uniform_filter1d(Z, size=9, axis=1, mode="nearest")
        if levels is None:
            loc = mpl.ticker.MaxNLocator(nbins=7, symmetric=(var == "strain"))
            lv = [l for l in loc.tick_values(np.nanmin(Zs), np.nanmax(Zs)) if l != 0 or var != "strain"]
        else:
            lv = levels
        cs = ax.contour(x, vals, Zs, levels=lv, colors=INK, linewidths=0.45)
        ax.clabel(cs, cs.levels[::2], fontsize=5, fmt="%g", inline_spacing=2)
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(VAR_LABEL.get(var, var))
    _xaxis(ax, window)
    ax.set_ylabel("Pump (arb. u.)" if inner == "pump" else "T$_0$ (K)")
    ax.set_title(title if title is not None else str(series), loc="left")
    return fig


# --- peak dynamics -------------------------------------------------------------
def fig_waterfall(ds, series, value, window="lang", ax=None, line=True, title=None):
    """Normalized q_z profiles over delay (colour) with the fitted peak position."""
    inner = _inner(ds)
    d = _sel_delay(ds.sel(series=series, **{inner: value}), window)
    P = d["profile"].values
    P = P / np.nanmax(P, axis=1, keepdims=True)
    ok = np.isfinite(P).any(axis=1)
    q_ref = float(ds.sel(series=series, **{inner: value}).center.sel(delay=slice(-20, -2)).mean())
    fig, ax = _ax(ax, (COL1, 2.4))
    if not ok.any() or not np.isfinite(q_ref):
        return _empty(ax, "keine Daten (oder keine Referenz vor t0)")
    y = (d.q_rel.values + _qwin(ds, series, inner, value) - q_ref) * 1e3
    pc = ax.pcolormesh(d.delay.values[ok], y, P[ok].T, cmap=SEQUENTIAL, shading="nearest",
                       rasterized=True, vmin=0, vmax=1)
    if line:
        ax.plot(d.delay.values, (d.center.values - q_ref) * 1e3, color=ACCENT, lw=1.0, label="Fit")
        ax.legend(loc="upper right")
    fig.colorbar(pc, ax=ax, pad=0.02).set_label("Intensität (norm.)")
    _xaxis(ax, window)
    ax.set_ylabel(r"q$_z$ − q$_{z,0}$ (10$^{-3}$ Å$^{-1}$)")
    ax.set_ylim(-25, 25)
    ax.set_title(title if title is not None else f"{series}, {value:g}", loc="left")
    return fig


def _qwin(ds, series, inner, value):
    """Absolute q of the profile window centre (stored per curve in q_window when available)."""
    if "q_window" in ds:
        return float(ds.q_window.sel(series=series, **{inner: value}))
    c = ds.sel(series=series, **{inner: value}).center
    return float(c.sel(delay=slice(-20, -2)).mean())


def fig_snail(ds, series, value, window="lang", ax=None, title=None):
    """'Schneckenspur': path of the peak in the (q_y, q_z) plane, coloured by delay."""
    inner = _inner(ds)
    d = _sel_delay(ds.sel(series=series, **{inner: value}), window)
    ref = ds.sel(series=series, **{inner: value}).sel(delay=slice(-20, -2))
    dqz = (d.center - float(ref.center.mean())).values * 1e3
    dqy = (d.qy_com - float(ref.qy_com.mean())).values * 1e3
    t = d.delay.values
    ok = np.isfinite(dqz) & np.isfinite(dqy)
    fig, ax = _ax(ax, (2.4, 2.2))
    if not ok.any():
        return _empty(ax)
    ax.plot(dqy[ok], dqz[ok], color="#c3c2b7", lw=0.6, zorder=1)
    w = WINDOWS[window] if isinstance(window, str) else window
    norm = (mpl.colors.SymLogNorm(linthresh=w.get("linthresh", 20), vmin=t[ok].min(), vmax=t[ok].max())
            if w.get("xscale") == "symlog" else mpl.colors.Normalize(t[ok].min(), t[ok].max()))
    sc = ax.scatter(dqy[ok], dqz[ok], c=t[ok], cmap=SEQUENTIAL, norm=norm, s=9, edgecolor=INK2,
                    linewidth=0.2, zorder=2)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Delay (ps)")
    ax.axhline(0, color=INK2, lw=0.5, ls=":")
    ax.axvline(0, color=INK2, lw=0.5, ls=":")
    ax.set_xlabel(r"Δq$_y$ (10$^{-3}$ Å$^{-1}$)")
    ax.set_ylabel(r"Δq$_z$ (10$^{-3}$ Å$^{-1}$)")
    ax.set_title(title if title is not None else f"{series}, {value:g}", loc="left")
    return fig


# --- summaries ("meta" plots) --------------------------------------------------
def window_mean(ds, delay_window, var="strain"):
    """Mean of var inside a delay window -> DataArray (series, inner)."""
    return ds[var].sel(delay=slice(*delay_window)).mean("delay")


def fig_vs_fluence(ds, delay_windows, var="strain", ax=None):
    """Strain in delay windows vs pump, one line per temperature (orange ramp); small multiples."""
    n = len(delay_windows)
    fig, axs = plt.subplots(1, n, figsize=(min(COL2, 2.2 * n + 0.4), 2.2), sharey=True,
                            constrained_layout=True, squeeze=False)
    temps = ds.temperature.values if "temperature" in ds.coords else np.arange(ds.sizes["series"])
    order = np.argsort(temps)
    cols = ramp(ORANGE_RAMP, len(order))
    for ax, (name, win) in zip(axs[0], delay_windows.items()):
        m = window_mean(ds, win, var)
        for c, i in zip(cols, order):
            y = m.isel(series=i)
            ok = np.isfinite(y)
            ax.plot(m.pump.values[ok], y.values[ok], "o-", color=c, ms=3.5,
                    label=str(ds.series.values[i]))
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_title(f"{name}: {win[0]:g}–{win[1]:g} ps", loc="left")
        ax.set_xlabel("Pump (arb. u.)")
    axs[0, 0].set_ylabel(VAR_LABEL.get(var, var))
    axs[0, -1].legend(loc="best")
    return fig


def fig_vs_temperature(ts, delay_windows, var="strain"):
    """Temperature plot: strain in delay windows vs start temperature, one line per pump."""
    n = len(delay_windows)
    fig, axs = plt.subplots(1, n, figsize=(min(COL2, 2.2 * n + 0.4), 2.2), sharey=True,
                            constrained_layout=True, squeeze=False)
    pumps = ts.pump.values if "pump" in ts.coords else np.arange(ts.sizes["series"])
    cols = ramp(BLUE_RAMP, len(pumps))
    for ax, (name, win) in zip(axs[0], delay_windows.items()):
        m = window_mean(ts, win, var)
        for c, i in zip(cols, np.argsort(pumps)):
            y = m.isel(series=i)
            ok = np.isfinite(y)
            ax.plot(m.temperature.values[ok], y.values[ok], "o-", color=c, ms=3.5,
                    label=str(ts.series.values[i]))
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_title(f"{name}: {win[0]:g}–{win[1]:g} ps", loc="left")
        ax.set_xlabel("T$_0$ (K)")
    axs[0, 0].set_ylabel(VAR_LABEL.get(var, var))
    axs[0, -1].legend(loc="best")
    return fig


# --- insets (standalone) -------------------------------------------------------
def fig_inset_profiles(ds, series, value, delays=(-5, 10, 30, 1000), size=(1.7, 1.4)):
    """Small standalone figure: q_z profiles at a few delays (for an inset)."""
    inner = _inner(ds)
    d = ds.sel(series=series, **{inner: value})
    q_ref = float(d.center.sel(delay=slice(-20, -2)).mean())
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    cols = ramp(BLUE_RAMP, len(delays))
    q = (d.q_rel.values + _qwin(ds, series, inner, value) - q_ref) * 1e3
    for t, c in zip(delays, cols):
        p = d.profile.sel(delay=t, method="nearest")
        ax.plot(q, p / np.nanmax(p), color=c, lw=1.0, label=f"{float(p.delay):.3g} ps")
    ax.set_xlabel(r"Δq$_z$ (10$^{-3}$ Å$^{-1}$)", fontsize=6)
    ax.set_yticks([])
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, handlelength=1)
    return fig


def fig_inset_rsm(rsm, size=(1.7, 1.6)):
    """Small standalone static RSM (q_x, q_z) from the saved <Probe>_RSM_<scan>.nc."""
    im = rsm["images"].fillna(0).sum("qy")
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    v = im.values.T
    ax.pcolormesh(im.qx.values, im.qz.values, np.log10(np.clip(v, v[v > 0].min(), None)),
                  cmap=SEQUENTIAL, shading="nearest", rasterized=True)
    ax.set_xlabel(r"q$_x$ (Å$^{-1}$)", fontsize=6)
    ax.set_ylabel(r"q$_z$ (Å$^{-1}$)", fontsize=6)
    ax.tick_params(labelsize=6)
    return fig


# --- overview figure ---------------------------------------------------------
def fig_overview(ds, series, contours=True):
    """2x2: lines and map over the long range (top) and the short range (bottom)."""
    fig, axs = plt.subplots(2, 2, figsize=(COL2, 4.6), constrained_layout=True)
    fig_lines(ds, series, "lang", ax=axs[0, 0], title=f"{series} – Überblick")
    fig_map(ds, series, "lang", contours=contours, ax=axs[0, 1], title="")
    fig_lines(ds, series, "kurz", ax=axs[1, 0], title="kurze Zeiten (magnetoelastisch)", shade_short=False)
    fig_map(ds, series, "kurz", contours=contours, ax=axs[1, 1], title="")
    return fig


def fig_loops(ds, series, value, window="kurz", ax=None, title=None):
    """Repetitions (loops) of one curve and their mean - how reproducible is a curve?"""
    inner = _inner(ds)
    d = _sel_delay(ds.sel(series=series, **{inner: value}), window)
    fig, ax = _ax(ax, (COL1, 2.2))
    loops = [i for i in d.loop.values if np.isfinite(d.strain_loop.sel(loop=i)).any()]
    if not loops:
        return _empty(ax)
    for i in loops:
        y = d.strain_loop.sel(loop=i)
        ok = np.isfinite(y)
        ax.plot(d.delay.values[ok], y.values[ok], color="#86b6ef" if i % 2 == 0 else "#c3c2b7", lw=0.8,
                label=f"Wiederholung {i + 1}")
    ok = np.isfinite(d.strain)
    ax.plot(d.delay.values[ok], d.strain.values[ok], color=INK, lw=1.3, label="Mittel")
    ax.axhline(0, color=INK2, lw=0.5)
    _xaxis(ax, window)
    ax.set_ylabel(VAR_LABEL["strain"])
    ax.legend(loc="best")
    ax.set_title(title if title is not None else f"{series}, {value:g}", loc="left")
    return fig
