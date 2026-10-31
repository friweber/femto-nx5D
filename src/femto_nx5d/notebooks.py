"""
Notebook-Vorlage pro Probe für die nx5d-Pipeline.

    python -m femto_nx5d.notebooks <zielordner> <Probe1,Probe2,...>
"""
import sys
from pathlib import Path


def make(sample, outdir="."):
    '''Schreibt <outdir>/<sample>_analysis.ipynb (RSM, Fluenz-/Temperaturserien, Abbildungen).'''
    import nbformat as nbf
    nb = nbf.v4.new_notebook(); C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: C.append(nbf.v4.new_code_cell(s))
    md(f"""# {sample}: RSM und Fluenzserien (FemtoMAX, nx5d)

1. **Katalog und Serien**: Die Serien werden aus dem Scan-Katalog erkannt. Eine Fluenzserie ist eine Starttemperatur (und Probenstelle) mit mindestens 4 Pump-Energien. Scans mit gleicher Pump-Energie (verschiedene Delay-Bereiche, Wiederholungen) werden zu einer Kurve zusammengeführt.
2. **Ingest**: Die benötigten Scans (RSM + Fluenzserien, ~1–3 TB Rohdaten) werden im Hintergrund umgewandelt. **Das dauert Stunden**; die Abschnitte 3–4 werten jeweils aus, was schon fertig ist, also später einfach erneut ausführen.
3. **RSM**: Der statische RSM liefert die Peakposition (Startwert für das qz-Fitfenster) und einen Vorschlag für den `theta_offset`.
4. **Fluenzserien**: Peakfit, Strain vs. Delay pro Temperatur, Zusammenfassung vs. Fluenz. Ergebnis als NetCDF in `results/`.
5. *(optional)* Temperaturserien bei fester Fluenz.

Alle Einstellungen stehen in `femtomax_config.json`, der Code im Paket `femto_nx5d`. Das Notebook ist für alle Proben identisch bis auf `SAMPLE`.""")
    code(f"""SAMPLE = "{sample}"

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from femto_nx5d import pipeline as fmx

cfg = fmx.load_config(os.environ.get("FMX_CONFIG", "femtomax_config.json"))
fmx.check_raw_dirs(cfg)
RESULTS = Path(cfg["_config_dir"]) / cfg.get("results_dir", "results")
RESULTS.mkdir(exist_ok=True)""")
    md("## 1 Katalog und Serien")
    code("""cat = fmx.catalog(cfg)          # einmalig ~1-2 min, danach aus der Katalog-Datei (catalog_file)
fmx.series_overview(cat, SAMPLE)  # Anzahl tRSS-Scans je (T, Probenstelle x) und Pump-Energie""")
    code("""RSM = fmx.rsm_scans(cat, SAMPLE)
FLUENCE = fmx.fluence_series(cat, SAMPLE)
# Auswahl einschränken, z. B. nur 80 K:  FLUENCE = {k: v for k, v in FLUENCE.items() if k[0] == 80}

print("RSM-Scans:", RSM)
for (T, x), by_pump in FLUENCE.items():
    scans = [s for v in by_pump.values() for s in v]
    print(f"{'T unbekannt' if T < 0 else f'{T:5.0f} K'}  x={x:+.1f}  pumps {sorted(by_pump)}  {len(scans)} Scans, "
          f"{cat.loc[scans, 'pilatus_GB'].sum():.0f} GB")""")
    md(f"""## 2 Ingest (Hintergrund)
Die Zelle startet die Umwandlung als eigenen Prozess: Er läuft weiter, wenn das Notebook geschlossen wird, und wird nicht doppelt gestartet. Er endet, wenn der JupyterHub-Server gestoppt wird; dann einfach erneut starten, fertige Scans werden übersprungen.
Reihenfolge: RSM, dann die kleinste Serie zuerst. Fortschritt: die Status-Zelle erneut ausführen oder im Terminal `tail -f ingest_{sample}.log`.
Stoppen: `fmx.stop_background_ingest(cfg, SAMPLE)`.""")
    code("""# Optional, einmalig (~1-2 min): Lesegeschwindigkeit auf dem Hub messen.
# Testet die Lesemodi und 2/4/8 parallele Prozesse auf einem langen Fluenz-Burst-Scan,
# den der Serien-Ingest nicht liest. Danach read_mode in der Config und JOBS unten setzen.
RUN_BENCHMARK = False
if RUN_BENCHMARK:
    bench_scan = cat[(cat["sample"] == SAMPLE) & (cat["type"] == "fRSS")]["pilatus_GB"].idxmax()
    fmx.benchmark_read(cfg, bench_scan, n_frames=60, parallel=(2, 4, 8))""")
    code("""needed = fmx.ingest_order(cat, SAMPLE, series="fluence")   # "all" = auch Temperaturserien
JOBS = 8          # parallele Prozesse; der Hub hat 128 Kerne, jeder Prozess liest ~140 MB/s (siehe Benchmark)

START_INGEST = True
if START_INGEST and any(not fmx.scan_file(cfg, SAMPLE, s).exists() for s in needed):
    fmx.start_background_ingest(cfg, SAMPLE, series="fluence", jobs=JOBS)
# Neu starten mit anderen Einstellungen: fmx.stop_background_ingest(cfg, SAMPLE), dann diese Zelle erneut""")
    code("""# Status (beliebig oft ausführen)
done = fmx.ingest_status(cfg, cat, SAMPLE, needed)""")
    md("""## 3 Statischer RSM
`repository()` nach jedem Ingest neu aufrufen, damit nx5d neue Scans sieht. Der rote Marker zeigt den Hauptpeak, dessen qz der Startwert für das Fitfenster der Serien ist.""")
    code("""repo = fmx.repository(cfg)
prop = fmx.open_proposal(repo, SAMPLE)     # None, solange nichts umgewandelt ist

QZ0 = None
if prop is not None and RSM and str(RSM[0]) in prop.all():
    rsm_run = prop.scan(key=str(RSM[0]))
    rsm, peak = fmx.analyze_rsm(rsm_run)
    fmx.plot_rsm(rsm, peak)
    plt.suptitle(f"{SAMPLE} RSM {RSM[0]}, T = {float(rsm_run.raw.temperature_a.mean()):.0f} K")
    QZ0 = peak["qz"]
    rsm.to_netcdf(RESULTS / f"{SAMPLE}_RSM_{RSM[0]}.nc")
    print("Hauptpeak:", {k: round(v, 4) for k, v in peak.items()})
elif prop is not None:
    print(f"RSM {RSM} noch nicht umgewandelt -> Fitfenster um das Maximum jeder Kurve")""")
    code("""# Vorschlag: theta_offset, mit dem der RSM-Peak bei qx = 0 liegt (symmetrischer Reflex).
# Nur übernehmen, wenn das physikalisch passt (Fehlschnitt? asymmetrischer Reflex?).
if QZ0 is not None:
    sug = fmx.suggest_theta_offset(rsm_run)
    print(f"theta_offset jetzt {sug['theta_offset']:.2f} (Peak qx={sug['peak']['qx']:+.3f}), "
          f"Vorschlag {sug['suggested_theta_offset']:.2f} (Peak qx={sug['peak_after']['qx']:+.3f}, "
          f"qz={sug['peak_after']['qz']:.4f})")

# Übernehmen (gilt ab dem ersten Scan dieser Probe):
# first = str(min(prop.all()))
# prop.spice.anchor(first, "offsets"); prop.spice.update(first, "offsets", theta=sug["suggested_theta_offset"])
# danach: repo = fmx.repository(cfg); prop = repo.proposal(key=SAMPLE) und RSM-Zelle neu ausführen""")
    md("""## 4 Fluenzserien
Pro Fluenz werden die Scans in **Wiederholungen** (aufeinanderfolgende Scannummern) aufgeteilt, jede für sich gebinnt und gefittet, auf ihre eigene Peakposition vor t₀ referenziert (**Drift-Korrektur**: der Peak wandert zwischen Wiederholungen auf dem Detektor) und dann gemittelt. Frames mit Ping-Ausreißern werden verworfen, Bins mit weniger als `MIN_FRAMES` Frames nicht gefittet, isolierte heiße Pixel maskiert.

`QX_QY_ZERO = True`: jede Detektorzeile bekommt q_z = 4π/λ·sin(2θ/2) aus ihrem Streuwinkel (q_x = q_y = 0, unabhängig vom Probenwinkel); `False`: volles q-Mapping mit xraydc.""")
    code("""QX_QY_ZERO = True          # q_x = q_y = 0: Profil über den Streuwinkel jeder Detektorzeile
DRIFT_CORRECTION = True    # Wiederholungen auf ihre eigene Peakposition vor t0 referenzieren
MIN_FRAMES = 5             # Bins mit weniger Frames werden nicht gefittet
OPTS = dict(qx_qy_zero=QX_QY_ZERO, drift_correction=DRIFT_CORRECTION, min_frames=MIN_FRAMES)""")
    code("""fl = fmx.analyze_fluence_series(prop, FLUENCE, cfg, qz_center=QZ0, **OPTS) if prop is not None else None
if fl is not None:
    fl.to_netcdf(RESULTS / f"{SAMPLE}_fluence_series.nc")
    import json, pandas as pd
    display(pd.DataFrame(json.loads(fl.attrs["log"])))   # Wiederholungen, Drift, verworfene Frames""")
    md("""## 5 Temperaturserien
Feste Pump-Energie, verschiedene Starttemperaturen (benötigt die Scans der Temperaturserien – Ingest mit `series="all"`).""")
    code("""TEMP = fmx.temperature_series(cat, SAMPLE)
for (p, x), by_T in TEMP.items():
    print(f"pump {p:g}, x={x:+.1f}: T = {sorted(by_T)}")

ts = fmx.analyze_temperature_series(prop, TEMP, cfg, qz_center=QZ0, **OPTS) if prop is not None and TEMP else None
if ts is not None:
    ts.to_netcdf(RESULTS / f"{SAMPLE}_temperature_series.nc")""")
    md("""## 6 Abbildungen (für das Paper)
Matplotlib, Spaltenbreite 3,4 in / 7 in, gespeichert als PDF + PNG in `results/figures/<Probe>/`. Jede Funktion zeichnet **ein** Panel (optional in ein vorhandenes `ax`) – zum späteren Zusammenstellen.

Zeitfenster: `lang` (−20 ps … 4 ns, symlog) für den Überblick, `kurz` (−5 … 60 ps, linear) für die magnetoelastische Dynamik – in `ff.WINDOWS` anpassbar. Farben: Fluenz = Blau-Skala, Temperatur = Orange-Skala, Dehnungskarten divergierend (blau = Kontraktion, rot = Expansion).""")
    code("""from femto_nx5d import figures as ff
ff.use_style()
FIG = RESULTS / "figures" / SAMPLE
# ff.WINDOWS["kurz"]["range"] = (-5, 80)     # Zeitfenster anpassen
# Überblick je Starttemperatur: Linien + Karte (mit Konturlinien), lang und kurz
if fl is not None:
    for ser in fl.series.values:
        f = ff.fig_overview(fl, ser, contours=True)
        ff.save(f, f"overview_{ser}".replace(" ", "_"), FIG)""")
    code("""# Einzelpanels zum Weiterbearbeiten (Beispiele; Serie/Fluenz anpassen)
SER, P = None, None
if fl is not None:
    # Serie mit den meisten Fluenzen; davon die zweithöchste Fluenz (anpassen)
    n_valid = np.isfinite(fl.strain).any("delay").sum("pump")
    SER = str(fl.series.values[int(n_valid.argmax())])
    valid = fl.pump.values[np.isfinite(fl.strain.sel(series=SER)).any("delay").values]
    P = float(valid[-2] if len(valid) > 1 else valid[-1])
if SER is not None:
    ff.save(ff.fig_lines(fl, SER, "kurz"), f"lines_short_{SER}".replace(" ", "_"), FIG)
    ff.save(ff.fig_map(fl, SER, "kurz", contours=False), f"map_short_nocontour_{SER}".replace(" ", "_"), FIG)
    ff.save(ff.fig_waterfall(fl, SER, P, "lang"), f"waterfall_{SER}_{P:g}".replace(" ", "_"), FIG)
    ff.save(ff.fig_snail(fl, SER, P, "kurz"), f"snail_{SER}_{P:g}".replace(" ", "_"), FIG)
    ff.save(ff.fig_loops(fl, SER, P, "kurz"), f"loops_{SER}_{P:g}".replace(" ", "_"), FIG)""")
    code("""# Meta-Plots: Dehnung in Zeitfenstern vs. Fluenz und vs. Starttemperatur
WIN = {"Expansion": (8, 20), "Plateau": (500, 2000)}
if fl is not None:
    ff.save(ff.fig_vs_fluence(fl, WIN), "vs_fluence", FIG)
if ts is not None:
    ff.save(ff.fig_vs_temperature(ts, WIN), "vs_temperature", FIG)
    for ser in ts.series.values:
        ff.save(ff.fig_map(ts, ser, "kurz"), f"temperature_map_{ser}".replace(" ", "_"), FIG)""")
    code("""# Insets als eigenständige kleine Abbildungen
if SER is not None:
    ff.save(ff.fig_inset_profiles(fl, SER, P), "inset_profiles", FIG)
rsm_file = RESULTS / f"{SAMPLE}_RSM_{RSM[0]}.nc" if RSM else None
if rsm_file is not None and rsm_file.exists():
    import xarray as xr
    ff.save(ff.fig_inset_rsm(xr.open_dataset(rsm_file)), "inset_rsm", FIG)
print("Abbildungen in", FIG)""")
    nb["cells"] = C
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = Path(outdir) / f"{sample}_analysis.ipynb"
    nbf.write(nb, str(path))
    return path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    for s in sys.argv[2].split(","):
        print(make(s, sys.argv[1]))


# ---------------------------------------------------------------------------
# Feb 2025: sample sets with one peak per layer (femto-nx5d setup-feb)

_FEB_SETUP = '''import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from femto_nx5d import pipeline as fmx, figures as ff

cfg = fmx.load_config(os.environ.get("FMX_CONFIG", "femtomax_config.json"))
fmx.check_raw_dirs(cfg)
RESULTS = Path(cfg["_config_dir"]) / cfg.get("results_dir", "results")
RESULTS.mkdir(exist_ok=True)
ff.use_style()

# Auswerte-Optionen (wie bei den Seltenen Erden):
OPTS = dict(
    drift_correction=True,   # Profile jedes Scans um seine Peakverschiebung vor t0 korrigieren (Drift)
    min_frames=3,            # Delay-Bins mit weniger Bildern nicht fitten (pro Scan ~10 Bilder je Delay)
    delay_tol_ps=3.0,        # Bilder mit Ping-Ausreißern (> 3 ps vom Sollwert-Median) verwerfen
    full_profiles=True,      # gebinnte q_z-Profile behalten (Karten, Export im alten Format)
)'''


def make_feb(name, outdir="."):
    """Notebook for one sample set of the Feb 2025 beamtime."""
    import nbformat as nbf
    nb = nbf.v4.new_notebook(); C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: C.append(nbf.v4.new_code_cell(s))
    md(f"""# {name}: Dehnung pro Schicht (FemtoMAX Feb 2025, nx5d)

1. **Scans**: Gruppen (Scans je Winkel) und Peaks aus `femtomax_config.json` (`sets["{name}"]`).
2. **Ingest**: Die Scans werden im Hintergrund ins nx5d-Format umgewandelt (läuft weiter, wenn das Notebook
   geschlossen wird). Die Abschnitte 3–5 werten aus, was schon fertig ist – später einfach erneut ausführen.
3. **RSM** eines Winkel-Scans der Probe: Lage der Peaks.
4. **Auswertung**: pro Scan die Peaklage vor t0 (Drift), Profile aller Scans um die Drift verschoben,
   zusammen gebinnt (wie bisher) und pro Delay gefittet: Gauß-Zentrum und Schwerpunkt (COM) → Dehnung.
   Label `Ni@Au` = Ni-Peak in den Scans am Au-Winkel.
5. **Abbildungen** und **Export im Format der bisherigen Auswertung** (`data_export`-Ordner), damit die
   Paper-Skripte (`Analysis/strain_extraction.py` …) die neuen Daten direkt lesen.

Zeitnullpunkt, Fitfenster oder Peaks ändern: in `femtomax_config.json` unter `sets["{name}"]`
(`delay_offset`, `peaks`) und Abschnitt 4 erneut ausführen; die Bilder werden dabei nicht neu gelesen.""")
    code(f'SET = "{name}"\n\n' + _FEB_SETUP)
    md("## 1 Scans")
    code("""cat = fmx.catalog(cfg)          # beim ersten Mal einige Minuten (liest alle Master-Dateien), danach catalog.csv
display(fmx.set_overview(cfg, SET))
print("Peaks:", {l: (p["element"], p["group"], p["qz"]) for l, p in cfg["sets"][SET]["peaks"].items()})""")
    md("## 2 Ingest (Hintergrund)")
    code("""JOBS = 6                 # parallele Prozesse (der Hub hat 8 Kerne)
START_INGEST = True

order = fmx.ingest_order(cat, SET, cfg=cfg)
print(f"{len(order)} Scans, {cat['pilatus_GB'].reindex(order).sum():.0f} GB Rohdaten")
if START_INGEST:
    fmx.start_background_ingest(cfg, SET, series="all", jobs=JOBS)""")
    code("""# Fortschritt (Zelle erneut ausführen); stoppen: fmx.stop_background_ingest(cfg, SET)
done = fmx.ingest_status(cfg, cat, SET, order)""")
    md("## 3 RSM")
    code("""repo = fmx.repository(cfg)
prop = fmx.open_proposal(repo, cfg["proposal"])
rsm = [s for s in cfg["sets"][SET]["rsm_scans"] if prop is not None and str(s) in prop.all()]
if rsm:
    m, pk = fmx.rsm_map(prop.scan(key=str(rsm[0])))
    fig = ff.fig_rsm_map(m, pk, title=f"{SET}: RSM Scan {rsm[0]}")
    for l, p in cfg["sets"][SET]["peaks"].items():
        if "@" not in l:
            fig.axes[0].axhline(p["qz"], color=ff.ELEMENT_COLOR.get(p["element"], "k"), lw=0.6, ls=":")
    print("Maximum:", pk)
else:
    print("Noch kein RSM-Scan umgewandelt:", cfg["sets"][SET]["rsm_scans"] or "keiner in which_sample.csv gefunden")""")
    md("## 4 Auswertung")
    code("""res = fmx.analyze_set(prop, cfg, SET, **OPTS) if prop is not None else {}
if "_log" in res:
    display(res["_log"])
OUT = RESULTS / SET
OUT.mkdir(parents=True, exist_ok=True)""")
    md("## 5 Abbildungen und Export")
    code("""FIG = RESULTS / "figures" / SET
if [l for l in res if not l.startswith("_")]:
    fig = ff.fig_set_overview(res, title=SET); ff.save(fig, f"{SET}_overview", FIG)
    for var in ("strain_com", "strain"):
        for w in ("lang", "kurz"):
            fig = ff.fig_set_lines(res, w, var=var, title=SET); ff.save(fig, f"{SET}_{var}_{w}", FIG)
    plt.show()
else:
    print("Noch nichts auszuwerten (Ingest abwarten).")""")
    code("""# pro Peak: Gauß vs. Schwerpunkt, Karte, Profile
for l in [l for l in res if not l.startswith("_")]:
    fig, ax = plt.subplots(1, 3, figsize=(ff.COL2 * 1.3, 2.4), constrained_layout=True)
    ff.fig_set_compare(res[l], "lang", ax=ax[0])
    ff.fig_set_map(res[l], "lang", ax=ax[1])
    ff.fig_set_profiles(res[l], ax=ax[2])
    ff.save(fig, f"{SET}_{l.replace('@', '_at_')}_detail", FIG)
    plt.show()""")
    code("""# Export im Format von Eval_by_Project/3. NiAu/data_export (für die Paper-Skripte)
if [l for l in res if not l.startswith("_")]:
    for d in fmx.export_set(res, cfg, SET, RESULTS / "paper_export"):
        print("geschrieben:", d)""")
    nb.cells = C
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = Path(outdir) / f"{name}_analysis.ipynb"
    nbf.write(nb, str(path))
    return path


def make_feb_overview(outdir="."):
    """Notebook for all paper sample sets at once (sets with "paper": true)."""
    import nbformat as nbf
    nb = nbf.v4.new_notebook(); C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: C.append(nbf.v4.new_code_cell(s))
    md("""# Alle Paper-Proben (FemtoMAX Feb 2025, nx5d)

Ingest aller Paper-Scans im Hintergrund, dann für jede Probe die Dehnung pro Schicht, der Export im Format
der bisherigen Auswertung (`results/paper_export/<Set>_<Gruppe>/`, Eingabe für die Paper-Skripte) und eine
Übersicht aller Proben. Einzelheiten pro Probe: `<Set>_analysis.ipynb`.""")
    code(_FEB_SETUP + '''

SETS = [n for n, st in cfg["sets"].items() if st.get("paper")]
print("Paper-Proben:", SETS)''')
    md("## 1 Katalog und Ingest")
    code("""cat = fmx.catalog(cfg)
JOBS = 6
START_INGEST = True
order = fmx.ingest_order(cat, "paper", cfg=cfg)
print(f"{len(order)} Scans, {cat['pilatus_GB'].reindex(order).sum():.0f} GB Rohdaten")
if START_INGEST:
    fmx.start_background_ingest(cfg, "paper", series="all", jobs=JOBS)""")
    code("""done = fmx.ingest_status(cfg, cat, "paper", order)""")
    md("## 2 Auswertung aller Proben")
    code("""repo = fmx.repository(cfg)
prop = fmx.open_proposal(repo, cfg["proposal"])
ALL, LOGS = {}, []
for name in SETS:
    r = fmx.analyze_set(prop, cfg, name, **OPTS) if prop is not None else {}
    ALL[name] = r
    if "_log" in r:
        LOGS.append(r["_log"].assign(Set=name))
    if [l for l in r if not l.startswith("_")]:
        fmx.export_set(r, cfg, name, RESULTS / "paper_export")
if LOGS:
    import pandas as pd
    display(pd.concat(LOGS).reset_index().set_index(["Set", "Peak"]))""")
    md("## 3 Übersicht")
    code("""FIG = RESULTS / "figures" / "Alle_Proben"
for var in ("strain_com", "strain"):
    for w in ("lang", "kurz"):
        fig = ff.fig_sets_grid(ALL, var=var, window=w)
        ff.save(fig, f"alle_{var}_{w}", FIG)
plt.show()""")
    md("""## 4 Paper-Skripte mit den neuen Daten

```bash
UXRD_DATA=<dieser Ordner>/results/paper_export python Analysis/strain_extraction.py
```
Die Ordnernamen (`NiAu100_400_Au`, …) und Peak-Namen entsprechen denen der bisherigen `data_export`-Ordner.""")
    nb.cells = C
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = Path(outdir) / "Alle_Proben.ipynb"
    nbf.write(nb, str(path))
    return path
