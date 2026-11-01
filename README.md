# femto-nx5D

Auswertung von Messungen an der FemtoMAX-Beamline (MAX IV): Sardana-Scans (`scan-N.h5` mit
`scan-N_pilatus.h5` und `scan-N_oscc_02_maui.h5`) → Bragg-Peak-Dynamik nach Laseranregung.

| Modul | Inhalt |
|---|---|
| `femto_nx5d.pipeline` | FemtoMAX → [nx5d](https://pypi.org/project/nx5d/): Scan-Katalog, Serien-Erkennung, Ingest (paralleler Hintergrundprozess, liest nur die ROI), Spice, Recipes, Auswertung von Fluenz- und Temperaturserien (Wiederholungen, Drift-Korrektur, q_x = q_y = 0) |
| `femto_nx5d.figures` | Abbildungen im Paper-Stil: Linien, Karten mit/ohne Konturen, Wasserfall, Peak-Spur, Wiederholungen, Meta-Plots vs. Fluenz und Temperatur, Insets |
| `femto_nx5d.feb` | Profil-Auswertung ohne nx5d: Suchpfade für Rohdaten und vorverarbeitete Profile, ROI-Leser (blockweise statt ganzer Pilatus-Stapel im Speicher), Status-Tabelle |
| `femto_nx5d.notebooks` | Notebook-Vorlage pro Probe für die nx5d-Pipeline |
| `femto_nx5d.scan_table` | kompakte Scan-Tabelle aller Master-Dateien (nur Skalare, ohne Bilder): `python -m femto_nx5d.scan_table <raw> <ziel>` |
| `femto_nx5d.minicopy` | verkleinerte Testkopie einzelner Scans (Master, ROI der ersten Bilder, Ping-Werte): `python -m femto_nx5d.minicopy <raw> <ziel> SCAN ...` |

## Installation

```bash
pip install --user "git+https://github.com/friweber/femto-nx5D"            # Grundpaket
pip install --user "femto-nx5d[nx5d] @ git+https://github.com/friweber/femto-nx5D"   # mit nx5d + xraydc
```

Aktualisieren: dasselbe mit `--upgrade --force-reinstall --no-deps`. Danach den Kernel neu starten.

## Verwendung

### nx5d-Pipeline
`examples/femtomax_config.example.json` als `femtomax_config.json` neben die Notebooks kopieren und
anpassen (Rohdaten-Ordner, Proposal-Ordner, ROI, Geometrie pro Probe). Dann:

```bash
python -m femto_nx5d.notebooks . SampleA,SampleB    # ein Notebook pro Probe
femto-nx5d catalog                                   # Scan-Katalog (liest nur Master-Dateien)
femto-nx5d ingest --sample SampleA --series fluence --jobs 8
```

```python
from femto_nx5d import pipeline as fmx, figures as ff
cfg = fmx.load_config("femtomax_config.json")
```

### Strahlzeit Feb 2025 (Schichtproben, ein Peak pro Schicht)
Auf dem Hub, im Arbeitsordner:

```bash
femto-nx5d setup-feb --raw <rohdaten-ordner> --out .
```

legt `femtomax_config.json`, ein `<Set>_analysis.ipynb` pro Proben-Set und `Alle_Proben.ipynb` an.
- Die Proben-Sets sind standardmäßig die korrigierten Scanlisten der 8 Paper-Proben bei 400 nm
  (`femto_nx5d/data/feb2025_scan_lists.json`). Eigene Listen `measurement_file_<Set>.txt` haben Vorrang
  (`--scan-files`, `<out>/scan_files`, `--search-share`).
- Die RSM-Scans kommen aus `which_sample.csv`.
- Die Geometrie je Scan (Mittelpixel, Austrittswinkel) wird aus den vorhandenen `process/scan-N_roi_profile.nc`
  gelesen, damit die q-Skala der bisherigen Auswertung erhalten bleibt.

Unterschiede zur Okt-Geometrie (alles in der Config):
- `scatter_axis: cols`: Der Detektor ist um 90° gedreht.
- `theta_motor: e_gon01_keta`.
- `geometry: fixed_exit` mit q_z = k (sin a_in + sin a_out).
- `ping_t0_ps` gilt abschnittsweise je Scanbereich.

Ausgewertet wird mit `loop_mode: merge`:
1. Die Drift jedes Scans wird aus seinem Profil vor t0 bestimmt.
2. Die Profile werden verschoben.
3. Danach werden alle Scans zusammen gebinnt und gefittet.

**Alles fürs Paper in einem Schritt:** `python -m femto_nx5d.pipeline paper` (bzw. `fmx.paper_export(cfg)`):
- Auswertung aller Paper-Sets mit q_x = q_y = 0 (|q| statt q_z-Komponente, `qx_qy_zero`)
- Export im alten Format
- RSMs mit Peaklage in q_x und Kippung (`results/rsm/`)
- `paper_summary.json` und die zip-Datei zum Herunterladen

`pipeline.export_set` schreibt die Ergebnisse im Format der bisherigen `data_export`-Ordner
(`binned_intensity_numpy_style.dat`, `results_export_*.dat`).

### Profil-Auswertung (`feb`)
`examples/feb_config.example.json` als `feb_config.json` in den Ordner der Notebooks kopieren:

```python
import femto_nx5d.feb as P
P.raw("scan-123456.h5")                       # erster Treffer in raw_dirs
P.processed("scan-123456_roi_profile.nc")     # eigener process-Ordner zuerst, dann die übrigen
images, x, y, overview = P.read_roi(123456, n_frames, roi_x=(500, 1000), roi_y=(100, 700), rotate=True)
P.status([123456, 123457])                    # Rohdaten / Profil vorhanden?
```

Dateien, die vor weniger als 10 min geändert wurden, gelten als noch in Übertragung (`P.MIN_AGE_S`).

## Hinweise
- Configs (`femtomax_config.json`, `feb_config.json`), Rohdaten und Ergebnisse gehören nicht ins Repo
  (siehe `.gitignore`).
- Getestet mit Python 3.12/3.13, nx5d 3.2.1, xraydc 0.4.1.
