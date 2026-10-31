# femto-nx5D

Auswertung von Messungen an der FemtoMAX-Beamline (MAX IV): Sardana-Scans (`scan-N.h5` mit
`scan-N_pilatus.h5` und `scan-N_oscc_02_maui.h5`) → Bragg-Peak-Dynamik nach Laseranregung.

| Modul | Inhalt |
|---|---|
| `femto_nx5d.pipeline` | FemtoMAX → [nx5d](https://pypi.org/project/nx5d/): Scan-Katalog, Serien-Erkennung, Ingest (paralleler Hintergrundprozess, liest nur die ROI), Spice, Recipes, Auswertung von Fluenz- und Temperaturserien (Wiederholungen, Drift-Korrektur, q_x = q_y = 0) |
| `femto_nx5d.figures` | Abbildungen im Paper-Stil: Linien, Karten mit/ohne Konturen, Wasserfall, Peak-Spur, Wiederholungen, Meta-Plots vs. Fluenz und Temperatur, Insets |
| `femto_nx5d.feb` | Profil-Auswertung ohne nx5d: Suchpfade für Rohdaten und vorverarbeitete Profile, ROI-Leser (blockweise statt ganzer Pilatus-Stapel im Speicher), Status-Tabelle |
| `femto_nx5d.notebooks` | Notebook-Vorlage pro Probe für die nx5d-Pipeline |

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
